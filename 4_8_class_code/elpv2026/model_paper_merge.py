"""LwNet 和 AdvEL-Net 模型实现，供 4class/train_merge_* 脚本调用。"""

import torch
from torch import nn
from torch.optim import Optimizer


class _LwBranch(nn.Module):
    """LwNet 并行分支：Conv + BN + LeakyReLU + MaxPool。"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class LwNet(nn.Module):
    """2024 Solar Energy 文献中的 LwNet。

    输入按论文使用 246x246。输出为 logits，直接用于 CrossEntropyLoss。
    """

    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        self.conv1_1 = _LwBranch(in_channels, 16, 3)
        self.conv1_2 = _LwBranch(in_channels, 32, 1)
        self.conv2_1 = _LwBranch(48, 8, 7)
        self.conv2_2 = _LwBranch(48, 8, 5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(32 * 61 * 61, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = torch.cat([self.conv1_1(x), self.conv1_2(x)], dim=1)
        x = torch.cat([self.conv2_1(x), self.conv2_2(x)], dim=1)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class _ConvBnRelu(nn.Module):
    """AdvEL-Net 基础卷积块。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CompactSE(nn.Module):
    """AdvEL-Net 文献中的 CSE：GAP + FC + Sigmoid。"""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.sigmoid(self.fc(weights)).view(b, c, 1, 1)
        return x * weights


class ExtendedDownsampling(nn.Module):
    """EDSM 三分支降采样。

    论文描述为 MaxPool、步长卷积和级联卷积分支相加。这里用深度可分离卷积
    控制参数量，同时保持输出尺寸与论文阶段表一致。
    """

    def __init__(self, channels):
        super().__init__()
        self.pool_branch = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cascade_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pool_branch(x) + self.conv_branch(x) + self.cascade_branch(x))


class _AdvELBlock(nn.Module):
    """AdvEL-Net 第 2-4 阶段：双卷积、CSE、残差相加、EDSM。"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = _ConvBnRelu(in_channels, mid_channels, kernel_size=3, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.cse = CompactSE(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ExtendedDownsampling(out_channels)

    def forward(self, x):
        identity = self.skip(x)
        features = self.conv2(self.conv1(x))
        x = self.relu(features + self.cse(features) + identity)
        return self.downsample(x)


class AdvELNet(nn.Module):
    """2026 Solar Energy 文献中的 AdvEL-Net。

    输入按论文使用 224x224。Softmax 由交叉熵损失内部处理，forward 返回 logits。
    """

    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        self.stage1 = nn.Sequential(
            _ConvBnRelu(in_channels, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = _AdvELBlock(64, 64, 64)
        self.stage3 = _AdvELBlock(64, 64, 128)
        self.stage4 = _AdvELBlock(128, 128, 256)
        self.stage5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class Lion(Optimizer):
    """AdvEL-Net 文献采用的 LION 优化器，可通过训练脚本开关启用/关闭。"""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
