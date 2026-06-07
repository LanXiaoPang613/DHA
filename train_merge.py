import os

from model import CNN7

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import model as M
from model import ModifiedVGG19,ModifiedResNet34
from sklearn.model_selection import train_test_split

if 1:
    from data_nas import ChallengeDataset
else:
    from data_nas import ChallengeDataset
from trainer_merge import Trainer
# from elpv_dataset.utils import load_dataset
import warnings
warnings.filterwarnings("ignore")
import random, os
import argparse
# dataset settings
parser = argparse.ArgumentParser('Main trainer')
parser.add_argument('--runpath', default='1', help='dataset')
# C:\Users\pc\Desktop\dataset\mini-imagenet
# parser.add_argument('--dataset_path', default=r'C:\Users\pc\Desktop\dataset\mini-imagenet', help='dataset path')
parser.add_argument('--num_class', default=2, type=int, help='classes')
#######################################################################################################################
# synthetic noise modes: for some datasets only
parser.add_argument('--backbone', type=str, default='vgg16', help='noise mode')
parser.add_argument('--merge_t', type=int, default=5, help='noise ratio')
parser.add_argument('--alpha', default=1., type=float, help='artifical noise ratio (default: 0.0)')
parser.add_argument('--lr', type=float, default=0.005, help='noise mode of mini-imagenet')

parser.add_argument('--gpuid', type=str, default='0', help='noise mode of mini-imagenet')

args = parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.cuda.set_device(int(args.gpuid))
seed_everything(42)
df = pd.read_csv('data.csv', sep=';')

class_map = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
stratify_labels = [class_map[(x, y)] for x, y in df[['crack', 'inactive']].to_numpy()]
train = None
test = None
# Set up data loading for the training and validation
train_dl = t.utils.data.DataLoader(ChallengeDataset(train, "train",args=args), batch_size=32, shuffle=True)
val_dl = t.utils.data.DataLoader(ChallengeDataset(test, "val",args=args), batch_size=32, shuffle=True)
# res_net = M.CNN7(n_outputs=2, dropout_rate=0.5)
# res_net2 = M.CNN7(n_outputs=2, dropout_rate=0.5)
if args.backbone == 'vgg16':
    res_net = tv.models.vgg16(pretrained=True)
    res_net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, args.num_class),
            )
    res_net2 = tv.models.vgg16(pretrained=True)
    res_net2.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, args.num_class),
    )
elif args.backbone == 'resnet18':
    res_net = tv.models.resnet18(pretrained=True)
    res_net.fc = nn.Linear(res_net.fc.in_features, args.num_class)
    res_net2 = tv.models.resnet18(pretrained=True)
    res_net2.fc = nn.Linear(res_net2.fc.in_features, args.num_class)
elif args.backbone == 'modified-vgg19':
    res_net = ModifiedVGG19().cuda()
    res_net2 = ModifiedVGG19().cuda()


# res_net = tv.models.resnet34(pretrained=True)
# res_net.fc = nn.Linear(res_net.fc.in_features, 2)
#
# # net2
# res_net2 = tv.models.resnet34(pretrained=True)
# res_net2.fc = nn.Linear(res_net2.fc.in_features, 2)


# res_net2.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 2),)

# Optimizer: SGD with Momentum
optim = t.optim.SGD(res_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00003)
optim2 = t.optim.SGD(res_net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00003)

# # Learning rate decay
scheduler = t.optim.lr_scheduler.MultiStepLR(optim, milestones=[100], gamma=0.1)
scheduler2 = t.optim.lr_scheduler.MultiStepLR(optim2, milestones=[100], gamma=0.1)

# epochs = 200
# scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=1E-6)
# scheduler2 = t.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=epochs, eta_min=1E-6)

loss=t.nn.CrossEntropyLoss()

# Start training
trainer = Trainer(res_net, res_net2, loss, optim, optim2, [scheduler, scheduler2], train_dl, val_dl, True)
res = trainer.fit(epochs=200,args=args)

# Plot train and validation loss
# plt.plot(np.arange(len(res[0])), res[0], label='train loss')
# plt.plot(np.arange(len(res[1])), res[1], label='train loss2')
# plt.plot(np.arange(len(res[2])), res[2], label='val loss')
# plt.axhline(y = 0.2, color = 'k', linestyle = 'dashed')
# plt.axhline(y = 0.1, color = 'k', linestyle = 'dashed')
# plt.yscale('log')
# ax = plt.gca()
# ax.set_ylim([0.01, 1])
# plt.legend()
# plt.savefig('./plots/losses.png')
