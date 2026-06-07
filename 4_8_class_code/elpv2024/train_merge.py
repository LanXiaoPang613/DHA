import numpy as np
import torch as t
import torchvision as tv
from model import *
if 1:
    from data_nas import ModifiedELPVDataset as ChallengeDataset
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
parser.add_argument('--num_class', default=4, type=int, help='classes')
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

train = None
test = None
# Set up data loading for the training and validation
# train_dl = t.utils.data.DataLoader(ChallengeDataset(train, "train"), batch_size=32, shuffle=True)
# val_dl = t.utils.data.DataLoader(ChallengeDataset(test, "val"), batch_size=32, shuffle=True)
train_ds = ChallengeDataset(
    csv_path="./modified_elpv_out/modified_elpv_8class.csv",
    root_dir="./modified_elpv_out",
    mode="train",
    num_classes=8
)
val_ds = ChallengeDataset(
    csv_path="./modified_elpv_out/modified_elpv_8class.csv",
    root_dir="./modified_elpv_out",
    mode="val",
    num_classes=8,
   class_to_idx=train_ds.class_to_idx,
   type_to_idx=train_ds.type_to_idx
)
train_dl = t.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = t.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)


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
    res_net = ModifiedVGG19(args.num_class).cuda()
    res_net2 = ModifiedVGG19(args.num_class).cuda()
# Optimizer: SGD with Momentum
optim = t.optim.SGD(res_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=3.0E-5)
optim2 = t.optim.SGD(res_net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=3.0E-5)

# # Learning rate decay
scheduler = t.optim.lr_scheduler.MultiStepLR(optim, milestones=[100,200], gamma=0.1)
scheduler2 = t.optim.lr_scheduler.MultiStepLR(optim2, milestones=[100,200], gamma=0.1)
loss=t.nn.CrossEntropyLoss()
# Start training
trainer = Trainer(res_net, res_net2, loss, optim, optim2, [scheduler, scheduler2], train_dl, val_dl, True)
res = trainer.fit(epochs=200,args=args)
