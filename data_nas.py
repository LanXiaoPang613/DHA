import numpy as np
import pandas as pd
import torchvision as tv
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb
# from elpv_dataset.utils import load_dataset


# Precomputed for normalization
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, mode: str,args=None):
        self.data = data
        self.mode = mode

        # Perform data augmentation
        self._transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(), 
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02)),
            tv.transforms.RandomResizedCrop((300, 300), scale=(0.98, 1.0), ratio=(1.0, 1.0)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        
        self._transform_val = tv.transforms.Compose([
            tv.transforms.ToPILImage(), 
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        self.class_map = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}

        if self.mode == "train":
            fname = './labels_train.txt'
            # images, probs, types = load_dataset()
        else:
            fname = './labels_val.txt'
        data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[
            'path', 'probability', 'type'])
        images = np.char.decode(data['path'])  # 路径，txt里写的是当前目录下的images文件夹
        probs = data['probability']
        types = np.char.decode(data['type'])
        labels = np.zeros(probs.shape).astype(np.int64)
        NUM_CLASSES = 2
        if NUM_CLASSES == 4:
            problist = np.unique(probs)
            for i in range(problist.size):
                oneprob = problist[i]
                labels[np.where(probs == oneprob)] = int(i)
        else:
            labels[np.where(probs > 0.5)] = int(1)

        imgtypes = np.zeros(types.shape).astype(np.int64)
        typelist = np.unique(types)
        for i in range(typelist.size):
            onetype = typelist[i]
            imgtypes[np.where(types == onetype)] = int(i)

        self.images, self.probs, self.types = images, probs, types
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        gray_image = imread(self.images[index])
        rgb_image = torch.from_numpy(np.transpose(gray2rgb(gray_image), (2, 0, 1)))

        # labels = torch.tensor([self.data.at[index, "crack"], self.data.at[index, "inactive"]])

        # labels = np.array([self.class_map[(self.data.at[idx, "crack"], self.data.at[idx, "crack"])] for idx in index]).astype(np.float_)
        labels = self.labels[index]
        imgtypes = self.types[index]
        prob = self.probs[index]
        trans_image = self._transform_train(rgb_image) if self.mode == "train" else self._transform_val(rgb_image)
        if self.mode == "train":
            return trans_image, labels, imgtypes, prob
        else:
            return trans_image, labels
