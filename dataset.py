import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk

def dir_helper(dir, label):
    file_names = os.listdir(dir)
    image_path = [os.path.join(dir, i) for i in file_names]
    labels = [label for _ in range(len(image_path))]
    label_image_path = dict(zip(image_path, labels))

    return image_path, labels, label_image_path 

class ChestImage(Dataset):

    """ Dataloader for loading in datasets """
    def __init__(self, args, transform=ToTensor(), target_transform=None):
        
        """ record label - imagePath dictionary """
        img_dir0 = "dataset/covid"
        img_dir1 = "dataset/normal"
        img_dir2 = "dataset/virus"
        
        image_path0, label_0, metaDict0 = dir_helper(img_dir0, 0)
        image_path1, label_1, metaDict1 = dir_helper(img_dir1, 1)
        image_path2, label_2, metaDict2 = dir_helper(img_dir2, 2)

        
        self.image_path = image_path0 + image_path1 + image_path2 
        self.labels = label_0 + label_1 + label_2 
        self.metaDict = metaDict0.copy()
        self.metaDict.update(metaDict1)
        self.metaDict.update(metaDict2)
        self.transform = transform
        self.target_transform = target_transform
        self.preprocess = args.preprocess
        # self.JACKPROCESS = args.modeJACK
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        
        image = cv2.imread(img_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(gray_image, (224, 224))
        # print(type(image))
        # print(image.dtype)
        label = self.labels[idx]
        if self.preprocess == "cannyedge":
            image = cv2.Canny(image, 100, 200)
        elif self.preprocess == "adaptivethreshold":
            image = cv2.adaptiveThreshold(image, 224, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
        elif self.preprocess == "shannonsentropy":
            image = entropy(image, disk(10))
        if self.transform:
            image = self.transform(image)
            if self.preprocess =="shannonsentropy":
                image = image.float()
        # elif self.preprocess == "":
        # elif self.preprocess == "":
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.tensor(label).long()

    def shuffle(self):
        # Shuffle label and imagePath
        pass


