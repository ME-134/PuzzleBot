from glob import glob
import numpy as np 
import pandas as pd
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

import argparse
import gc
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from piece_detector import *

import copy

import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms

def get_img_lists():
    img_groups = [glob(f'train_imgs/g{i} *') for i in range(1, 3)]
    img_lists = []
    for img_group in img_groups:

        detector = Detector()

        img = cv2.cvtColor(cv2.imread(img_group[0]), cv2.COLOR_BGR2RGB) 
        detector.process(img.copy())
        pieces_list = [p for p in detector.pieces]

        img_list = [[] for i in range(len(pieces_list))]

        for img_path in img_group:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
            detector.process(img.copy())

            for i, piece in enumerate(pieces_list):
                img_height, img_width, _ = img.shape
                x = piece.x
                y = piece.y
                w = piece.width + 50
                h = piece.height + 50
                img_chunk = img[max(y-h//2, 0):min(y+h//2, img_height), max(x-w//2, 0):min(x+w//2, img_width)]
                img_list[i].append(img_chunk)

        img_lists.append(img_list)
        
    return img_lists

class ImagesListDS(data.Dataset):
    def __init__(self, transform, multiplier = 10):
        self.img_lists = get_img_lists()
        self.len = len(self.img_lists) * len(self.img_lists[0]) * len(self.img_lists[0][0]) * multiplier
        self.transform = transform

    def __getitem__(self, index):
        i = np.random.randint(0, len(self.img_lists))
        l = self.img_lists[i]
        j = np.random.randint(0, len(l))
        ps = l[j]
        p1 = ps[np.random.randint(0, len(ps))]
        p2 = ps[np.random.randint(0, len(ps))]

        i = np.random.randint(0, len(self.img_lists))
        l = self.img_lists[i]
        j = np.random.randint(0, len(l))
        ps = l[j]
        n = ps[np.random.randint(0, len(ps))]



        anchor = self.transform(p1)
        positive = self.transform(p2)
        negative = self.transform(n)

        return anchor, positive, negative

    def __len__(self):
        # The number of samples in the dataset.
        return self.len


class ImagesRotateDS(data.Dataset):
    def __init__(self, transform, multiplier = 10):
        self.img_lists = get_img_lists()
        self.len = len(self.img_lists) * len(self.img_lists[0]) * len(self.img_lists[0][0]) * multiplier
        self.transform = transform

    def __getitem__(self, index):
        i = np.random.randint(0, len(self.img_lists))
        l = self.img_lists[i]
        j = np.random.randint(0, len(l))
        ps = l[j]
        m = np.random.randint(0, len(ps))

        k1 = np.random.choice([0,1,2,3])
        k2 = np.random.choice([0,1,2,3])

        p1 = np.rot90(ps[m], k = k1)
        p3 = np.rot90(ps[m], k = k2)


        anchor = self.transform(p1)
        positive = self.transform(p1)
        negative = self.transform(p3)

        return anchor, positive, negative

    def __len__(self):
        # The number of samples in the dataset.
        return self.len