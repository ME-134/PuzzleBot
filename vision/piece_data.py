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
                w = piece.width
                h = piece.height
                img_chunk = img[max(y-h//2, 0):min(y+h//2, img_height), max(x-w//2, 0):min(x+w//2, img_width)]
                img_list[i].append(img_chunk)

        img_lists.append(img_list)
        
    return img_lists

class ImagesListDS(data.Dataset):
    def __init__(self, img_lists):
        self.len = len(img_lists)
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.files[index]
        np_image = cv2.cvtColor(cv2.imread(np.random.choice(self.files)), cv2.COLOR_BGR2RGB)
        while min(np_image.shape[0], np_image.shape[1]) < 200:
            np_image= cv2.cvtColor(cv2.imread(np.random.choice(self.files)), cv2.COLOR_BGR2RGB)
        
        np_negative = cv2.cvtColor(cv2.imread(np.random.choice(self.files)), cv2.COLOR_BGR2RGB)
        while min(np_negative.shape[0], np_negative.shape[1]) < 200:
            np_negative= cv2.cvtColor(cv2.imread(np.random.choice(self.files)), cv2.COLOR_BGR2RGB)

        anchor = self.transform(np_image)
        positive = self.transform(np_image)
        negative = self.transform(np_negative)

        return anchor, positive, negative

    def __len__(self):
        # The number of samples in the dataset.
        return self.len