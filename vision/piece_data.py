from glob import glob
import numpy as np 
import pandas as pd
from tqdm import tqdm
import cv2
import warnings

from thomas_detector import ThomasDetector
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

good1s = glob('imgs/good1_*.jpg')
good2s = [p.replace('good1', 'good2') for p in good1s]

bas1s = glob('imgs/bas1_*.jpg')
bas2s = [p.replace('bas1', 'bas2') for p in good1s]

def get_clusters(line):
    high_pixels = np.array([i for i in range(len(line)) if line[i] > 128])
    clusters = []
    cluster = []
    for i in range(len(high_pixels)):
        if (i > 0) and (high_pixels[i] > high_pixels[i-1] + 1):

            clusters.append(np.array(cluster))
            cluster = []
        else:
            cluster.append(high_pixels[i])
    clusters.append(np.array(cluster))
    return clusters

def get_line_type(line):
    if (line[2:-2] > 128).mean() > 0.83:
        return 'flat'
    
    clusters = get_clusters(line)
    clusters = [c for c in clusters if len(c) > 2]
    if len(clusters) > 1:
        return 'inward'
    else:
        return 'outward'

def get_img_lists():
    img_groups = [glob(f'train_imgs/g{i} *') for i in range(1, 5)]
    img_lists = []
    for img_group in img_groups:

        detector = ThomasDetector()

        img = cv2.cvtColor(cv2.imread(img_group[0])[:, 160:], cv2.COLOR_BGR2RGB) 
        detector.process(img.copy())
        pieces_list = [p for p in detector.pieces if p.is_valid()]

        img_list = [[] for i in range(len(pieces_list))]

        for img_path in img_group:
            img = cv2.cvtColor(cv2.imread(img_path)[:, 160:], cv2.COLOR_BGR2RGB) 
            detector.process(img.copy())

            for i, piece in enumerate(pieces_list):
                # img_height, img_width, _ = img.shape
                # x = piece.x
                # y = piece.y
                # w = piece.width + 50
                # h = piece.height + 50
                # img_chunk = img[max(y-h//2, 0):min(y+h//2, img_height), max(x-w//2, 0):min(x+w//2, img_width)]
                # img_list[i].append(img_chunk)
                img_list[i].append(piece.natural_img * (piece.img.reshape(piece.img.shape[0], piece.img.shape[1], 1) > 128))
            
            for i, piece in enumerate(pieces_list):
                # img_height, img_width, _ = img.shape
                # x = piece.x
                # y = piece.y
                # w = piece.width + 50
                # h = piece.height + 50
                # img_chunk = img[max(y-h//2, 0):min(y+h//2, img_height), max(x-w//2, 0):min(x+w//2, img_width)]
                # img_list[i].append(img_chunk)
                img_list[i].append(piece.natural_img)

        img_lists.append(img_list)

        
    return img_lists

# import matplotlib.pyplot as plt

class ImagesListDS(data.Dataset):
    def __init__(self, transform, multiplier = 10):
        self.img_lists = get_img_lists()
        self.len = int(len(self.img_lists) * len(self.img_lists[0]) * len(self.img_lists[0][0]) * multiplier)
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


        # plt.imshow(p1)
        # plt.show()
        # plt.imshow(p2)
        # plt.show()
        # plt.imshow(n)
        # plt.show()

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