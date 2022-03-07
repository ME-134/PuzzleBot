from glob import glob
import numpy as np 
import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils import data
import copy

import accountant
import torch.multiprocessing as mp
from thomas_detector import ThomasDetector
from puzzle_grid import PuzzleGrid, get_xy_min

device = 'cpu'
torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b2_ns', pretrained=True)

name = 'efficientnetTune2_epoch8'
model = torch.load('..\\vision\\checkpoints\\'+name+'.cp').eval().to(device)
MODEL_OUT_DIM = 512

def run_model(img, image_size = 150):
    img = cv2.resize(img, (image_size, image_size))
    img = (img - img.mean()) / img.std()
#     img = ((img / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    ref = torch.from_numpy(img[:124, :124, :].reshape(1, 124, 124, 3)).float().permute(0, 3, 1, 2).to(device)
    ref_pred = model(ref)
    return ref_pred.cpu().detach().numpy()

def cvt_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class VisionMatcher():
    def __init__(self, base_image_path, width_n = 5, height_n = 4):
        self.base_img = cvt_color(cv2.imread(base_image_path))
        self.detector = ThomasDetector()
        self.detector.process(self.base_img)
        self.pieces = self.detector.pieces
        
        self.width_n = width_n
        self.height_n = height_n
        
        self.puzzle_grid = PuzzleGrid(
            width_n = width_n, height_n = height_n, spacing_height = 1.0/(height_n-1), spacing_width = 1.0/(width_n-1), offset_x = 0, offset_y = 0
            )
        
        self.max_x = np.max([p.x for p in self.pieces])
        self.min_x = np.min([p.x for p in self.pieces])
        self.max_y = np.max([p.y for p in self.pieces])
        self.min_y = np.min([p.y for p in self.pieces])

        self.piece_grid = [[None for j in range(height_n)] for i in range(width_n)]

        def scale_x(x):
            return (x - self.min_x) / (self.max_x - self.min_x)
        def scale_y(y):
            return (y - self.min_y) / (self.max_y - self.min_y)
        
        for piece in self.pieces:
            if (piece.is_valid()):
                (x_close, y_close) = get_xy_min(((self.puzzle_grid.grid_centers - np.array([scale_x(piece.x), scale_y(piece.y)]))**2).sum(axis = 2))
                self.piece_grid[x_close][y_close] = piece
        
        
        self.inferences = np.zeros((width_n, height_n, MODEL_OUT_DIM))
        
        for row in range(width_n):
            for col in range(height_n):
                piece = self.piece_grid[row][col]
                if(piece == None):
                    print("Missing piece", (row, col))
                else:
                    self.inferences[row, col] = run_model(piece.img)
            
    def calculate_xyrot(self, img):
        sims = np.zeros((self.width_n, self.height_n, 4))
        for k in range(4):
            sims[:, :, k] = ((self.inferences - run_model(np.rot90(img, k = k))) ** 2).sum(axis = 2)
        
        xy_min = get_xy_min(sims[:, :].min(axis = 2) + sims[:, :].mean(axis = 2))
        return xy_min, np.argmin(sims[xy_min[0], xy_min[1], :4])