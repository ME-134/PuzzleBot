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
accountant.load_accountant('desc\\efficientnetB0.df', evaluate = True)

def cvt_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class VisionMatcher():
    def __init__(self, base_image_path, width_n = 5, height_n = 4):
        self.base_img = cvt_color(cv2.imread(base_image_path))
        self.detector = ThomasDetector()
        self.detector.process(self.base_img)
        self.pieces = self.detector.pieces
    
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
            (x_close, y_close) = get_xy_min((self.puzzle_grid - np.array([scale_x(piece.x), scale_y(piece.y)]))**2)
            self.piece_grid[x_close][y_close] = piece
        

    def infer(self):
        