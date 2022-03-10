from glob import glob
import numpy as np 

import cv2

import torch
import torch.nn as nn
from torch.utils import data

import torch.multiprocessing as mp
from thomas_detector import ThomasDetector, get_piece_mask
from puzzle_grid import PuzzleGrid, get_xy_min
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


device = 'cpu'
torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b2_ns', pretrained=True)

name = 'efficientnetTune5_epoch10'
# vision_dir = '/home/me134/me134ws/src/HW1/vision'
vision_dir = '../vision'
model = torch.load(f'{vision_dir}/checkpoints/efficientnetTune5_epoch10.cp', map_location=torch.device('cpu')).eval().to(device)
# model = torch.load(f'{vision_dir}/checkpoints/efficientnetTune5_epoch10.cp', map_location=torch.device('cpu')).eval().to(device)
MODEL_OUT_DIM = 512

def norm(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 0.0001)

def run_model(img, image_size = 124):
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()
    img = cv2.resize(img, (image_size, image_size))
    img = (img - img.mean()) / img.std()
    # img = ((img / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    ref = torch.from_numpy(img[:image_size, :image_size, :].reshape(1, image_size, image_size, 3)).float().permute(0, 3, 1, 2).to(device)
    ref_pred = model(ref)
    return ref_pred.cpu().detach().numpy()

def run_model_masked(img, image_size = 124):
    img = cv2.resize(img, (image_size, image_size))
    mask = get_piece_mask(img).reshape((image_size, image_size, 1)) > 128
    img = img * mask
    img = (img - img.mean()) / img.std()
    # img = ((img / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    ref = torch.from_numpy(img[:image_size, :image_size, :].reshape(1, image_size, image_size, 3)).float().permute(0, 3, 1, 2).to(device)
    ref_pred = model(ref)
    return ref_pred.cpu().detach().numpy()

def calc_iou(img1, img2, image_size = 124):
    img1 = cv2.resize(img1, (image_size, image_size))
    mask1 = get_piece_mask(img1)> 128
    img2 = cv2.resize(img2, (image_size, image_size))
    mask2 = get_piece_mask(img2) > 128
    import matplotlib.pyplot as plt
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    return (mask1 * mask2).sum() / (mask1.sum() + mask2.sum())

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
                    val = piece.natural_img #* (piece.img.reshape(piece.img.shape[0], piece.img.shape[1], 1) > 128)
                    self.inferences[row, col] = run_model_masked(val)
        
        self.fit_rotation_pca()
            
    def calculate_xyrot(self, img):
        sims = np.zeros((self.width_n, self.height_n, 4))
        for k in range(4):
            sims[:, :, k] = ((self.inferences - run_model_masked(np.rot90(img, k = k))) ** 2).sum(axis = 2)
        
        xy_min = get_xy_min(sims[:, :].mean(axis = 2))

        base = self.piece_grid[xy_min[0]][xy_min[1]].natural_img
        ious = [calc_iou(np.rot90(img, k = k), base) for k in range(4)]
        argmin_basic = np.array(sims[xy_min[0], xy_min[1], :4])
        argmin_iou = 1-np.array(ious)
        sim_base = self.inferences[xy_min[0]][xy_min[1]]
        sims_rot = ((self.rotation_pca.transform(self.calculate_rotation_vectors(img).T) - self.rotation_pca.transform(sim_base.reshape(1, -1))) ** 2).sum(axis=1)
        argmin_pca = np.array(sims_rot)

        return xy_min, np.argmin(norm(argmin_basic) + norm(argmin_iou) + norm(argmin_pca))
    
    def calculate_rotation_difference_vectors(self, img, ref):
        sims = np.zeros((MODEL_OUT_DIM, 4))
        ref_inference = run_model_masked(ref)
        for k in range(4):
            sims[:, k] = (ref_inference - run_model_masked(np.rot90(img, k = k)))
        
        return sims
    
    def calculate_rotation_vectors(self, img):
        sims = np.zeros((MODEL_OUT_DIM, 4))
        for k in range(4):
            sims[:, k] = (run_model_masked(np.rot90(img, k = k)))
        return sims

    def fit_rotation_pca(self, dims = 10):
        imgs = glob(f'{vision_dir}/imgs/good1*.jpg')
        all_vectors = []
        for path in imgs:
            img = cvt_color(cv2.imread(path))
            vectors = self.calculate_rotation_vectors(img).T
            for v in vectors:
                all_vectors.append(v)

        all_vectors = np.array(all_vectors)
        self.rotation_pca = PCA(dims).fit(all_vectors)

    def fit_rotation_logistic(self):
        pass

    def fit_pca(self, dims = 5):
        all_vectors = []
        for piece in self.pieces:
            img = piece.img
            v = run_model_masked(img)
            all_vectors.append(v)

        all_vectors = np.array(all_vectors)
        self.pca = PCA(dims).fit(all_vectors)

    def match_all(self, pieces, method='greedy', order=True):
        # Greedily finds a 1-to-1 matching of pieces to reference pieces
        if method == 'greedy':
            scores = np.zeros((len(pieces), self.width_n, self.height_n, 4))
            for i, piece in enumerate(pieces):
                for k in range(4):
                    scores[i, :, :, k] = ((self.inferences - run_model_masked(np.rot90(piece.natural_img, k = k))) ** 2).sum(axis = 2)
            scores = scores.mean(axis=3)
            
            locations = np.zeros((len(pieces), 2))    
            for i in range(len(pieces)):
                print(np.argmin(scores, axis=-1))
                k, x, y = np.argmin(scores, axis=-1)
                locations[k] = np.array([x, y])
                scores[:, x, y] = np.inf
                scores[k, :, :] = np.inf

            if order:
                pass

            return locations  # locations[i] gives grid location of piece i
        else:
            raise Exception("Not Implemented")

    