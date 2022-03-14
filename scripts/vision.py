from glob import glob
import numpy as np 
import rospy
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
vision_dir = '/home/me134/me134ws/src/HW1/vision'
# vision_dir = '../vision'
model = torch.load(f'{vision_dir}/checkpoints/efficientnetTune5_epoch10.cp', map_location=torch.device('cpu')).eval().to(device)
# model = torch.load(f'{vision_dir}/checkpoints/efficientnetTune5_epoch10.cp', map_location=torch.device('cpu')).eval().to(device)
MODEL_OUT_DIM = 512

rotation_coefs = np.array([-4.49081824e-01,  1.12687836e-02,  4.85113796e-01,
         1.70662619e-01,  1.89955882e-02, -5.89564080e-01,
        -7.15204257e-01,  1.27510820e-01, -3.03986061e-01,
        -3.51322496e-01, -3.99617813e-01,  8.63904780e-01,
        -1.83382493e-01, -1.53791225e-01, -4.63326567e-01,
         9.18626120e-02, -4.11118605e-02, -3.12185556e-01,
         2.44983164e-01, -7.22996437e-01, -3.58611556e-02,
        -8.89574219e-02,  2.80122380e-01, -2.15861056e-01,
        -8.51508935e-01,  1.54787657e-01, -1.19536721e-01,
        -3.13194711e-02,  4.47863166e-01,  2.01475597e-01,
        -2.09516714e-03, -1.30780801e-01,  1.16117007e-01,
         4.29281071e-01,  3.01904183e-01, -4.13799253e-01,
         7.44954513e-02,  1.84409829e-01, -2.95420256e-01,
         3.59187507e-01, -1.22753103e+00,  3.38674211e-01,
         2.30627728e-01, -1.26686006e-01, -6.00702247e-01,
        -3.01241257e-01,  3.63822135e-02,  3.50812084e-02,
        -2.51459516e-01, -9.27914942e-01, -3.24184313e-02,
         2.00901422e-01,  1.01385002e-01, -1.65026681e-01,
        -1.16934172e-01,  1.55535720e-02, -3.40229604e-01,
        -4.19034121e-01, -1.51051745e-01,  5.90433948e-02,
         4.22249674e-01, -2.37192135e-02,  4.07363375e-02,
        -3.21967981e-01,  4.28466200e-01,  3.34861113e-01,
         2.14724423e-01, -1.10117557e-01,  2.86552233e-02,
        -5.96744481e-01,  1.28958066e-01,  3.48792654e-01,
         2.76660696e-01,  7.58979180e-02, -6.33471353e-01,
        -4.71356333e-01, -2.36349567e-01, -3.21924633e-01,
         2.04200779e-01, -1.45765512e-01,  6.78658228e-01,
        -6.84272790e-03, -3.16928983e-01,  4.39150130e-01,
         1.40720055e-01,  2.78746559e-01, -1.99890998e-01,
         1.02186732e-01, -3.00368123e-01,  2.06877113e-02,
        -1.96870767e-01, -6.52701999e-01,  1.62982875e-01,
         6.15440595e-02, -3.28999412e-01,  9.68438578e-01,
        -1.96306394e-01,  2.28945845e-01,  2.96183582e-02,
        -1.69630838e-01, -6.11137726e-03,  1.73919314e-01,
        -1.24840531e-01, -1.21247655e+00, -1.29634506e-01,
         2.14764333e-01,  5.36326873e-01, -5.53932068e-01,
         7.39554590e-02, -1.31032107e-01, -3.11485549e-01,
         2.44015560e-01, -3.51063939e-01,  2.21861071e-01,
        -4.55805874e-01, -4.52255737e-01, -1.16263205e-01,
        -3.23782421e-01, -1.30795369e+00, -9.11136769e-01,
         4.07615560e-01,  4.12691149e-01, -1.61221926e-01,
         2.16573028e-01,  4.38494526e-01, -5.73929595e-01,
         9.15103043e-02, -7.53079224e-01, -1.46197694e-01,
        -1.52767155e-01, -2.97382519e-02,  7.86810164e-01,
         3.57545551e-01,  6.12953294e-01, -3.48029864e-01,
        -1.88097664e-01, -3.15409418e-02,  1.53964569e-01,
         1.18962589e-01, -3.58199428e-01,  1.14258945e-01,
         3.70455049e-01, -7.64468354e-02,  6.12329622e-01,
        -2.28198815e-01,  3.47432775e-01, -4.20503819e-01,
         2.03874293e-01, -3.08244024e-01,  5.37397868e-02,
        -1.68519725e-01,  1.22723191e-01, -2.09610972e-01,
        -3.77196704e-01,  2.73715311e-01, -1.58597266e-01,
         3.94404037e-01, -2.54217415e-01, -4.37474617e-01,
        -7.11827583e-02, -3.43581509e-01,  2.09666940e-01,
        -2.28076464e-01, -4.55804019e-01, -1.32357193e-01,
        -1.23754833e-01, -4.79368085e-01,  3.09097807e-01,
         6.97669426e-02, -1.87596121e-01,  8.86352877e-02,
        -7.17474827e-02, -1.22182960e-01,  1.88113505e-01,
        -2.29538422e-01,  8.35787452e-01, -1.15642773e-01,
        -5.23573222e-01,  2.02686160e-01,  1.84512148e-01,
        -1.87638459e-01,  4.29196056e-01, -1.80044799e-01,
        -1.03911011e+00, -9.77515663e-02, -3.76666283e-01,
        -3.20124025e-01,  5.46661723e-01, -1.64028027e-01,
        -2.35854062e-01,  1.51351875e-03,  1.30150084e-01,
        -2.19699895e-01, -5.64326498e-01, -8.64185998e-01,
        -5.48847825e-02, -3.39879247e-01,  7.53453228e-03,
         9.64277796e-03, -2.12375519e-01, -1.45992235e-01,
         8.81716996e-02,  1.60359181e-01,  4.06743012e-02,
         8.72887010e-02, -3.54638323e-01, -1.89036319e-01,
         1.00846519e-01, -2.23536158e-01,  1.04290399e-02,
         9.23382385e-02,  4.18412999e-01,  7.43392767e-01,
         6.46651744e-01, -5.11025616e-01, -1.30426637e-01,
         7.59706208e-02, -3.75164863e-01, -2.33609517e-01,
         8.47060369e-02, -5.05031854e-01, -2.47857668e-01,
        -7.29147993e-01, -4.31993147e-01, -2.00323899e-01,
        -8.83935377e-03, -6.37830118e-02,  2.23868918e-01,
         1.30044902e-01, -2.20688469e-02, -3.84438955e-01,
        -1.76237178e-01,  4.16704167e-02,  4.46181691e-01,
        -2.92731316e-01, -1.54833350e-01, -3.20564661e-01,
         2.58819934e-01,  2.22295118e-01,  9.64461068e-03,
         4.01697333e-01,  4.77380626e-02, -2.53831307e-01,
         8.83415132e-02, -2.68795173e-01, -1.71080768e-01,
         3.83800291e-01,  1.88585534e-01,  2.12950433e-01,
         2.51003527e-01, -3.68128134e-01,  1.63827631e-01,
        -1.91185811e-01,  2.29751143e-01, -3.48222039e-01,
        -3.69501283e-01,  1.15199320e-01, -6.23949801e-01,
        -9.91216585e-01,  3.30483762e-01, -1.53248926e-01,
         3.04342533e-01, -6.07205322e-01, -5.57315745e-02,
         4.97566474e-02, -5.32596905e-01,  3.18239653e-01,
        -2.60811751e-01,  3.93631482e-01, -4.18649706e-01,
         5.34955011e-03, -3.54963523e-01,  3.29771138e-01,
         1.34596454e-01,  1.83323614e-01, -3.76941124e-01,
         3.09158099e-01, -6.95249535e-01,  2.41963918e-01,
        -1.32717523e-01,  1.95015957e-01,  2.92409270e-01,
        -2.88395124e-01,  1.15735649e-01, -2.91514079e-01,
         2.49829696e-01, -7.26490279e-02,  1.46796652e-01,
         4.96106930e-01,  8.09734736e-01,  3.73520372e-01,
        -4.05099190e-01,  3.57032796e-01,  6.37902295e-01,
         1.04604332e-01, -1.34527695e-01, -4.49618920e-01,
         1.20947896e-01,  4.85369330e-01,  7.08222171e-01,
        -5.98792627e-01,  6.01161832e-01,  2.33977659e-02,
         5.68861321e-02, -2.62020353e-01, -3.64184917e-01,
         4.01236679e-01,  1.62537950e-02,  1.42123234e-01,
        -4.26631835e-01, -7.90142462e-01,  5.17858114e-02,
         4.14064347e-01,  3.38919459e-04,  3.33650071e-01,
        -4.54193957e-01,  6.87850935e-04,  3.59154828e-02,
        -4.68169327e-02, -7.46718045e-02, -8.35737585e-02,
         3.27973917e-01, -2.38726903e-01, -2.74564181e-01,
        -2.50987611e-01,  5.39566656e-01, -9.37863717e-01,
        -8.51817355e-01,  2.67105547e-01, -5.07779697e-01,
        -3.96334672e-01, -3.07547307e-01, -3.48314512e-01,
         5.74850052e-01, -7.15960125e-02, -6.77572395e-01,
        -1.76872779e-01,  3.54873038e-01, -4.92165246e-02,
         7.18556216e-02,  2.53670392e-01, -7.86648478e-01,
         1.20564697e-01, -1.11779544e-01,  3.81238362e-01,
         2.53389081e-01,  9.70246476e-01,  5.19907422e-01,
        -1.88037098e-01,  1.79551810e-01,  1.11959911e-01,
         1.03009986e-01, -4.77580968e-01,  1.17874067e-01,
         3.68139266e-02, -5.83043920e-01, -3.98433546e-01,
         6.13229528e-01, -2.57458252e-01,  1.08655081e-01,
         2.65309157e-01, -2.60803715e-01, -2.95962210e-01,
        -2.54675496e-01, -1.71170391e-01, -1.31184618e-01,
        -5.69521955e-01, -2.19459156e-01, -4.47648617e-01,
         5.04717830e-01,  7.34360967e-01, -3.12864168e-01,
        -1.69402420e-02, -3.85530919e-01, -3.42185034e-01,
        -4.86333740e-01, -1.64116823e-01,  6.46896385e-01,
         2.83169140e-01,  2.66540726e-01,  7.68735665e-02,
         2.38074598e-02, -1.04748433e-01,  3.84826772e-01,
         1.07014755e-01, -2.67566628e-01, -1.65354869e-01,
         2.40297990e-01, -1.12384172e-01, -3.41623049e-01,
         2.59861030e-01, -3.84485249e-01,  1.53404863e-01,
        -8.82649774e-01,  1.40336235e-01,  7.89612183e-01,
         1.37022982e-01, -4.39361937e-01,  4.27945034e-01,
        -4.18953414e-01,  6.50661809e-02,  1.25488189e-01,
         5.13992689e-01,  3.14539869e-01, -2.48888935e-01,
         3.64168878e-01, -6.00338065e-02, -1.16983539e-01,
         2.59293768e-01, -8.25698754e-01,  1.30383705e-01,
        -7.56784812e-02,  2.16840469e-01,  9.05565595e-02,
        -8.01385705e-02, -1.46401407e-01, -4.41047180e-01,
         9.80584341e-02, -5.56127124e-02, -2.01275455e-01,
         1.37079834e-01,  3.89191198e-01,  5.76679060e-01,
         2.52477101e-01,  3.97929137e-01, -2.08480964e-01,
        -1.25133162e-01, -3.29575681e-02,  1.50105550e-01,
        -3.90655335e-02, -3.40123508e-01, -1.53146163e-02,
        -6.54475898e-02, -2.83434572e-01, -3.12314055e-01,
        -5.08397091e-01, -4.68584569e-01,  3.04541735e-01,
        -2.77318862e-01, -2.67565673e-01, -2.73255140e-01,
         3.49638136e-02,  2.45830682e-01, -1.85254856e-01,
        -1.06755957e-02,  2.98717147e-01, -6.84644823e-01,
        -2.59300080e-01,  5.87944305e-02,  1.82218429e-01,
        -9.51687059e-01, -2.14149171e-01,  2.54369114e-01,
        -5.66441949e-01,  2.84172543e-01, -2.22824074e-01,
        -1.04770577e-01,  4.29639678e-02, -2.72878893e-01,
         4.62659515e-01, -2.68434455e-01,  1.02750688e-01,
         4.27955846e-01,  5.19802894e-01, -1.16541223e-01,
        -2.45276750e-01,  3.11695301e-01,  1.24122815e-01,
        -3.90547195e-01, -6.87395252e-02, -2.76835895e-01,
         1.53614222e-01, -1.97549826e-01, -1.89711152e-01,
        -3.23552637e-01, -1.58365796e-01, -4.01555035e-01,
         1.35946968e-01,  2.38922601e-01,  3.15594863e-02,
        -7.92070284e-01,  3.49652499e-03,  2.68774587e-01,
        -1.83514325e-01, -5.80986520e-01,  2.19431429e-01,
        -2.23416122e-01, -2.91084704e-01, -1.99141823e-01,
         4.77258651e-01,  6.44214311e-01,  1.73078997e-01,
         1.10949716e-01, -1.32844631e-01,  4.13148983e-01,
         1.51298658e-02,  4.20944599e-02,  3.16868548e-01,
        -4.14816346e-01, -2.60606685e-01,  1.42218151e-02,
         3.03470397e-01, -1.90667005e-01,  1.65750449e-01,
         4.56620883e-01, -2.57279214e-01, -1.68089196e-01,
         3.71106647e-01,  1.05177727e+00, -3.08908253e-01,
        -3.53016225e-02, -1.35853189e-01,  3.47522520e+00])
rotation_coefs[-1] *= 2
# rotation_coefs[np.abs(rotation_coefs) < 0.01] = 0
rotation_intercept = 0.71418728

def norm(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 0.0001)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_rotation_model(base_img, rotation_image, k_init = 0):
    base_img = np.rot90(base_img, k = k_init)
    probs = []
    for k in range(4):
        rot_img = np.rot90(rotation_image, k = k + k_init)
        x = np.array(list(np.abs(run_model_masked(base_img)[0] - run_model_masked(rot_img)[0])) + [calc_iou(base_img, rot_img)])
        probs.append(sigmoid(np.dot(x, rotation_coefs) + rotation_intercept ))
    return probs

def run_rotation_model_big(base_img, rotation_image):
    probs = np.zeros(4)
    for i in range(4):
        probs += np.array(run_rotation_model(base_img, rotation_image, i)) / 4
    
    print("[VISION] : ", probs, np.argmax(probs))
    return probs

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
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    # img = ((img / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    ref = torch.from_numpy(img[:image_size, :image_size, :].reshape(1, image_size, image_size, 3)).float().permute(0, 3, 1, 2).to(device)
    ref_pred = model(ref)
    return ref_pred.cpu().detach().numpy()

def calc_iou(img1, img2, image_size = 124):
    img1 = cv2.resize(img1, (image_size, image_size))
    mask1 = get_piece_mask(img1)> 128
    img2 = cv2.resize(img2, (image_size, image_size))
    mask2 = get_piece_mask(img2) > 128
    # import matplotlib.pyplot as plt
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
    return (mask1 * mask2).sum() / (mask1.sum() + mask2.sum())


def calc_contour(filtered):
    filtered = get_piece_mask(filtered)
    contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    biggest_contour = max(contours, key = cv2.contourArea)
    return biggest_contour

from sklearn.neighbors import NearestNeighbors


def contour_similarity(c1, c2, approx_points = 100):
    c1, c2 = c1.reshape(-1, 2), c2.reshape(-1, 2)
    c1 = c1[::int(len(c1) / approx_points)]
    c2 = c2[::int(len(c2) / approx_points)]
    c1 = (c1 - c1.min(axis = 0)) / (c1.max(axis = 0) - c1.min(axis = 0))
    c2 = (c2 - c1.min(axis = 0)) / (c2.max(axis = 0) - c2.min(axis = 0))
    knn = NearestNeighbors(n_neighbors = 1, algorithm = 'brute')
    knn.fit(c1)
    dists, _ = knn.kneighbors(c2)
    return [dists.mean(), dists.std()]

def full_contour_similarity(img1, img2):
    c1 = calc_contour(img1)
    c2 = calc_contour(img2)
    return contour_similarity(c1, c2)

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
        rotation_vectors = [run_model_masked(np.rot90(img, k = k)) for k in range(4)]
        for k in range(4):
            sims[:, :, k] = ((self.inferences - rotation_vectors[k]) ** 2).sum(axis = 2)
        
        xy_min = get_xy_min(sims[:, :].mean(axis = 2))

        base = self.piece_grid[xy_min[0]][xy_min[1]].natural_img

        # ious = [calc_iou(np.rot90(img, k = k), base) for k in range(4)]
        # argmin_basic = np.array(sims[xy_min[0], xy_min[1], :4])
        # argmin_iou = 1-np.array(ious)
        # sim_base = self.inferences[xy_min[0]][xy_min[1]]
        # sims_rot = ((self.rotation_pca.transform(self.calculate_rotation_vectors(img).T) - self.rotation_pca.transform(sim_base.reshape(1, -1))) ** 2).sum(axis=1)
        # argmin_pca = np.array(sims_rot)
        rot_probs = run_rotation_model_big(base, img)
        rospy.loginfo("[VISION] CALCULATING XY-ROT : ", xy_min, np.argmax(rot_probs), rot_probs)
        return xy_min, np.argmax(rot_probs) #np.argmin(norm(argmin_basic) + norm(argmin_iou) + norm(argmin_pca))
    
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
        ''' Greedily finds a 1-to-1 matching of pieces to reference pieces
        Score is inf if no match found
        '''
        if method == 'greedy':
            scores_all = np.zeros((len(pieces), self.width_n, self.height_n, 4))
            for i, piece in enumerate(pieces):
                for k in range(4):
                    scores_all[i, :, :, k] = ((self.inferences - run_model_masked(np.rot90(piece.natural_img, k = k))) ** 2).sum(axis = 2)
            scores = scores_all.mean(axis=3)
            match_scores = np.zeros((len(pieces))) + np.inf
            
            locations = np.zeros((len(pieces), 2)).astype(int)
            for i in range(min(len(pieces), self.width_n*self.height_n)):
                best = np.argmin(scores)
                k = best // (self.width_n*self.height_n)
                x = (best - k*(self.width_n*self.height_n)) // self.height_n
                y = (best - k*(self.width_n*self.height_n) - x*self.height_n)
                if pieces[k].is_valid():
                    locations[k] = np.array([x, y])
                    match_scores[k] = scores[k,x,y]
                    scores[:, x, y] = np.inf
                else:
                    match_scores[k] = np.inf
                scores[k, :, :] = np.inf
            if order:
                pass
            
            rots = np.zeros((len(pieces)))
            for i in range(len(pieces)):
                base = self.piece_grid[locations[i, 0]][locations[i, 1]].natural_img
                img = pieces[i].natural_img
                ious = [calc_iou(np.rot90(pieces[i].natural_img, k = k), base) for k in range(4)]
                argmin_basic = np.array(scores_all[i, locations[i, 0], locations[i, 1], :4])
                argmin_iou = 1-np.array(ious)
                sim_base = self.inferences[locations[i, 0]][locations[i, 1]]
                sims_rot = ((self.rotation_pca.transform(self.calculate_rotation_vectors(pieces[i].natural_img).T) - self.rotation_pca.transform(sim_base.reshape(1, -1))) ** 2).sum(axis=1)
                argmin_pca = np.array(sims_rot)
                probs = run_rotation_model_big(base, img)

                print( "[NORMS] : ", norm(argmin_basic), norm(argmin_iou), norm(argmin_pca), norm(probs))
                rots[i] = np.argmin(norm(argmin_basic) + norm(argmin_iou) + norm(argmin_pca) - 4 * norm(probs))
            return locations, rots, match_scores  # locations[i] gives grid location of piece i
        else:
            raise Exception("Not Implemented")
    
    def get_largest_piece(self):
        return max(self.pieces, key=lambda x: x.area)

    