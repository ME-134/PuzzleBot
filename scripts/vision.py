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

rotation_coefs = np.array([-2.91785983e-01,  9.34488452e-02, -1.93939702e-01,
        -9.55246869e-02, -2.57486529e-01,  6.24093977e-02,
        -1.29381352e-01,  1.96460926e-01, -1.46661910e-01,
         1.05644753e-01, -3.86911258e-01, -1.79454328e-02,
        -2.46761392e-01,  1.22914485e-01,  3.56927131e-01,
        -2.53945488e-01,  8.53799470e-04, -1.31406403e-01,
         2.65717632e-02,  4.55777448e-02,  2.37693756e-01,
        -2.22413667e-01,  4.90686821e-01, -4.87787543e-02,
         1.66725629e-01,  1.06484413e-01,  8.68815060e-03,
         4.17079373e-01,  3.29352711e-01, -3.79872759e-01,
        -3.95831539e-01,  6.62233650e-02,  1.65020490e-01,
         8.16828985e-01, -5.36514696e-01, -9.54638580e-02,
        -1.63024041e-01, -3.38998706e-02, -7.98023658e-01,
         2.66623171e-01, -1.09475084e-01,  2.18604510e-01,
        -3.08468786e-01, -7.81259657e-02,  1.19007605e-01,
         1.09998252e-01, -7.46450183e-02,  2.42779687e-01,
        -3.50407853e-03, -5.61468253e-01, -2.59213296e-01,
        -1.40304059e-01,  2.18046836e-02, -3.38982995e-01,
        -2.48954835e-02, -6.93094599e-01, -4.32945515e-01,
         4.26407953e-02, -2.76473067e-01,  1.74340664e-01,
         5.29752907e-03,  4.35717799e-01, -3.86956636e-02,
        -7.96591556e-03, -3.34026783e-01,  2.06987088e-02,
         9.02738617e-02, -3.70134593e-02, -1.99177136e-01,
        -3.74218079e-01,  2.42055744e-01, -3.60859550e-01,
         4.98546924e-01,  6.06635481e-02, -4.63744336e-01,
        -1.04888690e-01,  1.84650386e-01, -7.47812223e-02,
         1.07320222e-01,  6.38841634e-02,  5.65180908e-01,
         2.83978166e-01,  2.19763119e-01, -8.40108135e-01,
        -4.16863541e-01, -8.63649433e-02, -6.50159574e-01,
        -6.26873750e-01,  2.79665126e-01,  2.03682481e-01,
         4.86343192e-01,  1.67379658e-01,  1.99281845e-01,
         2.50807273e-01, -4.41433625e-01,  9.94104589e-02,
        -1.58819747e-01, -2.22186680e-01,  9.06370516e-02,
        -6.16556921e-02, -5.70981335e-01, -1.76620433e-02,
        -1.04338918e-01, -1.06797031e+00, -4.74653427e-01,
        -3.29188875e-03,  2.60605847e-01,  6.90724058e-02,
         2.06272639e-01, -2.43064973e-02, -2.11238934e-01,
         1.88067409e-01, -7.98023623e-02,  4.50238390e-01,
        -1.51772382e-01, -1.63773520e-01, -1.00741015e-01,
        -3.34411318e-02, -6.10049848e-01, -3.65552496e-01,
         4.58534331e-01,  8.92065568e-03,  2.54546664e-01,
        -2.70683349e-01,  2.64559325e-02, -1.35829692e-01,
        -4.76718379e-02,  1.99101111e-01, -1.28808961e-01,
         4.69222167e-02, -3.53206404e-01,  4.36034608e-01,
        -7.15636877e-02,  6.39101243e-01, -2.14745286e-01,
        -3.57768269e-01, -7.53680644e-01, -4.80520333e-01,
        -7.26073713e-01,  2.14987790e-01, -2.04796095e-01,
         2.26989599e-01, -2.00315854e-01,  3.46442515e-01,
         2.45042526e-01, -9.02925328e-02,  6.46364163e-02,
         2.71354640e-01, -2.67466045e-02,  2.53137666e-01,
         3.34086682e-01, -9.93966921e-02,  2.55512713e-01,
        -1.00810749e-01,  2.27846654e-01,  6.69676879e-02,
         9.10392232e-02,  4.16450737e-01,  7.93424046e-02,
         3.43683204e-01, -1.21385918e-01, -2.18157219e-01,
        -3.96489021e-01, -9.29761061e-02, -7.39394235e-02,
        -3.98618630e-01,  9.92977014e-02, -5.74756435e-01,
        -1.94940739e-01, -1.53800444e-01, -2.31224822e-03,
         5.80095634e-02,  1.81095096e-01, -2.21242283e-01,
        -8.94752605e-02,  2.98492955e-01, -2.71740519e-01,
        -4.24732029e-01, -1.73658128e-01,  4.33464529e-01,
         6.33263075e-02,  3.57971879e-01, -2.17939893e-01,
        -2.49654760e-01, -1.18526165e-01, -1.50002093e-02,
         1.74397727e-01, -2.78323143e-02, -6.45026811e-02,
         2.38753391e-01,  7.22215029e-01,  3.97690655e-01,
         1.88445474e-02, -5.82673183e-01, -3.15036716e-01,
        -3.56449926e-01, -4.56648600e-01,  8.35931502e-02,
        -1.48264948e-01, -4.14429972e-01,  2.39983123e-01,
         4.02459372e-02,  6.05390383e-02, -5.15149170e-01,
        -5.01102898e-02,  5.54498286e-01, -1.12097349e-01,
        -3.70442340e-01, -3.83705703e-01, -7.69207324e-02,
        -1.89989363e-03, -7.44284729e-02,  1.76954341e-01,
        -2.79431121e-01, -4.13692523e-01,  2.33563281e-01,
        -2.24233700e-01,  3.85100449e-01,  6.75050666e-02,
         4.35397731e-01, -6.24265649e-02, -2.29845310e-01,
        -3.24847111e-01, -4.88223265e-01, -3.95634857e-01,
         1.43553237e-01, -6.05081431e-02,  6.73718191e-01,
         1.57692160e-01, -6.53968343e-01,  1.68237487e-01,
        -4.45824976e-01,  1.97730905e-02, -5.55966206e-02,
        -1.78466375e-01, -1.48773149e-01,  5.95264482e-02,
        -2.42719382e-01, -4.48966328e-01,  4.73738669e-01,
         4.72604894e-01,  4.46211642e-01, -3.85021101e-01,
        -4.38436911e-02,  1.32311859e-01,  1.52367269e-01,
         9.50610260e-02, -9.58343612e-02, -1.39823978e-01,
        -2.85338342e-02, -7.42496546e-01,  3.03727304e-01,
         1.52662884e-01, -1.90101918e-01, -4.67633798e-01,
        -3.29908433e-01,  6.03997699e-02,  2.03485424e-01,
        -2.36878565e-01, -1.61540735e-02,  2.56511791e-01,
         1.57036683e-01,  2.39362166e-01,  6.88482454e-02,
         2.13670040e-02, -6.48896466e-01, -4.82637243e-01,
        -2.27506585e-01,  2.01730244e-01,  5.18679840e-02,
        -3.35450201e-01, -6.60744005e-02,  1.36958297e-01,
         1.37852307e-01,  1.96026308e-01, -2.90461523e-01,
         2.22786015e-02, -1.00933104e-01, -1.93838981e-01,
         1.68878045e-01, -5.03720467e-01,  2.60452952e-01,
        -4.02204110e-01,  1.26508414e-01, -2.59025518e-02,
        -1.93040727e-01,  1.07080770e-01,  2.70678016e-01,
         1.18988953e-02,  1.52652435e-01, -4.88002358e-01,
        -6.22489995e-01,  6.97949758e-02,  1.52580029e-01,
        -6.24176077e-02,  6.80580721e-01, -3.89406206e-01,
        -3.91026872e-01,  1.31910609e-01,  2.79531277e-01,
        -7.45175290e-02,  3.04000232e-01, -8.51108344e-03,
        -2.47632299e-02,  1.35089944e-01, -2.63535886e-01,
         4.53258434e-01, -1.51977535e-01,  2.20650531e-01,
        -9.24290851e-01,  3.34291573e-02,  2.52248670e-01,
         1.83163864e-01, -1.36904566e-01,  7.30190884e-02,
         6.89720439e-01,  8.35183140e-02, -7.01379910e-02,
        -5.02715368e-01,  7.37751865e-02, -1.02196873e-01,
         2.87196276e-01, -6.14683881e-01,  1.16011579e-01,
         2.74788921e-02, -3.68458611e-01, -3.54120798e-01,
        -1.58303081e-02,  6.02769755e-01, -6.83444878e-03,
        -4.74124071e-02, -2.11884449e-01, -2.53743047e-01,
         1.05386537e-01, -5.65956013e-02, -2.09683827e-01,
        -2.86616700e-01, -9.75682890e-02, -2.84656810e-01,
        -2.53537889e-01,  1.77925383e-01, -6.08323086e-01,
         4.29058669e-01,  8.08038522e-04, -2.52758064e-02,
        -1.01284396e-02,  5.77455326e-01,  1.35145972e-01,
         4.72097043e-02,  2.07733537e-01,  8.20609298e-01,
         4.24608522e-01,  4.52917324e-01,  4.40304578e-01,
        -3.73876791e-01, -7.48326954e-03,  1.71600024e-02,
        -2.44523626e-01,  2.91542945e-01, -2.68188348e-01,
         4.22803612e-02,  3.11439145e-01,  9.17522257e-02,
         3.02028938e-01,  8.84342777e-02,  8.22547891e-02,
         1.99878992e-01, -3.43970810e-01,  4.12380024e-01,
         1.69280681e-01, -1.46531934e-01,  7.97833534e-02,
        -1.78213813e-02,  1.59781239e-01, -6.89144241e-01,
        -8.01093046e-02, -3.63227850e-01, -2.96488891e-01,
        -5.89228170e-01,  3.29018767e-01,  1.19464121e-01,
         2.67632845e-03, -4.68390175e-01, -8.19540811e-02,
         1.25703113e-01,  1.42866658e-02, -7.10609003e-02,
        -1.55499215e-01,  3.85523064e-01, -6.83413103e-02,
         2.03512876e-01, -4.17114434e-01, -5.80858358e-02,
        -5.91636892e-01, -1.62930170e-01,  9.64007455e-02,
         9.13954141e-02, -6.39961804e-01,  3.04683038e-01,
        -1.59211241e-01,  6.27356240e-01,  5.94011986e-01,
         4.19846730e-01,  2.68745376e-01, -4.25970241e-01,
        -3.66942608e-01,  1.09634589e-01, -1.99872552e-01,
        -4.74895230e-02, -8.07612786e-03,  9.59857062e-02,
        -2.75694329e-01,  1.25864244e-02, -2.60419454e-01,
        -2.40458041e-01,  4.13107358e-01,  1.70375692e-01,
         1.00274164e-01, -1.04778520e-01,  5.96103318e-02,
        -1.69251120e-01, -1.95043184e-01, -4.36966526e-02,
         1.26295223e-01,  9.72256337e-02, -5.43914477e-01,
         1.01937199e-01,  4.72518531e-02,  2.47032962e-02,
         2.42986289e-01,  1.30362385e-01,  2.17784471e-01,
        -1.10476362e-01, -6.61244523e-02,  5.02385998e-01,
        -3.98888551e-01, -1.17129292e-01, -1.95960376e-01,
        -2.11556313e-01, -5.10246819e-01, -1.69478688e-02,
         2.24134813e-01, -4.99869311e-02, -3.50332103e-01,
        -3.47561046e-01,  2.17322666e-01, -2.96812651e-01,
        -2.81390285e-01, -3.28575951e-01,  1.73065087e-01,
        -4.26874245e-01,  5.26387939e-01, -3.95042580e-01,
        -1.62937632e-01,  4.95463208e-01, -1.23184403e-01,
         5.14798161e-01,  5.39282084e-01,  8.11104864e-02,
        -1.72317887e-01, -1.73809657e-01,  1.04497012e-01,
         5.35120927e-01,  2.42950414e-01, -3.73482712e-01,
        -4.67044862e-01, -8.84617375e-01, -2.40162471e-02,
         3.68431014e-01, -1.97815891e-01,  1.48909074e-01,
         2.01286416e-01,  9.12555484e-03,  3.20532081e-02,
        -2.42595841e-01, -2.74002185e-01, -5.16082823e-01,
        -1.46847941e-01,  4.68037071e-02,  2.86277008e-01,
        -2.76224590e-01,  4.41456063e-04, -4.59803525e-02,
        -1.89536266e-01, -1.50970153e-01,  7.06992646e-02,
         8.44216810e-02, -5.26837071e-02,  1.22335757e-01,
        -2.52308651e-01, -3.20090717e-01,  1.26527448e-01,
         1.90725563e-01, -6.38253229e-01,  2.72911018e-01,
         1.85404424e-01,  1.79891680e-01,  2.77231408e-01,
         7.06403367e-03, -2.83942697e-01,  2.65126816e-01,
        -1.00684872e-01,  2.34816318e-01,  2.30760136e-01,
        -2.56843438e-01, -5.44260506e-02, -3.41263388e-01,
         3.81031323e-01, -5.73381623e-01, -4.26413161e-01,
         2.83260926e-01, -4.20856254e-01,  1.72534165e-01,
        -1.15368949e+00,  3.05293452e+00, -1.11882910e+00])
rotation_coefs[-2] *= 2
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
        x = np.array(list(np.abs(run_model_masked(base_img)[0] - run_model_masked(rot_img)[0])) + full_contour_similarity(base_img, rot_img) + [calc_iou(base_img, rot_img), calc_iou(base_img, rot_img, N = 20)])
        probs.append(sigmoid(np.dot(x, rotation_coefs) + rotation_intercept ))
    return probs

def run_rotation_model_big(base_img, rotation_image):
    probs = np.zeros(4)
    for i in range(4):
        probs += np.array(run_rotation_model(base_img, rotation_image, i)) / 4
    
    print("[VISION] : ", probs, np.argmax(probs))
    return probs

def run_model(img, image_size = 224):
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()
    img = cv2.resize(img, (image_size, image_size))
    img = (img - img.mean()) / img.std()
    # img = ((img / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    ref = torch.from_numpy(img[:image_size, :image_size, :].reshape(1, image_size, image_size, 3)).float().permute(0, 3, 1, 2).to(device)
    ref_pred = model(ref)
    return ref_pred.cpu().detach().numpy()

def run_model_masked(img, image_size = 224):
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

    