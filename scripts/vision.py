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

rotation_coefs = np.array([-4.43217370e-01, -3.54750184e-01,  4.32681003e-01,
        -3.07911142e-01,  2.83969383e-01, -3.53036316e-02,
        -6.41144918e-02,  5.16931090e-01,  3.66014299e-01,
        -4.22591484e-01, -3.93110993e-01,  2.37920661e-01,
        -4.17186199e-01, -4.11802206e-01, -2.81450882e-01,
         3.18158545e-02, -3.58844663e-01, -1.26828741e-01,
         4.29413520e-01, -2.70645472e-01,  2.16801927e-01,
         2.81460348e-01,  4.33267136e-02,  6.71886904e-02,
         6.04577451e-02,  4.94528650e-01,  1.22271409e-01,
        -6.45076117e-02,  2.28669824e-01, -1.05729420e-01,
         2.21131768e-01,  6.15391202e-01, -5.05119214e-01,
         1.07801955e-01, -2.27598969e-01,  5.21326773e-01,
         3.63420751e-01,  3.39573054e-01, -3.33682962e-01,
         5.90158252e-01, -4.54257697e-01,  3.30287664e-01,
        -4.98947508e-01, -2.10115437e-01, -4.38218778e-01,
         3.64251116e-01,  3.02243432e-01, -3.24815586e-01,
         3.05396391e-01,  8.01906134e-02,  2.05631266e-01,
        -4.04819462e-01,  5.05980899e-01, -1.11923353e-01,
         6.69209714e-02,  4.50700369e-01, -3.99116305e-01,
        -2.72902645e-01, -1.47975463e-01,  1.51376871e-01,
        -5.10611150e-02, -5.53239957e-01, -1.93964260e-01,
        -7.89584173e-01,  1.52071337e-01, -3.59084804e-02,
         1.18525418e-01, -3.00760852e-01, -1.32519837e-01,
        -2.31708355e-01,  5.01120634e-01, -1.78099936e-01,
        -7.41935203e-01, -2.05069159e-01,  5.74407972e-03,
        -2.67976005e-01, -3.33272552e-01, -2.03328458e-01,
         3.74982210e-01, -2.71271579e-01,  3.19497993e-01,
         2.35010674e-01,  6.19103254e-01,  5.91795079e-01,
         3.05091537e-01, -3.42496571e-01,  4.43417622e-01,
        -2.78271919e-01,  5.58293924e-01,  5.46988272e-02,
        -2.59942242e-01, -2.43893709e-01,  2.41831736e-01,
        -1.27890261e-01, -2.12596948e-01,  8.07294429e-01,
         6.37145529e-02, -2.79357789e-01,  1.42057955e-01,
         6.59660146e-01,  1.46017163e-02, -7.91534669e-02,
         3.32990200e-01, -3.13801658e-01,  6.51040479e-02,
         3.25226486e-01, -1.44500734e-01, -2.35246477e-01,
        -1.67697507e-01, -2.66187980e-01, -2.41755753e-01,
         7.06719500e-01,  4.12350495e-01,  3.85977513e-01,
        -2.45881343e-01,  6.49655745e-02, -5.88851470e-01,
         2.14932589e-01, -2.26877849e-01, -2.93865064e-01,
         7.88713086e-01,  3.19002457e-01,  2.56644485e-01,
         7.70155550e-03,  3.70215309e-01, -6.39761625e-02,
        -8.87930756e-02, -2.97348660e-01, -8.12071071e-01,
        -4.12371733e-01,  8.82351035e-01,  4.40685316e-01,
         2.61358567e-01, -2.48701392e-01, -1.30422284e-01,
         4.65920358e-01, -1.12526899e-01, -1.09380337e-01,
         4.14837747e-01, -4.07208938e-01, -5.95677515e-01,
        -1.94366691e-01, -7.28418284e-01,  2.16028682e-01,
         1.21134349e-01,  4.94407553e-01, -4.91323411e-02,
        -5.49895340e-01,  3.86012731e-01,  5.69481201e-01,
         7.96246630e-02,  1.09269001e-02,  5.58503347e-01,
         2.74376994e-01, -4.82795727e-01, -8.25819816e-01,
         1.04281074e-01, -2.56903137e-01,  1.01618689e-01,
         3.08348292e-01, -2.97319420e-01, -1.24612400e-01,
         2.56769508e-01, -1.68543710e-01, -1.42756492e-01,
        -6.67447767e-01, -3.03159994e-01,  4.11824687e-03,
        -3.65468893e-02,  3.77100487e-01,  7.73535976e-01,
        -1.48524289e-01, -8.94890933e-02, -4.09250530e-01,
        -4.43704718e-01,  2.87846202e-01, -7.57538614e-01,
        -1.66022209e-01,  2.67300789e-01,  2.52933870e-01,
         3.96971209e-01,  9.52116295e-02, -6.26280677e-01,
        -6.33633031e-01, -1.18051764e-02,  2.55663565e-01,
        -7.10136444e-01,  1.83555294e-01,  2.06072665e-01,
        -2.43755994e-01,  6.38626841e-01, -6.50395497e-02,
        -3.29091093e-01,  2.93258759e-01, -4.97656265e-02,
        -1.16568808e-01, -2.17486931e-01, -6.16381615e-01,
        -5.09012239e-01, -2.86880410e-01,  6.86295537e-01,
         2.78413160e-01,  4.76138760e-02,  2.12773083e-01,
         5.09916630e-01,  4.42973228e-01,  4.51894190e-01,
         2.69441136e-01, -2.76092931e-01,  6.59468663e-02,
        -2.81497250e-01, -2.61331436e-01,  1.61520108e-01,
        -3.84023709e-01, -1.05283929e-01, -3.56660509e-01,
        -9.49895354e-01,  4.17564970e-02, -8.65654624e-01,
         2.23581977e-01,  8.72246869e-02, -1.12995450e-01,
        -7.61122406e-01, -2.21834238e-01, -4.20556010e-01,
        -2.34175525e-01, -1.13218015e-01, -4.78881124e-01,
        -4.73776497e-01,  1.77774400e-01,  2.19015266e-01,
        -2.70303343e-01, -1.50907130e-01,  3.84949959e-01,
         1.00144060e-02,  1.78967314e-01,  1.70212268e-01,
        -3.26582806e-01, -9.45561756e-02,  1.61859455e-02,
        -3.07094990e-01,  4.72318465e-01,  6.36263070e-03,
        -7.75096102e-01,  7.18751439e-01, -1.30115855e-01,
         1.02947413e+00,  2.28182626e-01,  5.11349621e-01,
         2.60877819e-01, -5.94629313e-02,  1.24770420e-01,
        -1.23358752e-01,  4.23778623e-01, -4.66356859e-03,
         9.56405838e-02, -3.61239121e-01, -3.39562011e-01,
        -1.37046768e+00, -1.44124815e-01, -3.74623662e-02,
        -2.64045469e-01, -1.00279924e-02, -5.85063442e-01,
        -1.56366514e-01, -6.24949400e-01, -1.13470218e-02,
        -8.97320953e-02, -4.23164765e-01,  7.88553043e-02,
         4.95437352e-02,  1.49915913e-01,  2.90051716e-01,
         7.08025336e-02, -4.50721630e-01,  1.94643613e-01,
        -4.97234961e-02, -3.92695125e-01, -5.84372734e-01,
        -3.53691062e-01, -1.08213874e-01,  1.89302662e-01,
         3.05472939e-01,  2.36615732e-02,  6.03273844e-01,
         1.98860513e-01,  1.31129324e-02, -4.33807763e-01,
         2.43790363e-02,  2.63052604e-01, -6.32208974e-01,
        -4.99971521e-01,  5.03429817e-01, -2.46570570e-02,
        -6.73248671e-01, -4.13799079e-02,  3.15258026e-01,
        -3.48745798e-01,  5.01212916e-01,  3.67882821e-01,
         4.36388272e-01, -2.27960911e-01,  2.62350228e-01,
        -4.93581441e-01, -7.17624956e-01, -7.09525252e-02,
        -1.62594399e-01, -6.49772993e-01,  3.37403902e-01,
        -4.27290476e-01,  4.69640635e-01, -6.28954230e-02,
        -4.47194807e-01, -1.12957330e-01, -6.32243446e-01,
        -2.31959156e-01,  3.64880073e-01, -2.12015520e-01,
         1.67908717e-01,  1.33748181e-01, -2.96069432e-01,
        -2.88421993e-01, -3.87319524e-01, -1.90931259e-01,
        -6.18279817e-01, -4.66953126e-01, -5.07304386e-01,
         3.96110552e-01,  3.70949012e-01,  6.16917147e-01,
        -1.12680702e-01,  1.23171268e-01, -3.76893959e-01,
         2.13566663e-01,  4.44826091e-01, -3.38422088e-01,
        -2.14127210e-01,  3.61119573e-01, -2.19867981e-02,
         5.45343078e-02,  4.74889424e-01, -5.42610530e-01,
         2.86629289e-01,  3.10663549e-03, -3.93469441e-01,
        -1.14609353e-02,  6.66746168e-01,  9.32865900e-01,
         1.77945837e-01, -1.91417039e-02,  1.65795300e-01,
         5.18597488e-01,  7.57059287e-01,  2.34152233e-01,
        -3.46500627e-01,  3.03527789e-01, -3.11481749e-01,
        -1.51690255e-01, -7.86553518e-02, -6.11996766e-02,
        -5.04218957e-01,  1.33023193e-01, -1.59405660e-01,
        -4.66716029e-01, -1.09131160e-01,  1.69980442e-01,
        -5.27905181e-01, -1.39157842e-01, -6.86874174e-02,
         3.49170011e-02,  3.22874536e-01, -7.17706138e-02,
        -3.54340865e-02, -3.45990777e-01, -5.21054300e-01,
         3.51206022e-01,  6.81082236e-01, -9.91207164e-02,
         6.54787395e-01,  3.75405960e-01,  5.05011176e-01,
        -6.92172075e-01, -8.24183292e-01,  5.99862534e-01,
         1.10499834e+00, -9.03861444e-02,  6.70864080e-01,
        -2.56932028e-01,  5.15739465e-01,  5.93768320e-01,
        -2.04717519e-01, -1.96129873e-01, -1.40150846e-01,
        -9.43020535e-01, -2.85189943e-01,  2.32822974e-01,
        -3.32854425e-01, -3.98763385e-01, -1.40435043e-02,
        -4.15812667e-01, -2.52040784e-02,  2.04645322e-01,
         5.44082402e-01,  2.78667385e-01,  2.44614307e-01,
         4.16889902e-01, -7.68291327e-01, -5.93776845e-01,
         5.83483882e-02,  3.50627779e-01, -8.00888834e-01,
         9.26796548e-03, -1.93637538e-01,  1.62605031e-03,
        -3.98580345e-01,  2.70566732e-02,  1.90362732e-01,
        -3.64244895e-01, -1.29847599e-01,  2.56833820e-01,
         4.88084202e-01,  3.18268844e-01, -1.85592236e-01,
         1.69444905e-01,  4.39992324e-01, -2.03944279e-01,
         4.08859267e-01, -7.52799115e-02,  2.01780676e-01,
        -3.66497338e-01,  1.52852792e-01, -4.28409183e-01,
         1.82682484e-01,  2.06252026e-02, -8.15906143e-01,
         1.00451075e+00,  3.28013895e-01,  3.99982126e-01,
        -1.76092231e-01, -3.07782325e-01,  4.77769348e-01,
        -6.06164216e-01,  8.99512019e-02, -8.38912289e-01,
         2.02452537e-01, -7.53056159e-02, -2.31101259e-01,
         1.17186937e-01, -3.80273730e-01,  2.39526374e-02,
        -1.03859307e+00, -6.30765098e-02,  4.23263922e-01,
        -4.32721273e-01,  3.73487140e-02, -9.62812175e-01,
         5.98145907e-01, -1.96639027e-01, -3.06090925e-01,
         5.90288641e-01, -7.13453589e-01, -5.78252501e-01,
         2.66668694e-01,  4.65710191e-01,  8.41407065e-02,
        -4.60244585e-01, -2.29170599e-01, -2.89933774e-01,
         2.12671818e-02,  3.52517403e-01, -7.23688971e-01,
         6.23348532e-02, -7.83146823e-01, -1.32974371e+00,
         4.08579263e-01, -2.68377652e-01, -2.35253264e-01,
        -8.09783464e-01, -3.11141422e-01, -8.66035487e-02,
        -1.68878290e-01, -2.01848503e-01,  1.71270010e-01,
        -4.01529250e-01,  3.09937011e-01, -5.19798667e-01,
        -3.77654541e-01, -4.72274769e-01, -4.81113702e-01,
        -1.61931686e-01,  3.81943787e-01,  8.51005538e-02,
         2.10829185e-01, -2.24872137e-02, -1.05937390e-01,
        -4.58180690e-01, -3.20557634e-01, -2.43088369e-01,
        -2.68142076e-02, -3.07535707e-01, -8.68790627e-02,
        -6.98850478e-02, -1.12084817e-01, -3.25427145e-01,
        -7.96723010e-02, -4.05499318e-01, -7.79287105e-02,
         7.44906414e-01, -1.72817504e-01, -4.28755157e-01,
         1.20154032e-01,  2.22723217e-01,  4.66253408e+00])

rotation_intercept = -0.08146978

def norm(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 0.0001)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_rotation_model(base_img, rotation_image, k_init = 0):
    base_img = np.rot90(base_img, k = k_init)
    probs = []
    for k in range(4):
        rot_img = np.rot90(rotation_image, k = k)
        x = np.array(list(np.abs(run_model_masked(base_img)[0] - run_model_masked(rot_img)[0])) + [calc_iou(base_img, rot_img)])
        probs.append(sigmoid(np.dot(x, rotation_coefs) + rotation_intercept ))
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
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
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

        return xy_min, np.argmin(run_rotation_model(base, img) ) #np.argmin(norm(argmin_basic) + norm(argmin_iou) + norm(argmin_pca))
    
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
                ious = [calc_iou(np.rot90(pieces[i].natural_img, k = k), base) for k in range(4)]
                argmin_basic = np.array(scores_all[i, locations[i, 0], locations[i, 1], :4])
                argmin_iou = 1-np.array(ious)
                sim_base = self.inferences[locations[i, 0]][locations[i, 1]]
                sims_rot = ((self.rotation_pca.transform(self.calculate_rotation_vectors(pieces[i].natural_img).T) - self.rotation_pca.transform(sim_base.reshape(1, -1))) ** 2).sum(axis=1)
                argmin_pca = np.array(sims_rot)

                rots[i] = np.argmin(norm(argmin_basic) + norm(argmin_iou) + norm(argmin_pca))
            return locations, rots, match_scores  # locations[i] gives grid location of piece i
        else:
            raise Exception("Not Implemented")

    