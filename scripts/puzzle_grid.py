from typing import List
# from piece_outline_detector import PuzzlePiece
import numpy as np

def get_xy_min(arr):
    argmin = np.argmin(arr)
    y = argmin % arr.shape[1]
    x = (argmin - y) // arr.shape[1]

    return (x, y)

class PuzzleGrid():
    def __init__(self, width_n = 5, height_n = 4, offset = np.array([0, 0]), spacing_height = 200, spacing_width = 200, offset_x = 200, offset_y = 200):
        self.width_n = width_n
        self.height_n = height_n
        self.occupied = np.zeros((width_n, height_n))
        self.oriented = np.zeros((width_n, height_n))
        self.spacing_height = spacing_height
        self.spacing_width = spacing_width
        self.offset = offset
        self.grid_centers = np.array(
            [[[offset_x + i*spacing_width, offset_y + j*spacing_height]for j in range(height_n)] for i in range(width_n)]
        )
        self.piece = None

    
    def get_open_spot(self):
        '''
            Returns the x, y grid location for the first open spot, or none if there is no
            open location
        '''
        x, y = get_xy_min(self.occupied)

        if (self.occupied[x, y] == 0):
            return (x, y)
        else:
            return None

    
    def get_unoriented_spot(self):
        '''
            Returns the x, y grid location for the first non_oriented spot, or none if there is no
            open location
        '''
        x, y = get_xy_min(self.oriented)

        if (self.oriented[x, y] == 0):
            return (x, y)
        else:
            return None

    def process_filled(self, pieces):
        for piece in pieces:
            piece_xy = np.array(piece.x, piece.y)
            norm1 = np.abs(self.grid_centers - piece_xy).sum(axis = 2)
            min_norm1, (x_min, y_min) = np.min(norm1), get_xy_min(norm1)
            if (min_norm1 < 2*self.spacing):
                self.occupied[x_min, y_min] = 1
    
    def get_neighbors(self, loc):
        '''
            Returns neighbors in completed puzzle
        '''
        neighbors = []
        if loc[0]+1 < self.width_n and self.occupied[loc[0]+1, loc[1]] != 0:
            neighbors.append([loc[0]+1, loc[1]])
        if loc[0]-1 >= 0 and self.occupied[loc[0]-1, loc[1]] != 0:
            neighbors.append([loc[0]-1, loc[1]])
        if loc[1]+1 < self.height_n and self.occupied[loc[0], loc[1]+1] != 0:
            neighbors.append([loc[0], loc[1]+1])
        if loc[1]-1 >= 0 and self.occupied[loc[0], loc[1]-1] != 0:
            neighbors.append([loc[0], loc[1]-1])
        return np.array(neighbors)

    def does_mate(self, loc):
        '''
            Checks if a grid location mates with the completed puzzle so far
        '''
        if loc[0]+1 < self.width_n and self.occupied[loc[0]+1, loc[1]] != 0:
            return True
        if loc[0]-1 >= 0 and self.occupied[loc[0]-1, loc[1]] != 0:
            return True
        if loc[1]+1 < self.height_n and self.occupied[loc[0], loc[1]+1] != 0:
            return True
        if loc[1]-1 >= 0 and self.occupied[loc[0], loc[1]-1] != 0:
            return True
        return False

    def grid_to_pixel(self, loc):
        '''
            Returns position in pixel space for a piece on the grid
        '''
        if self.piece is not None:
            # Find the offset by top left corner
            drift = np.array([self.piece.xmin, self.piece.ymin]) - self.offset
        else:
            drift = 0

        pos = np.array([[[ 75,  60],
                [ 65, 170],
                [ 75, 280],
                [ 75, 393]],

                [[212,  58],
                [199 ,175],
                [207 ,285],
                [207 ,394]],

                [[354,  57],
                [339, 170],
                [339, 275],
                [339, 384]],

                [[481,  60],
                [481, 175],
                [476, 285],
                [471, 395]],

                [[613,  60],
                [603, 170],
                [608, 280],
                [603, 392]]])
        return pos[loc[0]][loc[1]] + self.offset + drift
        