from typing import List
# from piece_outline_detector import PuzzlePiece
import numpy as np

def get_xy_min(arr):
    argmin = np.argmin(arr)
    y = argmin % arr.shape[1]
    x = (argmin - y) // arr.shape[0]

    return (x, y)

class PuzzleGrid():
    def __init__(self, width_n = 5, height_n = 4, spacing_height = 200, spacing_width = 200, offset_x = 200, offset_y = 200):
        self.width_n = width_n
        self.height_n = height_n
        self.occupied = np.zeros((width_n, height_n))
        self.oriented = np.zeros((width_n, height_n))
        self.spacing_height = spacing_height
        self.spacing_width = spacing_width
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.grid_centers = np.array(
            [[[offset_x + i*spacing_width, offset_y + j*spacing_height]for j in range(height_n)] for i in range(width_n)]
        )

    '''
        Returns the x, y grid location for the first open spot, or none if there is no
        open location
    '''
    def get_open_spot(self):
        x, y = get_xy_min(self.occupied)

        if (self.occupied[x, y] == 0):
            return (x, y)
        else:
            return None

    '''
        Returns the x, y grid location for the first non_oriented spot, or none if there is no
        open location
    '''
    def get_unoriented_spot(self):
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
        