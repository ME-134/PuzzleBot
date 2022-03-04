import cv2
import numpy as np

class ThomasPuzzlePiece:
    def __init__(self, x, y, w, h, area):
        self.x = x
        self.y = y
        self.width  = w
        self.height = h
        self.area   = area

        self.color = tuple(map(int, np.random.random(size=3) * 255))

        self.matched = True
        self.removed = False

        self.img = None

    def get_location(self):
        return (self.x, self.y)

    def set_location(self, x, y):
        self.x = x
        self.y = y

    def set_img(self, img):
        self.img = img

    def get_color(self):
        return self.color

    def __repr__(self):
        return f"<Puzzle Piece at (x={self.x}, y={self.y}) with color={self.color}>"

    def matches(self, other):
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 < 4900

    def is_valid(self):
        if not (2000 < self.area):
            return False
        if not (50 < self.width):
            return False
        if not (50 < self.height):
            return False
        return True

    def is_multiple(self):
        # Assume valid
        if self.area > 5000:
            return True
        if self.width > 120 or self.height > 120:
            return True
        return False
    
    def get_bounding_box(self, threshold = 100, erosion = 1, dilation = 1):
        # Compute a contour and then use the largest countour to build a bounding box
        filtered = self.img
        filtered = cv2.dilate(filtered, np.ones((dilation, dilation), np.uint8))
        filtered = cv2.erode(filtered, np.ones((erosion, erosion), np.uint8))
        # import matplotlib.pyplot as plt
        # plt.imshow(filtered)
        # plt.show()
        contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(biggest_contour)
        self.box_raw = cv2.boxPoints(rect)
        
        return self.box_raw
    
    def get_rotation_to_align(self, compute_bounding_box = False, box_raw = None):
        # Use the vector of the top of the bounding box to orient the piece. 
        if compute_bounding_box:
            self.get_bounding_box()
        if (box_raw == None):
            box_raw = self.box_raw
            
        top_line_vector = box_raw[1] - box_raw[0]
        top_line_vector = top_line_vector / np.linalg.norm(top_line_vector)
        return np.arccos(top_line_vector.dot(np.array([1, 0])))
    

class ThomasDetector:
    def __init__(self):
        self.piece_centers = list()
        self.pieces = list()
     

    def process(self, img):

        # Filter out background
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        background_lower = (0, 0, 80)
        background_upper = (255, 30, 220)
        binary = cv2.inRange(hsv, background_lower, background_upper)
        
        # Part of the image which is the puzzle pieces
        blocks = 255 - binary
        
        # Remove noise
        blocks = cv2.dilate(blocks, None, iterations=1)
        blocks = cv2.erode(blocks, None, iterations=1)
        
        # Perform 2 iterations of eroding (by distance)
        piece_centers = blocks
        for i in range(2):
            dist_transform = cv2.distanceTransform(piece_centers,cv2.DIST_L2,5)
            _, piece_centers = cv2.threshold(dist_transform,2,255,0)
            piece_centers = piece_centers.astype(np.uint8)
    
        # One more eroding for good measure
        piece_centers = cv2.erode(piece_centers, None, iterations=4)

        n, markers, stats, centroids = cv2.connectedComponentsWithStats(piece_centers)
        
        # Increment so that background is not marked as "unknown"
        markers = markers + 1
        
        for i, stat in enumerate(stats):
            # Background

            xmin, ymin, width, height, area = tuple(stat)
            
            # This component is either noise or part of another piece that broke off
            # Mark its area as "unknown" to be filled by watershed
            if area < 200:
                markers[markers == (i+1)] = 0
        
        # Mark unknown regions
        # This is where it's part of a block but we're not sure which one it's part of.
        unknown = cv2.subtract(blocks, piece_centers)
        markers[unknown == 255] = 0
        
        # Convert image to RGB because watershed only works on RGB
        blobs = cv2.cvtColor(blocks, cv2.COLOR_GRAY2RGB)

        # Hooray
        cv2.watershed(blobs, markers)
        markers = markers.astype(np.uint8)

        # Outline pieces in original image as blue
#         img[markers == 255] = (255, 0, 0)

        #return img, markers
        
        piece_centers = list()
        pieces = list()

        # Unmatch all pieces
        for piece in self.pieces:
            piece.matched = False

        #print(stats)
        for i, stat in enumerate(stats):
            # Background
            if i == 0:
                continue
            
            if i+1 not in markers:
                continue
            
            xmin, ymin, width, height, area = tuple(stat)
            centroid = tuple(np.array(centroids[i]).astype(np.int32))
            piece = ThomasPuzzlePiece(centroid[0], centroid[1], width, height, area)
            #if piece.is_valid():
            if True:
                # First try to match the piece
                for existing_piece in self.pieces:
                    if not existing_piece.matched:
                        if existing_piece.matches(piece):
                            existing_piece.set_location(centroid[0], centroid[1])
                            existing_piece.width = width
                            existing_piece.height = height
                            existing_piece.matched = True
                            piece = existing_piece
                            break
                pieces.append(piece)

                cutout_img = blocks[max(ymin-100,0):ymin+height+100, max(xmin-100,0):xmin+height+100].copy()
#                 cutout_img = img[ymin:ymin+height, xmin:xmin+height].copy()
                piece.set_img(cutout_img)


        for piece in pieces:
            r = int(np.sqrt(piece.area) / 4) + 1
            color = piece.get_color()
            cv2.circle(img, piece.get_location(), r, color, -1)
            piece_centers.append(piece.get_location())

        #markers[res != 0] = 255

        self.piece_centers = piece_centers
        self.pieces = pieces

        return img, markers
