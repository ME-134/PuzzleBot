import cv2
import numpy as np

def calc_rotation(biggest_contour, step = 2, n_thetas = 90):
    derivatives = np.zeros_like(biggest_contour, dtype = np.float32)
    
    for i in range(len(biggest_contour)):
        prev_point = biggest_contour[(i - step) % len(biggest_contour)]
        next_point = biggest_contour[(i + step) % len(biggest_contour) ]
        v = next_point - prev_point
        derivatives[i] = (-v[1], v[0])
        derivatives[i] = derivatives[i] / (0.00001 + np.linalg.norm(derivatives[i]))

    thetas = np.linspace(0, np.pi/4, n_thetas)
    theta_sums = []
    for theta in thetas:
        rotation_vectors = np.array([[np.sin(theta), np.cos(theta)], 
                                     [np.cos(theta), -np.sin(theta)]])
        min_vec = 2*(np.min(np.square((derivatives @ rotation_vectors)), axis = 1))
        theta_sums.append(np.sum(min_vec))
    
    return thetas[np.argmin(theta_sums)]

def get_piece_mask(img):
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

    return blocks

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
        self.natural_img = None
        self.embedding = np.zeros((512, 4), dtype = np.float32)

    def get_location(self):
        return (self.x, self.y)

    def set_location(self, x, y):
        self.x = x
        self.y = y

    def set_img(self, img):
        self.img = img

    def set_natural_img(self, natural_img):
        self.natural_img = natural_img

    def get_color(self):
        return self.color

    def __repr__(self):
        return f"<Puzzle Piece at (x={self.x}, y={self.y}) with color={self.color}>"

    def matches(self, other):
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 < 4900

    def is_valid(self):
        if not (2000 < self.area):
            return False
        if not (70 < self.width):
            return False
        if not (70 < self.height):
            return False
        return True

    def is_multiple(self):
        # Assume valid
        if self.area > 5000:
            return True
        if self.width > 120 or self.height > 120:
            return True
        return False
    
    def get_bounding_box(self, threshold = 100, erosion = 3, dilation = 3, filter_iters = 0):
        # Compute a contour and then use the largest countour to build a bounding box
        filtered = self.img
        for i in range(filter_iters):
            filtered = cv2.dilate(filtered, np.ones((dilation, dilation), np.uint8))
            filtered = cv2.erode(filtered, np.ones((erosion, erosion), np.uint8))


        corner_vals = cv2.cornerHarris(filtered,7,7,0.09) > 1.5
        corners = []
        for i in range(corner_vals.shape[0]):
            for j in range(corner_vals.shape[1]):
                if corner_vals[i, j] > 0:
                    corners.append((j, i))
        corners = np.array(corners)    
        rect = cv2.minAreaRect(corners)
        self.box_raw = cv2.boxPoints(rect)
        
        return self.box_raw

    def get_largest_contour(self, threshold = 100, erosion = 5, dilation = 3, filter_iters = 2) :
        # Compute a contour and then use the largest countour to build a bounding box
        filtered = self.img
        for i in range(filter_iters):
            filtered = cv2.dilate(filtered, np.ones((dilation, dilation), np.uint8))
            filtered = cv2.erode(filtered, np.ones((erosion, erosion), np.uint8))
        
        # Old contour based box:
        contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key = cv2.contourArea)
        return biggest_contour

    def get_bounding_box_contour(self, threshold = 100, erosion = 5, dilation = 3, filter_iters = 2):
        # Compute a contour and then use the largest countour to build a bounding box
        biggest_contour = self.get_largest_contour(threshold = 100, erosion = 5, dilation = 3, filter_iters = 2)
        rect = cv2.minAreaRect(biggest_contour)
        self.box_raw = cv2.boxPoints(rect)
        
        return self.box_raw
    
    def get_rotation_to_align_old(self, compute_bounding_box = False, box_raw = None, maxWidth = 100, maxHeight = 100):
        # Use the vector of the top of the bounding box to orient the piece. 
        if compute_bounding_box:
            corner_box = self.get_bounding_box()
            contour_box = self.get_bounding_box_contour()
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
            M = cv2.getPerspectiveTransform(corner_box.astype(np.float32), dst)
            warped_corner = cv2.warpPerspective(self.img.copy(), M, (100, 100))
            M = cv2.getPerspectiveTransform(contour_box.astype(np.float32), dst)
            warped_contour = cv2.warpPerspective(self.img.copy(), M, (100, 100))

            if (warped_corner.mean() > (warped_contour.mean() + 0.25)):
                self.box_raw = corner_box
                box_raw = corner_box
            else:
                self.box_raw = contour_box
                box_raw = contour_box

        if (box_raw is None):
            box_raw = self.box_raw
            
        top_line_vector = box_raw[1] - box_raw[0]
        top_line_vector = top_line_vector / np.linalg.norm(top_line_vector)
        return np.arccos(top_line_vector.dot(np.array([1, 0])))

    def get_rotation_to_align(self, erosion = 5, dilation = 3, filter_iters = 2, **kwargs):
        biggest_contour = self.get_largest_contour(threshold = 100, erosion = 5, dilation = 3, filter_iters = 2)
        return calc_rotation(biggest_contour)
    
    def get_warped(self, maxWidth = 300, maxHeight = 300):
        box = self.get_bounding_box()
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
        warped = cv2.warpPerspective(self.natural_img, M, (maxWidth, maxHeight))
        return warped

class ThomasDetector:
    def __init__(self):
        self.piece_centers = list()
        self.pieces = list()
     

    def process(self, img):

        blocks = get_piece_mask(img)
        
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
                cutout_img = blocks[max(ymin-10,0):ymin+height+10, max(xmin-10,0):xmin+width+10].copy()
                piece.set_img(cutout_img)

                y_max, x_max = blocks.shape[:2]
                cutout_img = img[max(ymin-10,0):min(ymin+height+10, y_max),
                                    max(xmin-10,0):min(xmin+width+10, x_max)].copy()
                piece.set_natural_img(cutout_img)

        for piece in pieces:
            r = int(np.sqrt(piece.area) / 5) + 1
            color = piece.get_color()
            cv2.circle(img, piece.get_location(), r, color, -1)
            piece_centers.append(piece.get_location())

        #markers[res != 0] = 255

        self.piece_centers = piece_centers
        self.pieces = pieces

        return img, markers
