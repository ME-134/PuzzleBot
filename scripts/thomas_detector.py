import cv2
from matplotlib.pyplot import pie
import numpy as np
from itertools import combinations

# Set the approximate piece side length (in pixels).  This is used to
# sub-divide the long side of connected pieces.
SIDELEN = 125

# Set the number of points per side to match against another side.
SIDEPOINTS = 20

def calc_rotation(biggest_contour, step = 5, n_thetas = 90):
    biggest_contour = biggest_contour.reshape(-1, 2)
    derivatives = np.zeros_like(biggest_contour, dtype = np.float32)
    
    for i in range(len(biggest_contour)):
        prev_point = biggest_contour[(i - step) % len(biggest_contour)]
        next_point = biggest_contour[(i + step) % len(biggest_contour) ]
        v = next_point - prev_point
        derivatives[i] = (-v[1], v[0])
        derivatives[i] = derivatives[i] / (0.00001 + np.linalg.norm(derivatives[i]))

    thetas = np.linspace(0, np.pi, n_thetas)
    theta_sums = []
    for theta in thetas:
        rotation_vectors = np.array([[np.sin(theta), np.cos(theta)], 
                                     [np.cos(theta), -np.sin(theta)]])
        errors = np.square((derivatives @ rotation_vectors))
        
        min_vec = (np.min(errors, axis = 1))
        # min_vec = np.clip(min_vec, np.quantile(min_vec, 0.01), np.quantile(min_vec, 0.99))
        theta_sums.append(np.mean(min_vec))
    
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

def ToThomasPuzzlePiece(piece):
    tp = ThomasPuzzlePiece(0, 0, 0, 0, 0)
    tp.img = get_piece_mask(piece.natural_img)
    tp.natural_img = piece.natural_img
    return tp

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

    def get_location(self):
        return (self.x, self.y)

    def set_location(self, x, y):
        self.x = x
        self.y = y

    def set_img(self, img):
        cutout_img = np.zeros_like(img)
        cv2.drawContours(cutout_img, [self.get_largest_contour(image=img, erosion=0, dilation=0, filter_iters=0)], 0, color=255, thickness=cv2.FILLED)
        self.img = cutout_img

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
    
    def get_bounding_box(self, img = None, threshold = 100, erosion = 3, dilation = 3, filter_iters = 0):
        # Compute a contour and then use the largest countour to build a bounding box
        if img:
            filtered = img
        else:
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

    def get_largest_contour(self, image = None, threshold = 100, erosion = 5, dilation = 3, filter_iters = 0) :
        if image is None:
            image = self.img
        # Compute a contour and then use the largest countour to build a bounding box
        filtered = image
        for i in range(filter_iters):
            filtered = cv2.dilate(filtered, np.ones((dilation, dilation), np.uint8))
            filtered = cv2.erode(filtered, np.ones((erosion, erosion), np.uint8))
        
        # Old contour based box:
        contours, hierarchy = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

    def get_rotation_to_align(self, erosion = 1, dilation = 1, filter_iters = 0, **kwargs):
        # import matplotlib.pyplot as plt
        # plt.imshow(self.natural_img)
        # plt.show()
        biggest_contour = self.get_largest_contour(threshold = 100, erosion = erosion, dilation = dilation, filter_iters = filter_iters)
        return calc_rotation(biggest_contour)
    
    def get_warped(self, maxWidth = 300, maxHeight = 300):
        box = self.get_bounding_box()
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
        warped = cv2.warpPerspective(self.natural_img, M, (maxWidth, maxHeight))
        return warped

    def get_corners(self, ortho_threshold=.05):
        '''Finds the corners of the underlying polygon of the shape'''
        # Create a blank image, to allow the erosion and dilation without
        # interferring with other image elements.
        binary = np.zeros(np.array(self.img.shape)*3, dtype=np.uint8)
        binary[self.img.shape[0]:2*self.img.shape[0], self.img.shape[1]: 2*self.img.shape[1]] = self.img

        # kernel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=np.uint8)
        kernel = None
        # Dilate and erode to remove the holes.
        N = int(SIDELEN/8)
        binary = cv2.dilate(binary, kernel, iterations=N)
        binary = cv2.erode(binary,  kernel, iterations=N)

        # Erode and dilate to remove the tabs.
        N = int(SIDELEN/6)
        binary = cv2.erode(binary,  kernel, iterations=N)
        binary = cv2.dilate(binary, kernel, iterations=N)

        binary = binary[self.img.shape[0]:2*self.img.shape[0], self.img.shape[1]: 2*self.img.shape[1]]
        
        # Re-find the countour of the base shape.  Again, do not
        # approximate, so we get the full list of pixels on the boundary.
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return np.empty((0,1,2))
        base = contours[0]

        # Convert the base shape into a simple polygon.
        polygon = cv2.approxPolyDP(base, SIDELEN/5, closed=True)[:, 0, :]
        area = cv2.contourArea(polygon, oriented=False)
        n_pieces = np.round(area/SIDELEN/SIDELEN)

        def order_corners(corners):
            # Orders corners from top right clockwise
            mid = np.mean(corners, axis=0)
            new_corners = np.zeros((4, 2))
            for corner in corners - mid:
                if corner[0] >= 0 :
                    if corner[1] >= 0:
                        new_corners[0] = corner + mid
                    else:
                        new_corners[1] = corner + mid
                else:
                    if corner[1] >= 0:
                        new_corners[3] = corner + mid
                    else:
                        new_corners[2] = corner + mid
            return new_corners

        if len(polygon) > 4 and n_pieces == 1:
            # Not a rectangle!
            for corners in combinations(polygon, 4):
                ordered_corners = order_corners(corners)
                for i in range(len(ordered_corners)):
                    leg1 = ordered_corners[i-1]-ordered_corners[i]
                    leg1 = leg1/np.linalg.norm(leg1)
                    leg2 = ordered_corners[(i+1)%len(ordered_corners)]-ordered_corners[i]
                    leg2 = leg2/np.linalg.norm(leg2)
                    if abs(np.dot(leg1, leg2)) >= ortho_threshold:
                        break
                else:
                    polygon = corners
                    break
            else:
                # Couldn't find a good solution
                pass
        elif len(polygon) == 4:
            polygon = order_corners(polygon)
        else:
            # too few corners
            pass

        return np.array(polygon).astype(int)


    #
    #   Corner Indicies
    #
    #   Create a list of puzzle piece corners.  This also works on
    #   connected pieces, effectively sub-dividing long sides.
    #
    def _refineCornerIndex(self, contour, index):
        # Set up the parameters.
        N = len(contour)
        D = int(SIDELEN/6)          # Search a range +/- from the given
        d = int(SIDELEN/8)          # Compute the angle +/- this many pixels
        
        # Search for the best corner fit, checking +/- the given index.
        maxvalue = 0
        for i in range(index-D,index+D+1):
            p  = contour[(i  )%N, 0, :]
            da = contour[(i-d)%N, 0, :] - p
            db = contour[(i+d)%N, 0, :] - p
            value = (da[0]*db[1] - da[1]*db[0])**2
            if value > maxvalue:
                maxvalue = value
                index    = i%N

        # Return the best index.
        return(index)

    def _findCornerIndices(self, contour, polygon):
        # Prepare the list of corner indices.
        indices = []

        # Loop of the polygon points, sub-dividing long lines (across
        # multiple pieces) into single pieces.
        N = len(polygon)
        for i in range(N):
            p1 = polygon[ i,      :]
            p2 = polygon[(i+1)%N, :]

            # Sub-divide as appropriate.
            n  = int(round(np.linalg.norm(p2-p1) / SIDELEN))
            for j in range(n):
                p = p1*(n-j)/n + p2*j/n

                # Find the lowest distance to all contour points.
                d = np.linalg.norm(contour-p, axis=2)
                index = int(np.argmin(d, axis=0))

                # Refine the corner index for real corners.
                if (j == 0):
                    index = self._refineCornerIndex(contour, index)

                # Use that index.
                indices.append(index)

        # Return the indices.
        return(indices)

    
    #
    #   Find Sides
    #
    #   Process a contour (list of pixels on the boundary) into the sides.
    #
    def get_sides(self, contour=None):
        if contour is None:
            contour = self.get_largest_contour(threshold = 100, erosion = 0, dilation = 0, filter_iters = 0)
        # Create the base polygon.
        polygon = self.get_corners()

        # Get the indices to the corners.
        indices = self._findCornerIndices(contour, polygon)

        # Pull out the sides between the indicies.
        sides = []
        N = len(indices)
        for i in range(N):
            index1 = indices[i]
            index2 = indices[(i+1)%N]
            if (index1 <= index2):
                side = contour[index1:index2, 0, :]
            else:
                side = np.vstack((contour[index1:, 0, :],
                                contour[0:index2, 0, :]))
            sides.append(side)


        # Check the number of pieces (just for fun).
        A = cv2.contourArea(polygon, oriented=False)
        n = np.round(A/SIDELEN/SIDELEN)
        print("Guessing contour has %d pieces" % n)

        # Return the sides
        return sides


    #
    #   Check the Translation/Orientation/Match between 2 Sides
    #
    def compareSides(self, sideA, sideB):
        center = np.array(self.img.shape)/2
        # Grab the points from the two sides, relative to the center.
        M  = SIDEPOINTS
        iA = [int(round(j*(len(sideA)-1)/(M-1))) for j in range(M)]
        iB = [int(round(j*(len(sideB)-1)/(M-1))) for j in range(M-1,-1,-1)]
        pA = sideA[iA] - center
        pB = sideB[iB] - center

        # Pull out a list of the x/y coordinqtes.
        xA = pA[:,0].reshape((-1, 1))
        yA = pA[:,1].reshape((-1, 1))
        xB = pB[:,0].reshape((-1, 1))
        yB = pB[:,1].reshape((-1, 1))
        c0 = np.zeros((M,1))
        c1 = np.ones((M,1))

        # Build up the least squares problem for 4 parameters: dx, dy, cos, sin
        b  = np.hstack(( xA, yA)).reshape((-1,1))
        A1 = np.hstack(( c1, c0)).reshape((-1,1))
        A2 = np.hstack(( c0, c1)).reshape((-1,1))
        A3 = np.hstack((-yB, xB)).reshape((-1,1))
        A4 = np.hstack(( xB, yB)).reshape((-1,1))
        A  = np.hstack((A1, A2, A3, A4))

        param = np.linalg.pinv(A.transpose() @ A) @ (A.transpose() @ b)
        dtheta = np.arctan2(param[2][0], param[3][0])

        # Rebuild the least squares problem for 2 parameters: dx, dy
        b = b - A @ np.array([0, 0, np.sin(dtheta), np.cos(dtheta)]).reshape(-1,1)
        A = A[:, 0:2]

        param = np.linalg.pinv(A.transpose() @ A) @ (A.transpose() @ b)
        dx = param[0][0]
        dy = param[1][0]

        # Check the residual error.
        err = np.linalg.norm(b - A @ param) / np.sqrt(M)

        # Return the data.
        return (dx, dy, dtheta, err)

    def find_contour_match(self, other_piece, match_threshold=5, return_sides=False):
        # Finds the transform from this piece to other_piece based on contour

        def is_line(side, threshold=100):
            a = np.hstack(side[:, 0].reshape((-1, 1)), 1)
            b = side[:, 1]
            x, resid, _, _ = np.linalg.lstsq(a, b)
            return resid.sum() < threshold

        sidesA = other_piece.get_sides()
        sidesB = self.get_sides()
        best_err = np.inf
        ans = (0, 0, 0, [], []) if return_sides else (0, 0, 0)
        A_offset = np.array(other_piece.get_location()) + np.array(other_piece.img.shape)/2 - np.array(self.get_location()) - np.array(self.img.shape)/2
        for iA in range(len(sidesA)):
            if not is_line(sidesA[iA]):
                for iB in range(len(sidesB)):
                    if not is_line(sidesB[iB]):
                        (dx, dy, dtheta, err) = self.compareSides(sidesA[iA] + A_offset, sidesB[iB])
                        if err < match_threshold and best_err > err:
                            ans = (-dx, dy, dtheta, sidesA[iA], sidesB[iB]) if return_sides else (-dx, dy, dtheta)
        
        return ans
        


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
                spacing = 20
                pieces.append(piece)
                cutout_img = blocks[max(ymin-spacing,0):ymin+height+spacing, max(xmin-spacing,0):xmin+width+spacing].copy()
                piece.set_img(cutout_img)

                y_max, x_max = blocks.shape[:2]
                cutout_img = img[max(ymin-spacing,0):min(ymin+height+spacing, y_max),
                                    max(xmin-spacing,0):min(xmin+width+spacing, x_max)].copy()
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
