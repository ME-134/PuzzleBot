#!/usr/bin/env python3
#
#   detector.py
#
#   Detect the tennis balls with OpenCV.
#
#   Subscribers:    /usb_cam/image_raw     Source image
#
#   Publishers:     /detector/image_raw    Debug image
#

# ROS Imports
import rospy
import cv2
import cv_bridge
import numpy as np
import random

import os

from sensor_msgs.msg  import Image, CameraInfo

class PuzzlePiece:
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
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 < 10**2

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

#
#  Detector Node Class
#
class Detector:
    def __init__(self):
        # Grab an instance of the camera data.
        rospy.loginfo("Waiting for camera info...")
        msg = rospy.wait_for_message('/usb_cam/camera_info', CameraInfo);
        self.camD = np.array(msg.D).reshape(5)
        self.camK = np.array(msg.K).reshape((3,3))
        
        # Subscribe to the incoming image topic.  Using a queue size
        # of one means only the most recent message is stored for the
        # next subscriber callback.
        #rospy.init_node('detector')
        rospy.Subscriber("/usb_cam/image_raw", Image, self.getPiecesesAndPublish,
                         queue_size=1)

        # Set up the OpenCV Bridge.
        self.bridge = cv_bridge.CvBridge()
        

        # Publish to the processed image.  Store up to three images,
        # in case any clients need a little more time.
        self.pub_bluedots = rospy.Publisher("/detector/pieces", Image, queue_size=3)
        self.pub_binary   = rospy.Publisher("/detector/blocks", Image, queue_size=3)
        
        self.piece_centers = list()
        self.pieces = list()
        
        # ARUCO
        self.arucoDict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.latestImage = None

    def getPiecesesAndPublish(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latestImage = img
        bluedots_img, binary_img = self.better_process(img)
        self.pub_bluedots.publish(self.bridge.cv2_to_imgmsg(bluedots_img, "bgr8"))
        self.pub_binary.publish(self.bridge.cv2_to_imgmsg(binary_img))
        
    def init_aruco(self):
        image_msg = rospy.wait_for_message("/usb_cam/image_raw", Image)
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        (all_corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        rospy.loginfo(all_corners)
        if not all_corners:
            raise RuntimeError("Aruco markers not found!!")
        
        # Undistort corners
        all_corners = np.array(all_corners).reshape((-1,2))
        all_corners = cv2.undistortPoints(np.float32(all_corners), self.camK, self.camD)
        all_corners = np.array(all_corners).reshape((-1,2))
        
        ids = ids.flatten()
        '''A = np.append(all_corners, np.ones((len(all_corners[:, 0]), 1)), axis=1)
        points = np.array(
            [[1,1]
            ]
        )
        M, _, _, _ = np.linalg.lstsq(A, points, rcond=None)
        self.M = M.transpose()
        self.M = cv2.getPerspectiveTransform(all_corners, points)
        print(list(ids.flatten()).index(0))
        print(all_corners)'''

        box = all_corners[:4]
        xmin = np.min(box[:, 0])
        xmax = np.max(box[:, 0])
        ymin = np.min(box[:, 1])
        ymax = np.max(box[:, 1])
        #rospy.loginfo("xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

        w = 0.0195
        self.xb = (xmax - xmin) / w
        self.xa = xmin - self.xb * (-0.337)
        self.yb = -(ymax - ymin) / w
        self.ya = ymin - self.yb * (0.001)
        
    def world_to_screen(self, x, y):
        print(self.xa, self.xb, self.ya, self.yb, x, y)
        return (self.xa + x * self.xb, self.ya + y * self.yb)
        
    def screen_to_world(self, x, y):
        coords = cv2.undistortPoints(np.float32([[[x, y]]]), self.camK, self.camD)
        #points = cv2.perspectiveTransform(coords, self.M)
        x, y = coords[0,0,:]
        return ((x - self.xa) / self.xb, (y - self.ya) / self.yb)
        
    def get_random_piece_center(self):
        print("piece centers:", self.piece_centers)
        return random.choice(self.piece_centers)

    def get_random_piece(self):
        print("pieces centers:", self.pieces)
        return random.choice(self.pieces)

    def better_process(self, img):

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
        img[markers == 255] = (255, 0, 0)

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
            piece = PuzzlePiece(centroid[0], centroid[1], width, height, area)
            #if piece.is_valid():
            if True:
                # First try to match the piece
                for existing_piece in self.pieces:
                    if not existing_piece.matched:
                        if existing_piece.matches(piece):
                            existing_piece.set_location(centroid[0], centroid[1])
                            existing_piece.matched = True
                            piece = existing_piece
                            break
                pieces.append(piece)

                #cutout_img = img[ymin:ymin+height, xmin:xmin+height].copy()
                #piece.set_img(cutout_img)


        for piece in pieces:
            r = int(np.sqrt(piece.area) / 4) + 1
            color = piece.get_color()
            cv2.circle(img, piece.get_location(), r, color, -1)
            piece_centers.append(piece.get_location())

        #markers[res != 0] = 255

        self.piece_centers = piece_centers
        self.pieces = pieces

        return img, markers
        

    def process(self, img):

        edges = cv2.Canny(img, 20, 200)
        edges = cv2.dilate(edges, None, iterations=2)
        blocks = 255 - edges
        n, res, stats, centroids = cv2.connectedComponentsWithStats(blocks)
        res = res.astype(np.uint8)
        
        piece_centers = list()

        def isPuzzlePiece(stat):
            xmin, ymin, width, height, area = tuple(stat)
            if not (1000 < area < 4000):
                return False
            if not (50 < width  < 150):
                return False
            if not (50 < height < 150):
                return False
            return True
            '''
            if not (5000 < area < 10000):
                return False
            if not (60 < width  < 200):
                return False
            if not (60 < height < 200):
                return False
            return True
            '''
        #print(stats)
        for i, stat in enumerate(stats):
            area = stat[-1]
            centroid = tuple(np.array(centroids[i]).astype(np.int32))
            if isPuzzlePiece(stat):
                piece = (res == i)
                res[piece] = 255
                r = int(np.sqrt(area) / 4) + 1
                color = (np.random.random(size=3) * 255).astype(np.uint8)
                #print(centroid, r, color)
                #cv2.circle(img, centroid, r, color) 
                cv2.circle(img, centroid, r, (255, 0, 0), -1)
                #cv2.dilate(piece, None, iterations=1)
                piece_centers.append(centroid)
            else:
                res[res == i] = 0

        self.piece_centers = piece_centers

        #res = cv2.dilate(res, None, iterations=2)
        #res = cv2.erode(res, None, iterations=2)
        #res = cv2.dilate(res, None, iterations=2)
        #print(n, stats, centroids)
        #edges = cv2.erode(edges, None, iterations=1)
        #edges = cv2.erode(edges, None, iterations=1)
        #edges = cv2.dilate(edges, None, iterations=1)
        return img, res

#
#  Main Code
#
if __name__ == "__main__":
    # Prepare the node.  You can override the name using the
    # 'rosrun .... __name:=something' convention.
    rospy.init_node('piece_detector')

    # Instantiate the Detector object.
    detector = Detector()
    
    # Use Aruco markers to find transforms
    detector.init_aruco()
    print(detector.screen_to_world(2, 4))
    '''
    indir = './vision/sample_imgs'
    outdir = './vision/out'
    for filename in os.listdir(indir):
        filepath = os.path.join(indir, filename)
        img = cv2.imread(filepath)
        result = detector.process(img)
        outfilepath = os.path.join(outdir, filename)
        cv2.imwrite(outfilepath, result) 
    exit(1)
    '''
    
    # Continually process until shutdown.
    rospy.loginfo("Continually processing latest pending images...")
    rospy.spin()

    # Report completion.
    rospy.loginfo("Done!")
