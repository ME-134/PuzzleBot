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
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion

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
        self.imgsub = rospy.Subscriber("/usb_cam/image_raw", Image, self.getPiecesesAndPublish,
                                        queue_size=1)
        
        self.map_pub = rospy.Publisher('map', OccupancyGrid)
        self.map_data_pub = rospy.Publisher('map_metadata', 
                                             MapMetaData)
        self.map_pub_counter = 0
        self.map_pub_every = 50

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
        bluedots_img, binary_img = self.process(img)
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
        
        if len(all_corners) != 8:
            raise RuntimeError("Incorrect number of aruco marker corners:" + len(all_corners))
        
        #Real Coordinates
        world1 = np.array([-.3543, -.0046])
        world2 = np.array([.1696, 0.1574])
        
        box1 = all_corners[0:4]
        box2 = all_corners[4:8]
        screen1 = np.mean(box1, axis=0)
        screen2 = np.mean(box2, axis=0)
        #print(screen1, screen2)
        
        #screen1 = (box1])
        #screen2 = (all_corners[0][0])
        
        
        ids = ids.flatten()
        '''A = np.append(all_corners, np.ones((len(all_corners[:, 0]), 1)), axis=1)
        points = np.array(
            [[1,1]
            ]
        )
        M, _, _, _ = np.linalg.lstsq(A, points, rcond=None)
        self.M = M.transpose()
        self.M = cv2.getPerspectiveTransform(all_corners, points)'''

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
        
        #print(screen1, screen2)
        #print((world2[0] - world1[0]))
        #print((world2[1] - world1[1]))
        
        self.xb = (screen2[0] - screen1[0]) / (world2[0] - world1[0])
        self.xa = screen1[0] - self.xb * (world1[0])
        self.yb = (screen2[1] - screen1[1]) / (world2[1] - world1[1])* 0.16 / 0.14
        self.ya = screen1[1] - self.yb * (world1[1])
        
    def world_to_screen(self, x, y):
        #return self.a + x * self.b
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
        
    def publish_map_msg(self, img, force=False):
        self.map_pub_counter = (self.map_pub_counter + 1) % self.map_pub_every
        if not force and self.map_pub_counter > 0:
            return
        
        grid_msg = OccupancyGrid()

        # Set up the header.
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        # .info is a nav_msgs/MapMetaData message. 
        grid_msg.info.resolution = 0.00095
        grid_msg.info.width = img.shape[1]
        grid_msg.info.height = img.shape[0]
        
        # Rotated maps are not supported... quaternion represents no
        # rotation. 
        grid_msg.info.origin = Pose(Point(-0.38, -0.05, 0),
                               Quaternion(0, 0, 0, 1))

        # Flatten the numpy array into a list of integers from 0-100.
        # This assumes that the grid entries are probalities in the
        # range 0-1. This code will need to be modified if the grid
        # entries are given a different interpretation (like
        # log-odds).
        flat_grid = img.reshape((img.size,)) * (100 / 255)
        grid_msg.data = list(np.round(flat_grid).astype(int))
        self.map_data_pub.publish(grid_msg.info)
        self.map_pub.publish(grid_msg)

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
        
        # publish map
        self.publish_map_msg(binary[::-1])
        
        '''
        from pathlib import Path
        mappath = Path(__file__) / '../map/map.png'
        #print(mappath, mappath.resolve())
        mappath = '/home/me134/me134ws/src/HW1/map/map.png'
        if np.random.rand() < 0.01:
            cv2.imwrite(mappath, blocks)
        '''

        self.piece_centers = piece_centers
        self.pieces = pieces

        #        print(self.screen_to_world(self.piece_centers[0][0],self.piece_centers[0][1]))
        #print(self.xb, self.yb, 1/self.xb, 1/self.yb)
        return img, markers

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
