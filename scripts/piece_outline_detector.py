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

from std_msgs.msg      import Bool
from sensor_msgs.msg   import Image, CameraInfo
from nav_msgs.msg      import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion

class PuzzlePiece:
    # def __init__(self, bbox, centroid):
    #     self.x_center, self.y_center = centroid
    #     self.xmin, self.ymin, self.width, self.height, self.area = bbox

    def __init__(self, mask):
        self.update_mask(mask)
        
        self.color = tuple(map(int, np.random.random(size=3) * 255))

        self.matched = True
        self.removed = False

        self.img = None
        self.natural_img = None
    
    def update_mask(self, mask):
        ys, xs = np.where(mask)
        self.xmin, self.xmax = np.min(xs), np.max(xs)
        self.ymin, self.ymax = np.min(ys), np.max(ys)
        self.area   = np.sum(mask)
        self.width  = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.x_center = np.mean(xs).astype(int)
        self.y_center = np.mean(ys).astype(int)

        self.mask = mask.astype(np.float32)

    def copy(self):
        copy = PuzzlePiece(self.mask.copy())
        copy.img = self.img
        return copy

    def move_to(self, new_x_center, new_y_center):
        dx = new_x_center - self.x_center
        dy = new_y_center - self.y_center

        M = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])
        new_mask = cv2.warpAffine(self.mask, M, self.mask.shape[::-1])
        self.update_mask(new_mask)

    def move_to_no_mask(self, new_x_center, new_y_center):
        dx = new_x_center - self.x_center
        dy = new_y_center - self.y_center
        self.xmin += dx
        self.xmax += dx
        self.ymin += dy
        self.ymax += dy
        self.x_center += dx
        self.y_center += dy
        self.mask = None # invalidates mask

    def rotate(self, theta):
        # theta in radians
        center = (self.x_center, self.y_center)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=theta*180/np.pi, scale=1)
        new_mask = cv2.warpAffine(self.mask, rotate_matrix, self.mask.shape[::-1])
        self.update_mask(new_mask)

    def bounds_slice(self, padding=0):
        return (slice(max(0, self.ymin - padding), self.ymin+self.height+padding, 1), 
                slice(max(0, self.xmin - padding), self.xmin+self.width+padding, 1))

    def get_center(self):
        return (self.x_center, self.y_center)

    def set_img(self, img):
        self.img = img
    def set_natural_img(self, natural_img):
        self.natural_img = natural_img
    def get_color(self):
        return self.color

    def __repr__(self):
        return f"<Puzzle Piece at (x={self.x_center}, y={self.y_center}), width={self.width}, height={self.height} with color={self.color}>"

    def matches(self, other):
        return (self.x_center - other.x_center) ** 2 + (self.y_center - other.y_center) ** 2 < 10**2

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

    def fully_contained_in_region(self, region):
        xmin, ymin, xmax, ymax = region
        return (xmin < self.xmin < xmax) and \
               (xmin < self.xmax < xmax) and \
               (ymin < self.ymin < ymax) and \
               (ymin < self.ymax < ymax)

    def overlaps_with_region(self, region):
        xmin, ymin, xmax, ymax = region
        for (x, y) in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
            if (self.xmin < x < self.xmin + self.width) and \
               (self.xmin < y < self.xmin + self.height):
                return True
        for (x, y) in [(self.xmin, self.ymin),
                       (self.xmin, self.ymin+self.height),
                       (self.xmin+self.width, self.ymin),
                       (self.xmin+self.width, self.ymin+self.height)]:
            if (xmin < x < xmax) and (ymin < y < ymax):
                return True
        return False
#
#  Detector Node Class
#
class Detector:
    def __init__(self, continuous=False):
        # Grab an instance of the camera data.
        rospy.loginfo("Waiting for camera info...")
        msg = rospy.wait_for_message('/usb_cam/camera_info', CameraInfo)
        self.camD = np.array(msg.D).reshape(5)
        self.camK = np.array(msg.K).reshape((3,3))
        self.transform = None

        # Subscribe to the incoming image topic.  Using a queue size
        # of one means only the most recent message is stored for the
        # next subscriber callback.
        self.imgsub = rospy.Subscriber("/usb_cam/image_raw", Image, self.img_sub_callback, queue_size=1)

        # Takes the most recent image and runs the piece detector on it
        self.snapsub = rospy.Subscriber("/detector/snap", Bool, self.save_img, queue_size=1)

        # Binary map publisher for rviz
        self.map_pub = rospy.Publisher('map', OccupancyGrid, queue_size=1)
        self.map_data_pub = rospy.Publisher('map_metadata',
                                             MapMetaData, queue_size=1)
        self.map_pub_counter = 0
        self.map_pub_every = 50

        # Set up the OpenCV Bridge.
        self.bridge = cv_bridge.CvBridge()

        # Publish to the processed image.  Store up to three images,
        # in case any clients need a little more time.
        self.pub_bluedots  = rospy.Publisher("/detector/pieces", Image, queue_size=3)
        self.pub_binary    = rospy.Publisher("/detector/blocks", Image, queue_size=3)

        # If True, run detector on every new image
        # If False, only run detector on snap() or /detector/snap
        self.continuous = continuous
        
        # TODO: Lock?
        self.latestImage = None
        
        # List of puzzle piece objects
        # TODO: Lock?
        self.pieces = list()

        self.free_space_img = None

        # ARUCO
        self.arucoDict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()

    def img_sub_callback(self, msg):
        self.save_img(msg)
        if self.continuous:
            self.snap()

    def save_img(self, msg):
        self.latestImage = self.crop_raw(self.bridge.imgmsg_to_cv2(msg, "bgr8"))


    def snap(self):
        if self.latestImage is None:
            rospy.logwarn("[Detector] Waiting for image from camera...")
            while self.latestImage is None:
                pass
            rospy.loginfo("[Detector] Recieved image from camera")
        bluedots_img, binary_img, free_space_img = self.process(self.latestImage)
        self.pub_bluedots.publish(self.bridge.cv2_to_imgmsg(bluedots_img, "bgr8"))
        self.pub_binary.publish(self.bridge.cv2_to_imgmsg(binary_img))
        self.free_space_img = free_space_img

    def crop_raw(self, img):
        # Crops unnecessary parts of the image out
        return img[:, 200:, :]

    def init_aruco(self):
        image_msg = rospy.wait_for_message("/usb_cam/image_raw", Image)
        image = self.crop_raw(self.bridge.imgmsg_to_cv2(image_msg, "bgr8"))

        (all_corners, ids, rejected) = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        if not all_corners:
            raise RuntimeError("Aruco markers not found!!")

        # Undistort corners
        all_corners = np.array(all_corners).reshape((-1,2))
        #all_corners = cv2.undistortPoints(np.float32(all_corners), self.camK, self.camD)
        all_corners = np.array(all_corners).reshape((-1,2))

        if len(all_corners) != 16:
            rospy.loginfo(all_corners)
            rospy.loginfo(ids)
            raise RuntimeError("Incorrect number of aruco marker corners:" + str(len(all_corners)))

        #Real Coordinates
        world1 = np.array([-.4325, -.1519])
        world2 = np.array([.1982, 0.2867])
        world3 = np.array([-0.4146, 0.2654])
        world4 = np.array([0.2335, -0.1208])

        box1 = all_corners[0:4]
        box2 = all_corners[4:8]
        box3 = all_corners[8:12]
        box4 = all_corners[12:16]

        screen1 = np.mean(box1, axis=0)
        screen2 = np.mean(box2, axis=0)
        screen3 = np.mean(box3, axis=0)
        screen4 = np.mean(box4, axis=0)

        ids = ids.flatten()
        ids_reorder = np.zeros_like(ids)
        for i in range(len(ids)):
            ids_reorder[i] = np.where(ids == i)[0]
        screens = np.float32([screen1, screen2, screen3, screen4])[ids_reorder]
        worlds = np.float32([world1, world2, world3, world4])

        print(screens)
        print(worlds)

        self.transform = cv2.getPerspectiveTransform(screens, worlds)

    def world_to_screen(self, x, y):
        raise NotImplementedError()

    def screen_to_world(self, x, y):
        #coords = cv2.undistortPoints(np.float32([[[x, y]]]), self.camK, self.camD)
        coords = np.float32([[[x, y]]])
        x, y = cv2.perspectiveTransform(coords, self.transform)[0, 0]
        return (x, y)

    def get_random_piece(self):
        print("pieces:", self.pieces)
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
        grid_msg.info.resolution = 0.00035
        grid_msg.info.width = img.shape[1]
        grid_msg.info.height = img.shape[0]

        # Rotated maps are not supported... quaternion represents no
        # rotation.
        grid_msg.info.origin = Pose(Point(-0.44, -0.12, 0),
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
        img_orig = img.copy()
        # Filter out background
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        background_lower = (0, 0, 50)
        background_upper = (255, 30, 220)
        binary = cv2.inRange(hsv, background_lower, background_upper)

        # Part of the image which is the puzzle pieces
        blocks = 255 - binary

        # Remove noise
        blocks = cv2.dilate(blocks, None, iterations=2)
        blocks = cv2.erode(blocks, None, iterations=2)

        # Perform 3 iterations of eroding (by distance)
        piece_centers = blocks
        for i in range(3):
            dist_transform = cv2.distanceTransform(piece_centers,cv2.DIST_L2,5)
            _, piece_centers = cv2.threshold(dist_transform,7,255,0)
            piece_centers = piece_centers.astype(np.uint8)

        # One more eroding for good measure
        #piece_centers = cv2.erode(piece_centers, None, iterations=4)

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

        pieces = list()

        # Unmatch all pieces
        for piece in self.pieces:
            piece.matched = False

        for i in range(len(centroids)):
            # Background
            if i == 0:
                continue

            if i+1 not in markers:
                continue

            piece_mask = (markers == i+1)
            piece = PuzzlePiece(piece_mask)
            #if piece.is_valid():
            if True:
                # First try to match the piece
                for existing_piece in self.pieces:
                    if not existing_piece.matched:
                        if existing_piece.matches(piece):
                            existing_piece.update_mask(piece_mask)
                            existing_piece.matched = True
                            piece = existing_piece
                            break
                pieces.append(piece)

                cutout_img = img_orig[piece.bounds_slice(padding=30)].copy()
                piece.set_img(cutout_img)

                cutout_img = img_orig[piece.bounds_slice(padding=10)].copy()
                piece.set_natural_img(cutout_img)

        # Show a circle over each detected piece
        for piece in pieces:
            r = int(np.sqrt(piece.area) / 4) + 1
            color = piece.get_color()
            cv2.circle(img, piece.get_center(), r, color, -1)

        #markers[res != 0] = 255

        # publish map
        self.publish_map_msg(binary[::-1], force=not self.continuous)

        self.pieces = pieces

        table = 255 - blocks
        return img, markers, table

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
