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

from sensor_msgs.msg  import Image


#
#  Detector Node Class
#
class Detector:
    def __init__(self):
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
        
        # ARUCO
        self.arucoDict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.latestImage = None

    def getPiecesesAndPublish(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latestImage = img
        bluedots_img, binary_img = self.process(img)
        self.pub_bluedots.publish(self.bridge.cv2_to_imgmsg(bluedots_img, "bgr8"))
        self.pub2.publish(self.bridge.cv2_to_imgmsg(binary_img))
        
    def init_aruco(self):
        image_msg = rospy.wait_for_msg("/usb_cam/image_raw")
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        all_corners, _, _ = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        rospy.loginfo(all_corners)
        if not all_corners:
            raise RuntimeError("Aruco marker not found!!")

        box = all_corners[0][0]
        xmin = np.min(box[:, 0])
        xmax = np.max(box[:, 0])
        ymin = np.min(box[:, 1])
        ymax = np.max(box[:, 1])
        rospy.loginfo("xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

        w = 0.0195
        self.xb = (xmax - xmin) / w
        self.xa = xmin - self.xb * (-0.337)
        self.yb = -(ymax - ymin) / w
        self.ya = ymin - self.yb * (0.001)
        
    def world_to_screen(self, x, y):
        print(self.xa, self.xb, self.ya, self.yb, x, y)
        return (self.xa + x * self.xb, self.ya + y * self.yb)
        
    def screen_to_world(self, x, y):
        return ((x - self.xa) / self.xb, (y - self.ya) / self.yb)
        
    def get_random_piece_center(self):
        print("piece centers:", self.piece_centers)
        return random.choice(self.piece_centers)

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