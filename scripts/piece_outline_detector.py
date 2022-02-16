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

    def better_process(self, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        background_lower = (0, 0, 120)
        background_upper = (255, 30, 220)
        binary = cv2.inRange(hsv, background_lower, background_upper)
        blocks = 255 - binary
        
        n, res, stats, centroids = cv2.connectedComponentsWithStats(blocks)
        res = res.astype(np.uint8)
        
        piece_centers = list()

        def isPuzzlePiece(stat):
            xmin, ymin, width, height, area = tuple(stat)
            if not (2000 < area < 5000):
                return False
            if not (50 < width  < 120):
                return False
            if not (50 < height < 120):
                return False
            return True

        #print(stats)
        for i, stat in enumerate(stats):
            area = stat[-1]
            centroid = tuple(np.array(centroids[i]).astype(np.int32))
            if isPuzzlePiece(stat):
                piece = (res == i)
                #res[piece] = 255
                r = int(np.sqrt(area) / 4) + 1
                color = (np.random.random(size=3) * 255).astype(np.uint8)
                #print(centroid, r, color)
                #cv2.circle(img, centroid, r, color) 
                cv2.circle(img, centroid, r, (255, 0, 0), -1)
                #cv2.dilate(piece, None, iterations=1)
                piece_centers.append(centroid)
            else:
                pass
                #res[res == i] = 0
        res[res != 0] = 255

        self.piece_centers = piece_centers
        
        #res = cv2.dilate(res, None, iterations=2)
        #res = cv2.erode(res, None, iterations=2)
        #res = cv2.dilate(res, None, iterations=2)
        #print(n, stats, centroids)
        #edges = cv2.erode(edges, None, iterations=1)
        #edges = cv2.erode(edges, None, iterations=1)
        #edges = cv2.dilate(edges, None, iterations=1)
        return img, res
        

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
