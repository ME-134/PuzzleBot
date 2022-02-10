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
#import imutils

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
        #rospy.Subscriber("/usb_cam/image_raw", Image, self.process,
        #                 queue_size=1)

        # Set up the OpenCV Bridge.
        self.bridge = cv_bridge.CvBridge()
        
        self.centroid = [None,None]
        self.previouscentroid=[None,None]

        # Publish to the processed image.  Store up to three images,
        # in case any clients need a little more time.
        #self.puboutline = rospy.Publisher("/detector/ball_outline", Image,
        #                                  queue_size=3)
        #self.pubbinary  = rospy.Publisher("/detector/binary", Image,
        #                                  queue_size=3)

    def process(self, img):
        # Convert into OpenCV image.
        #image = self.bridge.imgmsg_to_cv2(imagemsg, "bgr8")
        #img = cv.imread(filename, 0)

        # Gaussian Blur
        #r = 1
        #image = cv2.GaussianBlur(image, (r, r), 0)
        edges = cv2.Canny(img, 20, 200)
        edges = cv2.dilate(edges, None, iterations=2)
        #out = img.copy()
        blocks = 255 - edges
        n, res, stats, centroids = cv2.connectedComponentsWithStats(blocks)
        res = res.astype(np.uint8)

        def isPuzzlePiece(stat):
            xmin, ymin, width, height, area = tuple(stat)
            if not (5000 < area < 10000):
                return False
            if not (60 < width  < 200):
                return False
            if not (60 < height < 200):
                return False
            return True

        for i, stat in enumerate(stats):
            area = stat[-1]
            centroid = tuple(np.array(centroids[i]).astype(np.int32))
            if isPuzzlePiece(stat):
                piece = (res == i)
                res[piece] = 255
                r = int(np.sqrt(area) / 4) + 1
                color = (np.random.random(size=3) * 255).astype(np.uint8)
                print(centroid, r, color)
                #cv2.circle(img, centroid, r, color) 
                cv2.circle(img, centroid, r, (255, 0, 0), -1)
                #cv2.dilate(piece, None, iterations=1)
            else:
                res[res == i] = 0

        #res = cv2.dilate(res, None, iterations=2)
        #res = cv2.erode(res, None, iterations=2)
        #res = cv2.dilate(res, None, iterations=2)
        #print(n, stats, centroids)
        #edges = cv2.erode(edges, None, iterations=1)
        #edges = cv2.erode(edges, None, iterations=1)
        #edges = cv2.dilate(edges, None, iterations=1)
        return img

        # Convert to HSV
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold

        # hue 17 47 hue
        # sat 39 233
        # val 105 255
        #greenLower = (30, 50, 6)
        #greenUpper = (64, 255, 255)
        
        greenLower = (17, 39, 90)
        greenUpper = (47, 255, 255)
        binary = cv2.inRange(hsv, greenLower, greenUpper)

        # Erode and Dilate
        binary = cv2.erode(binary, None, iterations=2)
        binary = cv2.dilate(binary, None, iterations=2)

        # Alternate erode/dilate/erode.
        #binary = cv2.erode(binary, None, iterations=10)
        #binary = cv2.dilate(binary, None, iterations=20)
        #binary = cv2.erode(binary, None, iterations=10)


        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        (contours, hierarchy) = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours on the original image for debugging.
        cv2.drawContours(image, contours, -1, (255,0,0), 2)

        # only proceed if at least one contour was found
        if len(contours) > 0:
            # Pick the largest contour.
            contour = max(contours, key=cv2.contourArea)

            # Find the enclosing circle (convert to pixel values)
            ((xr, yr), radius) = cv2.minEnclosingCircle(contour)
            xr     = int(xr)
            yr     = int(yr)
            radius = int(radius)

            # Find the centroid of the contour (in pixel values).
            M  = cv2.moments(contour)
            xc = int(M["m10"] / M["m00"])
            yc = int(M["m01"] / M["m00"])
            self.centroid = [xc,yc]

            # Report.
            #rospy.loginfo("Found Ball at (%d,%d), enclosed by radius %d about (%d,%d)",
            #              xc, yc, radius, xr, yr)

            # Only proceed if the radius meets a minimum size
            if radius > 5:
                # Draw the circle and centroid on the original image.
                cv2.circle(image, (xr, yr), int(radius), (0, 255, 255),  2)
                cv2.circle(image, (xc, yc), 5,           (0,   0, 255), -1)
            else:
                self.centroid = [None,None]
        else:
            self.centroid = [None,None]


        # Convert back into a ROS image and republish (for debugging).
        #self.puboutline.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

        # Alternatively, publish the black/white image.
        #self.pubbinary.publish(self.bridge.cv2_to_imgmsg(binary))


#
#  Main Code
#
if __name__ == "__main__":
    # Prepare the node.  You can override the name using the
    # 'rosrun .... __name:=something' convention.

    # Instantiate the Detector object.
    detector = Detector()
    indir = './vision/sample_imgs'
    outdir = './vision/out'
    for filename in os.listdir(indir):
        filepath = os.path.join(indir, filename)
        img = cv2.imread(filepath)
        result = detector.process(img)
        outfilepath = os.path.join(outdir, filename)
        cv2.imwrite(outfilepath, result) 
    exit(1)

    # Continually process until shutdown.
    rospy.loginfo("Continually processing latest pending images...")
    rospy.spin()

    # Report completion.
    _rospy.loginfo("Done!")
