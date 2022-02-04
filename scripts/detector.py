#!/usr/bin/env python3
#
#   detector.py
#
#   Detect the tennis balls with OpenCV.
#
#   Subscribers:    /usb_cam/image_raw     Source image
#
#   Publishers:     /detector/image_raw    Debug image

import rospy
import cv2
import cv_bridge
import imutils
from grip import GripPipeline

from sensor_msgs.msg  import Image


class Detector:
    def __init__(self):

        rospy.Subscriber("/usb_cam/image_raw", Image, self.process,
                         queue_size=1)

        self.bridge = cv_bridge.CvBridge()
        self.pubimage = rospy.Publisher("/detector/image_raw", Image,
                                        queue_size=1)
        self.grip = GripPipeline()

    def process(self, imagemsg):
        image = self.bridge.imgmsg_to_cv2(imagemsg, "bgr8")
        contours = self.grip.process(image)
        cv2.drawContours(image, contours, -1, (255,0,0), 2)
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

            # Report.
            rospy.loginfo("Found Ball at (%d,%d), enclosed by radius %d about (%d,%d)",
                          xc, yc, radius, xr, yr)

            # Only proceed if the radius meets a minimum size
            if radius > 10:
                # Draw the circle and centroid on the original image.
                cv2.circle(image, (xr, yr), int(radius), (0, 255, 255),  2)
                cv2.circle(image, (xc, yc), 5,           (0,   0, 255), -1)

        # Convert back into a ROS image and republish (for debugging).
        self.pubimage.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

    def process2(self, imagemsg):
        image = self.bridge.imgmsg_to_cv2(imagemsg, "bgr8")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        binary = cv2.inRange(hsv, greenLower, greenUpper)
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

            # Report.
            rospy.loginfo("Found Ball at (%d,%d), enclosed by radius %d about (%d,%d)",
                          xc, yc, radius, xr, yr)

            # Only proceed if the radius meets a minimum size
            if radius > 10:
                # Draw the circle and centroid on the original image.
                cv2.circle(image, (xr, yr), int(radius), (0, 255, 255),  2)
                cv2.circle(image, (xc, yc), 5,           (0,   0, 255), -1)

        # Convert back into a ROS image and republish (for debugging).
        self.pubimage.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

if __name__ == "__main__":
    rospy.init_node('detector')
    detector = Detector()

    # Continually process until shutdown.
    rospy.loginfo("Continually processing latest pending images...")
    rospy.spin()

    # Report completion.
    rospy.loginfo("Done!")
