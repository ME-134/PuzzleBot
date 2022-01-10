#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import scipy.optimize.least_squares as least_squares

from geometry_msgs.msg       import Pose
from kinematics              import *
from std_msgs.msg            import Float64, Int16
from sensor_msgs.msg         import JointState
from geometry_msgs.msg       import PointStamped, Point
from gazebo_msgs.srv         import GetModelState, GetLinkState
from sklearn.model_selection import train_test_split
from scipy                   import optimize
from gazebo_msgs.srv         import SetModelState
from gazebo_msgs.msg         import ModelState

class fit_parabolic:
    def fit(self):
        def f(coefs):
            return 
    def __init__(self, points=[]):
        self.points = np.array(points)
        
    def update(self, points):
        self.points = np.array(points)
        

# Spawn model into gazebo
def spawn_model(sdf_path, model_name, pose=[0]*3):
    os.system(f"rosrun gazebo_ros spawn_model -file {sdf_path} -sdf -model {model_name} -x {pose[0]} -y {pose[1]} -z {pose[2]}")
 
# Send model pose   
def set_model_pose(model_name, pose=[0]*6, vel=[0]*6):
    msg = ModelState() 
    msg.model_name = model_name
    if len(pose) == 3:
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = float(pose[2])
    elif len(pose) == 6:
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = float(pose[2])
        msg.pose.position.roll = float(pose[3])
        msg.pose.position.pitch = float(pose[4])
        msg.pose.position.yaw = float(pose[5])
        
    if len(vel) == 3:
        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = float(vel[2])
    elif len(vel) == 6:
        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = float(vel[2])
        msg.twist.angular.x = float(vel[3])
        msg.twist.angular.y = float(vel[4])
        msg.twist.angular.z = float(vel[5])
        
    rospy.wait_for_service('/gazebo/set_model_state')
    
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(msg)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

#
#  Basic Rotation Matrices
#
#  Note the angle is specified in radians.
#
def Rx(phi):
    return np.array([[ 1, 0          , 0          ],
                     [ 0, np.cos(phi),-np.sin(phi)],
                     [ 0, np.sin(phi), np.cos(phi)]])

def Ry(phi):
    return np.array([[ np.cos(phi), 0, np.sin(phi)],
                     [ 0          , 1, 0          ],
                     [-np.sin(phi), 0, np.cos(phi)]])

def Rz(phi):
    return np.array([[ np.cos(phi),-np.sin(phi), 0],
                     [ np.sin(phi), np.cos(phi), 0],
                     [ 0          , 0          , 1]])

#
#  Simple Vector Utilities
#
#  Just collect a 3x1 column vector, perform a dot product, or a cross product.
#
def vec(x,y,z):
    return np.array([[x], [y], [z]])

def dot(a,b):
    return a.T @ b

def cross(a,b):
    return np.cross(a, b, axis=0)

# Error functions
def ep(pd, pa):
    return (pd - pa)
    
def eR(Rd, Ra):
    return 0.5*(np.cross(Ra[:,0], Rd[:,0]) +
                np.cross(Ra[:,1], Rd[:,1]) +
                np.cross(Ra[:,2], Rd[:,2]))

# Int Publisher
class IntPublisher:
    def __init__(self, i=0):
        self.i = i
        self.pub_int = rospy.Publisher("/num_balls", Int16, queue_size=1)
        
    def publish(self, i=None):
        if i is not None:
            self.i = i
        self.pub_int.publish(self.i)

#
#  Point Publisher
#
#  This continually publishes the point and corresponding marker array.
#
class PointPublisher:
    def __init__(self, p = [0.0, 0.0, 0.0]):
        # Prepare the publishers (latching for new subscribers).
        self.pub_point = rospy.Publisher("/point",
                                         PointStamped,
                                         queue_size=1, latch=True)

        # Create the point.
        self.p = Point()
        self.p.x = p[0]
        self.p.y = p[1]
        self.p.z = p[2]

        # Create the point message.
        self.point = PointStamped()
        self.point.header.frame_id = "world"
        self.point.header.stamp    = rospy.Time.now()
        self.point.point           = self.p

    def update(self, p):
        self.p.x = p[0]
        self.p.y = p[1]
        self.p.z = p[2]
        self.point.point = self.p

    def publish(self, p=None):
        if p is not None:
            self.p.x = p[0]
            self.p.y = p[1]
            self.p.z = p[2]
            self.point.point = self.p
        # Publish.
        now = rospy.Time.now()
        self.point.header.stamp  = now
        self.pub_point.publish(self.point)

    def loop(self):
        # Prepare a servo loop at 10Hz.
        servo = rospy.Rate(10)
        rospy.loginfo("Point-Publisher publication thread running at 10Hz...")

        # Loop: Publish and sleep
        while not rospy.is_shutdown():
            self.publish()
            servo.sleep()

        # Report the cleanup.
        rospy.loginfo("Point-Publisher publication thread ending...")

#
#  Joint Command Publisher
#
#  Publish the commands on /joint_states (so RVIZ can see) as well as
#  on the /sevenbot/jointX_position/command topics (for Gazebo).
#
class JointCommandPublisher:
    def __init__(self, urdfnames, controlnames):
        # Make sure the name lists have equal length.
        assert len(urdfnames) == len(controlnames), "Unequal lengths"

        # Save the dofs = number of names/channels.
        self.n = len(urdfnames)

        # Create a publisher to /joint_states and pre-populate the message.
        self.pubjs = rospy.Publisher("/joint_states",
                                     JointState, queue_size=100)
        self.msgjs = JointState()
        for name in urdfnames:
            self.msgjs.name.append(name)
            self.msgjs.position.append(0.0)
            self.msgjs.velocity.append(0.0)

        # Prepare a list of publishers for each joint commands.
        self.pubX  = []
        for name in controlnames:
            topic = "/sevenbot/" + name + "/command"
            self.pubX.append(rospy.Publisher(topic, Float64, queue_size=100))

        # Wait until connected.  You don't have to wait, but the first
        # messages might go out before the connection and hence be lost.
        # rospy.sleep(0.5)

        # Report.
        rospy.loginfo("Ready to publish command for %d DOFs", self.n)

    def dofs(self):
        # Return the number of DOFs.
        return self.n

    def send(self, q, qdot):
        # Send each individual command and populate the joint_states.
        for i in range(self.n):
            self.pubX[i].publish(Float64(q[i]))
            self.msgjs.position[i] = q[i]
            self.msgjs.velocity[i] = qdot[i]

        # Send the command (with specified time).
        self.msgjs.header.stamp = rospy.Time.now()
        self.pubjs.publish(self.msgjs)


class EffortCommandPublisher:
    def __init__(self, urdfnames, controlnames):
        # Make sure the name lists have equal length.
        assert len(urdfnames) == len(controlnames), "Unequal lengths"

        # Save the dofs = number of names/channels.
        self.n = len(urdfnames)

        # Create a publisher to /joint_states and pre-populate the message.
        self.pubjs = rospy.Publisher("/joint_states",
                                     JointState, queue_size=100)
        self.msgjs = JointState()
        for name in urdfnames:
            self.msgjs.name.append(name)
            self.msgjs.position.append(0.0)
            self.msgjs.velocity.append(0.0)

        # Prepare a list of publishers for each joint commands.
        self.pubX  = []
        for name in controlnames:
            topic = "/sevenbot/" + name + "/command"
            self.pubX.append(rospy.Publisher(topic, Float64, queue_size=100))

        # Wait until connected.  You don't have to wait, but the first
        # messages might go out before the connection and hence be lost.
        # rospy.sleep(0.5)

        # Report.
        rospy.loginfo("Ready to publish command for %d DOFs", self.n)

    def dofs(self):
        # Return the number of DOFs.
        return self.n

    def send(self, q, qdot):
        # Send each individual command and populate the joint_states.
        for i in range(self.n):
            self.pubX[i].publish(Float64(q[i]))
            self.msgjs.position[i] = q[i]
            self.msgjs.velocity[i] = qdot[i]

        # Send the command (with specified time).
        self.msgjs.header.stamp = rospy.Time.now()
        self.pubjs.publish(self.msgjs)


#
#  Cubic Segment Objects
#
#  These compute/store the 4 spline parameters, along with the
#  duration and given space.  Note the space is purely for information
#  and doesn't change the computation.
#
class CubicSpline:
    # Initialize.
    def __init__(self, p0, v0, pf, vf, T, space='Joint'):
        # Precompute the spline parameters.
        self.T = T
        self.a = p0
        self.b = v0
        self.c =  3*(pf-p0)/T**2 - vf/T    - 2*v0/T
        self.d = -2*(pf-p0)/T**3 + vf/T**2 +   v0/T**2
        # Save the space
        self.usespace = space

    # Return the segment's space
    def space(self):
        return self.usespace

    # Report the segment's duration (time length).
    def duration(self):
        return(self.T)

    # Compute the position/velocity for a given time (w.r.t. t=0 start).
    def evaluate(self, t):
        # Compute and return the position and velocity.
        p = self.a + self.b * t +   self.c * t**2 +   self.d * t**3
        v =          self.b     + 2*self.c * t    + 3*self.d * t**2
        return (p,v)

class Goto(CubicSpline):
    # Use zero initial/final velocities (of same size as positions).
    def __init__(self, p0, pf, T, space='Joint'):
        CubicSpline.__init__(self, p0, 0*p0, pf, 0*pf, T, space)

class Hold(Goto):
    # Use the same initial and final positions.
    def __init__(self, p, T, space='Joint'):
        Goto.__init__(self, p, p, T, space)

class Stay(Hold):
    # Use an infinite time (stay forever).
    def __init__(self, p, space='Joint'):
        Hold.__init__(self, p, math.inf, space)


#
#  Quintic Segment Objects
#
#  These compute/store the 6 spline parameters, along with the
#  duration and given space.  Note the space is purely for information
#  and doesn't change the computation.
#
class QuinticSpline:
    # Initialize.
    def __init__(self, p0, v0, a0, pf, vf, af, T, space='Joint'):
        # Precompute the six spline parameters.
        self.T = T
        self.a = p0
        self.b = v0
        self.c = a0
        self.d = -10*p0/T**3 - 6*v0/T**2 - 3*a0/T    + 10*pf/T**3 - 4*vf/T**2 + 0.5*af/T
        self.e =  15*p0/T**4 + 8*v0/T**3 + 3*a0/T**2 - 15*pf/T**4 + 7*vf/T**3 -   1*af/T**2
        self.f =  -6*p0/T**5 - 3*v0/T**4 - 1*a0/T**3 +  6*pf/T**5 - 3*vf/T**4 + 0.5*af/T**3
        # Also save the space
        self.usespace = space

    # Return the segment's space
    def space(self):
        return self.usespace

    # Report the segment's duration (time length).
    def duration(self):
        return(self.T)

    # Compute the position/velocity for a given time (w.r.t. t=0 start).
    def evaluate(self, t):
        # Compute and return the position and velocity.
        p = self.a + self.b * t +   self.c * t**2 +   self.d * t**3 +   self.e * t**4 +   self.f * t**5
        v =          self.b     + 2*self.c * t    + 3*self.d * t**2 + 4*self.e * t**3 + 5*self.f * t**4
        return (p,v)

class Goto5(QuinticSpline):
    # Use zero initial/final velocities/accelerations (same size as positions).
    def __init__(self, p0, pf, T, space='Joint'):
        QuinticSpline.__init__(self, p0, 0*p0, 0*p0, pf, 0*pf, 0*pf, T, space)


