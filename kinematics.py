#!/usr/bin/env python3
#
#   kinematics.py
#
#   TO IMPORT, ADD TO YOUR CODE:
#   from kinematics import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz
#
#
#   Kinematics Class and Helper Functions
#
#   This computes the forward kinematics and Jacobian using the
#   kinematic chain.  It also includes test code when run
#   independently.
#
import rospy
import numpy as np

from urdf_parser_py.urdf import Robot


#
#  Kinematics Helper Functions
#
#  These helper functions simply convert the information between
#  different formats.  For example, between a python list and a NumPy
#  array.  Or between Euler Angles and a Rotation Matrix.  And so
#  forth.  They operate on:
#
#    NumPy 3x1  "p" Point vectors
#    NumPy 3x1  "e" Axes of rotation
#    NumPy 3x3  "R" Rotation matrices
#    NumPy 1x4  "q" Quaternions
#    NumPy 4x4  "T" Transforms
#
#  and may take inputs from the URDF tags <origin> and <axis>:
#
#    Python List 1x3:  <axis>          information
#    Python List 1x6:  <origin>        information
#    Python List 1x3:  <origin> "xyz"  vector of positions
#    Python List 1x3:  <origin> "rpy"  vector of angles
#

### Build T matrix from R/p.  Extract R/p from T matrix
def T_from_Rp(R, p):
    return np.vstack((np.hstack((R,p)),
                      np.array([0.0, 0.0, 0.0, 1.0])))
def p_from_T(T):
    return T[0:3,3:4]
def R_from_T(T):
    return T[0:3,0:3]

### Basic Rotation Matrices about an axis: Rotx/Roty/Rotz/Rot(axis)
def Rx(theta):
    c = np.cos(theta);
    s = np.sin(theta);
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
def Ry(theta):
    c = np.cos(theta);
    s = np.sin(theta);
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
def Rz(theta):
    c = np.cos(theta);
    s = np.sin(theta);
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def R_from_axisangle(axis, theta):
    ex = np.array([[     0.0, -axis[2],  axis[1]],
                   [ axis[2],      0.0, -axis[0]],
                   [-axis[1],  axis[0],     0.0]])
    return np.eye(3) + np.sin(theta) * ex + (1.0-np.cos(theta)) * ex @ ex

### Quaternion To/From Rotation Matrix
def R_from_q(q):
    norm2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    return - np.eye(3) + (2/norm2) * (
      np.array([[q[1]*q[1]+q[0]*q[0],q[1]*q[2]-q[0]*q[3],q[1]*q[3]+q[0]*q[2]],
                [q[2]*q[1]+q[0]*q[3],q[2]*q[2]+q[0]*q[0],q[2]*q[3]-q[0]*q[1]],
                [q[3]*q[1]-q[0]*q[2],q[3]*q[2]+q[0]*q[1],q[3]*q[3]+q[0]*q[0]]]))
def q_from_R(R):
    A = [1.0 + R[0][0] + R[1][1] + R[2][2],
         1.0 + R[0][0] - R[1][1] - R[2][2],
         1.0 - R[0][0] + R[1][1] - R[2][2],
         1.0 - R[0][0] - R[1][1] + R[2][2]]
    i = A.index(max(A))
    A = A[i]
    c = 0.5/np.sqrt(A)
    if   (i == 0):
        q = c*np.array([A, R[2][1]-R[1][2], R[0][2]-R[2][0], R[1][0]-R[0][1]])
    elif (i == 1):
        q = c*np.array([R[2][1]-R[1][2], A, R[1][0]+R[0][1], R[0][2]+R[2][0]])
    elif (i == 2):
        q = c*np.array([R[0][2]-R[2][0], R[1][0]+R[0][1], A, R[2][1]+R[1][2]])
    else:
        q = c*np.array([R[1][0]-R[0][1], R[0][2]+R[2][0], R[2][1]+R[1][2], A])
    return q
def q_from_T(T):
    return q_from_R(R_from_T(T))


### From URDF <ORIGIN> elements (and "xyz"/"rpy" sub-elements):
def T_from_URDF_origin(origin):
    return T_from_Rp(R_from_URDF_rpy(origin.rpy), p_from_URDF_xyz(origin.xyz))

def R_from_URDF_rpy(rpy):
    return Rz(rpy[2]) @ Ry(rpy[1]) @ Rx(rpy[0])

def p_from_URDF_xyz(xyz):
    return np.array(xyz).reshape((3,1))


### From URDF <AXIS> elements:
def T_from_URDF_axisangle(axis, theta):
    return T_from_Rp(R_from_axisangle(axis, theta), np.zeros((3,1)))

def e_from_URDF_axis(axis):
    return np.array(axis).reshape((3,1))



#
#   Kinematics Class
#
#   This encapsulates the kinematics functionality, storing the
#   kinematic chain elements.
#
class Kinematics:
    def __init__(self, robot, baseframe, tipframe):
        # Report what we are doing.
        rospy.loginfo("Kinematics: Setting up the chain from '%s' to '%s'...",
                      baseframe, tipframe)

        # Create the list of joints from the base frame to the tip
        # frame.  Search backwards, as this could be a tree structure.
        # Meantine while a parent may have multiple children, every
        # child has only one parent!  That makes the chain unique.
        self.joints = []
        frame = tipframe
        while (frame != baseframe):
            joint = next((j for j in robot.joints if j.child == frame), None)
            if (joint is None):
                rospy.logerr("Unable find joint connecting to '%s'", frame)
                raise Exception()
            if (joint.parent == frame):
                rospy.logerr("Joint '%s' connects '%s' to itself",
                             joint.name, frame)
                raise Exception()
            self.joints.insert(0, joint)
            frame = joint.parent

        # Report we found.
        self.dofs = sum(1 for j in self.joints if j.type != 'fixed')
        rospy.loginfo("Kinematics: %d active DOFs, %d total steps",
                      self.dofs, len(self.joints))


    def fkin(self, theta):
        # Check the number of joints
        if (len(theta) != self.dofs):
            rospy.logerr("Number of joint angles (%d) does not match URDF (%d)",
                         len(theta), self.dofs)
            return

        # Initialize the T matrix to walk up the chain
        T = np.eye(4)

        # As we are walking up, also store the position and joint
        # axis, ONLY FOR ACTIVE/MOVING/"REAL" joints.  We simply put
        # each in a python, and keep an index counter.
        plist = []
        elist = []
        index = 0

        # Walk the chain, one URDF <joint> entry at a time.  Each can
        # be "fixed" (just a transform) or "continuous" (a transform
        # AND a rotation).  NOTE the URDF entries are only the
        # step-by-step transformations.  That is, the information is
        # *not* in world frame - we have to append to the chain...
        for joint in self.joints:
            if (joint.type == 'fixed'):
                # Just append the fixed transform
                T = T @ T_from_URDF_origin(joint.origin)
                
            elif (joint.type == 'continuous'):
                # First append the fixed transform, then rotating
                # transform.  The joint angle comes from theta-vector.
                T = T @ T_from_URDF_origin(joint.origin)
                T = T @ T_from_URDF_axisangle(joint.axis, theta[index])

                # Save the position
                plist.append(p_from_T(T))

                # Save the joint axis.  The URDF <axis> is given in
                # the local frame, so multiply by the local R matrix.
                elist.append(R_from_T(T) @ e_from_URDF_axis(joint.axis))

                # Advanced the "active/moving" joint number 
                index += 1
    
            elif (joint.type != 'fixed'):
                # There shouldn't be any other types...
                rospy.logwarn("Unknown Joint Type: %s", joint.type)

        # Compute the Jacobian.  For that we need the tip information,
        # which is where the kinematic chain ended up, i.e. at T.
        ptip = p_from_T(T)
        J    = np.zeros((6,index))
        for i in range(index):
            J[0:3,i:i+1] = np.cross(elist[i], ptip-plist[i], axis=0)
            J[3:6,i:i+1] = elist[i]

        # Return the Ttip and Jacobian (at the end of the chain).
        return (T,J)


#
#  Main Code (only if run independently):
#
#  Test assuming a 3DOF URDF is loaded.
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node('kinematics')

    # Grab the robot's URDF from the parameter server.
    robot = Robot.from_parameter_server()

    # Instantiate the Kinematics.  Generally we care about tip w.r.t. world!
    kin = Kinematics(robot, 'world', 'tip')

    # Pick the test angles of the robot.
    theta = [np.pi/4, np.pi/6, np.pi/3]

    # Compute the kinematics.
    (T,J) = kin.fkin(theta)

    # Report.
    np.set_printoptions(precision=6, suppress=True)
    print("T:\n", T)
    print("J:\n", J)
    print("p/q: ", p_from_T(T).T, q_from_R(R_from_T(T)))
