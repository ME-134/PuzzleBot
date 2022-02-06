#!/usr/bin/env python3
#
#   kinematics_gravity.py
#
#   TO IMPORT, ADD TO YOUR CODE:
#   from kinematics_gravity import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz
#
#
#   Kinematics Class and Helper Functions
#
#   This computes the forward kinematics and Jacobian using the
#   kinematic chain.  It further computes the gravity feedforward
#   term.  It also includes test code when run independently.
#
import rospy
from numba import njit
import numpy as np

from functools import lru_cache

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

# Error functions
@njit
def ep(pd, pa):
    return (pd - pa)
@njit   
def eR(Rd, Ra):
    return 0.5*(np.cross(Ra[:,0], Rd[:,0]) +
                np.cross(Ra[:,1], Rd[:,1]) +
                np.cross(Ra[:,2], Rd[:,2]))

### Build T matrix from R/p.  Extract R/p from T matrix
def T_from_Rp(R, p):
    return np.vstack((np.hstack((R,p)),
                      np.array([0.0, 0.0, 0.0, 1.0])))
@njit
def p_from_T(T):
    return T[0:3,3:4]
@njit
def R_from_T(T):
    return T[0:3,0:3]

### Basic Rotation Matrices about an axis: Rotx/Roty/Rotz/Rot(axis)
@njit
def Rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
@njit
def Ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
@njit
def Rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def R_from_axisangle(axis, theta):
    ex = np.array([[     0.0, -axis[2],  axis[1]],
                   [ axis[2],      0.0, -axis[0]],
                   [-axis[1],  axis[0],     0.0]])
    return np.eye(3) + np.sin(theta) * ex + (1.0-np.cos(theta)) * ex @ ex

### Quaternion To/From Rotation Matrix
@njit
def R_from_q(q):
    norm2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    return - np.eye(3) + (2/norm2) * (
      np.array([[q[1]*q[1]+q[0]*q[0],q[1]*q[2]-q[0]*q[3],q[1]*q[3]+q[0]*q[2]],
                [q[2]*q[1]+q[0]*q[3],q[2]*q[2]+q[0]*q[0],q[2]*q[3]-q[0]*q[1]],
                [q[3]*q[1]-q[0]*q[2],q[3]*q[2]+q[0]*q[1],q[3]*q[3]+q[0]*q[0]]]))
@njit
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
@lru_cache
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
    def __init__(self, robot, baseframe, tipframe, inertial_params=None):
        # Report what we are doing.
        rospy.loginfo("Kinematics: Setting up the chain from '%s' to '%s'...",
                      baseframe, tipframe)
        self.inertial = inertial_params
        # Create the list of joints from the base frame to the tip
        # frame.  Search backwards, as this could be a tree structure.
        # Meantine while a parent may have multiple children, every
        # child has only one parent!  That makes the chain unique.
        self.links  = []
        self.joints = []
        frame = tipframe
        while (frame != baseframe):
            link  = next((l for l in robot.links  if l.name  == frame), None)
            joint = next((j for j in robot.joints if j.child == frame), None)
            if (link is None):
                rospy.logerr("Unable find link '%s'", frame)
                raise Exception()
            if (joint is None):
                rospy.logerr("Unable find joint connecting to '%s'", frame)
                raise Exception()
            if (joint.parent == frame):
                rospy.logerr("Joint '%s' connects '%s' to itself",
                             joint.name, frame)
                raise Exception()
            self.links.insert(0, link)
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


    def grav(self, theta):
        # Check the number of joints
        if (len(theta) != self.dofs):
            rospy.logerr("Number of joint angles (%d) does not match URDF (%d)",
                         len(theta), self.dofs)
            return

        # Initialize the T matrix to walk up the chain
        T = np.eye(len(theta))

        # Initialize the gravity torques.
        grav = np.zeros(self.dofs)
        if self.inertial is None:
            return grav
        '''
        # As we are walking up, also store the position and joint
        # axis, ONLY FOR ACTIVE/MOVING/"REAL" joints.  We simply put
        # each in a python, and keep an index counter.
        plist = []
        elist = []
        index = 0

        # Walk the chain, one URDF <joint> entry at a time (see fkin).
        # After each joint, consider the child link and project it's
        # gravity torques to all joints already passed (lower in the
        # kinematic chain).
        for (joint, link) in zip(self.joints, self.links):
            # Compute transform according to the joint.  See fkin().
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

		    # Now consider the child link.  Proceed if there is an
            # inertial element.
            if (self.inertial is not None):
                # Check whether to shift to the child link's center of mass.
                if (link.inertial.origin is not None):
                    Tlink = T @ T_from_URDF_origin(link.inertial.origin)
                else:
                    Tlink = T

                # Extract the mass times g and center of mass location.
                mg = link.inertial.mass * 9.81
                pc = p_from_T(Tlink)

                # Add all the torques (below the current index).  Note
                # the gravity vector is vertically up in world frame,
                # hence we simply take the 3rd element of the vector.
                for k in range(index):
                    grav[k] += mg * np.cross(elist[k], pc-plist[k], axis=0)[2]
        '''
        if self.inertial is not None:
            grav[2] = self.inertial[2, 0]*np.sin(-theta[2]+theta[1]) + self.inertial[2, 1]*np.cos(-theta[2]+theta[1])
            grav[1] = -grav[2] + self.inertial[1, 0]*np.sin(theta[1]) + self.inertial[1, 1]*np.cos(theta[1])
            grav[0] = 0.0
        # Return the gravity torques.
        return grav


#
#  Main Code (only if run independently):
#
#  Test assuming a 4DOF URDF is loaded.
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node('gravity')

    # Grab the robot's URDF from the parameter server.
    robot = Robot.from_parameter_server()

    # Instantiate the Kinematics.  Generally we care about tip w.r.t. world!
    kin = Kinematics(robot, 'world', 'tip')

    # Define the forward kinematics test.
    def test_fkin(theta):
        # Compute the kinematics and report.
        (T,J) = kin.fkin(theta)
        np.set_printoptions(precision=6, suppress=True)
        print("Fkin for: ", theta.T)
        print("T:\n", T)
        print("J:\n", J)
        print("p/q: ", p_from_T(T).T, q_from_R(R_from_T(T)))

    # Define the gravity test.
    def test_grav(theta):
        # Compute the gravity feedforward torques and report.
        G = kin.grav(theta)
        np.set_printoptions(precision=6, suppress=True)
        print("\nGrav for: ", theta.T)
        print("Torques : ", G.T)

    # Test the kinematics.
    test_fkin(np.array([np.pi/4, np.pi/6, np.pi/3, 0.0]).reshape((-1,1)))

    # Test the gravity.
    test_grav(np.array([0.0,      0.0,      0.0,       0.0    ]).reshape((-1,1)))
    test_grav(np.array([0.0,      0.0,      np.pi/2,   0.0    ]).reshape((-1,1)))
    test_grav(np.array([0.0,      np.pi/2,  0.0,       0.0    ]).reshape((-1,1)))
    test_grav(np.array([np.pi/2,  np.pi/2,  np.pi/2,   np.pi/2]).reshape((-1,1)))
    test_grav(np.array([np.pi/4,  np.pi/4,  np.pi/4,   np.pi/4]).reshape((-1,1)))
    test_grav(np.array([np.pi/6, -np.pi/3,  np.pi/6,  -np.pi/3]).reshape((-1,1)))

