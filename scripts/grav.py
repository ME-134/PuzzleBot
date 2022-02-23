#!/usr/bin/env python3
#
#   HW#4 P5 straightline.py
#
#   Visualize the 3DOF, moving up in a task-space move, down in a
#   joint-space move.
#
#   Publish:   /joint_states   sensor_msgs/JointState
#
import rospy
import numpy as np

from sensor_msgs.msg     import JointState
from std_msgs.msg        import Bool
from urdf_parser_py.urdf import Robot

# Import the kinematics stuff:
from force_kin import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz
# We could also import the whole thing ("import kinematics")oveit_configs  me134basic

# but then we'd have to write "kinematics.p_from_T()" ...

# Import the Spline stuff:
from splines import  CubicSpline, Goto, Hold, Stay, QuinticSpline, Goto5

# Uses cosine interpolation between two points
class SinTraj:
    # Initialize.
    def __init__(self, p0, pf, T, f, offset=0, space='Joint'):
        # Precompute the spline parameters.
        self.T = T
        self.f = f
        self.offset = offset
        self.p0 = p0
        self.pf = pf
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
        p = (self.pf - self.p0) * (.5+.5*np.cos(t*self.f*2*np.pi + self.offset)) + self.p0
        v = (self.pf - self.p0) * (-.5*np.sin(t*self.f*2*np.pi + self.offset)) * self.f*2*np.pi
        return (p,v)

#
#  Generator Class
#
class Generator:
    # Initialize.
    def __init__(self):
        # Create a publisher to send the joint commands.  Add some time
        # for the subscriber to connect.  This isn't necessary, but means
        # we don't start sending messages until someone is listening.
        self.pub = rospy.Publisher("/hebi/joint_commands", JointState, queue_size=10)
        self.rviz_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        rospy.sleep(0.25)

        # Grab the robot's URDF from the parameter server.
        robot = Robot.from_parameter_server()

        # Instantiate the Kinematics
        inertial_params = np.array([[0, 0],
                                  [-.1, 4.1],
                                  [-0.5, -2.5],])
        self.kin = Kinematics(robot, 'world', 'tip', inertial_params=inertial_params)

        # Initialize the current segment index and starting time t0.
        self.index = 0
        self.t0    = 0.0
        self.old_t0 = 0.0

        # Also initialize the storage of the last joint position
        # (angles) to where the first segment starts.

        # FIXME
        msg = rospy.wait_for_message('/hebi/joint_states', JointState)
        self.lasttheta_state = self.lasttheta = np.array(msg.position).reshape((5,1))
        self.lastthetadot_state = self.lastthetadot = np.array(msg.velocity).reshape((5,1))

        # Subscriber which listens to the motors' positions and velocities
        self.state_sub = rospy.Subscriber('/hebi/joint_states', JointState, self.state_update_callback)

    def state_update_callback(self, msg):
        # Update our knowledge of true position and velocity of the motors
        #rospy.loginfo("Recieved state update message " + str(msg))
        self.lasttheta_state = np.array(msg.position)
        self.lastthetadot_state = np.array(msg.velocity)

    def plot_errors(self):
        theta_error = self.lasttheta_state - self.lasttheta
        thetadot_error = self.lastthetadot_state - self.lastthetadot

        
    # Update is called every 10ms!
    def update(self, t):
        # Create and send the command message.  Note the names have to
        # match the joint names in the URDF.  And their number must be
        # the number of position/velocity elements.
        cmdmsg = JointState()
        cmdmsg.name         = ['Thor/1', 'Thor/6', 'Thor/3', 'Thor/4', 'Thor/2']
        cmdmsg.position     = [np.nan, np.nan, np.nan, np.nan, np.nan]#theta
        cmdmsg.velocity     = [np.nan, np.nan, np.nan, np.nan, np.nan]
        cmdmsg.effort       = self.kin.grav(self.lasttheta_state)
        cmdmsg.header.stamp = rospy.Time.now()
        self.pub.publish(cmdmsg)
        cmdmsg.position     = self.lasttheta_state
        cmdmsg.velocity     = self.lastthetadot_state
        self.rviz_pub.publish(cmdmsg)
        


#
#  Main Code
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node('straightline')

    # Instantiate the trajectory generator object, encapsulating all
    # the computation and local variables.
    generator = Generator()

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))


    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    starttime = rospy.Time.now()
    while not rospy.is_shutdown():

        # Current time (since start)
        servotime = rospy.Time.now()
        t = (servotime - starttime).to_sec()

        # Update the controller.
        generator.update(t)

        # Wait for the next turn.  The timing is determined by the
        # above definition of servo.
        servo.sleep()
