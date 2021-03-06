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
import matplotlib.pyplot as plt

from sensor_msgs.msg     import JointState
from std_msgs.msg        import Bool
from urdf_parser_py.urdf import Robot

# Import the kinematics stuff:
from kinematics import Kinematics, p_from_T, R_from_T, Rx, Ry, Rz
# We could also import the whole thing ("import kinematics"),
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
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        rospy.sleep(0.25)

        # Grab the robot's URDF from the parameter server.
        robot = Robot.from_parameter_server()

        # Instantiate the Kinematics
        self.kin = Kinematics(robot, 'world', 'tip')

        # Set the tip targets (in 3x1 column vectors).
        xA = np.array([ 0.1, 0.2, 0.01]).reshape((3,1))    # Bottom.
        xB = np.array([-0.1, 0.2, 0.01]).reshape((3,1))    # Top.

        # Pick the initial estimate (in a 3x1 column vector).
        theta0 = np.array([0.0, np.pi/2, -np.pi/2]).reshape((3,1))

        # Figure out the start/stop joint values.
        thetaA = (self.ikin(xA, theta0))

        # Create the splines (cubic for now, use Goto5() for HW#5P1).
        self.sin_traj = SinTraj(xA, xB, np.inf, .5, space="Cart")
        self.segments = [self.sin_traj]
        self.segment_q = list()

        # Initialize the current segment index and starting time t0.
        self.index = 0
        self.t0    = 0.0
        self.old_t0 = 0.0

        # Also initialize the storage of the last joint position
        # (angles) to where the first segment starts.
        self.lasttheta = thetaA
        self.lastthetadot = thetaA * 0.0

        # Flips between 1 and -1 every time the robot does a flip.
        # Not critical for functionality, but ensures that the robot doesn't
        # keep spinning the same way on the vertical axis.
        self.orientation = 1

        # Subscribe to "/switch" which causes the robot to do a flip
        self.switch_sub = rospy.Subscriber("/switch", Bool, self.switch_callback)

        self.theta_history = []
        self.thetadot_history = []

    def flip(self, duration = 4):
        # Convert all angles to be between 0 and 2pi
        rounds = np.floor_divide(self.lasttheta, np.pi*2)

        # Hard-code solution to flipped arm
        thetaInit = np.remainder(self.lasttheta, np.pi*2)
        thetaGoal = (thetaInit.T * np.array([1, -1, -1])
                                 + np.array([self.orientation * np.pi, np.pi, 0])).reshape((3, 1))

        # Ensures that the robot arm segments don't collide with each other.
        if thetaInit[2, 0] > np.pi:
            thetaGoal[2, 0] += 4*np.pi

        thetaDotInit = self.lastthetadot
        thetaDotGoal = (thetaDotInit.T * np.array([1, -1, -1])).reshape((3, 1))

        # Flips between -1 and 1
        self.orientation *= -1

        # Convert angles back to original space
        thetaGoal += rounds * 2*np.pi
        thetaInit += rounds * 2*np.pi
        self.segment_q.append(CubicSpline(thetaInit, thetaDotInit, thetaGoal, thetaDotGoal, duration))

    def switch_callback(self, msg):
        # msg is unused
        self.flip()

    # Newton-Raphson static (indpendent of time/motion) Inverse Kinematics:
    # Iterate to find the joints values putting the tip at the goal.
    def ikin(self, pgoal, theta_initialguess):
        # Start iterating from the initial guess
        theta = theta_initialguess

        # Iterate at most 50 times (just to be safe)!
        for i in range(50):
            # Figure out where the current joint values put us:
            # Compute the forward kinematics for this iteration.
            (T,J) = self.kin.fkin(theta)

            # Extract the position and use only the 3x3 linear
            # Jacobian (for 3 joints).  Generalize this for bigger
            # mechanisms if/when we need.
            p = p_from_T(T)
            J = J[0:3, 0:3]

            # Compute the error.  Generalize to also compute
            # orientation errors if/when we need.
            e = pgoal - p

            # Take a step in the appropriate direction.  Using an
            # "inv()" though ultimately a "pinv()" would be safer.
            theta = theta + np.linalg.inv(J) @ e

            # Return if we have converged.
            if (np.linalg.norm(e) < 1e-6):
                return theta

        # After 50 iterations, return the failure and zero instead!
        rospy.logwarn("Unable to converge to [%f,%f,%f]",
                      pgoal[0], pgoal[1], pgoal[2]);
        return 0 * theta


    # Update is called every 10ms!
    def update(self, t):

        if self.segment_q:
            self.index = len(self.segments)
            self.segments += self.segment_q
            self.segment_q = list()
            self.old_t0 = self.t0
            self.t0 = t

        # If the current segment is done, shift to the next.
        dur = self.segments[self.index].duration()
        if (t - self.t0 >= dur):
            self.t0    = (self.t0    + dur)
            #self.index = (self.index + 1)                       # not cyclic!
            self.index = (self.index + 1) % len(self.segments)  # cyclic!
            if self.index < len(self.segments) and isinstance(self.segments[self.index], SinTraj):
                self.t0 = self.old_t0

        # Check whether we are done with all segments.
        if (self.index >= len(self.segments)):
            rospy.signal_shutdown("Done with motion")
            return

        # Decide what to do based on the space.
        if (self.segments[self.index].space() == 'Joint'):
            # Grab the spline output as joint values.
            (theta, thetadot) = self.segments[self.index].evaluate(t - self.t0)
        else:
            # Grab the spline output as task values.
            (p, v)    = self.segments[self.index].evaluate(t - self.t0)

            # Compute theta, starting the Newton Raphson at the last theta.
            theta     = self.ikin(p, self.lasttheta)

            # From that compute the Jacobian, so we can use J^-1 for thetadot.
            (T,J)     = self.kin.fkin(theta)
            J         = J[0:3, :]
            thetadot  = np.linalg.inv(J) @ v

        # Save the position (to be used as an estimate next time).
        self.lasttheta = theta
        self.lastthetadot = thetadot

        # Create and send the command message.  Note the names have to
        # match the joint names in the URDF.  And their number must be
        # the number of position/velocity elements.
        cmdmsg = JointState()
        cmdmsg.name         = ['Thor/1', 'Thor/2', 'Thor/3']
        cmdmsg.position     = theta
        cmdmsg.velocity     = thetadot
        cmdmsg.header.stamp = rospy.Time.now()
        self.pub.publish(cmdmsg)

        self.update_plot()

    def update_plot(self):
        self.theta_history.append(self.lasttheta[0])
        if len(self.theta_history) > 0:
            if len(self.theta_history) > 240:
                self.theta_history = self.theta_history[-240:]
            
            plt.clf()
            plt.plot(self.theta_history)
            plt.draw()
            plt.pause(1e-6)

#
#  Main Code
#
if __name__ == "__main__":
    plt.ion()
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
