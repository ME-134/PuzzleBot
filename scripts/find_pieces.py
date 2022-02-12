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
import sys, time

from sensor_msgs.msg     import JointState
from std_msgs.msg        import Bool, Duration
from urdf_parser_py.urdf import Robot

from force_kin import *
from splines import *

from piece_outline_detector import Detector

#
#  Controller Class
#
class Controller:
    # Initialize.
    def __init__(self, sim=False):
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
                                    [-.1, .7],
                                    [0, -0.1],])
        self.kin = Kinematics(robot, 'world', 'tip', inertial_params=inertial_params)

        # Initialize the current segment index and starting time t0.
        self.t0    = 0.0
        
        self.last_t = None
        self.sim = sim

        # Also initialize the storage of the last joint position
        # (angles) to where the first segment starts.

        if self.sim == False:
            msg = rospy.wait_for_message('/hebi/joint_states', JointState)
            self.lasttheta_state = self.lasttheta = np.array(msg.position).reshape((3,1))
            self.lastthetadot_state = self.lastthetadot = np.array(msg.velocity).reshape((3,1))
        else:
            self.lasttheta_state = self.lasttheta = np.pi * np.random.rand(3, 1)
            self.lastthetadot_state = self.lastthetadot = self.lasttheta * 0.01
            
        # Create the splines.
        self.segment = None
        
        self.is_resetting = False

        # Add spline which goes to the correct starting position
        self.reset(duration = 4)

        # Subscriber which listens to the motors' positions and velocities
        # Used for touch detection
        self.state_sub = rospy.Subscriber('/hebi/joint_states', JointState, self.state_update_callback)

        self.detector = Detector()
        detector.init_aruco()


    def fix_goal_theta(self, goal_theta, init_theta=None, assume_init_theta_valid=True):
        if init_theta is None:
            init_theta = self.lasttheta

        # Define bounds
        # Note that axis #1 is has a minimum of 0, so it is always above the table.
        # Note that axis #2 is cut off at np.pi, so the arm cannot go through itself.
        theta_min = np.array([-np.pi/2,     0, -np.pi]).reshape((3, 1))
        theta_max = np.array([ np.pi/2, np.pi,  np.pi]).reshape((3, 1))

        # Mod by 2pi, result should be in the range [-pi, pi]
        goal_theta = np.remainder(goal_theta + np.pi, 2*np.pi) - np.pi

        if assume_init_theta_valid:
            if np.any(theta_min > init_theta) or np.any(theta_max < init_theta):
                raise ValueError(f"Initial theta [{init_theta}] is outside the valid bounds!")
        if np.any(theta_min > goal_theta) or np.any(theta_max < goal_theta):
            raise ValueError(f"Goal theta [{goal_theta}] is outside the valid bounds!")

        return goal_theta

    def change_segment(self, segment):
        self.segment = segment
        self.t0 = self.last_t

    def reset(self, duration = 10):
        # TODO rewrite this function
        rospy.loginfo("Resetting robot")
        # Compute desired theta, starting the Newton Raphson at the last theta.
        goal_pos = np.array([ 0.07, 0.04, 0.15]).reshape((3,1))
        goal_theta = np.fmod(self.ikin(goal_pos, self.lasttheta), 4*np.pi)

        # Choose fastest path to the start
        #for i in range(goal_theta.shape[0]):
        #    candidate = np.remainder(goal_theta[i,0], 2*np.pi)
        #    if abs(candidate - self.lasttheta[i,0]) < abs(goal_theta[i,0] - self.lasttheta[i,0]):
        #        goal_theta[i,0] = candidate

        goal_theta = self.fix_goal_theta(goal_theta)
            
        goal_theta = self.ikin(goal_pos, goal_theta)

        #QuinticSpline(self.lasttheta, self.lastthetadot, 0, goal_theta, 0, 0, duration)
        spline = CubicSpline(self.lasttheta, -self.lastthetadot, goal_theta, 0, duration, rm=True)
        self.change_segment(spline)
        self.is_resetting = True

    def state_update_callback(self, msg):
        # Update our knowledge of true position and velocity of the motors
        self.lasttheta_state = np.array(msg.position).reshape((3,1))
        self.lastthetadot_state = np.array(msg.velocity).reshape((3, 1))

    def is_contacting(self):
        theta_error = np.sum(np.abs(self.lasttheta_state.reshape(-1) - self.lasttheta.reshape(-1)))
        thetadot_error = np.sum(np.abs(self.lastthetadot_state.reshape(-1) - self.lastthetadot.reshape(-1)))
        return (theta_error > 0.075)
    
    # Like ikin but only do 1 round
    def ikin_fast(self, pgoal, theta_initialguess, return_J=False):

        # Figure out where the current joint values put us:
        # Compute the forward kinematics for this iteration.
        (T,J) = self.kin.fkin(theta_initialguess)

        # Extract the position and use only the 3x3 linear
        # Jacobian (for 3 joints).  Generalize this for bigger
        # mechanisms if/when we need.
        p = p_from_T(T)

        # Compute the error.  Generalize to also compute
        # orientation errors if/when we need.
        e = pgoal - p

        # Take a step in the appropriate direction.  Using an
        # "inv()" though ultimately a "pinv()" would be safer.
        theta = theta_initialguess + np.linalg.inv(J[0:3, 0:3]) @ e

        if return_J:
            return theta, J
        return theta

    def ikin(self, pgoal, theta_initialguess):
        # Start iterating from the initial guess
        theta = theta_initialguess

        # Iterate at most 50 times (just to be safe)!
        for i in range(50):

            theta = self.ikin_fast(pgoal, theta)

            # Return if we have converged.
            if (np.linalg.norm(e) < 1e-6):
                return theta

        # After 50 iterations, return the failure and zero instead!
        rospy.logwarn("Unable to converge to [%f,%f,%f]",
                      pgoal[0], pgoal[1], pgoal[2]);
        return theta_initialguess


    # Update is called every 10ms!
    def update(self, t):

        dt = t - self.last_t
        self.last_t = t

        # If the current segment is done, replace the semgent with a new one
        dur = self.segment.duration()
        if (t - self.t0 >= dur):
            if self.is_resetting:
                self.is_resetting = False
                
            # FIND NEW PUZZLE PIECE
            x, y = self.detector.get_random_piece_center()
            x, y = self.detector.screen_to_world(x, y)
            pgoal = np.array([x, y, 0.02]).reshape((3, 1))
            goal_theta = self.ikin(pgoal, self.lasttheta)
            goal_theta = np.fmod(goal_theta, 2*np.pi)
            goal_theta = self.fix_goal_theta(goal_theta)
            print("chose location:", pgoal)
            print("goal theta: ", goal_theta)
            spline = CubicSpline(self.lasttheta, self.lastthetadot, goal_theta, 0, 3, rm=True)
            self.change_segment(spline)
            if np.linalg.norm(goal_theta - self.lasttheta) > 6:
                rospy.signal_shutdown("error with ikin")
                raise RuntimeError("it pop")

        # Decide what to do based on the space.
        if (self.segment.space() == 'Joint'):
            # Grab the spline output as joint values.
            (theta, thetadot) = self.segment.evaluate(t - self.t0)
        else:
            # Grab the spline output as task values.
            (p, v)    = self.segment.evaluate(t - self.t0)
            
            print(self.detector.world_to_screen(p[0, 0], p[1, 0]))
            
            theta, J = self.ikin_fast(p, self.lasttheta, return_J=True)
            thetadot  = np.linalg.inv(J[0:3, :]) @ v

        # Save the position (to be used as an estimate next time).
        self.lasttheta = theta
        self.lastthetadot = thetadot

        # Create and send the command message.  Note the names have to
        # match the joint names in the URDF.  And their number must be
        # the number of position/velocity elements.
        cmdmsg = JointState()
        cmdmsg.name         = ['Thor/1', 'Thor/4', 'Thor/3']
        cmdmsg.position     = self.lasttheta
        cmdmsg.velocity     = self.lastthetadot
        cmdmsg.effort       = self.kin.grav(self.lasttheta_state)
        cmdmsg.header.stamp = rospy.Time.now()
        if not self.sim:
            self.pub.publish(cmdmsg)
        self.rviz_pub.publish(cmdmsg)

        if not self.sim and self.is_contacting():
            self.reset(duration=4)
#
#
#  Main Code
#
if __name__ == "__main__":
    
    # Prepare/initialize this node.
    rospy.init_node('straightline')

    # Instantiate the controller object, encapsulating all
    # the computation and local variables.
    doSim = sys.argv[1] != "False" if len(sys.argv) > 1 else True
    if doSim:
        rospy.loginfo("Running simulator only")
    else:
        rospy.loginfo("Sending real commands")
    controller = Controller(sim=doSim)

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))

    timing_pub = rospy.Publisher("/update_time", Duration, queue_size=10)

    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    starttime = rospy.Time.now()
    controller.last_t = starttime.to_sec()
    while not rospy.is_shutdown():

        # Current time (since start)
        servotime = rospy.Time.now()
        t = (servotime - starttime).to_sec()

        # Update the controller.
        start = time.time()
        controller.update(t)
        t = (time.time() - start)
        tms = t * 1000
        if tms > 5:
            rospy.logwarn(f"Update took {int(tms)}ms")
        timing_pub.publish(t)

        # Wait for the next turn.  The timing is determined by the
        # above definition of servo.
        servo.sleep()
