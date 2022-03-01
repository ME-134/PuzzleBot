#!/usr/bin/env python3
#
#   solve_puzzle.py
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
from std_msgs.msg        import Bool, Duration, UInt16, Float32
from urdf_parser_py.urdf import Robot

from force_kin import *
from splines import *

from piece_outline_detector import Detector
from solver import Solver, Status
import enum

class State(enum.Enum):
    idle   = 0
    grav   = 1
    reset  = 2
    pickup = 3

motor_names = ['Thor/1', 'Thor/6', 'Thor/3', 'Thor/4', 'Thor/2']

class IKinException(Exception):
    pass
class BoundsException(Exception):
    pass
class InvalidSpaceException(Exception):
    pass

class FuncSegment:
    # Calls function with no arguments
    # Can be added to segments list
    def __init__(self, func):
        self.func = func

    def duration(self):
        return .05

    # Return the segment's space
    def space(self):
        return 'Func'

    def __call__(self):
        self.func()

class Bounds:
    # Note that axis #1 is has a minimum of 0, so it is always above the table.
    # Note that axis #2 is cut off at np.pi, so the arm cannot go through itself.
    theta_min = np.array([-np.pi/2,      -0.1, -np.pi*0.9, -np.pi, -np.inf]).reshape((5, 1))
    theta_max = np.array([ np.pi/2,     np.pi,  np.pi*0.9,  np.pi,  np.inf]).reshape((5, 1))

    # I don't know
    thetadot_max = np.array([np.pi, np.pi, np.pi, np.pi, np.inf]).reshape((5, 1))/2
    thetadot_min = -thetadot_max

    @staticmethod
    def is_theta_valid(theta, axis=None):
        if axis is None:
            return np.all(theta <= Bounds.theta_max) and np.all(theta >= Bounds.theta_min)
        return theta[axis] <= Bounds.theta_max[axis] and theta[axis] >= Bounds.theta_min[axis]

    @staticmethod
    def is_thetadot_valid(thetadot, axis=None):
        if axis is None:
            return np.all(thetadot <= Bounds.thetadot_max) and np.all(thetadot >= Bounds.thetadot_min)
        return thetadot[axis] <= Bounds.thetadot_max[axis] and thetadot[axis] >= Bounds.thetadot_min[axis]

    @staticmethod
    def assert_theta_valid(theta):
        if not Bounds.is_theta_valid(theta):
            errmsg = f"Given motor angle is out of bounds: \ntheta={theta}\nmin={Bounds.theta_min}\nmax={Bounds.theta_max}"
            rospy.signal_shutdown(errmsg)
            raise BoundsException(errmsg)

    @staticmethod
    def assert_thetadot_valid(thetadot):
        if not Bounds.is_thetadot_valid(thetadot):
            errmsg = f"Given motor angular velocity is out of bounds: thetadot={thetadot}, min={Bounds.thetadot_min}, max={Bounds.thetadot_max}"
            rospy.signal_shutdown(errmsg)
            raise BoundsException(errmsg)

#
#  Controller Class
#
class Controller:
    # Initialize.
    def __init__(self, sim=False):
        # Create a publisher to send the joint commands.  Add some time
        # for the subscriber to connect.  This isn't necessary, but means
        # we don't start sending messages until someone is listening.
        self.pub = rospy.Publisher("/hebi/joint_commands", JointState, queue_size=1)
        self.rviz_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)
        self.pump_pub = rospy.Publisher("/toggle_pump", UInt16, queue_size=1)
        rospy.sleep(0.25)

        # Grab the robot's URDF from the parameter server.
        robot = Robot.from_parameter_server()

        # Instantiate the Kinematics
        inertial_params = np.array([[0, 0],
                                  [-.1, 4.4],
                                  [-0.5, -3.3],])
        self.kin = Kinematics(robot, 'world', 'tip', inertial_params=inertial_params)

        # Initialize the current segment index and starting time t0.
        self.state = 0
        self.index = 0
        self.t0    = 0.0
        self.last_t = 0.0
        self.sim = sim

        self.pump_value = 0

        # Also initialize the storage of the last joint position
        # (angles) to where the first segment starts.

        if self.sim == False:
            msg = rospy.wait_for_message('/hebi/joint_states', JointState)
            self.lasttheta_state = self.lasttheta = np.array(msg.position).reshape((5,1))
            self.lastthetadot_state = self.lastthetadot = np.array(msg.velocity).reshape((5,1))
        else:
            self.lasttheta_state = self.lasttheta = np.array([np.pi/12, np.pi/6, np.pi/4, 0, 0]).reshape((5, 1))#np.pi * np.random.rand(3, 1)
            self.lastthetadot_state = self.lastthetadot = self.lasttheta * 0.01

        # Create the splines.
        self.segments = []

        # Point where the robot resets to
        self.reset_pos = np.array([ 0.06, -0.11, 0.15, 0, 0]).reshape((5,1))
        self.mean_theta = np.array([-0.15, 1.15, 1.7, -0.30, -1.31]).reshape((5,1))

        # Add spline which goes to the correct starting position
        self.reset(duration = 4)

        # Subscriber which listens to the motors' positions and velocities
        # Used for touch detection and gravity compensation
        self.state_sub = rospy.Subscriber('/hebi/joint_states', JointState, self.state_update_callback)
        self.state_sub = rospy.Subscriber('/current_draw', Float32, self.current_callback)
        self.current = 0.0

        self.detector = Detector()
        self.detector.init_aruco()
        self.solver = Solver(self.detector)


    def change_segments(self, segments):
        self.segments = segments
        self.t0 = self.last_t
        self.index = 0

    def reset(self, duration = 4):
        # Assumes that the initial theta is valid.
        # Will not attempt to reset a theta which is out of bounds.

        rospy.loginfo("Resetting robot")
        self.state = State.reset

        guess = np.array([-1.31, 1.88, 2.26, -0.38, -1.31]).reshape((5,1))
        goal_theta = self.ikin(self.reset_pos, guess)
        print('\n\n\n')
        print(goal_theta)

        Bounds.assert_theta_valid(goal_theta)
        Bounds.assert_theta_valid(self.lasttheta)
        Bounds.assert_thetadot_valid(self.lastthetadot)

        spline = CubicSpline(self.lasttheta, -self.lastthetadot, goal_theta, 0, duration, rm=True)
        self.change_segments([spline])

    def move_piece(self, piece_origin, piece_destination, turn=0, jiggle=False, space='Joint'):
        # piece_origin and piece_destination given in pixel space
        rospy.loginfo("[Controller] Moving piece from {piece_origin} to {piece_destination}")

        pickup_height = -0.01
        hover_amount  = 0.06

        # move from current pos to piece_origin
        def get_piece_and_hover_thetas(pixel_coords, turn=0):
            x, y = pixel_coords
            x, y = self.detector.screen_to_world(x, y)
            # print(pixel_coords, x, y)
            pgoal = np.array([x, y, pickup_height, turn, 0]).reshape((5, 1))
            hover = pgoal+np.array([0, 0, hover_amount, 0, 0]).reshape((5, 1))

            if space == 'Joint':
                # ASK Hayama: why not self.lasttheta?
                hover_theta = self.ikin(hover, self.mean_theta)
                rospy.loginfo(f"[Controller] Chose location: {hover.flatten()}\n\t Goal theta: {hover_theta.flatten()}")
                goal_theta = self.ikin(pgoal, hover_theta)
                rospy.loginfo(f"[Controller] Chose location: {pgoal.flatten()}\n\t Goal theta: {goal_theta.flatten()}")
                return goal_theta, hover_theta
            elif space == 'Task':
                rospy.loginfo(f"[Controller] Chose location: {hover.flatten()}")
                rospy.loginfo(f"[Controller] Chose location: {pgoal.flatten()}")
                return pgoal, hover
            else:
                errmsg = f"Space \"{space}\" is invalid"
                rospy.signal_shutdown(errmsg)
                raise InvalidSpaceException(errmsg)

        splines = list()
        origin_goal, origin_hover = get_piece_and_hover_thetas(piece_origin)
        dest_goal,   dest_hover   = get_piece_and_hover_thetas(piece_destination, turn=turn)
        splines.append(CubicSpline(self.lasttheta, self.lastthetadot, origin_hover, 0, 3, space=space))
        splines.append(CubicSpline(origin_hover, 0, origin_goal, 0, 1, space=space))
        splines.append(FuncSegment(lambda: self.set_pump(True)))
        splines.append(CubicSpline(origin_goal, 0, origin_goal, 0, 0.5, space=space))
        splines.append(CubicSpline(origin_goal, 0, origin_hover, 0, 1, space=space))
        splines.append(CubicSpline(origin_hover, 0, dest_hover, 0, 3, space=space))
        splines.append(CubicSpline(dest_hover, 0, dest_goal, 0, 1, space=space))
        splines.append(FuncSegment(lambda: self.set_pump(False)))
        splines.append(CubicSpline(dest_goal, 0, dest_goal, 0, 0.7, space=space))
        splines.append(CubicSpline(dest_goal, 0, dest_hover, 0, 1, space=space))
        self.change_segments(splines)
        self.state = State.pickup

    def idle(self):
        self.change_segments([])
        self.state = State.grav

    def state_update_callback(self, msg):
        # Update our knowledge of true position and velocity of the motors
        self.lasttheta_state = np.array(msg.position).reshape((5, 1))
        self.lastthetadot_state = np.array(msg.velocity).reshape((5, 1))

    def current_callback(self, msg):
        # Storing current the pump is drawing
        self.current = msg.data

    def set_pump(self, value):
        # Turns on/off pump
        msg = UInt16()
        self.pump_value = 1 if value else 0
        msg.data = self.pump_value
        self.pump_pub.publish(msg)
        return True

    def is_contacting(self):
        theta_error = np.sum(np.abs(self.lasttheta_state.reshape(-1) - self.lasttheta.reshape(-1)))
        thetadot_error = np.sum(np.abs(self.lastthetadot_state.reshape(-1) - self.lastthetadot.reshape(-1)))
        return (theta_error > 0.11)

    def fix_goal_theta(self, goal_theta, goal_pos):

        def put_in_range(t):
            return np.remainder(t + np.pi, 2*np.pi) - np.pi

        # Mod by 2pi, result should be in the range [-pi, pi]
        goal_theta = put_in_range(goal_theta)

        # ikin 3DOF arm gives has 2 solutions
        # Try to find the complementary solution, it might be valid
        if not Bounds.is_theta_valid(goal_theta):
            rospy.loginfo("Attempting to correct to valid position")

            # Check if first axis is out of bounds, and correct
            # This would be
            if not Bounds.is_theta_valid(goal_theta, axis=0):
                rospy.loginfo("Corrected 0th axis of goal_theta")
                goal_theta *= np.array([    1,       -1, -1, -1, 1]).reshape((5, 1))
                goal_theta += np.array([np.pi,  np.pi/2,  0, 0, 0]).reshape((5, 1))
                goal_theta = put_in_range(goal_theta)

            # Check if second axis is out of bounds, and correct
            if not Bounds.is_theta_valid(goal_theta, axis=1):
                rospy.loginfo("Corrected 1st axis of goal_theta")
                # Only works if arms are similar lengths
                goal_theta *= np.array([ 1, -1, -1, -1, 1]).reshape((5, 1))
                #goal_theta += np.array([0, 0, 0, 0, 0]).reshape((5, 1))
                goal_theta = put_in_range(goal_theta)

            goal_theta = self.ikin(goal_pos, goal_theta)

        Bounds.assert_theta_valid(goal_theta)

        return goal_theta

    def ikin(self, xgoal, theta_initialguess, return_J=False, max_iter=50, warning=True):
        # Start iterating from the initial guess
        theta = theta_initialguess

        # Iterate at most 50 times (just to be safe)!
        for i in range(max_iter):
            # Figure out where the current joint values put us:
            # Compute the forward kinematics for this iteration.
            (T,J) = self.kin.fkin(theta)
            # 3rd row is 'z' angle in world frame
            # 4th row keeps end effector parallel
            angs = np.array([
                [1, 0, 0, 0, -1],
                [0, 1, -1, -1, 0]
            ])
            J = np.append(J[:3], angs, axis=0)

            # Extract the position
            p = p_from_T(T)

            # Compute the error
            pgoal = xgoal[:3]
            e_p = ep(pgoal, p).reshape((3, 1))
            e_R = xgoal[3:5] - angs @ theta.reshape((5, 1))

            e = np.vstack((e_p, e_R))

            # Take a step in the appropriate direction.
            theta = theta + np.linalg.pinv(J) @ e

            # Return if we have converged.
            if (np.linalg.norm(e) < 1e-6):
                break

        # If we never converge
        else:
            raise RuntimeError(f"Unable to converge to {pgoal.flatten()}")

        #theta = self.fix_goal_theta(theta, xgoal)


        if return_J:
            return theta, J
        return theta


    # Update is called every 10ms!
    def update(self, t):
        dt = t - self.last_t
        self.last_t = t

        #print(self.state)

        if self.state == State.idle:
            # FIND NEW PUZZLE PIECE
            x, y = self.detector.get_random_piece_center()
            x, y = self.detector.screen_to_world(x, y)
            pgoal = np.array([x, y, 0.02, 0, 0]).reshape((5, 1))
            hover = pgoal+np.array([0, 0, .06, 0, 0]).reshape((5, 1))
            hover_theta = self.ikin(hover, self.mean_theta)
            if hover_theta is None:
                return
            rospy.loginfo("chose location:" + str(pgoal))
            rospy.loginfo("goal theta: " + str(hover_theta))
            spline1 = CubicSpline(self.lasttheta, self.lastthetadot, hover_theta, 0, 3)
            goal_theta = self.ikin(pgoal, hover_theta)
            spline2 = CubicSpline(hover_theta, 0, goal_theta, 0, 3)
            spline3 = CubicSpline(goal_theta, 0, hover_theta, 0, 3)
            self.change_segments([spline1, spline2, spline3])
            self.state = State.pickup

        elif self.state == State.grav:
            cmdmsg = JointState()
            cmdmsg.name         = motor_names
            cmdmsg.position     = [np.nan, np.nan, np.nan, np.nan, np.nan]
            cmdmsg.velocity     = [np.nan, np.nan, np.nan, np.nan, np.nan]
            cmdmsg.effort       = self.kin.grav(self.lasttheta_state)
            cmdmsg.header.stamp = rospy.Time.now()
            self.pub.publish(cmdmsg)
            cmdmsg.position     = self.lasttheta_state
            cmdmsg.velocity     = self.lastthetadot_state
            self.rviz_pub.publish(cmdmsg)
            return

        # If the current segment is done, replace the segment with a new one
        dur = self.segments[self.index].duration()
        if (t - self.t0 >= dur):
            self.index = (self.index + 1)
            self.t0 = t

            # if self.state == State.pickup:
            #     if self.index == 2 and not self.pump_value:
            #         self.set_pump(True)
            #     elif self.index == 6 and self.pump_value:
            #         self.set_pump(False)

            if self.index >= len(self.segments):
                self.state = State.idle
                self.segments = None

                status = Status.Ok
                self.solver.notify_action_completed(status)
                self.solver.apply_next_action(self)

        #print(self.segments[self.index].space())
        # Decide what to do based on the space.
        if (self.segments[self.index].space() == 'Joint'):
            # Grab the spline output as joint values.
            (theta, thetadot) = self.segments[self.index].evaluate(t - self.t0)
            
        elif (self.segments[self.index].space() == 'Task'):
            # Grab the spline output as task values.
            # Dim 0-2 are cartesian positions
            # Dim 3 controls tip angle about z axis when end is parallel with table
            # Dim 4 keeps end parallel with table when 0, can be used for flipping
            (p, v)    = self.segments[self.index].evaluate(t - self.t0)

            rospy.loginfo("Screen coordinates of arm: ",
                          self.detector.world_to_screen(p[0, 0], p[1, 0]))

            theta, J = self.ikin(p, self.lasttheta, return_J=True, max_iter=1, warning=False)
            thetadot  = np.linalg.pinv(J[:, :]) @ v

        elif (self.segments[self.index].space() == 'Func'):
            self.segments[self.index]()
            return

        else:
            errmsg = f"Space \"{self.segments[self.index].space()}\" is invalid"
            rospy.signal_shutdown(errmsg)
            raise InvalidSpaceException(errmsg)


        # Save the position (to be used as an estimate next time).
        self.lasttheta = theta
        self.lastthetadot = thetadot

        effort = self.kin.grav(self.lasttheta_state)
        self.safe_publish_cmd(theta, thetadot, effort)

        #if not self.sim and self.is_contacting():
        #    if self.state == State.reset:
        #        self.state == State.grav
        #        return
        #    self.reset(duration=4)

    def safe_publish_cmd(self, position, velocity, effort):
        '''
        Publish a command to update the robot's position, velocity, effort.
        Check that the values are within reasonable bounds.
        Meant as a safeguard so we don't break anything in real life.
        '''
        Bounds.assert_theta_valid(position)
        Bounds.assert_thetadot_valid(velocity)

        # Don't command sudden changes in position
        threshold = 1 # radian, feel free to change
        diff = np.linalg.norm(position - self.lasttheta)
        if diff > threshold:
            errmsg = f"Commanded theta is too far from current theta: lasttheta={self.lasttheta}, command={position}, distance={diff}"
            rospy.logerror(errmsg)
            rospy.signal_shutdown(errmsg)
            raise RuntimeError(errmsg)

        # Create and send the command message.  Note the names have to
        # match the joint names in the URDF.  And their number must be
        # the number of position/velocity elements.
        cmdmsg = JointState()
        cmdmsg.name         = motor_names
        cmdmsg.position     = position
        cmdmsg.velocity     = velocity
        cmdmsg.effort       = effort
        cmdmsg.header.stamp = rospy.Time.now()
        if not self.sim:
            self.pub.publish(cmdmsg)
        self.rviz_pub.publish(cmdmsg)

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
    rate  = 100
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))

    timing_pub = rospy.Publisher("/update_time", Duration, queue_size=10)

    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    starttime = rospy.Time.now()
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

