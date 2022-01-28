#!/usr/bin/env python3
#
#   threesinusoids.py
#
#   Create simple motor commands, moving sinusoidally from the initial
#   position.
#
#   Publish:   /hebi/joint_states       sensor_msgs/JointState
#
#   Subscribe: /hebi/joint_commands     sensor_msgs/JointState
#
import rospy
import math

from sensor_msgs.msg   import JointState

#
#  Controller Class
#
class Controller:
    # Initialize.
    def __init__(self):
        # Collect the motor names, which defines the dofs (useful to know).
        self.motors = ['Thor/1', 'Thor/2', 'Thor/3']
        self.dofs   = len(self.motors)

        # Create a publisher to send the joint commands.  Add some time
        # for the subscriber to connect.  This isn't necessary, but means
        # we don't start sending messages until the recipient is listening.
        self.pub = rospy.Publisher("/hebi/joint_commands", JointState,
                                   queue_size = 3)
        rospy.sleep(0.25)

        # Find the starting positions.  This will block, but that's
        # appropriate as we don't want to start until we have this
        # information.
        msg = rospy.wait_for_message('/hebi/joint_states', JointState);
        for i in range(self.dofs):
            if (msg.name[i] != self.motors[i]):
                raise ValueError("Motor names don't match")

        self.initpos = msg.position
        for i in range(self.dofs):
            rospy.loginfo("Starting motor[%d] '%s' at pos: %f rad",
                          i, self.motors[i], self.initpos[i])


    # Update every 10ms!
    def update(self, t, dt):
        # Create a joint state message and set the motor names.
        cmdmsg = JointState()
        cmdmsg.name = self.motors

        # For the first 3.14 seconds, move to zero.
        if (t <= math.pi):
            cmdmsg.position = [self.initpos[0] * 0.5 * (math.cos(t) + 1.0),
                               self.initpos[1] * 0.5 * (math.cos(t) + 1.0),
                               self.initpos[2] * 0.5 * (math.cos(t) + 1.0)]
            cmdmsg.velocity = [self.initpos[0] * 0.5 * (- math.sin(t)),
                               self.initpos[1] * 0.5 * (- math.sin(t)),
                               self.initpos[2] * 0.5 * (- math.sin(t))]
        else:
            t1 = t - math.pi
            cmdmsg.position = [1.0 * (1.0 - math.cos(1.0 * t1)),
                               1.0 * (1.0 - math.cos(2.0 * t1)),
                               0.5 * (1.0 - math.cos(4.0 * t1))]
            cmdmsg.velocity = [1.0 * (1.0 * math.sin(1.0 * t1)),
                               1.0 * (2.0 * math.sin(2.0 * t1)),
                               0.5 * (4.0 * math.sin(4.0 * t1))]

        # Add no gravity torques.
        cmdmsg.effort = [0.0, 0.0, 0.0]

        # Send the command (with the current time).
        cmdmsg.header.stamp = rospy.Time.now()
        self.pub.publish(cmdmsg)


#
#  Main Code
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node('demo')

    # Instantiate the controller, encapsulating all the computation.
    control = Controller()

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
        control.update(t, dt)

        # Wait for the next turn.  The timing is determined by the
        # above definition of servo.
        servo.sleep()
