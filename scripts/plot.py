#!/usr/bin/env python3

import rospy   
   
import numpy as np
import matplotlib.pyplot as plt
from time import time

from sensor_msgs.msg     import JointState

plt.ion()
fig, axs = plt.subplots(3, 2, sharex = True)
pos_axs = [axs[i, 0] for i in range(3)]
vel_axs = [axs[i, 1] for i in range(3)]


rospy.init_node('plotter')

value_to_plot = 120

realtime_hist = []
realpos_hist = []
realvel_hist = []

cmdtime_hist = []
cmdpos_hist = []
cmdvel_hist = []

msg = rospy.wait_for_message('/hebi/joint_states', JointState)
realpos_hist.append(msg.position)
realvel_hist.append(msg.velocity)

msg = rospy.wait_for_message('/hebi/joint_commands', JointState)
cmdpos_hist.append(msg.position)
cmdvel_hist.append(msg.velocity)

def realcallback(msg):
    realtime_hist.append(time())
    realpos_hist.append(msg.position)
    realvel_hist.append(msg.velocity)
def cmdcallback(msg):
    cmdtime_hist.append(time())
    cmdpos_hist.append(msg.position)
    cmdvel_hist.append(msg.velocity)

real_sub = rospy.Subscriber('/hebi/joint_states', JointState, realcallback)
cmd_sub = rospy.Subscriber('/hebi/joint_commands', JointState, cmdcallback)

while True:
    plt.clf()
    
    realtime = np.array(realtime_hist)
    realpos = np.array(realpos_hist)
    
    for i in range(3):
        pos_axs[i].clear()
        pos_axs[i].plot(realpos[:, i])
    
    plt.draw()
    plt.pause(0.001)
    
