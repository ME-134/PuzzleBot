#!/usr/bin/env python3

import rospy   
   
import numpy as np
import matplotlib.pyplot as plt
from time import time

from sensor_msgs.msg     import JointState

plt.ion()
plt.tight_layout()
fig, axs = plt.subplots(4, 3)
fig = None
pos_axs = [axs[i, 0] for i in range(4)]
vel_axs = [axs[i, 1] for i in range(4)]
eff_axs = [axs[i, 2] for i in range(4)]

rospy.init_node('plotter')

value_to_plot = 300

realtime_hist = []
realpos_hist = []
realvel_hist = []
realeff_hist = []

poserror_hist = []
velerror_hist = []
efferror_hist = []

cmdtime_hist = []
cmdpos_hist = []
cmdvel_hist = []
cmdeff_hist = []

msg = rospy.wait_for_message('/hebi/joint_states', JointState)
realtime_hist.append(time())
realpos_hist.append(msg.position)
realvel_hist.append(msg.velocity)
realeff_hist.append(msg.effort)
poserror_hist.append(0)
velerror_hist.append(0)
efferror_hist.append(0)


msg = rospy.wait_for_message('/hebi/joint_commands', JointState)
cmdtime_hist.append(time())
cmdpos_hist.append(msg.position)
cmdvel_hist.append(msg.velocity)
cmdeff_hist.append(msg.effort)

def realcallback(msg):
    realtime_hist.append(time())
    realpos_hist.append(np.array(msg.position))
    realvel_hist.append(np.array(msg.velocity))
    realeff_hist.append(np.array(msg.effort))
    poserror_hist.append(np.sum(np.abs(realpos_hist[-1] - cmdpos_hist[-1])))
    velerror_hist.append(np.sum(np.abs(realvel_hist[-1] - cmdvel_hist[-1])))
    efferror_hist.append(np.sum(np.abs(realeff_hist[-1] - cmdeff_hist[-1])))

def cmdcallback(msg):
    cmdtime_hist.append(time())
    cmdpos_hist.append(np.array(msg.position))
    cmdvel_hist.append(np.array(msg.velocity))
    cmdeff_hist.append(np.array(msg.effort))

real_sub = rospy.Subscriber('/hebi/joint_states', JointState, realcallback)
cmd_sub = rospy.Subscriber('/hebi/joint_commands', JointState, cmdcallback)

while True:
    try:

        # realtime = np.array(realtime_hist)

        if len(realpos_hist) > value_to_plot:
            realpos_hist = realpos_hist[-value_to_plot:]
        if len(realvel_hist) > value_to_plot:
            realvel_hist = realvel_hist[-value_to_plot:]
        if len(realeff_hist) > value_to_plot:
            realeff_hist = realeff_hist[-value_to_plot:]
        
        if len(poserror_hist) > value_to_plot:
            poserror_hist = poserror_hist[-value_to_plot:]
        if len(velerror_hist) > value_to_plot:
            velerror_hist = velerror_hist[-value_to_plot:]
        if len(efferror_hist) > value_to_plot:
            efferror_hist = efferror_hist[-value_to_plot:]

        realpos = np.array(realpos_hist)
        realvel = np.array(realvel_hist)
        realeff = np.array(realeff_hist)

        poserror = np.array(poserror_hist)
        velerror = np.array(velerror_hist)
        efferror = np.array(efferror_hist)

        cmdpos = np.array(cmdpos_hist)
        cmdvel = np.array(cmdvel_hist)
        cmdeff = np.array(cmdeff_hist)

        if len(cmdpos_hist) > value_to_plot:
            cmdpos_hist = cmdpos_hist[-value_to_plot:]
        if len(cmdvel_hist) > value_to_plot:
            cmdvel_hist = cmdvel_hist[-value_to_plot:]
        if len(cmdeff_hist) > value_to_plot:
            cmdeff_hist = cmdeff_hist[-value_to_plot:]
        
        for i in range(3):
            pos_axs[i].clear()
            vel_axs[i].set_title(f'Position Motor {i}')
            pos_axs[i].plot(realtime_hist[-len(realpos[:, i]):], realpos[:, i])
            pos_axs[i].plot(cmdtime_hist[-len(cmdpos[:, i]):], cmdpos[:, i])

        pos_axs[3].clear()
        pos_axs[3].set_title('Position Error')
        pos_axs[3].plot(poserror)

        for i in range(3):
            vel_axs[i].clear()
            vel_axs[i].set_title(f'Velocity Motor {i}')
            vel_axs[i].plot(realtime_hist[-len(realvel[:, i]):], realvel[:, i])
            vel_axs[i].plot(cmdtime_hist[-len(cmdvel[:, i]):], cmdvel[:, i])

        vel_axs[3].clear()
        vel_axs[3].set_title('Velocity Error')
        vel_axs[3].plot(velerror)
        
        for i in range(3):
            eff_axs[i].clear()
            eff_axs[i].set_title(f'Effort Motor {i}')
            eff_axs[i].plot(realtime_hist[-len(realeff[:, i]):], realeff[:, i])
            eff_axs[i].plot(cmdtime_hist[-len(cmdeff[:, i]):], cmdeff[:, i])

        eff_axs[3].clear()
        eff_axs[3].set_title('Effort Error')
        eff_axs[3].plot(efferror)
        
        plt.draw()
        plt.pause(0.01)
    
    except:
        print('PLZ ' * 100)
        plt.close('all')
        break
    
