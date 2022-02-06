#!/usr/bin/env python3

import rospy   
   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import time
from threading import Lock

from sensor_msgs.msg     import JointState

#plt.ion()
#plt.tight_layout()
fig, axs = plt.subplots(4, 3)
#fig = None
pos_axs = [axs[i, 0] for i in range(4)]
vel_axs = [axs[i, 1] for i in range(4)]
eff_axs = [axs[i, 2] for i in range(4)]

rospy.init_node('plotter')

buffer_size = 300

do_plot_real = True
do_plot_cmd = True

cmd_hist  = list()
real_hist = list()
err_hist  = list()

cmd_lock = Lock()
real_lock = Lock()
err_lock = Lock()

def realcallback(msg):
    t = time() * np.ones(3)
    pos = np.array(msg.position)
    vel = np.array(msg.velocity)
    eff = np.array(msg.effort)

    real_lock.acquire()
    real_hist.append([t, pos, vel, eff])
    if len(real_hist) > buffer_size:
        real_hist.pop(0)
    real_lock.release()

    if do_plot_cmd and len(cmd_hist) > 0:
        last  = cmd_hist[-1]

        err_lock.acquire()
        err_hist.append([t, pos-last[0], vel-last[1], eff-last[2]])
        if len(err_hist) > buffer_size:
            err_hist.pop(0)
        err_lock.release()

def cmdcallback(msg):
    t = time() * np.ones(3)
    pos = np.array(msg.position)
    vel = np.array(msg.velocity)
    eff = np.array(msg.effort)

    cmd_lock.acquire()
    cmd_hist.append([t, pos, vel, eff])
    if len(cmd_hist) > buffer_size:
        cmd_hist.pop(0)
    cmd_lock.release()

try:
    msg = rospy.wait_for_message('/joint_states', JointState, timeout=1)
    realcallback(msg)
except rospy.ROSException:
    do_plot_real = False

try:
    msg = rospy.wait_for_message('/hebi/joint_commands', JointState, timeout=1)
    cmdcallback(msg)
except rospy.ROSException:
    do_plot_cmd = False

real_sub = rospy.Subscriber('/joint_states', JointState, realcallback)
cmd_sub = rospy.Subscriber('/hebi/joint_commands', JointState, cmdcallback)

'''
fig = plt.figure(figsize=(12,6), facecolor='#DEDEDE')
ax = plt.subplot(121)
ax1 = plt.subplot(122)
ax.set_facecolor('#DEDEDE')
ax1.set_facecolor('#DEDEDE')
'''

def loop(t):

    if do_plot_real:
        real_lock.acquire()
        real_hist_arr = np.array(real_hist)
        real_lock.release()
        realt   = real_hist_arr[:, 0]
        realpos = real_hist_arr[:, 1]
        realvel = real_hist_arr[:, 2]
        realeff = real_hist_arr[:, 3]
    
    if do_plot_cmd:
        cmd_lock.acquire()
        cmd_hist_arr = np.array(cmd_hist)
        cmd_lock.release()
        cmdt   = cmd_hist_arr[:, 0]
        cmdpos = cmd_hist_arr[:, 1]
        cmdvel = cmd_hist_arr[:, 2]
        cmdeff = cmd_hist_arr[:, 3]

    if do_plot_real and do_plot_cmd:
        err_lock.acquire()
        err_hist_arr = np.array(err_hist)
        err_lock.release()
        errt   = err_hist_arr[:, 0]
        errpos = err_hist_arr[:, 1]
        errvel = err_hist_arr[:, 2]
        erreff = err_hist_arr[:, 3]

    for i in range(3):
        pos_axs[i].clear()
        pos_axs[i].set_title(f'Position Motor {i}')
        if do_plot_real:
            pos_axs[i].plot(realt[:, i], realpos[:, i])
        if do_plot_cmd:
            pos_axs[i].plot(cmdt[:, i], cmdpos[:, i])

    if do_plot_real and do_plot_cmd:
        pos_axs[3].clear()
        pos_axs[3].set_title('Position Error')
        pos_axs[3].plot(errt[:, 0], np.sum(errpos, axis=1))

    for i in range(3):
        vel_axs[i].clear()
        vel_axs[i].set_title(f'Velocity Motor {i}')
        if do_plot_real:
            vel_axs[i].plot(realt[:, i], realvel[:, i])
        if do_plot_cmd:
            vel_axs[i].plot(cmdt[:, i], cmdvel[:, i])

    if do_plot_real and do_plot_cmd:
        vel_axs[3].clear()
        vel_axs[3].set_title('Velocity Error')
        vel_axs[3].plot(errt[:, 0], np.sum(errvel, axis=1))
    
    for i in range(3):
        eff_axs[i].clear()
        eff_axs[i].set_title(f'Effort Motor {i}')
        if do_plot_real:
            eff_axs[i].plot(realt[:, i], realeff[:, i])
        if do_plot_cmd:
            eff_axs[i].plot(cmdt[:, i], cmdeff[:, i])

    if do_plot_real and do_plot_cmd:
        eff_axs[3].clear()
        eff_axs[3].set_title('Effort Error')
        vel_axs[3].plot(errt[:, 0], np.sum(erreff, axis=1))
    
# animate
ani = FuncAnimation(fig, loop, interval=300, repeat=True)
plt.show()
