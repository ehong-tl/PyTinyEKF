#!/usr/bin/env python

from PyTinyEKF import ekf
import numpy as np
import matplotlib.pyplot as plt

def update_eq(cls, t):
    # Set n x n Jacobian matrix F
    cls.setF(0,0,1)
    cls.setF(0,1,t)
    cls.setF(1,1,1)

    # Set m x n Jacobian matrix H
    cls.setH(0,0,1)
    cls.setH(1,0,1)
    cls.setH(2,1,1)

    # Set n x 1 f(x) matrix
    cls.setfx(0, ekf.x[0] + t * ekf.x[1])
    cls.setfx(1, ekf.x[1])

    # Set m x 1 h(x) matrix
    cls.sethx(0, ekf.x[0])
    cls.sethx(1, ekf.x[0])
    cls.sethx(2, ekf.x[1])

    # Set n x n Q matrix
    cls.setQ(0,0,0.0001)
    cls.setQ(1,1,0.0001)

    # Set m x m R matrix
    cls.setR(0,0,0.04)
    cls.setR(1,1,0.25)
    cls.setR(2,2,0.01)


n = 2 # number of states
m = 3 # number of observations
ekf = ekf(n,m)

# Set initial state n x 1 X matrix
ekf.setX(0,0)
ekf.setX(1,0)

# Set initial covariance n x n P matrix
ekf.setP(0,0,0.1)
ekf.setP(1,1,0.1)

time = np.arange(0, 60, 0.1)
y = np.sin(2*np.pi/60*time) + np.random.normal(0, 0.2, len(time))
y1 = np.sin(2*np.pi/60*time) + np.random.normal(0, 0.5, len(time))
v = 2*np.pi/60*np.cos(2*np.pi/60*time) + np.random.normal(0, 0.1, len(time))
yr = np.sin(2*np.pi/60*time)
vr = 2*np.pi/60*np.cos(2*np.pi/60*time)

prev = 0
y_f = []
v_f = []
py = []
pv = []
gy1 = []
gy2 = []
gv = []
for i in range(len(time)):
    t = float(time[i]) - prev
    prev = float(time[i])
    z = [float(y[i]),float(y1[i]),float(v[i])] # Sensor observations
    ekf.step(z, update_eq, ekf, t) # EKF cycle
    x_filtered = ekf.getX() # Get filtered state
    p_filtered = ekf.getP() # Get filtered covariance
    g_filtered = ekf.getG() # Get Kalman gain
    y_f.append(x_filtered[0])
    v_f.append(x_filtered[1])
    py.append(p_filtered[0])
    pv.append(p_filtered[3])
    gy1.append(g_filtered[0])
    gy2.append(g_filtered[1])
    gv.append(g_filtered[5])

yf = np.array(y_f)
plt.plot(time,y_f,'r',label='EKF')
plt.plot(time,y,'g',label='Raw 1')
plt.plot(time,y1,'b',label='Raw 2')
plt.plot(time,yr,'black',label='Truth')
plt.legend()
plt.title('Displacement')
plt.figure()
plt.plot(time,v_f,'r',label='EKF')
plt.plot(time,v,'g',label='Raw')
plt.plot(time,vr,'black',label='Truth')
plt.legend()
plt.title('Velocity')
plt.figure()
py = np.array(py)
pv = np.array(pv)
plt.plot(time, py, 'r',label='Displacement')
plt.plot(time, pv, 'g',label='Velocity')
plt.legend()
plt.title('Covariance')
plt.figure()
gy1 = np.array(gy1)
gy2 = np.array(gy2)
gv = np.array(gv)
plt.plot(time, gy1, 'r',label='Displacement 1')
plt.plot(time, gy2, 'g',label='Displacement 2')
plt.plot(time, gv, 'b',label='Velocity')
plt.legend()
plt.title('Kalman Gain')
plt.show()
