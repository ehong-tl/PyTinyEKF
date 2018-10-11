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

    cls.flatten()

n = 2 # number of states
m = 3 # number of observations
ekf = ekf(n,m)

# Set n x n P matrix
ekf.setP(0,0,0.1)
ekf.setP(1,1,0.1)
ekf.flattenP()

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
for i in range(len(y)):
    t = float(time[i]) - prev
    prev = float(time[i])
    z = [float(y[i]),float(y1[i]),float(v[i])] # Sensor observations
    update_eq(ekf, t) # Update matrix
    ekf.step(z) # EKF cycle
    x_filtered = ekf.getX() # Get filtered state
    y_f.append(x_filtered[0])
    v_f.append(x_filtered[1])
    py.append(ekf.getP()[0])
    pv.append(ekf.getP()[3])

yf = np.array(y_f)
plt.plot(time,y_f,'r')
plt.plot(time,y,'g')
plt.plot(time,y1,'b')
plt.plot(time,yr,'black')
plt.figure()
plt.plot(time,v_f,'r')
plt.plot(time,v,'g')
plt.plot(time,vr,'black')
plt.figure()
py = np.array(py)
pv = np.array(pv)
plt.plot(time, py, 'r')
plt.plot(time, pv, 'g')
plt.show()
