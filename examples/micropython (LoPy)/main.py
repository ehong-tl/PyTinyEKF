# EKF to filter BLE beacon RSSI value

from network import Bluetooth
import ubinascii
import utime
from PyTinyEKF import ekf

def update_eq(cls, t):
    # Set n x n Jacobian matrix F
    cls.setF(0,0,1)
    cls.setF(0,1,t)
    cls.setF(1,1,1)

    # Set m x n Jacobian matrix H
    cls.setH(0,0,1)

    # Set n x 1 f(x) matrix
    cls.setfx(0, ekf.x[0] + t * ekf.x[1])
    cls.setfx(1, ekf.x[1])

    # Set m x 1 h(x) matrix
    cls.sethx(0, ekf.x[0])

    # Set n x n Q matrix
    cls.setQ(0,0,0.001)
    cls.setQ(1,1,0.001)

    # Set m x m R matrix
    cls.setR(0,0,0.1)

n = 2 # number of states
m = 1 # number of observations
ekf = ekf(n,m)

# Set initial state n x 1 X matrix
ekf.setX(0,-70)
ekf.setX(1,0)

# Set initial covariance n x n P matrix
ekf.setP(0,0,100)
ekf.setP(1,1,100)

# Setup bluetooth to scan bluetooth devices
bt = Bluetooth()
bt.stop_scan()
bt.start_scan(-1)

prev = utime.ticks_ms()
while True:
    adv = bt.get_advertisements()
    if len(adv) > 0:
        for ad in adv:
            if bt.resolve_adv_data(ad.data, bt.ADV_NAME_CMPL) == 'RECO':
                if ubinascii.hexlify(ad.mac) == b'fd46b42dfe28':
                    now =  utime.ticks_ms()
                    t = utime.ticks_diff(now, prev)/1000
                    prev = now
                    z = [ad.rssi] # Sensor observations
                    ekf.step(z, update_eq, ekf, t) # EKF cycle
                    x_filtered = ekf.getX() # Get filtered states
                    print(x_filtered[0], ',', z[0])
