# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 20:07:18 2026

@author: nsast
"""

import numpy as np
import matplotlib.pyplot as plt

#function that takes an np.array() vector and turns it into its corresponding skew matrix
#usful since matrix multiplication with skew matrix is more efficient that cross product calculation
def cross(vec): 
    a1 = vec[0][0]
    a2 = vec[1][0]
    a3 = vec[2][0]
    cross_matrix = np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    return cross_matrix

#--------------------Inputs----------------------------------------------------

#time clock
t = 0 #initial value set at 0

#inclination in radians
incl = 4 #placeholder

#orbit altitude (assumed to be constant)
R_orb = 550000 #placeholder

#satellite arguement of latitude in radians at time t=0
u0 = 1 #placeholder

roll = 4 #placeholder
pitch = 4 #placeholder
yaw = 4 #placeholder

#total strength of the axial dipole model of the Earth geomagnetic field (ampere metre squared)
mu_ad = 400 #placeholder

#orbital angular rate
omega_orbital = 1 #placeholder

#body reference frame coordinates of the angular velocity of the body frame with respect to the orbital frame
omega_r = np.array([[1], [2], [3]]) #placeholders

#inertia matrix
J = np.diag([1, 2, 3]) #placeholders for Jx, Jy, and Jz

#coordinates of the nadir axis in the body frame (determined by Earth horizon sensors)
n = np.array([[1], [2], [3]]) #placeholders

#coordinates of these axis in the body reference frame to be pointed in the nadir direction
e3 = np.array([[0], [0], [1]])

#scalar gains for the magnetorquer
kp = 1 #placeholder
kd = 1 #placeholder


#--------------------Calculated Variables--------------------------------------

#rotation matrices between orbital and body reference frames for nadir pointing
Rx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
Ry = np.array([[np.cos(pitch), 0 , -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
Rz = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
R = Rx @ Ry @ Rz

#time-dependent axial dipole model of the Earth geomagnetic field
bot = (mu_ad / (R_orb**3)) * np.array([[np.sin(incl)*np.cos(omega_orbital*t+u0)], [-np.cos(incl)], [2*np.sin(incl)*np.sin(omega_orbital*t+u0)]])

#body reference frame coordinates of the geomagnetic induction at the satellite
bRt = R @ bot

gammat = np.transpose(bot) @ bot * np.diag([1, 1, 1]) - bot @ np.transpose(bot)

#physical input for the magnetorquers
#mc = np.cross(bRt, kp*np.cross(e3, n)-kd*omega_r)

#mathematical control input for the magnetorquers
u = kp*cross(e3) @ n - kd*omega_r

#orbital angluar rate vector
omega_o = np.array[[0], [omega_orbital], [0]]


#--------------------Torques---------------------------------------------------

#gravitational gradient torque
Tgg = 3 * (omega_orbital**3) * cross(n) @ (J @ n)

#other disturbance torques
Td = np.array([[0], [0], [0]]) #assumed to be zero

#control torque
Tc = R @ gammat @ np.transpose(R) @ u


#--------------------Attitude Dynamics-----------------------------------------

#rate of change of the angular velocity of the body frame with respect to the orbital frame
omega_r_dot = np.linalg.inv(J) @ ((-J@cross(R@omega_o) -cross(omega_r)@J -cross(R@omega_o)@J)@omega_r -cross(R @ omega_o)@J@R@omega_o + Tgg + Td + Tc)

 