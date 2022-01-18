import sys
import os
sys.path.insert(1, os.path.realpath(os.path.pardir))

import copy
import numpy as np
import scipy.signal
import scipy.linalg
from matplotlib import pyplot as plt

from flywheel import Flywheel
from util import Motor, Encoder, Talon, TalonControlMode, Sim

SIM_DT = 0.001 # simulation and motor controller
CTRL_DT = 0.02 # main robot controller (roboRIO)

# motor and system constants

# falcon 500
V_nominal = 12 # V
w_free =  6380 * (2 * np.pi / 60) # RPM -> rad/s
T_stall = 4.69 # Nm
I_stall = 257 # A
I_free = 1.5 # A
n_motors = 1

G_ratio = 1
r_flywheel = 0.076 # m
m_flywheel = 0.23 # kg
n_flywheel = 2
J_flywheel = 1/2 * m_flywheel * r_flywheel**2 * n_flywheel # kg*m^2, solid disk

motor = Motor(V_nominal, w_free, T_stall, I_stall, I_free, n_motors)
flywheel = Flywheel(motor, G_ratio, J_flywheel)
encoder = Encoder(G=G_ratio, measurement_window=100, rolling_avg_period=64, dt=SIM_DT)
base_talon = Talon(encoder, dt=SIM_DT)
ticks_per_rev, ticks_per_100ms_per_rev_per_second = encoder.get_conversions()

# Current limits (limit current (A), trigger current (A), trigger threshold time (s))
base_talon.stator_limit_config = (20, 30, 0)
base_talon.stator_limit_enable = True
base_talon.supply_limit_config = (np.inf, np.inf, 0)
base_talon.supply_limit_enable = False

# PID Gains
base_talon.P = 0.05
base_talon.I = 0
base_talon.D = 0
base_talon.F = 1023 / (w_free / G_ratio * ticks_per_100ms_per_rev_per_second)
base_talon.izone = 0

def sim_offboard(sim_ts, x_0, w_goal, tau_ext):
    """Simulates velocity PIDF control loop on the motor controller
    """

    sim = Sim(flywheel, x_0)
    talon = copy.deepcopy(base_talon)
    talon.encoder.set_rate(x_0[1] * ticks_per_100ms_per_rev_per_second)
    talon.set(TalonControlMode.Velocity, w_goal * ticks_per_100ms_per_rev_per_second)

    # run simulation
    for i, t in enumerate(sim_ts):

        talon.update()
        V = talon.get_motor_output_voltage()
        sim.step((V, tau_ext[i]), SIM_DT)
        talon.encoder.push_reading(sim.xs[-1][0] * ticks_per_rev)
        talon.current = np.abs(motor.get_I(V, sim.xs[-1][1]))

    return sim

# LQR Gains

K_ff = 12 / (w_free / G_ratio) # feedforward gain

A_d, B_d, C_d, _, _ = scipy.signal.cont2discrete((np.atleast_2d(flywheel.A[1,1]), np.atleast_2d(flywheel.B[1,0]), np.atleast_2d(1), np.atleast_2d(0)), CTRL_DT)
Q_fb = 1
R_fb = 2000

# Solve discrete algebriac ricatti eq (DARE) to obtain feedback gain K_fb
# https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Infinite-horizon,_discrete-time_LQR
P_fb = scipy.linalg.solve_discrete_are(A_d, B_d, Q_fb, R_fb)
K_fb = np.linalg.solve(R_fb + B_d.T @ P_fb @ B_d, B_d.T @ P_fb @ A_d)

Q_k = 1
R_k = 0.01

# Solve DARE for dual system to obtain steady-state kalman gain K_k
P_k = scipy.linalg.solve_discrete_are(A_d.T, C_d.T, Q_k, R_k)
K_k = np.linalg.solve(R_k + C_d @ P_k @ C_d.T, C_d @ P_k).T # TODO : should it be multiplied by A_d.T ???

print("feedback gain (K_fb): ", K_fb)
print("kalman gain (K_k): ", K_k)

def sim_onboard(sim_ts, x_0, w_goal, tau_ext):
    """Simulates velocity feedback control w/ kalman observer on the roboRIO
    """

    sim = Sim(flywheel, x_0)
    talon = copy.deepcopy(base_talon)
    talon.encoder.set_rate(x_0[1] * ticks_per_100ms_per_rev_per_second)
    talon.set(TalonControlMode.Velocity, w_goal * ticks_per_100ms_per_rev_per_second)

    w_hat = x_0[1]

    ctrl_ts = np.arange(int(sim_ts[-1]/CTRL_DT)) * CTRL_DT

    update_ctrl = np.zeros_like(sim_ts, dtype=bool)
    update_ctrl[np.searchsorted(sim_ts, ctrl_ts)] = True

    # run simulation
    for i, t in enumerate(sim_ts):

        if update_ctrl[i]:

            y = talon.encoder.sensor_rate / ticks_per_100ms_per_rev_per_second # read measurement
            w_hat += K_k * (y - w_hat) # observer update step
            percent_out = (K_fb * (w_goal - w_hat) + K_ff * w_goal) / 12.0 # compute control

            talon.set(TalonControlMode.PercentOutput, percent_out)

        talon.update()
        V = talon.get_motor_output_voltage()
        sim.step((V, tau_ext[i]), SIM_DT)
        talon.encoder.push_reading(sim.xs[-1][0] * ticks_per_rev)
        talon.current = np.abs(motor.get_I(V, sim.xs[-1][1]))

        if update_ctrl[i]:
            w_hat = A_d @ w_hat + B_d @ np.atleast_1d(V) # observer predict step
    
    return sim

def sim_spinup(sim_func, t, w_goal):
    """Simulates spinning up a shooter
    """

    sim_ts = np.arange(int(t/SIM_DT)) * SIM_DT
    tau_ext = np.zeros_like(sim_ts)

    sim = sim_func(sim_ts, np.array([0, 0]), w_goal, tau_ext)

    return sim

def sim_shoot(sim_func, t, w_goal, shooter_type='single'):
    """Simulates firing a ball
    """

    # print("flywheel inertia:", J_flywheel)
    # print("flywheel energy:", 1/2 * J_flywheel * w_goal**2)

    tau_dist, steps_dist = calc_ball_ext_torque(w_goal, shooter_type)

    # print("ball inertia:", J_ball)
    # print("ball energy: ",  tau_dist * wrap_angle)

    # print("accel time: ", steps_dist * SIM_DT)

    idx_start = int(0.2*t / SIM_DT)
    idx_end = idx_start + steps_dist

    t = idx_end * SIM_DT if idx_end * SIM_DT > t else t

    sim_ts = np.arange(int(t/SIM_DT)) * SIM_DT
    tau_ext = np.zeros_like(sim_ts)
    tau_ext[idx_start:idx_end] = tau_dist
    
    sim = sim_func(sim_ts, np.array([0, w_goal]), w_goal, tau_ext)

    return sim

# Ball disturbance
# assumes the ball is accelerated linearly over the entrire wrap of the hood
# note: wrap is not very well defined, especially for double flywheels, but it doesn't have a significant impact under these assumptions

m_ball = 0.27 # kg
r_ball = 0.12 # m
J_ball = 2/3 * m_ball * r_ball ** 2 # kg*m^2, thin spherical shell
wrap_angle = np.pi / 4 # rad

def calc_ball_ext_torque(w_flywheel, shooter_type='single'):
    """Calculate external torque applied to flywheel by ball over a certain number of simulation steps

    shooter_type=single: hooded shooter with v_ball = 1/2 * r_flywheel * w_flywheel
    shooter_type=double: double flywheel with v_ball = r_flywheel * w_flywheel where each flywheel contributes half of ball energy
    """

    if shooter_type == 'single':
        v_ball = 1/2 * r_flywheel * w_flywheel
    elif shooter_type == 'double':
        v_ball = r_flywheel * w_flywheel
    
    dir = np.sign(w_flywheel)

    if shooter_type == 'single':
        K_trans = 1/2 * m_ball * v_ball ** 2
        K_rot = 1/2 * J_ball * (v_ball / r_ball) ** 2
        K = K_trans + K_rot
    elif shooter_type == 'double':
        K = 1/2 * (1/2 * m_ball * v_ball ** 2) # half of ball energy

    w_flywheel_final = dir * np.sqrt(w_flywheel ** 2 - 2 / J_flywheel * K) # 1/2*J*w_f^2 = 1/2*J*w_0^2 - K

    print("initial speed:", w_flywheel * 60 / (2 * np.pi))
    print("final speed:", w_flywheel_final * 60 / (2 * np.pi))

    t = np.abs(wrap_angle / ((w_flywheel + w_flywheel_final) / 2))
    steps = int(t / SIM_DT)
    
    tau_ext = dir * K / (wrap_angle * steps * SIM_DT / t)

    # print("actual disturbance: ", t, K/wrap_angle)
    # print("adjusted disturbance: ", steps * SIM_DT, tau_ext)

    return tau_ext, steps

if __name__ == "__main__":
    
    # simulation duration
    t_sim = 5

    # flywheel setpoint
    w_goal = w_free / 2

    # construct desired simulation type
    # sim = sim_spinup(sim_onboard, t_sim, w_goal)
    sim = sim_shoot(sim_onboard, t_sim, w_goal, shooter_type='double')
    xs, us, ts = sim.get_result()

    plt.figure(figsize=(6, 8))

    # plot flywheel speed
    plt.subplot(3, 1, 1)
    plt.plot((ts[0], ts[-1]), (w_goal * 60 / (2*np.pi), w_goal * 60 / (2*np.pi)))
    plt.plot(ts, xs[:,1] * 60 / (2*np.pi))
    plt.legend(('setpoint', 'actual'))
    plt.ylabel('flywheel speed (rpm)')

    # plot motor voltage
    plt.subplot(3, 1, 2)
    plt.plot(ts, us[:, 0])
    plt.legend(('motor voltage',))
    plt.ylabel('voltage (V)')

    # plot currents
    plt.subplot(3, 1, 3)
    plt.plot(ts[1:], motor.get_I(us[:-1, 0], xs[1:, 1]))
    plt.plot(ts[1:], motor.get_I(us[:-1, 0], xs[1:, 1])*us[:-1, 0]/12)
    plt.legend(('motor current', 'supply current'))
    plt.xlabel('time (s)')
    plt.ylabel('current (A)')

    plt.show()