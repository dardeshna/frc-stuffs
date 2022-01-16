from enum import Enum
import numpy as np

from .encoder import Encoder

class TalonControlMode(Enum):
    PercentOutput = 0
    Position = 1
    Velocity = 2
    # Current = 3
    # Follower = 5
    # MotionProfile = 6
    # MotionMagic = 7
    # MotionProfileArc = 10
    Disabled = 15

class Talon():

    def __init__(self, encoder=Encoder(), dt=0.001):

        self.encoder = encoder
        self.dt = dt

        self.mode = TalonControlMode.Disabled

        self.P = 0
        self.I = 0
        self.D = 0
        self.F = 0
        self.izone = 0

        self.setpoint = 0
        self.error = 0
        self.i_accum = 0
        self.reset = 0
        self.output = 0
        self.prev_output = 0

        self.forwards_peak_output = 1
        self.reverse_peak_output = 1
        self.ramp_rate = 0
        self.deadband = 0.04
        self.voltage_comp_saturation = 12.0
    
    def set(self, mode, setpoint):

        # print(f"Control Mode:{mode}\nSetpoint: {setpoint}")

        if self.mode != mode:
            self.reset = True
        self.mode = mode
        self.setpoint = setpoint
    
    def update(self):

        self.prev_output = self.output

        if self.mode == TalonControlMode.PercentOutput:
            self.output = int(self.setpoint * 1023)
            self.error = 0
        elif self.mode == TalonControlMode.Position:
            self.pid(self.encoder.sensor_readings[0], self.encoder.sensor_readings[1])
        elif self.mode == TalonControlMode.Velocity:
            self.pid(self.encoder.sensor_rate, self.encoder.prev_sensor_rate)
        else:
            self.output = 0
            self.error = 0

        if self.ramp_rate != 0:
            if np.sign(self.output)*(self.output - self.prev_output) > 1023/self.ramp_rate * self.dt:
                self.output = int(self.prev_output + np.sign(self.output - self.prev_output) * 1023/self.ramp_rate * self.dt)
    
    def pid(self, pos, prev_pos):

        self.error = self.setpoint - pos
        if self.reset:
            self.i_accum = 0
        
        if self.izone == 0 or abs(self.error) < self.izone:
            self.i_accum += self.error
        else:
            self.i_accum = 0
        
        d_err = prev_pos - pos

        if self.reset:
            d_err = 0
            self.reset = False
         
        self.output = int(max(min(1023*self.forwards_peak_output, self.error*self.P + d_err*self.D + self.i_accum*self.I + self.setpoint*self.F), -1023*self.reverse_peak_output))

    def get_motor_output_voltage(self, battery_voltage=12.0):

        if abs(self.output) <= self.deadband*1023:
            return 0
        else:
            return max(-battery_voltage, min(battery_voltage, self.output/1023.0 * self.voltage_comp_saturation))

    def get_motor_output_percent(self, battery_voltage=12.0):
        return self.get_motor_output_voltage(battery_voltage)/battery_voltage