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

        self.P = 0.1
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

        self.input_voltage = 12.0

        # limit current, trigger current, trigger threshold time
        self.supply_limit_config = (np.inf, np.inf, 0) 
        self.supply_limit_enable = False
        self.stator_limit_config = (np.inf, np.inf, 0) # limit, trigger, trigger threshold time
        self.stator_limit_enable = False

        self.motor_current = 0
        self.input_current = 0
        self.supply_limiting = False
        self.supply_limit_counter = 0
        self.stator_limiting = False
        self.stator_limit_counter = 0

    @property
    def current(self):
        return self.motor_current
    
    @current.setter
    def current(self, value):
        self.motor_current = value
        self.input_current = np.abs(self.get_motor_output_percent()) * value
    
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
            self._pid(self.encoder.sensor_readings[0], self.encoder.sensor_readings[1])
        elif self.mode == TalonControlMode.Velocity:
            self._pid(self.encoder.sensor_rate, self.encoder.prev_sensor_rate)
        else:
            self.output = 0
            self.error = 0

        if self.stator_limit_enable and not self.stator_limiting:
                if self.motor_current > self.stator_limit_config[1]:
                    self.stator_limit_counter += 1
                else:
                    self.stator_limit_counter = 0
                if self.stator_limit_counter > int(self.stator_limit_config[2]/self.dt):
                    self.stator_limiting = True
                    self.stator_limit_counter = 0

        if self.supply_limit_enable and not self.supply_limiting:
                if self.input_current > self.supply_limit_config[1]:
                    self.supply_limit_counter += 1  
                else:
                    self.supply_limit_counter = 0
                if self.supply_limit_counter > int(self.supply_limit_config[2]/self.dt):
                    self.supply_limiting = True
                    
                    self.supply_limit_counter = 0
        
        # TODO: replace ramp rate kludge with PID current control
        if (self.stator_limit_enable and self.stator_limiting and self.motor_current > self.stator_limit_config[0] or
                self.supply_limit_enable and self.supply_limiting and self.input_current > self.supply_limit_config[0]):
            self.output = self._ramp_output(0, self.prev_output, 0.1)
        elif (self.stator_limit_enable and self.stator_limiting or
                self.supply_limit_enable and self.supply_limiting):
            self.output = self._ramp_output(self.output, self.prev_output, 0.1)
            if np.sign(self.prev_output)*self.output < np.abs(self.prev_output):
                if self.motor_current < self.stator_limit_config[0]:
                    self.stator_limiting = False
                if self.input_current < self.supply_limit_config[0]:
                    self.supply_limiting = False
                
        if self.ramp_rate != 0:
            self.output = self._ramp_output(self.output, self.prev_output, self.ramp_rate, True)
    
    def _pid(self, pos, prev_pos):

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

    def _ramp_output(self, output, prev_output, ramp_rate, unidirectional=False):

        if not unidirectional and np.abs(output - prev_output) > 1023/ramp_rate * self.dt:
            return int(prev_output + np.sign(output - prev_output) * 1023/ramp_rate * self.dt)
        elif unidirectional and np.sign(self.output)*(self.output - self.prev_output) > 1023/ramp_rate * self.dt:
            return int(prev_output + np.sign(output - prev_output) * 1023/ramp_rate * self.dt)
        else:
            return output

    def get_motor_output_voltage(self):

        if abs(self.output) <= self.deadband*1023:
            return 0
        else:
            return max(-self.input_voltage, min(self.input_voltage, self.output/1023.0 * self.voltage_comp_saturation))

    def get_motor_output_percent(self):
        return self.get_motor_output_voltage()/self.input_voltage