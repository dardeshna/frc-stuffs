import numpy as np

class Encoder():
    """Simulates an encoder connected to a motor controller

    Uses CTRE standard units of ticks for position and ticks per 100ms for velocity

    Velocity averaging with adjustable measurement window and sampling period
    """

    def __init__(self, cpr=4096, measurement_window=100, rolling_avg_period=64, dt=0.001, G=1, r=1):

        self.cpr = cpr
        self.measurement_window = measurement_window
        self.rolling_avg_period = rolling_avg_period
        self.dt = dt
        
        self.G = G
        self.r = r

        self.reset()

    def get_conversions(self):

        ticks_per_unit = self.cpr * self.G / (self.r * 2 * np.pi)
        ticks_per_100ms_per_unit_per_second = self.cpr * self.G / (self.r * 2 * np.pi * 10)

        return ticks_per_unit, ticks_per_100ms_per_unit_per_second

    def push_reading(self, reading):
        
        temp = np.zeros(self.measurement_window+self.rolling_avg_period)
        temp[1:] = self.sensor_readings[:-1]
        temp[0] = int(reading)
        self.sensor_readings = temp

        avg = (self.sensor_readings[:self.rolling_avg_period] - self.sensor_readings[self.measurement_window:self.measurement_window+self.rolling_avg_period]).sum() / self.rolling_avg_period
        
        self.prev_sensor_rate = self.sensor_rate
        self.sensor_rate = avg * (100 / self.measurement_window) * (0.001 / self.dt)
        print(self.sensor_rate)
    
    def zero_sensor(self):
        self.sensor_readings = self.sensor_readings - self.sensor_readings[0]

    def reset(self):

        self.sensor_readings = np.zeros(self.measurement_window+self.rolling_avg_period)
        self.prev_sensor_rate = 0
        self.sensor_rate = 0