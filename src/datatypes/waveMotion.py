import numpy as np
##################################################################
#
# WaveMotion
# - Generates the three independent wave motion processes for roll, 
# pitch and yaw.
# - The wave motion is added on the camera rig.
#
##################################################################
class WaveMotion:
    def __init__(self):
        """OBS: These parameters need tuning"""

        self.roll_1 = 0
        self.pitch_1 = 0
        self.yaw_1 = 0

        self.roll_2 = 0
        self.pitch_2 = 0
        self.yaw_2 = 0

        self.roll_a = 0.99
        self.roll_b = 0.99
        self.pitch_a = 0.99
        self.pitch_b = 0.99
        self.yaw_a = 0.99
        self.yaw_b = 0.99

        self.roll_sigma = 0.0001
        self.pitch_sigma = 0.001
        self.yaw_sigma = 0.0001

    def generate_wave(self):
        self.roll_1 = self.roll_a*self.roll_1 + np.random.normal(scale=self.roll_sigma)
        self.roll_2 = self.roll_b*self.roll_2 + self.roll_1

        self.pitch_1 = self.pitch_a*self.pitch_1 + np.random.normal(scale=self.pitch_sigma)
        self.pitch_2 = self.pitch_b*self.pitch_2 + self.pitch_1

        self.yaw_1 = self.yaw_a*self.yaw_1 + np.random.normal(scale=self.yaw_sigma)
        self.yaw_2 = self.yaw_b*self.yaw_2 + self.yaw_1
        return self.roll_2, self.pitch_2, self.yaw_2