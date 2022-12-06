import numpy as np
from superSimpleTrackGenerator import SuperSimpleTrackGenerator
from datatypes.vessel import Vessel

class DynamicSceneGenerator():

    def __init__(self, vessels):
        """
        Vessels: List of vessels (of type Vessel) in the environment
        """
        self.vessels = vessels
        self.superSimpleTrackGenerator = SuperSimpleTrackGenerator()
    
    def __init__(self):
        self.superSimpleTrackGenerator = SuperSimpleTrackGenerator()
        self.vessels = []

    def generate_random_tracks(self, start_time=0, end_time=200, min_radius=1, max_radius=10, w_min=0.01, w_max=0.3, p_0=[0,0], frequency=1):
        if max_radius < len(self.vessels):
            raise Exception("Can not generate a collision free model. Increase radius and decrease number of vessels.")
        used_radii = []
        for vessel in self.vessels:

            # To avoid collision all vessels should have different radii
            radius = np.random.randint(min_radius, max_radius)
            while radius in used_radii:
                radius = np.random.randint(min_radius, max_radius)

            theta_start = np.radians(np.random.randint(0, 360))
            w = np.random.uniform(w_min, w_max)
            vessel.set_track(self.superSimpleTrackGenerator.generate_track(start_time, end_time, frequency, radius, theta_start, p_0, w))
            used_radii.append(radius)

    def set_vessels(self, number):
        self.vessels = []
        for _ in range(number):
            self.vessels.append(Vessel())

    def get_vessels(self):
        return self.vessels