import numpy as np
from superSimpleTrackGenerator import SuperSimpleTrackGenerator
from datatypes.vessel import Vessel

# Airdraft, beam, length
vessel_sizes = {'sailboat': [20, 4, 14], 'motorboat': [4, 3, 7], 'ferry': [15, 11, 64]}

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
    
    def check_legal_radius(self, radius, used_radii, required_distance):
        if used_radii.size == 0: return True

        used_radii_lower = used_radii[used_radii<radius]
        if used_radii_lower.size != 0:
            if not ((radius-required_distance) > np.amax(used_radii_lower)): return False
        used_radii_upper = used_radii[used_radii>radius]
        if used_radii_upper.size !=0:
            if not ((radius+required_distance) < np.amin(used_radii_upper)): return False
        return True


    def generate_random_tracks(self, start_time=0, end_time=200, min_radius=5, max_radius=180, w_min=0.005, w_max=0.03, p_0=[200,200], frequency=1):
        if max_radius < len(self.vessels):
            raise Exception("Can not generate a collision free model. Increase radius and decrease number of vessels.")
        used_radii = np.array([])
        for vessel in self.vessels:

            # To avoid collision all vessels should have different radii and
            # we assume the boats keep at least the distance of their beam between each other
            # If the required distance is to large, the random radius generator might take to much time
            required_distance = vessel.get_beam()
            radius = np.random.choice(list(set(range(min_radius, max_radius)) - set(used_radii)))
            while not self.check_legal_radius(radius, used_radii, required_distance):
            #while ((radius in used_radii) or ((radius+vessel.get_beam()) in used_radii) or ((radius-vessel.get_beam()) in used_radii)):
                # To speed up process:
                radius = np.random.choice(list(set(range(min_radius, max_radius)) - set(used_radii)))
                # radius = np.random.randint(min_radius, max_radius)

            theta_start = np.radians(np.random.randint(0, 360))
            w = np.random.uniform(w_min, w_max)
            vessel.set_track(self.superSimpleTrackGenerator.generate_track(start_time, end_time, frequency, radius, theta_start, p_0, w, vessel.get_beam(), vessel.get_length()))
            used_radii = np.append(used_radii,radius)
            for x in range(1, required_distance):
                used_radii = np.append(used_radii,radius+x)
                used_radii = np.append(used_radii,radius-x)



    def set_random_vessels(self, number):
        self.vessels = []
        for n in range(number):
            if n%3==0:
                self.vessels.append(Vessel(vessel_sizes['sailboat'][0], vessel_sizes['sailboat'][1], vessel_sizes['sailboat'][2], 'sailboat'))
            elif n%5==0:
                self.vessels.append(Vessel(vessel_sizes['ferry'][0], vessel_sizes['ferry'][1], vessel_sizes['ferry'][2], 'ferry'))
            else:
                self.vessels.append(Vessel(vessel_sizes['motorboat'][0], vessel_sizes['motorboat'][1], vessel_sizes['motorboat'][2], 'motorboat'))

    def get_vessels(self):
        return self.vessels