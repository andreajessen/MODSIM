import numpy as np
from superSimpleTrackGenerator import SuperSimpleTrackGenerator
from datatypes.vessel import Vessel

# Example sizes of vessels. The elements are the sizes of the airdraft, beam and length
vessel_sizes = {'sailboat': [20, 4, 14], 'motorboat': [4, 3, 7], 'ferry': [15, 11, 64]}

##################################################################
#
# Dynamic scene generator
# - Generates a dynamic scene
#
##################################################################
class DynamicSceneGenerator():

    def __init__(self, vessels=[]):
        """
        Vessels: List of vessels (of type Vessel) in the environment
        """
        self.vessels = vessels
        self.superSimpleTrackGenerator = SuperSimpleTrackGenerator()
        self.vessel_track_parameters = {}
    
    def get_vessels(self):
        return self.vessels
    
    def check_legal_radius(self, radius, used_radii, required_distance):
        '''
        Check if the chosen radius is allowed
        Input:
        - radius (int)
        - used_radii (array)
        - required_distance (int): the required distance the vessels need to keep
        Output:
        - Boolean: True if allowed, else false
        '''
        if used_radii.size == 0: return True

        used_radii_lower = used_radii[used_radii<radius]
        if used_radii_lower.size != 0:
            if not ((radius-required_distance) > np.amax(used_radii_lower)): return False
        used_radii_upper = used_radii[used_radii>radius]
        if used_radii_upper.size !=0:
            if not ((radius+required_distance) < np.amin(used_radii_upper)): return False
        return True


    def generate_random_tracks(self, start_time=0, end_time=200, min_radius=5, max_radius=980, w_min=0.005, w_max=0.03, p_0=[1000,1000], frequency=1):
        '''
        Generates random tracks for each vessel in the scene
        Input:
        - start_time (int, seconds): start time of track
        - end_time (int, seconds): end time of track
        - min_radius (int): minimum radius of the circle
        - max_radius (int): max radius of the circle
        - w_min (double): minimum rotation rate (radians/seconds)
        - w_max (double): max rotation rate (radians/seconds)
        - p_0 (array): position vector of the circular track
        - frequency (int, seconds): time difference between the discrete time steps
        '''
        self.frequency = frequency
        self.p_0 = p_0
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
            vessel.set_track(self.superSimpleTrackGenerator.generate_track(start_time, end_time, self.frequency, radius, theta_start, p_0, w))
            used_radii = np.append(used_radii,radius)
            for x in range(1, required_distance):
                used_radii = np.append(used_radii,radius+x)
                used_radii = np.append(used_radii,radius-x)
        
        self.largest_radius = np.max(used_radii)
    
    def set_initial_vessel_tracks(self, min_radius=5, max_radius=980, w_min=0.005, w_max=0.03, p_0=[1000,1000], frequency=1):
        '''
        - min_radius (int): minimum radius of the circle
        - max_radius (int): max radius of the circle
        - w_min (double): minimum rotation rate (radians/seconds)
        - w_max (double): max rotation rate (radians/seconds)
        - p_0 (array): position vector of the circular track
        - frequency (int, seconds): time difference between the discrete time steps
        '''
        self.frequency = frequency
        self.p_0 = p_0
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
            
            self.vessel_track_parameters[vessel] = {'radius': radius, 'theta_start': theta_start, 'w': w}
            used_radii = np.append(used_radii,radius)
            for x in range(1, required_distance):
                used_radii = np.append(used_radii,radius+x)
                used_radii = np.append(used_radii,radius-x)
        
        self.largest_radius = np.max(used_radii)

    
    def generate_random_tracks_t(self, t):
        '''
        Generates next track position for the next time step for all vessels
        Input:
        - Next time step
        '''
        for vessel in self.vessels:
            parameters = self.vessel_track_parameters[vessel]
            self.superSimpleTrackGenerator.calculate_next_position(t, self.frequency, self.p_0, vessel.get_track(), parameters['radius'], parameters['theta_start'], parameters['w'])


    def set_random_vessels(self, number):
        '''
        Sets size and label for the given number of vessels 
        Input:
        - number (int): number of vessels that should be in the scene
        '''
        self.vessels = []
        for n in range(number):
            vesselID = f'vessel{n}'
            if n%3==0:
                self.vessels.append(Vessel(vesselID, vessel_sizes['sailboat'][0], vessel_sizes['sailboat'][1], vessel_sizes['sailboat'][2], 'sailboat'))
            elif n%5==0:
                self.vessels.append(Vessel(vesselID, vessel_sizes['ferry'][0], vessel_sizes['ferry'][1], vessel_sizes['ferry'][2], 'ferry'))
            else:
                self.vessels.append(Vessel(vesselID, vessel_sizes['motorboat'][0], vessel_sizes['motorboat'][1], vessel_sizes['motorboat'][2], 'motorboat'))
    
    def get_larges_radius(self):
        if not hasattr(self, 'largest_radius'):
            raise NameError("Largest radius is not sat")
        return self.largest_radius


    
    def get_path_centre(self):
        if not hasattr(self, 'p_0'):
            raise NameError("P_0 is not sat")
        return self.p_0
