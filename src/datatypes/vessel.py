import numpy as np
##################################################################
#
# Vessel
# - Holds information about the vessel and the vessel's track
#
##################################################################
class Vessel():

    def __init__(self, air_draft=2, beam=2, length=4, label=""):
        """
        Input:
        Beam (int): The width of the widest point of the boat
        air_draft (int): The distance between the ship's waterline and the highest point of the boat; indicates the distance the vessel can safely pass under
        Length (int): Length of the boat
        Label (string): boat class (Sailboat, motorboat etc)
        """
        self.air_draft = air_draft
        self.beam = beam
        self.length = length
        self.label = label
        self.track = None
    
    def set_track(self, track):
        self.track = track
    
    def get_track(self):
        return self.track

    def get_beam(self):
        return self.beam
    
    def get_length(self):
        return self.length
    
    def calculate_cornerpoints(self, direction_vector, position):
        '''
        Calculates the cornerpoints of the vessel given a position and direction vector
        Input:
        - direction_vector (array)
        - position (array)
        Output:
        - cornerpoints (array)
        '''
        rotation_matrix = np.array([[direction_vector[0], -direction_vector[1]], [direction_vector[1], direction_vector[0]]])
        cornerpoints_VRF = np.array([[self.length/2, -self.beam/2], [-self.length/2, -self.beam/2], [-self.length/2, self.beam/2], [self.length/2, self.beam/2]])
        cornerpoints_WRF = [np.dot(rotation_matrix,uv)+position for uv in cornerpoints_VRF]
        return np.array(cornerpoints_WRF)