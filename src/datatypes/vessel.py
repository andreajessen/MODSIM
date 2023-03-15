import numpy as np
##################################################################
#
# Vessel
# - Holds information about the vessel and the vessel's track
#
##################################################################
class Vessel():

    def __init__(self, id, air_draft=2, beam=2, length=4, label=""):
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
        self.id = id
    
    def set_track(self, track):
        self.track = track
    
    def get_track(self):
        return self.track
    
    def get_track_dict(self):
        return self.track.get_track_dict()

    def get_beam(self):
        return self.beam
    
    def get_length(self):
        return self.length
    
    def get_air_draft(self):
        return self.air_draft
    
    def get_rotation_matrix(self, time_stamp):
        direction_vector = self.track.get_direction_vector(time_stamp) # Order of cornerpoints (length, beam): Front back, back back, back front, front front 
        rotation_matrix = np.array([[direction_vector[0], -direction_vector[1], 0], [direction_vector[1], direction_vector[0], 0], [0, 0, 1]])
        return rotation_matrix
    
    def calculate_2D_cornerpoints(self, time_stamp):
        '''
        Calculates the cornerpoints of the vessel given a position and direction vector
        Input:
        - direction_vector (array)
        - position (array)
        Output:
        - cornerpoints (array)
        '''
        direction_vector = self.track.get_direction_vector(time_stamp) # Order of cornerpoints (length, beam): Front back, back back, back front, front front 
        position = self.track.get_position(time_stamp)[:2]
        rotation_matrix = np.array([[direction_vector[0], -direction_vector[1]], [direction_vector[1], direction_vector[0]]])
        cornerpoints_VRF = np.array([[self.length/2, -self.beam/2], [-self.length/2, -self.beam/2], [-self.length/2, self.beam/2], [self.length/2, self.beam/2]])
        cornerpoints_WRF = [np.dot(rotation_matrix,uv)+position for uv in cornerpoints_VRF]
        return np.array(cornerpoints_WRF)
    
    def calculate_3D_cornerpoints(self, time_stamp): # Order of cornerpoints (length, beam, height): Front back lower, back back lower, back front lower, front front lower, Front back upper, back back upper, back front upper, front front upper,
        cornerpoints_2D = self.calculate_2D_cornerpoints(time_stamp)
        cornerpoints_3D = [np.append(cornerpoint, 0) for cornerpoint in cornerpoints_2D]
        cornerpoints_3D.extend([np.append(cornerpoint, self.air_draft) for cornerpoint in cornerpoints_2D])
        return np.array(cornerpoints_3D)
    