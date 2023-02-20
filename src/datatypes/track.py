import numpy as np
##################################################################
#
# Track
# - Holds position and direction of a vessel for each time step
#
##################################################################

class Track:
    def __init__(self):
       self.x_values = []
       self.y_values = []
       self.z_values = []
       self.time_stamps = []
       self.track_dict={}
    

    def addPosition(self, x, y, z, heading_rad, time_stamp):
        '''
        Adds position and direction of the vessel for a new timestep
        Input:
        - x (int): X position of the vessel
        - y (int): Y position of the vessel
        - z (int): Z position of the vessel
        - heading_rad (int): the angle between the direction and x axis in radians
        - time_stamp (int): time stamp of the position
        '''
        if self.time_stamps and time_stamp <= self.time_stamps[-1]:
            raise Exception("Invalid position. Timestamp has already happened")
        self.x_values.append(x)
        self.y_values.append(y)
        self.z_values.append(z)
        #self.heading_rads.append(heading_rad)
        #direction_vector = [np.cos(heading_rad), np.sin(heading_rad)]
        #self.direction_vectors.append(direction_vector)
        self.time_stamps.append(time_stamp)
        self.track_dict[time_stamp] = {'center_position_m': [x, y, z], 'heading_rad': heading_rad}
    
    def get_position(self, time_stamp):
        '''
        Returns vessel's position at the given time stamp
        Input:
        - time_stamp (int)
        Output:
        - position (array)
        '''
        return self.track_dict[time_stamp]['center_position_m']
    
    def get_x_values(self):
        return self.x_values
    
    def get_y_values(self):
        return self.y_values
    
    def get_z_values(self):
        return self.z_values
    
    def get_time_stamps(self):
        return self.time_stamps
    
    def get_track_dict(self):
        '''
        Returns a dictionary of all the vessels positions with timestamp as key, and position and heading rad as items.
        '''
        return self.track_dict
    
    def get_direction_vector(self, time_stamp):
        '''
        Returns the vessel's direction at the given time stamp
        Input:
        - time_stamp (int)
        Output:
        - direction_vector (array)
        '''
        heading_rad = self.get_heading_rad(time_stamp)
        return [np.cos(heading_rad), np.sin(heading_rad)]
    
    def get_heading_rad(self, time_stamp):
        return self.track_dict[time_stamp]['heading_rad']


    
