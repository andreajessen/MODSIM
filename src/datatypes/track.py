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
       self.direction_vectors = []
       self.time_stamps = []
    

    def addPosition(self, x, y, z, direction_vector, time_stamp):
        '''
        Adds position and direction of the vessel for a new timestep
        Input:
        - x (int): X position of the vessel
        - y (int): Y position of the vessel
        - z (int): Z position of the vessel
        - direction_vector (array): a vector indicating the direction of the vessel
        - time_stamp (int): time stamp of the position
        '''
        if self.time_stamps and time_stamp <= self.time_stamps[-1]:
            raise Exception("Invalid position. Timestamp has already happened")
        self.x_values.append(x)
        self.y_values.append(y)
        self.z_values.append(z)
        self.direction_vectors.append(direction_vector)
        self.time_stamps.append(time_stamp)
    
    def get_position(self, time_stamp):
        '''
        Returns vessel's position at the given time stamp
        Input:
        - time_stamp (int)
        Output:
        - position (array)
        '''
        index = self.time_stamps.index(time_stamp)
        return [self.x_values[index], self.y_values[index], self.z_values[index]]
    
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
        Returns a dictionary of all the vessels positions with timestamp as key, and position as item.
        '''
        merged_points = list(zip(self.x_values, self.y_values, self.z_values))
        return dict(zip(self.time_stamps, merged_points))
    
    def get_direction_vector(self, time_stamp):
        '''
        Returns the vessel's direction at the given time stamp
        Input:
        - time_stamp (int)
        Output:
        - direction_vector (array)
        '''
        index = self.time_stamps.index(time_stamp)
        return self.direction_vectors[index]
    
    def get_direction_vectors(self):
        return self.direction_vectors()
    


    
