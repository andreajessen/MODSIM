import numpy as np

class Track:
    def __init__(self):
       self.x_values = []
       self.y_values = []
       self.z_values = []
       self.time_stamps = []
    

    def addPosition(self, x, y, z, time_stamp):
        if self.time_stamps and time_stamp <= self.time_stamps[-1]:
            raise Exception("Invalid position. Timestamp has already happened")
        self.x_values.append(x)
        self.y_values.append(y)
        self.z_values.append(z)
        self.time_stamps.append(time_stamp)
    
    def get_position(self, time_stamp):
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
        merged_points = list(zip(self.x_values, self.y_values, self.z_values))
        return dict(zip(self.time_stamps, merged_points))

    
