import numpy as np
from datatypes.track import Track

class SuperSimpleTrackGenerator():

    def generate_track(self, start_time, end_time, frequency, radius, theta_start, p_0, w, vessel):
        """ Generate a super simple track
        Input: 
        - start_time (int, seconds): start time of track
        - end_time (int, seconds): end time of track
        - frequency (int, seconds): time difference between the discrete time steps
        - radius (int): radius of the circle
        - p_0 (np.array): the position vector of the centre of the circle
        - w (int): rotation rate rad/sec
        
        return Track
        """
        time_steps =int((end_time - start_time)/frequency)
        track = Track()
        
        for n in range(time_steps):
            theta = w*n*frequency+theta_start
            x = p_0[0]+radius*np.cos(theta)
            y = p_0[1]+radius*np.sin(theta)
            z = 0
            # direction_vector = [-np.sin(theta), np.cos(theta)]
            heading_rad = theta+np.pi/2
            time_stamp = start_time + frequency*n
            track.addPosition(x, y, z, heading_rad, time_stamp)

        return track