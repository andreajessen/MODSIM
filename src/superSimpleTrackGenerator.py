import numpy as np
from datatypes.track import Track

class SuperSimpleTrackGenerator():

    def generate_track(self, start_time, end_time, frequency, radius, theta_start, p_0, w, vessel_beam, vessel_length):
        """ Generate a super simple track
        start_time (int, seconds): start time of track
        end_time (int, seconds): end time of track
        frequency (int, seconds): time difference between the discrete time steps
        radius (int): radius of the circle
        p_0 (np.array): the position vector of the centre of the circle
        w (int): rotation rate rad/sec
        
        return Track
        """
        time_steps =int((end_time - start_time)/frequency)
        track = Track()
        
        for n in range(time_steps):
            theta = w*n*frequency+theta_start
            x = p_0[0]+radius*np.cos(theta)
            y = p_0[1]+radius*np.sin(theta)
            z = 0
            direction_vector = [-np.sin(theta), np.cos(theta)]

            rotation_matrix = np.array([[-np.sin(theta), -np.cos(theta)], [np.cos(theta), -np.sin(theta)]])
            position = np.array([x, y])
            cornerpoints_VRF = np.array([[vessel_length/2, -vessel_beam/2], [-vessel_length/2, -vessel_beam/2], [-vessel_length/2, vessel_beam/2], [vessel_length/2, vessel_beam/2]])
            cornerpoints_WRF = [np.dot(rotation_matrix,uv)+position for uv in cornerpoints_VRF]

            time_stamp = start_time + frequency*n
            track.addPosition(x, y, z, direction_vector, cornerpoints_WRF, time_stamp)

        return track