##################################################################
#
# CameraRig
# - Holds the camera setup. Only one camera (NB should be expandend 
# to support a multiple camera setup).
# - If the camera is not placed on a vessel -> Vessel = None.
# - If the camera is placed on a vessel, initialize the CameraRig 
# with the vessel the camera is placed on.
#
##################################################################
import numpy as np

class CameraRig:
    def __init__(self, camera, vessel=None, wave=True):
        self.camera = camera
        self.vessel = vessel
        self.wave = wave
        if self.wave:
            self.roll_1 = 0
            self.pitch_1 = 0
            self.yaw_1 = 0

            self.roll_2 = 0
            self.pitch_2 = 0
            self.yaw_2 = 0

            self.roll_a = 0.99
            self.roll_b = 0.99
            self.pitch_a = 0.99
            self.pitch_b = 0.99
            self.yaw_a = 0.99
            self.yaw_b = 0.99

            self.roll_sigma = 0.0001
            self.pitch_sigma = 0.001
            self.yaw_sigma = 0.0001
    
    def take_photo(self, vessel_points, timestamp):
        if self.vessel:
            track = self.vessel.get_track()
            self.camera.update_camera_world_pos(track.get_position(timestamp), track.get_heading_rad(timestamp), self.vessel.get_rotation_matrix(timestamp))
        
        if self.wave:
            self.roll_1 = self.roll_a*self.roll_1 + np.random.normal(scale=self.roll_sigma)
            self.roll_2 = self.roll_b*self.roll_2 + self.roll_1

            self.pitch_1 = self.pitch_a*self.pitch_1 + np.random.normal(scale=self.pitch_sigma)
            self.pitch_2 = self.pitch_b*self.pitch_2 + self.pitch_1

            self.yaw_1 = self.yaw_a*self.yaw_1 + np.random.normal(scale=self.yaw_sigma)
            self.yaw_2 = self.yaw_b*self.yaw_2 + self.yaw_1

            self.camera.add_wave_motion(self.roll_2, self.pitch_2, self.yaw_2)
        return self.camera.project_points(vessel_points)
    
    def get_camera_position(self, timestamp):
        # Might need to change this to include wave motion
        if self.vessel:
            track = self.vessel.get_track()
            pos = track.get_position(timestamp) + self.vessel.get_rotation_matrix(timestamp).dot(self.camera.get_position_vcf())
        else:
            pos = self.camera.get_position_wcf()
        return pos
    
    def get_camera_orientation(self, timestamp):
        # Might need to change this to include wave motion
        if self.vessel:
            track = self.vessel.get_track()
            orientation = track.get_direction_vector(timestamp)
        else:
            orientation = self.camera.get_orientation_vector()
        return orientation
