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
from datatypes.waveMotion import WaveMotion

class CameraRig:
    def __init__(self, camera, vessel=None, include_wave=True):
        self.camera = camera
        self.vessel = vessel
        self.include_wave = include_wave
        if self.include_wave:
            self.wave_motion = WaveMotion()    
    
    def take_photo(self, vessel_points, timestamp):
        if self.vessel:
            track = self.vessel.get_track()
            self.camera.update_camera_world_pos(track.get_position(timestamp), track.get_heading_rad(timestamp), self.vessel.get_rotation_matrix(timestamp))
        
        if self.include_wave:
            wave_roll, wave_pitch, wave_yaw = self.wave_motion.generate_wave()
            self.camera.add_wave_motion(wave_roll, wave_pitch, wave_yaw)
        return self.camera.project_points(vessel_points)
    
    def get_camera_position(self, timestamp):
        if self.vessel:
            track = self.vessel.get_track()
            pos = track.get_position(timestamp) + self.vessel.get_rotation_matrix(timestamp).dot(self.camera.get_position_vcf())
        else:
            pos = self.camera.get_position_wcf()
        return pos
    
    def get_camera_orientation(self, timestamp):
        # Do we want this function to also include the wave induced yaw?  
        if self.vessel:
            track = self.vessel.get_track()
            orientation = track.get_direction_vector(timestamp)
        else:
            orientation = self.camera.get_orientation_vector()
        return orientation
