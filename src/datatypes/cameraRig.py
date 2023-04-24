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
    def __init__(self, cameras, vessel=None, include_wave=True):
        '''
        Input:
        - Cameras: Dictionary of all cameras and their ID
        '''
        self.cameras = cameras
        self.vessel = vessel
        self.include_wave = include_wave
        if self.include_wave:
            self.wave_motion = WaveMotion()  
        
        self.horizon = {}
        for cameraID in cameras.keys():
            self.horizon[cameraID] = {}

    
    def take_photos(self, all_vessel_points, timestamp):
        if self.include_wave:
            wave_roll, wave_pitch, wave_yaw = self.wave_motion.generate_wave()
        else:
            wave_roll, wave_pitch, wave_yaw = None, None, None
        projected_points_cam = {}
        for cameraID, camera in self.cameras.items():
            projected_points_cam[cameraID] = self.take_photo(camera, cameraID, all_vessel_points, timestamp, wave_roll, wave_pitch, wave_yaw)
        return projected_points_cam


    def take_photo(self, camera, cameraID, all_vessel_points, timestamp, wave_roll=None, wave_pitch=None, wave_yaw=None):
        '''
        Input:
        vessel_points: {vessel.id: vessel.calculate_3D_cornerpoints(t) for vessel in vessels}
        '''
        if self.vessel:
            track = self.vessel.get_track()
            camera.update_camera_world_pos(track.get_position(timestamp), track.get_heading_rad(timestamp), self.vessel.get_rotation_matrix(timestamp))
        if self.include_wave:
                camera.add_wave_motion(wave_roll, wave_pitch, wave_yaw)
        self.horizon[cameraID][timestamp] = camera.get_horizon()
        projected_points = {vesselID: camera.project_points(vessel_points) for vesselID, vessel_points in all_vessel_points.items()}
        return projected_points
    
    def get_camera_position(self, cameraID, timestamp):
        camera = self.cameras[cameraID]
        if self.vessel:
            track = self.vessel.get_track()
            pos = track.get_position(timestamp) + self.vessel.get_rotation_matrix(timestamp).dot(camera.get_position_vcf())
        else:
            pos = camera.get_position_wcf()
        return pos
    
    def get_camera_orientation(self, cameraID, timestamp):
        # Do we want this function to also include the wave induced yaw?  
        if self.vessel:
            track = self.vessel.get_track()
            orientation = track.get_direction_vector(timestamp)
        else:
            orientation = self.cameras[cameraID].get_orientation_vector()
        return orientation
