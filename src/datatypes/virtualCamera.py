import numpy as np
from datatypes.projectedPoint import ProjectedPoint

class VirtualCamera:
    
    def __init__(self, focal_length, p_x, p_y, principal_point, image_bounds, skew=0):
        '''
        Input:
        principal_point: Principal point of image plane (c_x,c_y) in pixel coordinates
        '''
        self.image_bounds = image_bounds # In pixels (x,y)
        # Position and orientation in VCF coordinates
        # Initialized as 0, changed if the camera is placed on a vessel (self.place_camera_on_vessel)
        self.position_nvcf = np.array([0,0,0])
        self.roll_nvcf = 0
        self.pitch_nvcf = 0
        self.yaw_nvcf = 0

        self.roll_wvcf = 0
        self.pitch_wvcf = 0
        self.yaw_wvcf = 0
        # Position in WCF coordinates
        # If the camera is placed on a vessel, these parameters are updated for each timestep. It is the CameraRig that knows the track of the camera. 
        self.position_wcf = np.array([0,0,0])
        self.roll_wcf = 0
        self.pitch_wcf = 0
        self.yaw_wcf = 0

        # Intrinsic parameters
        self.focal_length = focal_length
        self.p_x = p_x
        self.p_y = p_y
        self.principal_point = principal_point
        self.skew = skew

        # Intrinsic camera matrix
        self.f_x = 1/self.p_x*self.focal_length
        self.f_y = 1/self.p_y*self.focal_length
        self.c_x = principal_point[0]
        self.c_y = principal_point[1]


        self.K = np.array([[self.f_x, self.skew, self.c_x], [0, self.f_y, self.c_y], [0, 0, 1]])
        # self.calculate_projection_matrix()

    def place_camera_on_vessel(self, position_nvcf, roll_nvcf, pitch_nvcf, yaw_nvcf):
        self.position_nvcf = position_nvcf
        self.roll_nvcf = roll_nvcf
        self.pitch_nvcf = pitch_nvcf
        self.yaw_nvcf = yaw_nvcf
    
    def place_camera_in_world(self, position_wcf, roll_wcf, pitch_wcf, yaw_wcf):
        self.position_wcf = position_wcf
        self.roll_wcf = roll_wcf
        self.pitch_wcf = pitch_wcf
        self.yaw_wcf = yaw_wcf
        self.calculate_projection_matrix()

    def update_camera_world_pos(self, position_wcf, yaw_wcf, vcf_wcf_rotation_matrix):
        self.position_wcf = vcf_wcf_rotation_matrix.dot(self.position_nvcf) + position_wcf
        self.yaw_wcf = yaw_wcf
        self.calculate_projection_matrix()

    def add_wave_motion(self, roll_wvcf, pitch_wvcf, yaw_wvcf):
        self.roll_wvcf = roll_wvcf
        self.pitch_wvcf = pitch_wvcf
        self.yaw_wvcf = yaw_wvcf
        self.calculate_projection_matrix()

    def get_intrinsic_camera_matrix(self):
        return self.K
    
    def get_position_vcf(self):
        return self.position_nvcf
    
    def get_position_wcf(self):
        return self.position_wcf
    
    def calculate_rotation_matrix(self):
        R_roll = np.round(np.array([[np.cos(self.roll_wcf + self.roll_nvcf + self.roll_wvcf), -np.sin(self.roll_wcf + self.roll_nvcf + self.roll_wvcf), 0], [np.sin(self.roll_wcf + self.roll_nvcf + self.roll_wvcf), np.cos(self.roll_wcf + self.roll_nvcf + self.roll_wvcf), 0], [0,0,1]]),5)
        R_yaw = np.round(np.array([[np.cos(self.yaw_wcf + self.yaw_nvcf + self.yaw_wvcf), 0, np.sin(self.yaw_wcf + self.yaw_nvcf + self.yaw_wvcf)],[0,1,0],[-np.sin(self.yaw_wcf + self.yaw_nvcf + self.yaw_wvcf), 0, np.cos(self.yaw_wcf + self.yaw_nvcf + self.yaw_wvcf)]]),5)
        R_pitch = np.round(np.array([[1, 0, 0], [0, np.cos(self.pitch_wcf + self.pitch_nvcf + self.pitch_wvcf), -np.sin(self.pitch_wcf + self.pitch_nvcf + self.pitch_wvcf)], [0, np.sin(self.pitch_wcf + self.pitch_nvcf + self.pitch_wvcf), np.cos(self.pitch_wcf + self.pitch_nvcf + self.pitch_wvcf)]]),5)
        # np.round to 5 decimals because the number π cannot be represented exactly as a floating-point number.
        R_orientation = R_roll.dot(R_pitch.dot(R_yaw))

        R_align = np.array([[0,-1,0],[0, 0, -1], [1,0,0]])
        rotation_matrix = R_orientation.dot(R_align)
        return rotation_matrix
    
    def calculate_projection_matrix(self):
        self.R = self.calculate_rotation_matrix()
        t = self.R.dot(-self.position_wcf)
        self.M = np.hstack((self.R, t[:, None]))
        self.P = self.K.dot(self.M)

    def get_point_in_ccf(self, p_W_homogeneous):
        return self.M.dot(p_W_homogeneous)

    def project_point(self, p_W):
        p_W_homogeneous = np.append(p_W, 1)
        p_C = self.get_point_in_ccf(p_W_homogeneous)
        depth = p_C[2] # If depth is negative, the point should not be in the image
        p_I = self.P.dot(p_W_homogeneous)
        p_I_scaled = p_I/p_I[-1]
        image_coordinate = np.array([p_I_scaled[0], p_I_scaled[1]])
        projected_point = ProjectedPoint(image_coordinate, depth)
        return projected_point

    def project_points(self, points):
        projected_points = [self.project_point(point) for point in points]
        return np.array(projected_points)
    
    def get_orientation_vector(self):
        y = round(np.cos(np.pi/2-(self.yaw_wcf + self.yaw_nvcf + self.yaw_wvcf)),5)
        x = round(np.cos(self.yaw_wcf + self.yaw_nvcf + self.yaw_wvcf),5)
        return np.array([x,y])
    
    def get_horizon(self):
        cam_pos = self.get_position_wcf()
        point_at_horizon1 = np.array([cam_pos[0], cam_pos[1]]) + self.get_orientation_vector() * 10000 + np.array([1, 1])
        point_at_horizon2 = np.array([cam_pos[0], cam_pos[1]]) + self.get_orientation_vector() * 10000 - np.array([1, 1])
        point1 = np.append(point_at_horizon1, 0)
        point2 = np.append(point_at_horizon2, 0)
        horizon_points = self.project_points([point1, point2])
        return horizon_points