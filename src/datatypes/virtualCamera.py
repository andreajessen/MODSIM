import numpy as np

class VirtualCamera:
    
    def __init__(self, position_WRF, roll, yaw, pitch, focal_length, p_x, p_y, principal_point, image_bounds, skew=0):
        '''
        Input:
        principal_point: Principal point of image plane (c_x,c_y) in pixel coordinates
        '''
        self.image_bounds = image_bounds # In pixels (x,y)
        # Position in WRF coordinates
        self.position_WRF = position_WRF
        self.roll = roll # Roll is a counterclockwise rotation of φ about the x-axis
        self.yaw = yaw # The yaw is a counterclockwise rotation of θ about the z-axis. 
        self.pitch = pitch # The pitch is a counterclockwise rotation of ψ about the y-axis.

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
        self.calculate_projection_matrix()

    
    def get_intrinsic_camera_matrix(self):
        return self.K
    
    def get_position(self):
        return self.position_WRF
    
    def calculate_rotation_matrix(self):
        R_roll = np.round(np.array([[np.cos(self.roll), -np.sin(self.roll), 0], [np.sin(self.roll), np.cos(self.roll), 0], [0,0,1]]),5)
        R_yaw = np.round(np.array([[np.cos(self.yaw), 0, np.sin(self.yaw)],[0,1,0],[-np.sin(self.yaw), 0, np.cos(self.yaw)]]),5)
        R_pitch = np.round(np.array([[1, 0, 0], [0, np.cos(self.pitch), -np.sin(self.pitch)], [0, np.sin(self.pitch), np.cos(self.pitch)]]),5)
        # np.round to 5 decimals because the number π cannot be represented exactly as a floating-point number.
        R_orientation = R_roll.dot(R_pitch.dot(R_yaw))

        R_align = np.array([[0,-1,0],[0, 0, -1], [1,0,0]])
        rotation_matrix = R_orientation.dot(R_align)
        return rotation_matrix

    def update_camera_angle(self, roll, yaw, pitch):
        self.roll = roll # Roll is a counterclockwise rotation of φ about the x-axis
        self.yaw = yaw # The yaw is a counterclockwise rotation of θ about the z-axis. 
        self.pitch = pitch # The pitch is a counterclockwise rotation of ψ about the y-axis.
        self.calculate_projection_matrix()
    
    def calculate_projection_matrix(self):
        self.R = self.calculate_rotation_matrix()
        t = self.R.dot(-self.position_WRF)
        self.M = np.hstack((self.R, t[:, None]))
        self.P = self.K.dot(self.M)

    def project_point(self, p_W):
        p_W_homogeneous = np.append(p_W, 1)
        p_I = self.P.dot(p_W_homogeneous)
        return p_I/p_I[-1]

    def project_points(self, points):
        projected_points = [self.project_point(point) for point in points]
        return np.array(projected_points)
    
    def get_orientation_vector(self):
        y = round(np.cos(np.pi/2-self.yaw),5)
        x = round(np.cos(self.yaw),5)
        return np.array([x,y])
    