import numpy as np

class VirtualCamera:
    
    def __init__(self, position_WRF, orientation, focal_length, p_x, p_y, principal_point, skew=0):
        # Position in WRF coordinates
        self.position_WRF = position_WRF
        self.orientation = orientation # Same as cameras z axis

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


        self.K = np.array([[self.f_x, self.skew, self.cx], [0, self.f_y, self.c_y], [0, 0, 1]])

    
    def get_intrinsic_camera_matrix(self):
        return self.K
    
    def get_position(self):
        return self.position_WRF