from dynamicSceneGenerator import DynamicSceneGenerator
from visualize import visualize
from datatypes.virtualCamera import VirtualCamera
import numpy as np
# Generate dynamic scene with random tracks
dsg = DynamicSceneGenerator()
dsg.set_random_vessels(6)
dsg.generate_random_tracks()
vessels = dsg.get_vessels()
# visualize(vessels)

focal_length = 300*10**-6
px = py = 150*10**-6
position_WRF = np.array([500,200,0])
roll = 0
yaw = np.pi
pitch = 0
principal_point = (150,150)
image_bound = (300,300)
time_stamp = 166

camera = VirtualCamera(position_WRF, roll, yaw, pitch, focal_length, px, py, principal_point)

points = [vessel.calculate_3D_cornerpoints(time_stamp) for vessel in vessels]
projected_points = [camera.project_points(vessel_points) for vessel_points in points]

from visualize import plot_image
plot_image(projected_points, image_bound, time_stamp)


