import numpy as np
import json
import os

from utils import dict_to_json, json_to_tracks, json_to_vessels

from datatypes.virtualCamera import VirtualCamera
from dynamicSceneGenerator import DynamicSceneGenerator
from datatypes.boundingBox import BoundingBox

#########################################################
#       Functionality for creating a dynamic scene with 
#       random tracks
##########################################################


def create_dynamic_scene_with_random_tracks(number_of_vessels, writeToJson=False, path=None):
    # Generate dynamic scene with random tracks
    dsg = DynamicSceneGenerator()
    dsg.set_random_vessels(number_of_vessels)
    dsg.generate_random_tracks()
    if writeToJson and path:
        vessels = dsg.get_vessels()
        tracks_to_json(vessels, path)
        vessels_to_json(vessels, path)
    return dsg


#########################################################
#       Functionality for placing a camera in the scene
#########################################################

def calculate_camera_position_WRF(focal_length, width_of_sensor, largest_radius, path_centre, camera_height):
    angle_of_view = 2*np.arctan(width_of_sensor/(2*focal_length)) # In radians
    d = largest_radius/np.tan(angle_of_view/2)
    # Assume camera parallel to x axis and always pointing towards the centre of the path circle
    x = d+path_centre[0]
    y = path_centre[1]
    z = camera_height
    return np.array([x,y,z])

def calculate_camera_pitch(camera_position):
    alpha = np.arctan(camera_position[2]/camera_position[0]) # arctan(height/distance)
    return alpha

def create_and_place_simple_legacy_camera(largest_radius, path_centre, height=60): # This should maybe get a better name, and should be moved somewhere?
    '''
    Function for placing a Simple legacy photo camera in the dynamic scene
    '''
    # NB! Make sure everything is in meters
    focal_length = 50*10**-3
    image_bounds = (3600, 2400) # Pixels (x,y)
    film_size = (36*10**-3, 24*10**-3)
    px = film_size[0]/image_bounds[0]
    py = film_size[1]/image_bounds[1]
    principal_point = (image_bounds[0]/2,image_bounds[1]/2)
    width_of_sensor = 36*10**-3 # Width of sensor
    camera_height = height # metre

    position_WRF = calculate_camera_position_WRF(focal_length, width_of_sensor, largest_radius, path_centre, camera_height)
    roll = 0
    yaw = np.pi
    pitch = calculate_camera_pitch(position_WRF)
    
    camera = VirtualCamera(position_WRF, roll, yaw, pitch, focal_length, px, py, principal_point, image_bounds)

    return camera

#########################################################
#       Functionality for projecting all points
#########################################################

def project_all_points(camera, vessels, writeToJson=False, folder_path=None):
    '''
    Projects corner points of the vessels given the camera
    Input:
        Camera (Camera)
        vessels (list of Vessel)
    Ouputs:
        Dictionary with timestamp as key and the
        items are a list of projected points for each vessel
    '''
    all_projected_points = {}
    for t in vessels[0].get_track().get_time_stamps():
        points = {vessel.id: vessel.calculate_3D_cornerpoints(t) for vessel in vessels}
        projected_points = {vesselID: camera.project_points(vessel_points) for vesselID, vessel_points in points.items()}
        all_projected_points[t] = projected_points
    if writeToJson and folder_path:
        projectedPoints_to_json(all_projected_points, folder_path)

    return all_projected_points

def project_all_points_from_json(camera, folder_path, writeToJson=True):
    '''
    Inputs json path with tracks and vessels, and projects all corner points
    '''
    vessel_path = os.path.join(folder_path, 'vehicle_characteristics.json')
    track_path = os.path.join(folder_path, 'tracks.json')
    vessels = json_to_vessels(vessel_path)
    tracks = json_to_tracks(track_path)
    for vesselID, track in tracks.items(): 
        vessels[vesselID].set_track(track)
    vessel_list = list(vessels.values())
    projected_points = project_all_points(camera, vessel_list)
    if writeToJson:
        projectedPoints_to_json(projected_points, folder_path)
    return projected_points

#########################################################
#       Functionality for creating ground truth BBs
#########################################################

def create_bound_boxes_json(projected_points):
    bbs = []
    for vesselID, vessel_dict in projected_points.items():
        x_vals = list(map(lambda v : v['x'], vessel_dict.values()))
        y_vals = list(map(lambda v : v['y'], vessel_dict.values()))
        depth_vals = list(map(lambda v : v['depth'], vessel_dict.values()))
        max_x = np.max(x_vals)
        min_x = np.min(x_vals)
        max_y = np.max(y_vals)
        min_y = np.min(y_vals)
        width = max_x-min_x
        height = max_y - min_y
        centre = [min_x+width/2, min_y+height/2]
        depth = np.average(depth_vals) #OBS which depth should we use
        bounding_box = BoundingBox(vesselID, centre, width, height, depth)
        # Make sure these are in the correct order as in master thesis
        bbs.append(bounding_box)
    return bbs

# NB! Might need change depending on the structure of projected points.
def create_all_bbs_from_json(folder_path, occlusion=True, writeToJson=False):
    filepath = os.path.join(folder_path, 'projectedPoints.json')
    with open(filepath, 'r') as f:
        all_projected_points = json.load(f)
    all_bbs = {}
    for t in all_projected_points.keys():
        if occlusion:
            all_bb=create_bound_boxes_json(all_projected_points[t])
            all_bbs[t] = handle_covered_bbs(all_bb)
        else:
            all_bbs[t]=create_bound_boxes_json(all_projected_points[t])
    if writeToJson and folder_path:
        bbs_to_json(all_bbs, folder_path)
    return all_bbs

def create_bound_boxes(projected_points):
    bbs = []
    for vesselID, vessel in projected_points.items():
        if vessel.size > 0:
            x_vals = np.array([point.image_coordinate[0] for point in vessel])
            y_vals = np.array([point.image_coordinate[1] for point in vessel])
            depth_vals = np.array([point.depth for point in vessel])
            max_x = np.max(x_vals)
            min_x = np.min(x_vals)
            max_y = np.max(y_vals)
            min_y = np.min(y_vals)
            width = max_x-min_x
            height = max_y - min_y
            centre = [min_x+width/2, min_y+height/2]
            depth = np.average(depth_vals) #OBS which depth should we use
            bounding_box = BoundingBox(vesselID, centre, width, height, depth)
            # Make sure these are in the correct order as in master thesis
            bbs.append(bounding_box)
    return bbs

def handle_covered_bbs(bounding_boxes):
    bbs = []
    sorted_bbs = sorted(bounding_boxes, key=lambda bb: bb.depth, reverse=True)
    for i in range(len(bounding_boxes)-1):
        bb = sorted_bbs[i]
        fully_covered = False
        covering_bbs = []
        for j in range(i+1, len(bounding_boxes)):
            if bb.check_fully_covered(sorted_bbs[j]):
                fully_covered = True
            elif bb.check_overlap(sorted_bbs[j]):
                covering_bbs.append(sorted_bbs[j])
        if len(covering_bbs) > 0:
            bb.update_bb_if_covered(covering_bbs)
        if not fully_covered:
            bbs.append(bb)
    bbs.append(sorted_bbs[-1])
    return bbs

def create_all_bbs(all_projected_points, occlusion=True, writeToJson=False, folder_path=None):
    all_bbs = {}
    for t in all_projected_points.keys():
        if occlusion:
            all_bb=create_bound_boxes(all_projected_points[t])
            all_bbs[t] = handle_covered_bbs(all_bb)
        else:
            all_bbs[t]=create_bound_boxes(all_projected_points[t])
    if writeToJson and folder_path:
        bbs_to_json(all_bbs, folder_path)
    return all_bbs




#########################################################
#       Save objects to json files
#########################################################

def tracks_to_json(vessels, path):
    # NB! Assumes all vessels have the same timestamp
    all_tracks = {key: {vessel.id: vessel.get_track_dict()[key] for vessel in vessels} for key in vessels[0].get_track_dict().keys()}
    filename = os.path.join(path, 'tracks.json')
    dict_to_json(filename, all_tracks)

def vessels_to_json(vessels, path):
    filename = os.path.join(path, 'vehicle_characteristics.json')
    vessel_dict = {vessel.id: {'air_draft_m': vessel.air_draft, 'beam_m': vessel.beam, 'length_m': vessel.length, 'label': vessel.label} for vessel in vessels}
    dict_to_json(filename, vessel_dict)

def projectedPoints_to_json(projected_points, path):
    # All projected points will probably be on a different format. Did this to get a file.
    projected_points_dict = {time_stamp: {vesselID: {cornerNumber: {'x': projected_points[time_stamp][vesselID][cornerNumber].image_coordinate[0], 'y': projected_points[time_stamp][vesselID][cornerNumber].image_coordinate[1], 'depth': projected_points[time_stamp][vesselID][cornerNumber].depth} for cornerNumber in range(len(projected_points[time_stamp][vesselID]))} for vesselID in projected_points[time_stamp]} for time_stamp in projected_points.keys()}
    filename = os.path.join(path, 'projectedPoints.json')
    dict_to_json(filename, projected_points_dict)

def bbs_to_json(bbs, folder_path):
    # OBS: because we create bounding boxes based on the depth of the vessels in the CCF, the bbs are created in a order with decreasing depth
    bb_dict = {time_stamp: {bb.vesselID: {'centre': {'x':  bb.centre[0], 'y': bb.centre[1]}, 'height': bb.height, 'width': bb.width, 'depth':  bb.depth} for bb in bbs[time_stamp]} for time_stamp in bbs.keys()}
    save_path = os.path.join(folder_path, 'boundingBoxes.json')
    dict_to_json(save_path, bb_dict) 
