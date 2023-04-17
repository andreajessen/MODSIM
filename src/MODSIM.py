import numpy as np
import json
import os

from utils import *
from datatypes.virtualCamera import VirtualCamera
from dynamicSceneGenerator import DynamicSceneGenerator
from datatypes.boundingBox import BoundingBox
from errorGenerator import ErrorGenerator
from datatypes.detection import Detection
from datatypes.annotation import Annotation
#########################################################
#       Functionality for initializing DSG
##########################################################
def initialize_dynamic_scene_with_random_tracks(number_of_vessels, writeToJson=False, path=None):
    dsg = DynamicSceneGenerator()
    dsg.set_random_vessels(number_of_vessels)
    dsg.set_initial_vessel_tracks()
    if writeToJson and path:
        vessels = dsg.get_vessels()
        vessels_to_json(vessels, path)
    return dsg

#########################################################
#       Functionality for Track generation
##########################################################

def generate_positions_t(dsg, t, writeToJson=False, path=None):
    '''
    Generates tracks for one time step
    '''
    dsg.generate_random_tracks_t(t)
    if writeToJson and path:
        vessels = dsg.get_vessels()
        append_track_to_json(vessels, path, t)


def create_dynamic_scene_with_random_tracks(number_of_vessels, writeToJson=False, path=None):
    '''
    Generates tracks for default number of time steps (200)
    '''
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
    
    camera = VirtualCamera(focal_length, px, py, principal_point, image_bounds)
    camera.place_camera_in_world(position_WRF, roll, pitch, yaw)

    return camera

#########################################################
#       Functionality for projecting all points
#########################################################
def project_points_t(t, camera_rig, vessels, writeToJson=False, folder_path=None):
    '''
    Projects points for the given time step
    '''
    points = {vessel.id: vessel.calculate_3D_cornerpoints(t) for vessel in vessels}
    projected_points = {vesselID: camera_rig.take_photo(vessel_points, t) for vesselID, vessel_points in points.items()}
    if writeToJson and folder_path:
        update_projectedPoints_json(projected_points, folder_path, t)
    return projected_points


def project_all_points(camera_rig, vessels, writeToJson=False, folder_path=None):
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
        projected_points = {vesselID: camera_rig.take_photo(vessel_points, t) for vesselID, vessel_points in points.items()}
        all_projected_points[t] = projected_points
    if writeToJson and folder_path:
        projectedPoints_to_json(all_projected_points, folder_path)

    return all_projected_points


def project_all_points_from_json(camera_rig, folder_path, writeToJson=True):
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
    projected_points = project_all_points(camera_rig, vessel_list)
    if writeToJson:
        projectedPoints_to_json(projected_points, folder_path)
    return projected_points



#########################################################
#       Functionality for creating ground truth BBs
#########################################################
def create_bb(vesselID, pps, image_bounds):
    '''
    Create a bounding box around the vessel using its projected points.

    Inputs:
    - vessel_id: Id of vessel to create BB around
    - projected_points: list of projected points
    - image_bounds: tuple of (xbound, ybound)

    Returns:
    - BoundingBox object or None
    '''
    try:
        x_vals = []
        y_vals = []
        depth_vals = []
        for point in pps:
            x_vals.append(point.get_x())
            y_vals.append(point.get_y())
            depth_vals.append(point.get_depth())
        max_x = np.max(x_vals)
        min_x = np.min(x_vals)
        max_y = np.max(y_vals)
        min_y = np.min(y_vals)
        if check_inside_imagebounds(max_x, min_x, max_y, min_y, image_bounds):
            # BB should be created
            depth = np.average(depth_vals) #OBS which depth should we use
            if depth < 0:
                # BB is behind camera. The BB should not be created. 
                return None
            # Ensure bounding box does not exceed image bounds
            max_x = min(max_x, image_bounds[0])
            min_x = max(min_x, 0)
            max_y = min(max_y, image_bounds[1])
            min_y = max(min_y, 0)
            width, height = max_x - min_x, max_y - min_y
            centre = [min_x + width / 2, min_y + height / 2]
            bounding_box = BoundingBox(vesselID, centre, width, height, depth)
            return bounding_box
        # Not inside image bounds, and BB should not be created
        return None
    except NameError as e:
        # Handle exceptions
        print(f"VesselID do not have any projected points {vesselID}: {str(e)}")
        return None
    
def create_annotations_t(projected_points, vessels, image_bounds, time_stamp, annotation_mode=0, writeToJson=False, folder_path=None):
    '''
    Create annotations for vessels based on projected points and image bounds.
    
    Inputs:
    - projected_points: list of projected points
    - vessels: list of vessels
    - image_bounds: tuple of (xmin, ymin, xmax, ymax)
    - time_stamp: timestamp for annotations
    - annotation_mode: mode for creating annotations (default 0)
    - write_to_json: boolean indicating whether to write annotations to json file (default False)
    - folder_path: path to folder to write json file (default None)
    
    Returns:
    - List of Annotation objects
    '''
    vessel_dict = {vessel.id: vessel for vessel in vessels}
    bound_boxes = create_bound_boxes_t(projected_points, image_bounds, annotation_mode=annotation_mode)
    annotations = [Annotation(bb, vessel_dict[bb.vesselID].label, bb.vesselID) for bb in bound_boxes]
    if writeToJson and folder_path:
        update_annots_json(annotations, folder_path, time_stamp)
    return annotations


def create_bound_boxes_t(projected_points, image_bounds, annotation_mode=0):
    '''
    Creates bounding boxes for the given time step
    Input:
    - Projected points: dict with vesselID as key and array of projected points for that vessel as item.
    - Image bounds: Array of len 2 with x and y image bounds
    - Time stamp
    Return
    - bbs: List of bounding boxes
    '''
    bbs = []
    for vesselID, pps in projected_points.items():
        bb = create_bb(vesselID, pps, image_bounds)
        if bb:
            bbs.append(bb)
    if len(bbs) > 1: 
        bbs = handle_covered_bbs(bbs, annotation_mode)
    return bbs

def check_inside_imagebounds(max_x, min_x, max_y, min_y, image_bounds):
    if ((max_x <= image_bounds[0] and max_x >= 0) or (min_x <= image_bounds[0] and min_x >= 0)) and ((max_y <= image_bounds[1] and max_y >= 0) or (min_y <= image_bounds[1] and min_y >= 0)):
        return True
    return False

def handle_covered_bbs(bounding_boxes, annotation_mode):
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
            bb.update_visibility(covering_bbs)
            if annotation_mode == 2:
                bb.update_bb_if_covered(covering_bbs)
        if not fully_covered:
            bbs.append(bb)
    bbs.append(sorted_bbs[-1])
    return bbs


#########################################################
#       Perform full cycle for one time step
#########################################################
def perform_one_time_step(dsg, errorGenerator, camera_rig, t, annotation_mode=0, writeToJson=False, path=None):
    """
    Performs a full cycle for one time step.

    Parameters:
    dsg (DynamicScene): The dynamic scene
    error_generator (Error generator): Error generator object
    camera_rig (Camera rig): Camera rig object
    t (int): Time step
    write_to_json (bool): Whether to write results to JSON files (default is False)
    folder_path (str): Folder path to write JSON files (default is None)

    Returns:
    Dictionaries containing positions, bounding boxes, and error bounding boxes for each time step.
    """
    generate_positions_t(dsg, t, writeToJson=writeToJson, path=path)
    pps = project_points_t(t, camera_rig, dsg.get_vessels(), writeToJson=writeToJson, folder_path=path)
    annots = create_annotations_t(pps, dsg.get_vessels(), camera_rig.camera.image_bounds, t, annotation_mode=annotation_mode, writeToJson=writeToJson, folder_path=path)
    detections =  errorGenerator.generate_detections_t(annots, t, writeToJson=writeToJson, folder_path=path)
    return pps, annots, detections

def perform_time_steps(t_start, t_end, dsg, errorGenerator, camera_rig, annotation_mode=0, writeToJson=False, path=None):
    pps, bbs, eBBs = {}, {}, {}
    for t in range(t_start, t_end):
        pps[t], bbs[t], eBBs[t] = perform_one_time_step(dsg, errorGenerator, camera_rig, t, annotation_mode=annotation_mode, writeToJson=writeToJson, path=path)
    return pps, bbs, eBBs

#############################################################
#       Perform full cycle for one time step from pose data
############################################################
def perform_one_time_step_poseData(dsg, errorGenerator, camera_rig, t, annotation_mode=0, writeToJson=False, path=None):
    pps = project_points_t(t, camera_rig, dsg.get_vessels(), writeToJson=writeToJson, folder_path=path)
    annots = create_annotations_t(pps, dsg.get_vessels(), camera_rig.camera.image_bounds, t, annotation_mode=annotation_mode, writeToJson=writeToJson, folder_path=path)
    detections =  errorGenerator.generate_detections_t(annots, t, writeToJson=writeToJson, folder_path=path)
    return pps, annots, detections