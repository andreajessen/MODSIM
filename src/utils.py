import os
import json

from datatypes.track import Track
from datatypes.vessel import Vessel
from datatypes.boundingBox import BoundingBox
from datatypes.projectedPoint import ProjectedPoint

def find_path_to_next_simulation():
    arr = os.listdir('./simulations/')
    arr.sort()
    timestamp = 0 if not arr else int(arr[-1])+1
    filename = f'./simulations/{timestamp}/'
    return filename

def dict_to_json(path, dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w+') as f:
        json.dump(dict, f, indent = 4)

def update_json(path, dict):
    '''
    Append to current json file without loading it. 
    '''
    if os.path.exists(path) == False:
        dict_to_json(path, dict)
    else:
        with open (path, mode="r+") as file:
            file.seek(os.stat(path).st_size -1)
            file.write( ",{}".format(json.dumps(dict, indent = 4)[1:-1]))
            file.write('}')


def json_to_tracks(path):
    '''
    Loads the track json file and creates a list 
    of corresponding track classes
    Input:
        Path (string): path to track file
    Output:
        tracks (list of Track)
    '''
    with open(path, 'r') as f:
        track_dict = json.load(f)
    init = True
    for t, vessels in track_dict.items():
        if init:
            tracks = {vesselId: Track() for vesselId in vessels.keys()}
            init = False
        for vessel, position in vessels.items():
            tracks[vessel].addPosition(position['center_position_m'][0], 
                                       position['center_position_m'][1], 
                                       position['center_position_m'][2], 
                                       position['heading_rad'], float(t))
    return tracks

def json_to_vessels(path):
    '''
    Loads the vessel json file and creates a list 
    of corresponding vessel classes
    Input:
        Path (string): path to track file
    Output:
        vessels (list of Vessel)
    '''
    with open(path, 'r') as f:
        vessels_dict = json.load(f)
    vessels = {key: Vessel(key, item['air_draft_m'], item['beam_m'], item['length_m'], item['label']) for key, item in vessels_dict.items()}
    return vessels

def json_to_projectedPoints(path):
    with open(path, 'r') as f:
        pp_dict = json.load(f)
    pps = {time_stamp: {vessel_id: [ProjectedPoint((point['x'], point['y']), point['depth']) for point in points.values()] for vessel_id, points in time_info.items()} for time_stamp, time_info in pp_dict.items()}
    return pps


def json_to_bb(path):
    '''
    Loads the BB json file and creates a list 
    of corresponding BB classes
    Input:
        Path (string): path to track file
    Output:
        BBs (list of BoundingBoxes)
    '''
    with open(path, 'r') as f:
        bb_dict = json.load(f)
    # (self, vesselID, centre, width, height, depth)
    bbs = {key: [BoundingBox(vesselID, (bb['centre']['x'], bb['centre']['y']), bb['height'], bb['width'], bb['depth']) for vesselID, bb in vessels.items()] for key, vessels in bb_dict.items()}
    return bbs

#########################################################
#       Save objects to json files
#########################################################
def tracks_to_json(vessels, path):
    # NB! Assumes all vessels have the same timestamp
    all_tracks = {key: {vessel.id: vessel.get_track_dict()[key] for vessel in vessels} for key in vessels[0].get_track_dict().keys()}
    filename = os.path.join(path, 'tracks.json')
    dict_to_json(filename, all_tracks)

def append_track_to_json(vessels, path, t_start, t_end=None):
    if not t_end:
        tracks = {t_start: {vessel.id: vessel.get_track_dict()[t_start] for vessel in vessels}}
    else:
        tracks = {t: {vessel.id: vessel.get_track_dict()[t] for vessel in vessels} for t in range(t_start, t_end)}
    filename = os.path.join(path, 'tracks.json')
    update_json(filename, tracks)


def vessels_to_json(vessels, path):
    filename = os.path.join(path, 'vehicle_characteristics.json')
    vessel_dict = {vessel.id: {'air_draft_m': vessel.air_draft, 'beam_m': vessel.beam, 'length_m': vessel.length, 'label': vessel.label} for vessel in vessels}
    dict_to_json(filename, vessel_dict)

def projectedPoints_to_json(projected_points, path):
    # All projected points will probably be on a different format. Did this to get a file.
    projected_points_dict = {time_stamp: {vesselID: {cornerNumber: {'x': projected_points[time_stamp][vesselID][cornerNumber].image_coordinate[0], 'y': projected_points[time_stamp][vesselID][cornerNumber].image_coordinate[1], 'depth': projected_points[time_stamp][vesselID][cornerNumber].depth} for cornerNumber in range(len(projected_points[time_stamp][vesselID]))} for vesselID in projected_points[time_stamp]} for time_stamp in projected_points.keys()}
    filename = os.path.join(path, 'projectedPoints.json')
    dict_to_json(filename, projected_points_dict)

def update_projectedPoints_json(projected_points, path, time_stamp):
    '''
    Input:
    - Projected points for the given time stamp
    - Path to save file
    - Time stamp
    '''
    projected_points_dict = {time_stamp: {vesselID: {cornerNumber: {'x': projected_points[vesselID][cornerNumber].image_coordinate[0], 'y': projected_points[vesselID][cornerNumber].image_coordinate[1], 'depth': projected_points[vesselID][cornerNumber].depth} for cornerNumber in range(len(projected_points[vesselID]))} for vesselID in projected_points}}
    filename = os.path.join(path, 'projectedPoints.json')
    update_json(filename, projected_points_dict)

def bbs_to_json(bbs, folder_path):
    # OBS: because we create bounding boxes based on the depth of the vessels in the CCF, the bbs are created in a order with decreasing depth
    bb_dict = {time_stamp: {bb.vesselID: {'centre': {'x':  bb.centre[0], 'y': bb.centre[1]}, 'height': bb.height, 'width': bb.width, 'depth':  bb.depth} for bb in bbs[time_stamp]} for time_stamp in bbs.keys()}
    save_path = os.path.join(folder_path, 'boundingBoxes.json')
    dict_to_json(save_path, bb_dict)

def update_bbs_json(bbs, folder_path, time_stamp):
    '''
    Input:
    - Projected points for the given time stamp
    - Path to save file
    - Time stamp
    '''
    bb_dict = {time_stamp: {bb.vesselID: {'centre': {'x':  bb.centre[0], 'y': bb.centre[1]}, 'height': bb.height, 'width': bb.width, 'depth':  bb.depth} for bb in bbs}}
    save_path = os.path.join(folder_path, 'boundingBoxes.json')
    update_json(save_path, bb_dict)

def error_bbs_to_json(error_bbs, folder_path):
    # OBS: because we create bounding boxes based on the depth of the vessels in the CCF, the bbs are created in a order with decreasing depth
    error_bb_dict = {time_stamp: {bb.vesselID: {'centre': {'x':  bb.centre[0], 'y': bb.centre[1]}, 'height': bb.height, 'width': bb.width, 'depth':  bb.depth} for bb in error_bbs[time_stamp]} for time_stamp in error_bbs.keys()}
    save_path = os.path.join(folder_path, 'distortedBoundingBoxes.json')
    dict_to_json(save_path, error_bb_dict)

def update_eBBs_json(error_bbs, folder_path, time_stamp):
    '''
    Input:
    - Projected points for the given time stamp
    - Path to save file
    - Time stamp
    '''
    eBB_dict = {time_stamp: {bb.vesselID: {'centre': {'x':  bb.centre[0], 'y': bb.centre[1]}, 'height': bb.height, 'width': bb.width, 'depth':  bb.depth} for bb in error_bbs}}
    save_path = os.path.join(folder_path, 'distortedBoundingBoxes.json')
    update_json(save_path, eBB_dict)