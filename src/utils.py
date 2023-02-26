import os
import json

from datatypes.track import Track
from datatypes.vessel import Vessel

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
                                       position['heading_rad'], int(t))
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



