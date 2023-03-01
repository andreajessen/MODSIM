from datatypes.boundingBox import BoundingBox
import numpy as np 
import yaml

class ErrorGenerator:

    def __init__(self, detector_stats_path):

        # Read YAML file
        with open(detector_stats_path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        # How to get this in the best way
        self.drop_out = data_loaded['confusionMatrix']['FN']
        self.sigma_cx = data_loaded['errorStats']['sigma_cx'] # Standard deviation
        self.mu_cx = data_loaded['errorStats']['mu_cx'] # Expected value

        self.sigma_cy = data_loaded['errorStats']['sigma_cy'] # Standard deviation
        self.mu_cy = data_loaded['errorStats']['mu_cy'] # Expected value

        self.sigma_h = data_loaded['errorStats']['sigma_h'] # Standard deviation
        self.mu_h = data_loaded['errorStats']['mu_h'] # Expected value

        self.sigma_w = data_loaded['errorStats']['sigma_w'] # Standard deviation
        self.mu_w = data_loaded['errorStats']['mu_w'] # Expected value
    

    def generate_error_BB(self, BB: BoundingBox):
        '''
        Input: 
        boundingBox (BoundingBox)
        '''
        if self.is_dropout():
            # Return no BB, because dropout happened
            return None
        
        # Introduce error based on normal distribution
        e_cx = np.random.normal(self.mu_cx, self.sigma_cx, 1)
        e_cy = np.random.normal(self.mu_cy, self.sigma_cy, 1)
        e_w = np.random.normal(self.mu_w, self.sigma_w, 1)
        e_h = np.random.normal(self.mu_h, self.sigma_h, 1)

        new_centre = BB.centre + [e_cx, e_cy]
        new_w = BB.width + e_w
        new_h = BB.height + e_h

        errorBB = BoundingBox(BB.vesselID, new_centre, new_w, new_h, BB.depth)
        return errorBB

    

    def is_dropout(self):
        # Should BB size affect dropout
        dropout = np.random.choice([True, False], p=[self.drop_out, 1-self.drop_out])
        return dropout


    def generate_all_error_BBs(self, bb_dict):
        '''
        Input:
        Dictionary of timestamp: list of BBs
        '''
        error_bbs = {time_stamp: list(filter(lambda item: item is not None, [self.generate_error_BB(bb) for bb in bbs])) for time_stamp, bbs in bb_dict.items()}
        return error_bbs