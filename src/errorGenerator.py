from datatypes.boundingBox import BoundingBox
from datatypes.detection import Detection
from datatypes.temporalModel import TemporalModel
import numpy as np 
import yaml
from utils import update_detections_json

class ErrorGenerator:

    def __init__(self, detector_stats_path, temporal_model=False, transition_matrix=None, states=None, start_state=None):
        '''
        Input:
        - detector_stats_path: path to detector statistics
        - Transition matrix (np array): Consists of conditional probabilities describing the probability of moving from state sj to state si
        - States (dict): key: Id/index in transition matrix, value: ConditionState class
        - Start state (int): ID/index of start state
        '''
        # Read YAML file
        with open(detector_stats_path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        # How to get this in the best way
        self.sigma_cx = data_loaded['errorStats']['sigma_cx'] # Standard deviation
        self.mu_cx = data_loaded['errorStats']['mu_cx'] # Expected value

        self.sigma_cy = data_loaded['errorStats']['sigma_cy'] # Standard deviation
        self.mu_cy = data_loaded['errorStats']['mu_cy'] # Expected value

        self.sigma_h = data_loaded['errorStats']['sigma_h'] # Standard deviation
        self.mu_h = data_loaded['errorStats']['mu_h'] # Expected value

        self.sigma_w = data_loaded['errorStats']['sigma_w'] # Standard deviation
        self.mu_w = data_loaded['errorStats']['mu_w'] # Expected value


        # Drop out rate if we don't have a temporal model. But we always assume we have a temporal model?
        self.drop_out = data_loaded['confusionMatrix']['FN']

        if temporal_model:
            if (transition_matrix is None and states is None and start_state is None):
                raise ValueError('You need to provide transition_matrix, states and start_state')
            self.temporal_model = TemporalModel(transition_matrix, states, start_state)
        else:
            self.temporal_model = None



    

    def generate_error_BB(self, BB: BoundingBox):
        '''
        Input: 
        boundingBox (BoundingBox)
        '''
        if self.is_dropout():
            # Return no BB, because dropout happened
            return None
        
        # Introduce error based on normal distribution
        e_cx = float(np.random.normal(self.mu_cx, self.sigma_cx, 1))
        e_cy = float(np.random.normal(self.mu_cy, self.sigma_cy, 1))
        e_w = float(np.random.normal(self.mu_w, self.sigma_w, 1))
        e_h = float(np.random.normal(self.mu_h, self.sigma_h, 1))

        new_centre = BB.centre + [e_cx, e_cy]
        new_w = BB.width + e_w
        new_h = BB.height + e_h

        errorBB = BoundingBox(BB.vesselID, new_centre, new_w, new_h, BB.depth)
        return errorBB
    
    def generate_error_label(self, true_label):
        return true_label

    def is_dropout(self):
        drop_out = self.drop_out
        if self.temporal_model:
            drop_out = self.temporal_model.get_dropout()
        # Should BB size affect dropout
        dropout = np.random.choice([True, False], p=[drop_out, 1-drop_out])
        return dropout

    
    def generate_error(self, annot):
        eBB = self.generate_error_BB(annot.bb)
        if not eBB: return None
        label = self.generate_error_label(annot.label)
        return Detection(eBB, label, annot.vesselID)

    
    def generate_detections_t(self, annots_t, t, writeToJson=False, folder_path=None, log=False):
        detections = list(filter(lambda item: item is not None, [self.generate_error(annot) for annot in annots_t]))
        self.temporal_model.perform_one_time_step(t, log)
        if writeToJson and folder_path:
            update_detections_json(detections, folder_path, t)
        return detections