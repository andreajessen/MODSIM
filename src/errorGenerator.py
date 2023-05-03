from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np 
import yaml
import copy

from datatypes.annotation import Annotation
from datatypes.boundingBox import BoundingBox
from datatypes.detection import Detection
from datatypes.temporalModel import TemporalModel
from utils import update_detections_json

@dataclass
class DetectorStats:
    sigma_cx: float
    mu_cx: float
    sigma_cy: float
    mu_cy: float
    sigma_h: float
    mu_h: float
    sigma_w: float
    mu_w: float
    labels: Optional[List[str]]
    false_positives_labels: Optional[List[float]]
    confusion_matrix: Optional[Dict[str, Dict[str, float]]]
    confidence_threshold: float


class ErrorGenerator:

    def __init__(self, detector_stats_path: str, temporal_model: bool, transition_matrix, states, start_state):
        '''
        Input:
        - detector_stats_path: path to detector statistics
        - temporal model (boolean)
        - Transition matrix (np array): Consists of conditional probabilities describing the probability of moving from state sj to state si
        - States (dict): key: Id/index in transition matrix, value: ConditionState class
        - Start state (int): ID/index of start state
        '''

        # Read YAML file
        with open(detector_stats_path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        # Sigma = standard deviation
        # Mu = expected values
        self.stats = DetectorStats(
            sigma_cx=data_loaded['errorStats']['sigma_cx'], 
            mu_cx=data_loaded['errorStats']['mu_cx'], 
            sigma_cy=data_loaded['errorStats']['sigma_cy'], 
            mu_cy=data_loaded['errorStats']['mu_cy'], 
            sigma_h=data_loaded['errorStats']['sigma_h'],
            mu_h=data_loaded['errorStats']['mu_h'],
            sigma_w=data_loaded['errorStats']['sigma_w'],
            mu_w=data_loaded['errorStats']['mu_w'],
            labels=data_loaded.get('labels'),
            false_positives_labels=data_loaded.get('falsePositivesLabels'),
            confusion_matrix=data_loaded.get('confusionMatrix'),
            confidence_threshold=data_loaded['confidenceThreshold']
        )

        self.classification = bool(self.stats.labels)
        self.BACKGROUND = 'Background'
        self.possible_labels = self.stats.labels + [self.BACKGROUND] if self.classification else None
        self.confidence_threshold = self.stats.confidence_threshold
        self.drop_out = self.stats.confusion_matrix['FN'] if self.stats.confusion_matrix else None
        self.false_positives = self.stats.confusion_matrix['FP'] if self.stats.confusion_matrix else None
        self.false_positives_labels = self.stats.false_positives_labels
        self.temporal_model = TemporalModel(transition_matrix, states, start_state) if temporal_model else None



    def generate_error_BB(self, annot: Annotation):
        '''
        Input: 
        annot (Annotation)
        '''
        BB = annot.bb
        # Introduce error based on normal distribution
        e_cx = float(np.random.normal(self.stats.mu_cx, self.stats.sigma_cx, 1))
        e_cy = float(np.random.normal(self.stats.mu_cy, self.stats.sigma_cy, 1))
        e_w = float(np.random.normal(self.stats.mu_w, self.stats.sigma_w, 1))
        e_h = float(np.random.normal(self.stats.mu_h, self.stats.sigma_h, 1))

        new_centre = BB.centre + [e_cx, e_cy]
        new_w = BB.width + e_w
        new_h = BB.height + e_h

        errorBB = BoundingBox(BB.vesselID, new_centre, new_w, new_h, BB.depth)
        return errorBB
    
    def generate_error_label(self, true_label):
        '''
        Generates error in the classification label
        '''
        # Confusion matrix excludes background label.
        # For np choice the p array must sum to one, so we need to include the label background
        # If the label is background it is a drop-out.
        p = self.temporal_model.get_confusion_matrix_labels()[true_label]
        p_w_drop_out = p.append(1-sum(p))
        pred_label = np.random.choice(self.possible_labels, p=p_w_drop_out)
        return pred_label
    
    def generate_confidence_score(self):
        '''
        Generates a confidence score for the detected bounding boxes
        '''
        # Confidence score should maybe be rounded to 2 or 3 decimals?
        # Analyze the confidence scores of yolov8 to determine a probability distribution
        # Should confidence score be dependent on size and visibility of the bounding box?
        confidence_score = np.random.normal(0.67, 0.10)
        if confidence_score < self.confidence_threshold:
            confidence_score = self.confidence_threshold
        elif confidence_score > 1:
            confidence_score = 1
        return round(confidence_score,3)

    def is_dropout(self):
        '''
        Calculates dropout when drop out rate is not dependent on classification
        '''
        drop_out = self.temporal_model.get_dropout()
        # Should BB size affect dropout
        dropout = np.random.choice([True, False], p=[drop_out, 1-drop_out])
        return dropout

    def generate_error_class(self, annot):
        '''
        Generates error when classification is true
        - annot (Annotation)
        '''
        label = self.generate_error_label(annot.label)
        if label == self.BACKGROUND:
            return None
        eBB = self.generate_error_BB(annot)
        confidence_score = self.generate_confidence_score()
        return Detection(eBB, label, annot.vesselID, confidence_score)
    
    def generate_error_detection(self, annot):
        '''
        Generates error when classification is false
        - annot (Annotation)
        '''
        label = None
        if self.is_dropout():
            return None
        eBB = self.generate_error_BB(annot)
        confidence_score = self.generate_confidence_score()
        return Detection(eBB, label, annot.vesselID, confidence_score)

        
    def generate_error(self, annot: Annotation):
        '''
        Generates detections (erroneous bounding boxes (and labels))
        '''
        if self.classification:
            return self.generate_error_class(annot)
        return self.generate_error_detection(annot)

    def generate_detections_t(self, annots_t, t, image_bounds, horizon, writeToJson=False, filename=None, log=False):
        '''
        Generates a all detections in the given time step.
        Input:
        - annots_t (array of Annotations): all annotations a the current time step
        - t (int): current time step
        - Image bounds (array of len 2): The hight and width of the image
        - Horizon (array of 2 ProjectedPoints): Two points at the horizon
        '''
        detections = list(filter(lambda item: item is not None, [self.generate_error(annot) for annot in annots_t]))
        false_detections = self.generate_false_positives(image_bounds, horizon, len(detections))
        if false_detections:
            detections.extend(false_detections)
        if writeToJson and filename:
            update_detections_json(detections, filename, t)
        self.temporal_model.perform_one_time_step(t, log)
        return detections
    

    def create_random_detection(self, image_bounds, horizon, label):
        '''
        Creates a random false detection / clutter
        Input
        - Image bounds (array of len 2): The hight and width of the image
        - Horizon (array of 2 ProjectedPoints): Two points at the horizon
        - label (str): Classification label or None
        '''
        cx = np.random.uniform(0, image_bounds[0])
        cy = np.random.uniform(int(horizon[0].get_y()), image_bounds[1])
        h = 100 # What should the size be?
        w = 100 # What should the size be?
        bb = BoundingBox(None, [cx, cy], w, h, None)

        # How to set this?
        confidence_score = round(np.random.uniform(self.confidence_threshold, self.confidence_threshold+0.2),3) 

        return Detection(bb, label, None, confidence_score)

    def generate_false_positives(self, image_bounds, horizon, numb_detections):
        '''
        Generates false positives/clutter based on the current false positive rate. 
        Input
        - Image bounds (array of len 2): The hight and width of the image
        - Horizon (array of 2 ProjectedPoints): Two points at the horizon
        - num_detections (int): Number of detections at the current time step
        '''
        fp_prob = self.temporal_model.get_false_positives()

        # Higher confidence threshold should give less clutter
        # The maximum number of false positives are dependent on how many detections there are in the image
        # to ensure that the number of false positive bounding boxes is balanced with the number of true positive bounding boxes,
        # Confidence threshold should influence the amount of clutter
        max_fp = max(0, round(fp_prob*numb_detections-self.confidence_threshold))
        if max_fp == 0: return None
        false_detections = []
        for _ in range(max_fp):
            is_fp = np.random.choice([True, False], p=[fp_prob, 1-fp_prob])
            if is_fp:
                # false_positives_labels is the probability of the FP to be each class when we know there is a FP
                label = np.random.choice(self.possible_labels[:-1], p = self.false_positives_labels) if self.classification else None
                detection = self.create_random_detection(image_bounds, horizon, label)
                false_detections.append(detection)
        return false_detections
