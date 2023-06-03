import numpy as np 
import copy

from datatypes.annotation import Annotation
from datatypes.boundingBox import BoundingBox
from datatypes.detection import Detection
from datatypes.temporalModel import TemporalModel
from utils import update_detections_json


class ErrorGenerator:

    def __init__(self, labels, background_label, confidence_threshold, transition_matrix, states, start_state):
        '''
        Input:
        - detector_stats_path: path to detector statistics
        - temporal model (boolean)
        - Transition matrix (np array): Consists of conditional probabilities describing the probability of moving from state sj to state si
        - States (dict): key: Id/index in transition matrix, value: ConditionState class
        - Start state (int): ID/index of start state
        '''
        self.background_label = background_label
        self.possible_labels = labels + [background_label]
        self.confidence_threshold = confidence_threshold
        self.temporal_model = TemporalModel(transition_matrix, states, start_state)


    def generate_error_BB(self, annot: Annotation):
        '''
        Input: 
        annot (Annotation)
        '''
        BB = annot.bb
        bb_error_stats = self.temporal_model.get_bb_error_stats()
        # Introduce error based on normal distribution

        error_vector = np.random.multivariate_normal(bb_error_stats.mean_error_vector, bb_error_stats.error_covariance_matrix)

        e_cx = error_vector[0]
        e_cy = error_vector[1]
        e_w = error_vector[2]
        e_h = error_vector[3]

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
        p = self.temporal_model.get_confusion_matrix()[true_label]
        pred_label = np.random.choice(self.possible_labels, p=p)
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

    def generate_error(self, annot: Annotation):
        '''
        Generates error when classification is true
        - annot (Annotation)
        '''
        label = self.generate_error_label(annot.label)
        if label == self.background_label:
            return None
        eBB = self.generate_error_BB(annot)
        confidence_score = self.generate_confidence_score()
        return Detection(eBB, label, annot.vesselID, confidence_score)


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
        fdr = self.temporal_model.get_false_discovery_rate()

        # Higher confidence threshold should give less clutter
        # The maximum number of false positives are dependent on how many detections there are in the image
        # to ensure that the number of false positive bounding boxes is balanced with the number of true positive bounding boxes,
        # Confidence threshold should influence the amount of clutter
        max_fp = max(0, round(fdr*numb_detections-self.confidence_threshold))
        if max_fp == 0: return None
        false_detections = []
        for _ in range(max_fp):
            is_fp = np.random.choice([True, False], p=[fdr, 1-fdr])
            if is_fp:
                # false_positives_labels is the probability of the FP to be each class when we know there is a FP
                p = self.temporal_model.get_confusion_matrix()[self.background_label][:-1]
                label = np.random.choice(self.possible_labels[:-1], p = p)
                detection = self.create_random_detection(image_bounds, horizon, label)
                false_detections.append(detection)
        return false_detections
