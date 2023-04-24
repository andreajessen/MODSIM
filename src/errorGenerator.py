from datatypes.annotation import Annotation
from datatypes.boundingBox import BoundingBox
from datatypes.detection import Detection
import numpy as np 
import yaml
from utils import update_detections_json

class ErrorGenerator:

    def __init__(self, detector_stats_path):

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


        # Check if classification should be active
        self.classification = True if data_loaded.get('labels') else False

        # Check if there is a general confusion matrix (without labels)
        if data_loaded.get('confusionMatrix'):
            self.drop_out = data_loaded['confusionMatrix']['FN']

        # If classification is True, get label specific confusion matrix
        if self.classification:
            self.possible_labels = data_loaded['labels'] # Possible classification labels
            # Confusion matrix, background not included
            self.confusion_matrix_labels = data_loaded['confusionMatrixLabels']
            self.calculate_drop_out_labels()

        self.confidence_threshold = data_loaded['confidenceThreshold']
    
    
    def calculate_drop_out_labels(self):
        self.possible_labels.append('Background')
        for label, values in self.confusion_matrix_labels.items():
            drop_out = 1 - sum(values)
            self.confusion_matrix_labels[label].append(drop_out)
            


    def generate_error_BB(self, annot: Annotation):
        '''
        Input: 
        annot (Annotation)
        '''
        BB = annot.bb
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
        pred_label = np.random.choice(self.possible_labels, p=self.confusion_matrix_labels[true_label])
        return pred_label
    
    def generate_confidence_score(self):
        # Confidence score should maybe be rounded to 2 or 3 decimals?
        # Analyze the confidence scores of yolov8 to determine a probability distribution
        # Should confidence score be dependent on size and visibility of the bounding box?
        confidence_score = np.random.normal(0.67, 0.10)
        if confidence_score < self.confidence_threshold:
            confidence_score = self.confidence_threshold
        elif confidence_score > 1:
            confidence_score = 1
        return confidence_score

    def is_dropout(self):
        '''
        Calculates dropout when drop out rate is not dependent on classification
        '''
        # Should BB distance affect dropout
        is_drop_out = np.random.choice([True, False], p=[self.drop_out, 1-self.drop_out])
        return is_drop_out

    def generate_error_class(self, annot):
        label = self.generate_error_label(annot.label)
        if label == 'Background':
            return None
        eBB = self.generate_error_BB(annot)
        confidence_score = self.generate_confidence_score()
        return Detection(eBB, label, annot.vesselID, confidence_score)
    
    def generate_error_detection(self, annot):
        label = None
        if self.is_dropout():
            return None
        eBB = self.generate_error_BB(annot)
        confidence_score = self.generate_confidence_score()
        return Detection(eBB, label, annot.vesselID, confidence_score)

        
    def generate_error(self, annot: Annotation):
        if self.classification:
            return self.generate_error_class(annot)
        return self.generate_error_detection(annot)

    
    def generate_detections_t(self, annots_t, t, writeToJson=False, folder_path=None):
        detections = list(filter(lambda item: item is not None, [self.generate_error(annot) for annot in annots_t]))
        if writeToJson and folder_path:
            update_detections_json(detections, folder_path, t)
        return detections