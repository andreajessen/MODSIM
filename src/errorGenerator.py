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
            self.false_positives = data_loaded['confusionMatrix']['FP']

        # If classification is True, get label specific confusion matrix
        if self.classification:
            self.BACKGROUND = 'Background'
            self.possible_labels = data_loaded['labels'] # Possible classification labels
            # Confusion matrix, background not included
            self.confusion_matrix_labels = data_loaded['confusionMatrixLabels']
            self.calculate_drop_out_labels()
            self.false_positives_class = data_loaded.get('falsePositives')
            # Add probability for Not false positives
            self.false_positives_class.append(max(0,1-sum(self.false_positives_class)))

        self.confidence_threshold = data_loaded['confidenceThreshold']
    
    
    def calculate_drop_out_labels(self):
        self.possible_labels.append(self.BACKGROUND)
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
        return round(confidence_score,3)

    def is_dropout(self):
        '''
        Calculates dropout when drop out rate is not dependent on classification
        '''
        # Should BB distance affect dropout
        is_drop_out = np.random.choice([True, False], p=[self.drop_out, 1-self.drop_out])
        return is_drop_out

    def generate_error_class(self, annot):
        label = self.generate_error_label(annot.label)
        if label == self.BACKGROUND:
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

    
    def generate_detections_t(self, annots_t, t, image_bounds, horizon, writeToJson=False, filename=None):
        detections = list(filter(lambda item: item is not None, [self.generate_error(annot) for annot in annots_t]))
        false_detections = self.generate_false_positives(image_bounds, horizon, len(detections))
        if false_detections:
            detections.extend(false_detections)
        if writeToJson and filename:
            update_detections_json(detections, filename, t)
        return detections
    

    def create_random_detection(self, image_bounds, horizon, label):
        cx = np.random.uniform(0, image_bounds[0])
        cy = np.random.uniform(int(horizon[0].get_y()), image_bounds[1])
        h = 100 # What should the size be?
        w = 100 # What should the size be?
        bb = BoundingBox(None, [cx, cy], w, h, None)
        confidence_score = round(np.random.uniform(self.confidence_threshold, self.confidence_threshold+0.2),3) # How to set this?
    
        return Detection(bb, label, None, confidence_score)

    def generate_false_positives(self, image_bounds, horizon, numb_detections):
        total_fp = sum(self.false_positives_class[:-1]) if self.classification else self.false_positives
        # Higher confidence threshold should give less clutter
        # The maximum number of false positives are dependent on how many detections there are in the image
        # to ensure that the number of false positive bounding boxes is balanced with the number of true positive bounding boxes,
        # Confidence threshold should influence the amount of clutter
        max_fp = max(0, round(total_fp*numb_detections-self.confidence_threshold))
        if max_fp == 0: return None
        if self.classification:
            false_detections = []
            for _ in range(max_fp):
                label = np.random.choice(self.possible_labels, p = self.false_positives_class)
                if label!=self.BACKGROUND: 
                    detection = self.create_random_detection(image_bounds, horizon, label)
                    false_detections.append(detection)
        else:
            false_detections = []
            for _ in range(max_fp):
                is_fp = np.random.choice([True, False], p=[self.false_positives, 1-self.false_positives])
                if is_fp:
                    detection = self.create_random_detection(image_bounds, horizon, None)
                    false_detections.append(detection)
        return false_detections



        

