from dataclasses import dataclass

@dataclass
class BBStats:
    sigma_cx: float
    mu_cx: float
    sigma_cy: float
    mu_cy: float
    sigma_h: float
    mu_h: float
    sigma_w: float
    mu_w: float

class ConditionState:

    def __init__(self, name, id, confusion_matrix, bb_stats, fdr):
        '''
        Input:
        - Name (String)
        - Id (int)
        - Confusion matrix: {'Label': [List], 'Label': [List]}. Ex: {'Vessel': [0.867, 0.133], 'Background': [1, None]}
        - fdr (float): False discovery rate
        '''
        self.name = name
        self.id = id
        self.confusion_matrix = confusion_matrix
        self.bb_stats = BBStats(
            sigma_cx=bb_stats['sigma_cx'], 
            mu_cx=bb_stats['mu_cx'], 
            sigma_cy=bb_stats['sigma_cy'], 
            mu_cy=bb_stats['mu_cy'], 
            sigma_h=bb_stats['sigma_h'],
            mu_h=bb_stats['mu_h'],
            sigma_w=bb_stats['sigma_w'],
            mu_w=bb_stats['mu_w']
        )
        self.fdr = fdr

    def get_dropout(self, label):
        return self.confusion_matrix[label][-1]
    
    def get_false_discovery_rate(self):
        return self.fdr

    