from dataclasses import dataclass

# (bb_error_stats.mean_error_vector, bb_error_stats.error_covariance_matrix)
@dataclass
class BBStats:
    mean_error_vector: any
    error_covariance_matrix: any

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
            mean_error_vector=bb_stats['mean_error_vector'], 
            error_covariance_matrix=bb_stats['error_covariance_matrix'], 
        )
        self.fdr = fdr

    def get_dropout(self, label):
        return self.confusion_matrix[label][-1]
    
    def get_false_discovery_rate(self):
        return self.fdr

    