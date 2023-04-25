class ConditionState:

    def __init__(self, name, id, confusion_matrix, confusion_matrix_labels=None):
        '''
        Input:
        - Name (String)
        - Id (int)
        - Confusion matrix: {'FN': 0.2, 'FP': 0.2, 'TP': 0.8, 'TN': 0.8} 
        '''
        self.name = name
        self.id = id
        self.confusion_matrix = confusion_matrix
        self.confusion_matrix_labels = confusion_matrix_labels


    def get_dropout(self):
        return self.confusion_matrix['FN']
    
    def get_false_positives(self):
        return self.confusion_matrix['FP']
    
    def set_confusion_matrix_labels(self, confusion_matrix_labels):
        self.confusion_matrix_labels = confusion_matrix_labels