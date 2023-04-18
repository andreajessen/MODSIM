class ConditionState:

    def __init__(self, name, id, confusion_matrix):
        '''
        Input:
        - Name (String)
        - Id (int)
        - Confusion matrix: {'FN': 0.2, 'FP': 0.2, 'TP': 0.8, 'TN': 0.8} 
        '''
        self.name = name
        self.id = id
        self.confusion_matrix = confusion_matrix

    def get_dropout(self):
        return self.confusion_matrix['FN']