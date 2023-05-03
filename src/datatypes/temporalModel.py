import yaml
import numpy as np
import logging




class TemporalModel:

    def __init__(self, transition_matrix, states, start_state):
        '''
        Input:
        - Transition matrix (np array): Consists of conditional probabilities describing the probability of moving from state sj to state si
        - States (dict): key: Id/index in transition matrix, value: ConditionState class
        - Start state (int): ID/index of start state
        '''
        self.states = states
        self.state_names = {id : state.name for id, state in states.items()}
        self.numb_states = len(states)
        self.TM = transition_matrix
        self.start_state = start_state
        self.current_state = start_state
        self.previous_states = {}
    
    def get_confusion_matrix(self):
        return self.states[self.current_state].confusion_matrix
    
    def get_dropout(self):
        return self.states[self.current_state].get_dropout()
    
    def get_false_positives(self):
        return self.states[self.current_state].get_false_positives()
    
    def get_confusion_matrix_labels(self):
        return self.states[self.current_state].confusion_matrix_labels

    def perform_one_time_step(self, t, log=False):
        # Add current state to previous states memory
        self.previous_states[t] = self.current_state

        # Create state probability vector for current state
        p_current = np.zeros(self.numb_states)
        p_current[self.current_state] = 1

        # Calculate state probability vector for next state
        p_next = np.dot(self.TM, p_current)

        self.current_state = np.random.choice(self.numb_states, p=p_next)
        if log:
            print(f'In state {self.current_state}: {self.states[self.current_state].name}')
