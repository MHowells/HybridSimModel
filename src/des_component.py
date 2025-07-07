import ciw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_activity_dictionaries(alphabet, start_value=2):
    """
    Converts a list of activity letters into a dictionaries. One mapping each 
    letter to an integer starting from `start_value`, the other mapping the 
    integers back to the letters.

    Parameters
    ----------
    alphabet : list
        List of activity letters (e.g., ['A', 'B', 'C', ...]).
    start_value : int, optional
        Starting integer value for mapping (default is 2).

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - The first dictionary maps each letter to an integer.
        - The second dictionary maps each integer back to its corresponding letter.
    """
    activity_dict = {letter: idx for idx, letter in enumerate(alphabet, start=start_value)}
    inverted_dict = {v: k for k, v in activity_dict.items()}
    inverted_dict[1] = ''
    return activity_dict, inverted_dict


class PDFARouting(ciw.routing.NodeRouting):
    """
    A class to implement PDFA-based routing in a Ciw discrete-event simulation.
    This class inherits from `ciw.routing.NodeRouting` and implements the
    `next_node` method to determine the next node based on a PDFA matrix and 
    an alphabet of activity letters.

    Attributes
    ----------
    pdfa_matrix : np.ndarray
        A 3D numpy array representing the PDFA transition probabilities.
    alphabet : list
        A list of activity letters corresponding to the PDFA transitions.
    activity_dict : dict
        A dictionary mapping activity letters to their corresponding indices.
    """
    def __init__(self, pdfa_matrix, alphabet, activity_dict):
        """
        Initializes the PDFARouting instance with a PDFA matrix, an alphabet,
        and a dictionary mapping activity letters to their indices.
        Parameters
        ----------
        pdfa_matrix : np.ndarray
            A 3D numpy array representing the PDFA transition probabilities.
        alphabet : list
            A list of activity letters corresponding to the PDFA transitions.
        activity_dict : dict
            A dictionary mapping activity letters to their corresponding indices.
        """
        super().__init__()
        self.p_matrix = pdfa_matrix
        self.alphabet = alphabet
        self.activity_dict = activity_dict

    def next_node(self, ind):
        """
        Determines the next node for an individual based on the PDFA transition
        probabilities and the individual's current route position.
        Parameters
        ----------
        ind : ciw.Individual
            The individual for whom the next node is to be determined.
        Returns
        -------
        ciw.Node
            The next node in the simulation for the individual.
        """
        if not hasattr(ind, "route_position"):
            ind.route_position = 1 # Or initial state if different

        leaving_row = ind.route_position
        p_values = []
        possible_next_state = [] 
        possible_next_activity = []

        for letter in range(len(self.alphabet)):
            trans_probs = self.p_matrix[letter, leaving_row, :]
            if trans_probs.sum() > 0:
                p_values.append(trans_probs.sum())
                possible_next_state.append(np.where(trans_probs > 0)[0][0])
                possible_next_activity.append(letter)

        if len(p_values) > 0:
            final_prob = 1 - sum(p_values)
        else:
            final_prob = 1
        if final_prob > 0:
            p_values.append(final_prob)
            possible_next_state.append("tau")
            possible_next_activity.append(-1)

        next_activity = np.random.choice(a=possible_next_activity, p=p_values)
        next_state = possible_next_state[possible_next_activity.index(next_activity)]

        if next_activity == -1:
            ind.route_position = -1
            return self.simulation.nodes[-1]
        else:
            next_node = self.activity_dict[self.alphabet[next_activity]]
            ind.route_position = next_state
            return self.simulation.nodes[next_node]