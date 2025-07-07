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