import ciw
import numpy as np
import pandas as pd
import des_component as des


def test_make_activity_dictionaries():
    alphabet = ['A', 'B', 'C']
    start_value = 2
    activity_dict, inverted_dict = des.make_activity_dictionaries(alphabet, start_value)

    expected_activity_dict = {'A': 2, 'B': 3, 'C': 4}
    expected_inverted_dict = {2: 'A', 3: 'B', 4: 'C', 1: ''}

    assert activity_dict == expected_activity_dict
    assert inverted_dict == expected_inverted_dict