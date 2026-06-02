import ciw
import numpy as np
import pandas as pd
import unittest
from collections import Counter
import des_component as des


def test_make_activity_dictionaries():
    alphabet = ['A', 'B', 'C']
    start_value = 2
    activity_dict, inverted_dict = des.make_activity_dictionaries(alphabet, start_value)

    expected_activity_dict = {'A': 2, 'B': 3, 'C': 4}
    expected_inverted_dict = {2: 'A', 3: 'B', 4: 'C', 1: ''}

    assert activity_dict == expected_activity_dict
    assert inverted_dict == expected_inverted_dict


N = ciw.create_network(
    arrival_distributions=[
        ciw.dists.Exponential(rate=1.0),
        None,
        None,
    ],
    service_distributions=[
        ciw.dists.Exponential(rate=2.0),
        ciw.dists.Exponential(rate=2.0),
        ciw.dists.Exponential(rate=2.0),
    ],
    number_of_servers=[2, 2, 2],
    routing=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
)


class TestPDFARouting(unittest.TestCase):
    def test_transition(self):
        pdfa_matrix = np.zeros((2, 3, 3))
        pdfa_matrix[0, 1, 2] = 1.0
        pdfa_matrix[1, 2, 1] = 1.0
        alphabet = ['A', 'B']
        activity_dictionary = {'A': 2, 'B': 3}
        R = des.PDFARouting(pdfa_matrix, alphabet, activity_dictionary)
        ciw.seed(0)
        Q = ciw.Simulation(N)
        R.initialise(Q, 1)
        ind = ciw.Individual(1)
        samples = Counter([r.id_number for r in [R.next_node(ind) for _ in range(100)]])
        self.assertEqual([samples[i] for i in range(1, 4)], [0, 50, 50])

    def test_endpoint_transition(self):
        pdfa_matrix = np.zeros((1, 3, 3))
        alphabet = ['A']
        activity_dictionary = {'A': 2}
        R = des.PDFARouting(pdfa_matrix, alphabet, activity_dictionary)
        ciw.seed(0)
        Q = ciw.Simulation(N)
        R.initialise(Q, 1)
        ind = ciw.Individual(1)
        samples = [r.id_number for r in [R.next_node(ind) for _ in range(100)]]
        self.assertTrue(all(r == -1 for r in samples))
