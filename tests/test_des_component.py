import ciw
import numpy as np
import pytest
import unittest
from collections import Counter
import hybridsim.des_component as des


def test_get_activity_dictionaries():
    alphabet = ['A', 'B', 'C']
    start_value = 2
    activity_dict, inverted_dict = des.get_activity_dictionaries(alphabet, start_value)

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


def make_deterministic_pdfa(
    alphabet,
    activity_letter,
    from_state=1,
    to_state=2,
    n_states=3,
):
    pdfa = np.zeros((len(alphabet), n_states, n_states))
    activity_index = alphabet.index(activity_letter)
    pdfa[activity_index, from_state, to_state] = 1.0
    return pdfa


def make_routing_test_individual(
    customer_class,
    level,
    node=1,
    route_position=1,
):
    ind = ciw.Individual(1)
    ind.original_class = level
    ind.customer_class = customer_class
    ind.level = level
    ind.node = node
    ind.route_position = route_position
    ind.pre_op = False
    return ind


def make_pdfa_routing_for_severity_and_subspecialty_tests():
    alphabet = ["A", "B", "C"]

    activity_dict = {
        "A": 2,
        "B": 3,
        "C": 4,
    }

    subspec_dict = {
        "Foot": 0,
        "Knee": 1,
    }

    pdfa_matrices = [
        # Foot: Low, Medium, High
        make_deterministic_pdfa(alphabet, "A"),
        make_deterministic_pdfa(alphabet, "B"),
        make_deterministic_pdfa(alphabet, "C"),
        # Knee: Low, Medium, High
        make_deterministic_pdfa(alphabet, "B"),
        make_deterministic_pdfa(alphabet, "C"),
        make_deterministic_pdfa(alphabet, "A"),
    ]

    alphabets = [alphabet] * len(pdfa_matrices)

    routing = des.PDFARouting(
        pdfa_matrices=pdfa_matrices,
        alphabets=alphabets,
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    return routing, activity_dict


def make_routing_test_simulation():
    network = ciw.create_network(
        arrival_distributions=[None] * 8,
        service_distributions=[
            ciw.dists.Deterministic(value=0.0)
            for _ in range(8)
        ],
        number_of_servers=[1] * 8,
        routing=[
            [0.0] * 8
            for _ in range(8)
        ],
    )

    return ciw.Simulation(network)


@pytest.mark.parametrize(
    "customer_class, level, expected_node_id, expected_route_position",
    [
        ("Hip", "Low", 2, 2),
        ("Hip", "Medium", 3, 2),
        ("Hip", "High", 4, 2),
        ("Knee", "Low", 6, 2),
        ("Knee", "Medium", 7, 2),
        ("Knee", "High", 5, 2),
    ],
)
def test_pdfa_routing_selects_correct_activity_for_subspecialty_and_severity(
    customer_class,
    level,
    expected_node_id,
    expected_route_position,
):
    routing, _ = make_pdfa_routing_for_severity_and_subspecialty_tests()
    simulation = make_routing_test_simulation()

    routing.initialise(simulation, 1)

    ind = make_routing_test_individual(
        customer_class=customer_class,
        level=level,
    )

    next_node = routing.next_node(ind)

    assert next_node.id_number == expected_node_id
    assert ind.route_position == expected_route_position

class TestPDFARouting(unittest.TestCase):
    def make_test_individual(self):
        ind = ciw.Individual(1)
        ind.original_class = "Low"
        ind.customer_class = "TestSubspec"
        ind.level = "Low"
        ind.node = 1
        ind.route_position = 1
        ind.pre_op = False
        return ind
    
    def test_transition(self):
        alphabet = ["A", "B", "C", "D"]

        pdfa_matrix = np.zeros((len(alphabet), 3, 3))
        pdfa_matrix[0, 1, 2] = 1.0
        pdfa_matrix[1, 2, 1] = 1.0
        p_matrices = [pdfa_matrix, pdfa_matrix, pdfa_matrix]
        alphabets = [alphabet, alphabet, alphabet]
        activity_dictionary = {
            "A": 2,
            "B": 3,
            "C": 4,
            "D": 5,
        }
        subspec_dictionary = {"TestSubspec": 0}

        R = des.PDFARouting(
            p_matrices,
            alphabets,
            activity_dictionary,
            subspec_dictionary,
            pre_op_letter="C",
            elective_surgery_letter="D",
        )

        ciw.seed(0)
        Q = ciw.Simulation(N)
        R.initialise(Q, 1)
        ind = self.make_test_individual()
        samples = Counter(
            [r.id_number for r in [R.next_node(ind) for _ in range(100)]]
        )
        self.assertEqual([samples[i] for i in range(1, 4)], [0, 50, 50])

    def test_endpoint_transition(self):
        alphabet = ["A", "C", "D"]
        pdfa_matrix = np.zeros((len(alphabet), 3, 3))
        p_matrices = [pdfa_matrix, pdfa_matrix, pdfa_matrix]
        alphabets = [alphabet, alphabet, alphabet]
        activity_dictionary = {
            "A": 2,
            "C": 3,
            "D": 4,
        }
        subspec_dictionary = {"TestSubspec": 0}

        R = des.PDFARouting(
            p_matrices,
            alphabets,
            activity_dictionary,
            subspec_dictionary,
            pre_op_letter="C",
            elective_surgery_letter="D",
        )

        ciw.seed(0)
        Q = ciw.Simulation(N)
        R.initialise(Q, 1)
        ind = self.make_test_individual()
        samples = [r.id_number for r in [R.next_node(ind) for _ in range(100)]]
        self.assertTrue(all(r == -1 for r in samples))
