import ciw
import numpy as np
from types import SimpleNamespace
import pytest
import hybridsim.des_component as des


# Activity dictionary tests
# -------------------------

def test_get_activity_dictionaries():
    alphabet = ['A', 'B', 'C']
    start_value = 2
    activity_dict, inverted_dict = des.get_activity_dictionaries(alphabet, start_value)

    expected_activity_dict = {'A': 2, 'B': 3, 'C': 4}
    expected_inverted_dict = {2: 'A', 3: 'B', 4: 'C', 1: ''}

    assert activity_dict == expected_activity_dict
    assert inverted_dict == expected_inverted_dict


# Helper functions
# ----------------

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
        "Hip": 0,
        "Knee": 1,
    }

    pdfa_matrices = [
        # Hip: Low, Medium, High
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


# PDFA routing tests
# ------------------

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


def test_pdfa_routing_sends_patient_to_exit_when_current_state_has_no_outgoing_transitions():
    alphabet = ["A", "B", "C"]
    activity_dict = {
        "A": 2,
        "B": 3,
        "C": 4,
    }
    subspec_dict = {
        "Hip": 0,
    }

    pdfa = np.zeros((len(alphabet), 3, 3))

    routing = des.PDFARouting(
        pdfa_matrices=[pdfa, pdfa, pdfa],
        alphabets=[alphabet, alphabet, alphabet],
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    simulation = make_routing_test_simulation()
    routing.initialise(simulation, 1)

    ind = make_routing_test_individual(
        customer_class="Hip",
        level="Low",
        route_position=1,
    )

    next_node = routing.next_node(ind)

    assert next_node.id_number == -1
    assert ind.route_position == -1


# Pre-operative routing tests
# ---------------------------

def make_service_record(
    node,
    service_end_date,
):
    return SimpleNamespace(
        node=node,
        service_end_date=service_end_date,
    )


def make_pre_op_routing(
    pdfa,
):
    alphabet = ["A", "B", "C"]
    activity_dict = {
        "A": 2,
        "B": 3,
        "C": 4,
    }
    subspec_dict = {
        "Hip": 0,
    }

    routing = des.PDFARouting(
        pdfa_matrices=[pdfa, pdfa, pdfa],
        alphabets=[alphabet, alphabet, alphabet],
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    simulation = make_routing_test_simulation()
    routing.initialise(simulation, 1)

    return routing, alphabet, activity_dict


def test_pdfa_routing_sets_pre_op_flag_after_pre_op_assessment():
    pdfa = make_deterministic_pdfa(
        alphabet=["A", "B", "C"],
        activity_letter="A",
    )

    routing, _, activity_dict = make_pre_op_routing(pdfa)

    ind = make_routing_test_individual(
        customer_class="Hip",
        level="Low",
        node=activity_dict["B"],
        route_position=1,
    )

    routing.next_node(ind)

    assert ind.pre_op is True


def test_pdfa_routing_prevents_repeat_pre_op_assessment_when_alternative_route_exists():
    alphabet = ["A", "B", "C"]
    pdfa = np.zeros((len(alphabet), 3, 3))
    pdfa[alphabet.index("A"), 1, 2] = 0.5
    pdfa[alphabet.index("B"), 1, 2] = 0.5

    routing, _, activity_dict = make_pre_op_routing(pdfa)

    ind = make_routing_test_individual(
        customer_class="Hip",
        level="Low",
        node=activity_dict["B"],
        route_position=1,
    )

    next_node = routing.next_node(ind)

    assert ind.pre_op is True
    assert next_node.id_number == activity_dict["A"]
    assert ind.route_position == 2


def test_pdfa_routing_resets_pre_op_flag_after_elective_surgery():
    pdfa = make_deterministic_pdfa(
        alphabet=["A", "B", "C"],
        activity_letter="A",
    )

    routing, _, activity_dict = make_pre_op_routing(pdfa)

    ind = make_routing_test_individual(
        customer_class="Hip",
        level="Low",
        node=activity_dict["C"],
        route_position=1,
    )
    ind.pre_op = True

    routing.next_node(ind)

    assert ind.pre_op is False


# Pre-operative expiry distribution tests
# ---------------------------------------

def make_pre_op_expiry_distribution():
    activity_dict = {
        "A": 2,
        "B": 3,
        "C": 4,
    }
    subspec_dict = {
        "Hip": 0,
    }

    expiry_distribution = des.PreOpExpiryDist(
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    return expiry_distribution, activity_dict


def make_pre_op_expiry_test_individual(
    data_records,
    customer_class="Hip",
):
    return SimpleNamespace(
        customer_class=customer_class,
        data_records=data_records,
    )


def test_pre_op_expiry_returns_infinity_when_patient_has_no_pre_op_assessment():
    expiry_distribution, _ = make_pre_op_expiry_distribution()

    ind = make_pre_op_expiry_test_individual(
        data_records=[],
    )

    obtained = expiry_distribution.sample(
        t=100.0,
        ind=ind,
    )

    assert obtained == float("inf")


def test_pre_op_expiry_returns_remaining_validity_after_pre_op_assessment():
    expiry_distribution, activity_dict = make_pre_op_expiry_distribution()

    ind = make_pre_op_expiry_test_individual(
        data_records=[
            make_service_record(
                node=activity_dict["B"],
                service_end_date=100.0,
            ),
        ],
    )

    obtained = expiry_distribution.sample(
        t=110.0,
        ind=ind,
    )

    expected = des.PRE_OP_VALIDITY_DAYS - 10.0

    assert obtained == expected


def test_pre_op_expiry_returns_infinity_when_surgery_occurs_after_pre_op_assessment():
    expiry_distribution, activity_dict = make_pre_op_expiry_distribution()

    ind = make_pre_op_expiry_test_individual(
        data_records=[
            make_service_record(
                node=activity_dict["B"],
                service_end_date=100.0,
            ),
            make_service_record(
                node=activity_dict["C"],
                service_end_date=120.0,
            ),
        ],
    )

    obtained = expiry_distribution.sample(
        t=130.0,
        ind=ind,
    )

    assert obtained == float("inf")


def test_pre_op_expiry_uses_latest_pre_op_assessment_after_surgery():
    expiry_distribution, activity_dict = make_pre_op_expiry_distribution()

    ind = make_pre_op_expiry_test_individual(
        data_records=[
            make_service_record(
                node=activity_dict["B"],
                service_end_date=100.0,
            ),
            make_service_record(
                node=activity_dict["C"],
                service_end_date=120.0,
            ),
            make_service_record(
                node=activity_dict["B"],
                service_end_date=140.0,
            ),
        ],
    )

    obtained = expiry_distribution.sample(
        t=150.0,
        ind=ind,
    )

    expected = des.PRE_OP_VALIDITY_DAYS - 10.0

    assert obtained == expected


def test_pre_op_expiry_returns_negative_remaining_time_when_assessment_has_expired():
    expiry_distribution, activity_dict = make_pre_op_expiry_distribution()

    ind = make_pre_op_expiry_test_individual(
        data_records=[
            make_service_record(
                node=activity_dict["B"],
                service_end_date=100.0,
            ),
        ],
    )

    obtained = expiry_distribution.sample(
        t=100.0 + des.PRE_OP_VALIDITY_DAYS + 1.0,
        ind=ind,
    )

    assert obtained == -1.0