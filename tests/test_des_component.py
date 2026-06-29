import ciw
import numpy as np
import pandas as pd
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


# Network node list tests
# -----------------------

def test_get_list_of_nodes_returns_referral_nodes_and_activity_blocks_by_subspecialty():
    alphabets = [
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["A", "C", "D"],
        ["A", "C", "D"],
        ["A", "C", "D"],
    ]
    subspecialties = ["Hip", "Knee"]

    obtained = des.get_list_of_nodes(
        alphabets=alphabets,
        subspecialties=subspecialties,
    )

    assert obtained == [
        "*",
        "*",
        "A",
        "B",
        "C",
        "D",
        "A",
        "B",
        "C",
        "D",
    ]


def test_get_list_of_nodes_removes_duplicate_activity_letters():
    alphabets = [
        ["C", "A", "B"],
        ["B", "A"],
        ["C", "B"],
    ]
    subspecialties = ["Hip"]

    obtained = des.get_list_of_nodes(
        alphabets=alphabets,
        subspecialties=subspecialties,
    )

    assert obtained == [
        "*",
        "*",
        "A",
        "B",
        "C",
    ]


def test_get_list_of_nodes_repeats_activity_block_for_each_subspecialty():
    alphabets = [
        ["A", "B"],
        ["A", "B"],
        ["A", "B"],
    ]
    subspecialties = ["Hip", "Knee", "Spine"]

    obtained = des.get_list_of_nodes(
        alphabets=alphabets,
        subspecialties=subspecialties,
    )

    assert obtained == [
        "*",
        "*",
        "A",
        "B",
        "A",
        "B",
        "A",
        "B",
    ]


# PDFA activity removal tests
# ---------------------------

def test_remove_activity_from_pdfa_removes_selected_activity_transitions():
    pdfa = np.zeros((3, 3, 3))

    pdfa[0, 1, 2] = 0.5
    pdfa[1, 1, 2] = 0.5

    obtained = des.remove_activity_from_pdfa(
        pdfa=pdfa,
        activity_index=1,
    )

    assert np.all(obtained[1] == 0.0)


def test_remove_activity_from_pdfa_renormalises_remaining_transitions():
    pdfa = np.zeros((3, 3, 3))

    pdfa[0, 1, 2] = 0.2
    pdfa[1, 1, 2] = 0.3
    pdfa[2, 1, 2] = 0.5

    obtained = des.remove_activity_from_pdfa(
        pdfa=pdfa,
        activity_index=1,
    )

    assert obtained[0, 1, 2] == pytest.approx(0.2 / 0.7)
    assert obtained[1, 1, 2] == 0.0
    assert obtained[2, 1, 2] == pytest.approx(0.5 / 0.7)

    assert obtained[:, 1, :].sum() == pytest.approx(1.0)


def test_remove_activity_from_pdfa_leaves_empty_state_row_when_no_transitions_remain():
    pdfa = np.zeros((3, 3, 3))
    pdfa[1, 1, 2] = 1.0

    obtained = des.remove_activity_from_pdfa(
        pdfa=pdfa,
        activity_index=1,
    )

    assert np.all(obtained[:, 1, :] == 0.0)


def test_remove_activity_from_pdfa_does_not_modify_original_pdfa():
    pdfa = np.zeros((3, 3, 3))
    pdfa[0, 1, 2] = 0.4
    pdfa[1, 1, 2] = 0.6

    original_pdfa = pdfa.copy()

    des.remove_activity_from_pdfa(
        pdfa=pdfa,
        activity_index=1,
    )

    assert np.array_equal(pdfa, original_pdfa)


# Closed PDFA tests
# -----------------

def test_find_closed_pdfa_states_identifies_self_loop_without_tau_exit():
    pdfa = np.zeros((1, 2, 2))
    pdfa[0, 1, 1] = 1.0

    obtained = des.find_closed_pdfa_states(
        p_matrix=pdfa,
    )

    assert obtained == [[1]]


def test_find_closed_pdfa_states_identifies_multi_state_cycle_without_tau_exit():
    pdfa = np.zeros((1, 3, 3))

    pdfa[0, 1, 2] = 1.0
    pdfa[0, 2, 1] = 1.0

    obtained = des.find_closed_pdfa_states(
        p_matrix=pdfa,
    )

    assert obtained == [[1, 2]]


def test_find_closed_pdfa_states_does_not_flag_state_with_tau_exit():
    pdfa = np.zeros((1, 2, 2))
    pdfa[0, 1, 1] = 0.8

    obtained = des.find_closed_pdfa_states(
        p_matrix=pdfa,
    )

    assert obtained == []


def test_find_closed_pdfa_states_does_not_flag_cycle_with_transition_outside_group():
    pdfa = np.zeros((1, 4, 4))

    pdfa[0, 1, 2] = 1.0
    pdfa[0, 2, 1] = 0.5
    pdfa[0, 2, 3] = 0.5

    obtained = des.find_closed_pdfa_states(
        p_matrix=pdfa,
    )

    assert obtained == []


# Add tau escape tests
# --------------------

def test_add_tau_escape_to_states_rescales_outgoing_probabilities():
    pdfa = np.zeros((2, 3, 3))

    pdfa[0, 1, 1] = 0.4
    pdfa[1, 1, 2] = 0.6

    obtained = des.add_tau_escape_to_states(
        pdfa=pdfa,
        states=[1],
        tau_prob=0.2,
    )

    assert obtained[0, 1, 1] == pytest.approx(0.32)
    assert obtained[1, 1, 2] == pytest.approx(0.48)

    assert obtained[:, 1, :].sum() == pytest.approx(0.8)


def test_add_tau_escape_to_states_preserves_relative_transition_probabilities():
    pdfa = np.zeros((2, 3, 3))

    pdfa[0, 1, 1] = 0.25
    pdfa[1, 1, 2] = 0.75

    obtained = des.add_tau_escape_to_states(
        pdfa=pdfa,
        states=[1],
        tau_prob=0.1,
    )

    obtained_ratio = (
        obtained[0, 1, 1]
        / obtained[1, 1, 2]
    )

    expected_ratio = 0.25 / 0.75

    assert obtained_ratio == pytest.approx(expected_ratio)
    assert obtained[:, 1, :].sum() == pytest.approx(0.9)


def test_add_tau_escape_to_states_only_changes_selected_states():
    pdfa = np.zeros((1, 3, 3))

    pdfa[0, 1, 1] = 1.0
    pdfa[0, 2, 2] = 1.0

    obtained = des.add_tau_escape_to_states(
        pdfa=pdfa,
        states=[1],
        tau_prob=0.25,
    )

    assert obtained[0, 1, 1] == pytest.approx(0.75)
    assert obtained[0, 2, 2] == pytest.approx(1.0)


def test_add_tau_escape_to_states_leaves_zero_outgoing_state_unchanged():
    pdfa = np.zeros((1, 3, 3))

    obtained = des.add_tau_escape_to_states(
        pdfa=pdfa,
        states=[1],
        tau_prob=0.2,
    )

    assert np.all(obtained[:, 1, :] == 0.0)


def test_add_tau_escape_to_states_does_not_modify_original_pdfa():
    pdfa = np.zeros((1, 3, 3))
    pdfa[0, 1, 1] = 1.0

    original_pdfa = pdfa.copy()

    des.add_tau_escape_to_states(
        pdfa=pdfa,
        states=[1],
        tau_prob=0.2,
    )

    assert np.array_equal(pdfa, original_pdfa)


# Helper functions for PDFA routing tests
# ---------------------------------------

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

def make_service_record(
    node,
    service_end_date,
):
    return SimpleNamespace(
        node=node,
        service_end_date=service_end_date,
    )


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


# Jockey routing tests
# --------------------

def make_jockey_routing():
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

    pdfa = make_deterministic_pdfa(
        alphabet=alphabet,
        activity_letter="A",
    )

    routing = des.JockeyRouting(
        pdfa_matrix=[pdfa] * 6,
        alphabet=[alphabet] * 6,
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    simulation = make_routing_test_simulation()
    routing.initialise(simulation, 1)

    return routing, activity_dict


@pytest.mark.parametrize(
    "customer_class, expected_node_id",
    [
        ("Hip", 3),
        ("Knee", 6),
    ],
)
def test_jockey_routing_sends_patient_to_subspecialty_pre_op_node(
    customer_class,
    expected_node_id,
):
    routing, _ = make_jockey_routing()

    ind = make_routing_test_individual(
        customer_class=customer_class,
        level="Low",
    )

    next_node = routing.next_node_for_jockeying(ind)

    assert next_node.id_number == expected_node_id


# GP arrival distribution tests
# -----------------------------

def test_make_gp_arrival_rates_creates_one_distribution_per_severity_level():
    lambdas = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    t = np.array([0.0, 1.0, 2.0])

    obtained = des.make_gp_arrival_rates(
        lambdas=lambdas,
        t=t,
        max_sample_date=10.0,
    )

    assert len(obtained) == 3
    assert all(
        isinstance(distribution, ciw.dists.PoissonIntervals)
        for distribution in obtained
    )


def test_make_gp_arrival_rates_uses_low_medium_high_columns_in_order():
    lambdas = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    t = np.array([0.0, 1.0, 2.0])

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )
    )

    assert low_arrivals.rates == [1.0, 4.0]
    assert medium_arrivals.rates == [2.0, 5.0]
    assert high_arrivals.rates == [3.0, 6.0]

    assert low_arrivals.endpoints == [1.0, 2.0]
    assert medium_arrivals.endpoints == [1.0, 2.0]
    assert high_arrivals.endpoints == [1.0, 2.0]


@pytest.mark.parametrize(
    "lambdas",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([[[1.0, 2.0, 3.0]]]),
    ],
)
def test_make_gp_arrival_rates_rejects_non_two_dimensional_lambdas(
    lambdas,
):
    t = np.array([0.0, 1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="Expected lambdas to be two-dimensional",
    ):
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )


def test_make_gp_arrival_rates_rejects_mismatched_lambda_and_time_lengths():
    lambdas = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    t = np.array([0.0, 1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="lambdas.shape\\[0\\] to match len\\(t\\)",
    ):
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )


@pytest.mark.parametrize(
    "lambdas",
    [
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ),
    ],
)
def test_make_gp_arrival_rates_rejects_non_three_severity_columns(
    lambdas,
):
    t = np.array([0.0, 1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="Expected three severity columns",
    ):
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )


# Other referral arrival distribution tests
# -----------------------------------------

def test_make_other_referral_arrival_rates_creates_one_distribution_per_severity_level():
    weekday_rates = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    endpoints = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    obtained = des.make_other_referral_arrival_rates(
        weekday_rates=weekday_rates,
        endpoints=endpoints,
        max_sample_date=10.0,
    )

    assert len(obtained) == 3
    assert all(
        isinstance(distribution, ciw.dists.PoissonIntervals)
        for distribution in obtained
    )


def test_make_other_referral_arrival_rates_splits_weekday_rates_by_severity_proportion():
    weekday_rates = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    endpoints = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_other_referral_arrival_rates(
            weekday_rates=weekday_rates,
            endpoints=endpoints,
            max_sample_date=10.0,
            severity_proportions=(0.5, 0.3, 0.2),
        )
    )

    assert low_arrivals.rates == [
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
    ]
    assert medium_arrivals.rates == [
        3.0,
        6.0,
        9.0,
        12.0,
        15.0,
        18.0,
        21.0,
    ]
    assert high_arrivals.rates == [
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
    ]

    assert low_arrivals.endpoints == endpoints
    assert medium_arrivals.endpoints == endpoints
    assert high_arrivals.endpoints == endpoints


def test_make_other_referral_arrival_rates_respects_custom_severity_proportions():
    weekday_rates = [10.0, 20.0]
    endpoints = [1.0, 2.0]

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_other_referral_arrival_rates(
            weekday_rates=weekday_rates,
            endpoints=endpoints,
            max_sample_date=10.0,
            severity_proportions=(0.1, 0.2, 0.7),
        )
    )

    assert low_arrivals.rates == [1.0, 2.0]
    assert medium_arrivals.rates == [2.0, 4.0]
    assert high_arrivals.rates == [7.0, 14.0]


def test_make_other_referral_arrival_rates_returns_zero_rates_when_weekday_rates_are_zero():
    weekday_rates = [0.0] * 7
    endpoints = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_other_referral_arrival_rates(
            weekday_rates=weekday_rates,
            endpoints=endpoints,
            max_sample_date=10.0,
        )
    )

    assert low_arrivals.rates == [0.0] * 7
    assert medium_arrivals.rates == [0.0] * 7
    assert high_arrivals.rates == [0.0] * 7


# Arrival distribution assignment tests
# -------------------------------------

def test_get_arrival_distributions_assigns_gp_and_other_arrivals_to_referral_nodes():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    gp_arrival_rates = [
        "gp_low",
        "gp_medium",
        "gp_high",
    ]
    other_arrival_rates = [
        "other_low",
        "other_medium",
        "other_high",
    ]

    obtained = des.get_arrival_distributions_for_nodes(
        nodes=nodes,
        subspecialties=subspecialties,
        gp_arrival_rates=gp_arrival_rates,
        other_arrival_rates=other_arrival_rates,
    )

    assert obtained["Low"] == [
        "gp_low",
        "other_low",
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert obtained["Medium"] == [
        "gp_medium",
        "other_medium",
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert obtained["High"] == [
        "gp_high",
        "other_high",
        None,
        None,
        None,
        None,
        None,
        None,
    ]


def test_get_arrival_distributions_assigns_no_direct_arrivals_to_subspecialty_classes():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    obtained = des.get_arrival_distributions_for_nodes(
        nodes=nodes,
        subspecialties=subspecialties,
        gp_arrival_rates=[
            "gp_low",
            "gp_medium",
            "gp_high",
        ],
        other_arrival_rates=[
            "other_low",
            "other_medium",
            "other_high",
        ],
    )

    assert obtained["Hip"] == [None] * len(nodes)
    assert obtained["Knee"] == [None] * len(nodes)


def test_get_arrival_distributions_uses_no_arrivals_when_inputs_are_not_supplied():
    nodes = ["*", "*", "A", "B", "C"]
    subspecialties = ["Hip"]

    obtained = des.get_arrival_distributions_for_nodes(
        nodes=nodes,
        subspecialties=subspecialties,
    )

    expected_no_arrivals = [None] * len(nodes)

    assert obtained["Low"] == expected_no_arrivals
    assert obtained["Medium"] == expected_no_arrivals
    assert obtained["High"] == expected_no_arrivals
    assert obtained["Hip"] == expected_no_arrivals


# Service distribution assignment tests
# -------------------------------------

def test_make_deterministic_service_distributions_converts_values_to_ciw_distributions():
    service_values = [
        [1.5, 2.0, 3.5],
        [4.0, 5.0, 6.0],
    ]

    obtained = des.make_deterministic_service_distributions(
        service_values=service_values,
    )

    assert len(obtained) == 2
    assert len(obtained[0]) == 3
    assert len(obtained[1]) == 3

    assert all(
        isinstance(distribution, ciw.dists.Deterministic)
        for subspecialty_distributions in obtained
        for distribution in subspecialty_distributions
    )

    assert obtained[0][0].sample() == 1.5
    assert obtained[0][1].sample() == 2.0
    assert obtained[0][2].sample() == 3.5
    assert obtained[1][0].sample() == 4.0
    assert obtained[1][1].sample() == 5.0
    assert obtained[1][2].sample() == 6.0


def test_get_service_distributions_assigns_zero_service_to_referral_classes():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    subspecialty_services = des.make_deterministic_service_distributions(
        service_values=[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
    )

    obtained = des.get_service_distributions_for_nodes(
        nodes=nodes,
        subspecialties=subspecialties,
        subspecialty_services=subspecialty_services,
    )

    for severity_class in ["Low", "Medium", "High"]:
        assert len(obtained[severity_class]) == len(nodes)

        assert all(
            distribution.sample() == 0.0
            for distribution in obtained[severity_class]
        )


def test_get_service_distributions_assigns_services_to_correct_subspecialty_node_blocks():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    hip_services = [
        ciw.dists.Deterministic(value=1.0),
        ciw.dists.Deterministic(value=2.0),
        ciw.dists.Deterministic(value=3.0),
    ]
    knee_services = [
        ciw.dists.Deterministic(value=4.0),
        ciw.dists.Deterministic(value=5.0),
        ciw.dists.Deterministic(value=6.0),
    ]

    obtained = des.get_service_distributions_for_nodes(
        nodes=nodes,
        subspecialties=subspecialties,
        subspecialty_services=[
            hip_services,
            knee_services,
        ],
    )

    assert [
        distribution.sample()
        for distribution in obtained["Hip"]
    ] == [
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
    ]

    assert [
        distribution.sample()
        for distribution in obtained["Knee"]
    ] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.0,
        5.0,
        6.0,
    ]


# Server allocation tests
# -----------------------

def test_get_servers_assigns_infinite_servers_to_referral_nodes():
    nodes = ["*", "*", "A", "B", "C"]

    obtained = des.get_servers(
        nodes=nodes,
    )

    assert obtained == [
        float("inf"),
        float("inf"),
        1,
        1,
        1,
    ]


def test_get_servers_assigns_infinite_servers_to_emergency_nodes():
    nodes = ["*", "*", "A", "B", "C"]

    obtained = des.get_servers(
        nodes=nodes,
        emergency_nodes=["B"],
    )

    assert obtained == [
        float("inf"),
        float("inf"),
        1,
        float("inf"),
        1,
    ]


def test_get_servers_assigns_one_server_to_elective_activity_nodes():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]

    obtained = des.get_servers(
        nodes=nodes,
        emergency_nodes=["C"],
    )

    assert obtained[0] == float("inf")
    assert obtained[1] == float("inf")

    assert obtained[2] == 1
    assert obtained[3] == 1
    assert obtained[4] == float("inf")

    assert obtained[5] == 1
    assert obtained[6] == 1
    assert obtained[7] == float("inf")


# Routing matrix assignment tests
# -------------------------------

def test_get_routing_assigns_leave_routing_to_referral_classes():
    nodes = ["*", "*", "A", "B", "C"]
    subspecialties = ["Hip"]

    supplied_routing = object()

    obtained = des.get_routing(
        nodes=nodes,
        subspecialties=subspecialties,
        subspecialty_class=supplied_routing,
    )

    for severity_class in ["Low", "Medium", "High"]:
        network_routing = obtained[severity_class]

        assert isinstance(
            network_routing,
            ciw.routing.NetworkRouting,
        )

        assert len(network_routing.routers) == len(nodes)

        assert all(
            isinstance(router, ciw.routing.Leave)
            for router in network_routing.routers
        )


def test_get_routing_assigns_supplied_routing_to_each_subspecialty_node():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    supplied_routing = object()

    obtained = des.get_routing(
        nodes=nodes,
        subspecialties=subspecialties,
        subspecialty_class=supplied_routing,
    )

    for subspecialty in subspecialties:
        network_routing = obtained[subspecialty]

        assert isinstance(
            network_routing,
            ciw.routing.NetworkRouting,
        )

        assert len(network_routing.routers) == len(nodes)

        assert all(
            router is supplied_routing
            for router in network_routing.routers
        )


# Class change matrix tests
# -------------------------

def test_get_class_change_matrices_assigns_severity_specific_subspecialty_probabilities():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    obtained = des.get_class_change_matrices(
        nodes=nodes,
        subspecialties=subspecialties,
        subspec_probs_low=[0.7, 0.3],
        subspec_probs_medium=[0.4, 0.6],
        subspec_probs_high=[0.2, 0.8],
    )

    referral_class_changes = obtained[0]

    assert referral_class_changes["Low"] == {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
        "Hip": 0.7,
        "Knee": 0.3,
    }

    assert referral_class_changes["Medium"] == {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
        "Hip": 0.4,
        "Knee": 0.6,
    }

    assert referral_class_changes["High"] == {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
        "Hip": 0.2,
        "Knee": 0.8,
    }


@pytest.mark.parametrize(
    "severity_class",
    ["Low", "Medium", "High"],
)
def test_get_class_change_matrices_referral_probabilities_sum_to_one(
    severity_class,
):
    nodes = ["*", "*", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    obtained = des.get_class_change_matrices(
        nodes=nodes,
        subspecialties=subspecialties,
        subspec_probs_low=[0.7, 0.3],
        subspec_probs_medium=[0.4, 0.6],
        subspec_probs_high=[0.2, 0.8],
    )

    referral_class_changes = obtained[0]

    assert sum(
        referral_class_changes[severity_class].values()
    ) == 1.0


def test_get_class_change_matrices_applies_same_class_changes_at_both_referral_nodes():
    nodes = ["*", "*", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    obtained = des.get_class_change_matrices(
        nodes=nodes,
        subspecialties=subspecialties,
        subspec_probs_low=[0.7, 0.3],
        subspec_probs_medium=[0.4, 0.6],
        subspec_probs_high=[0.2, 0.8],
    )

    assert obtained[0] == obtained[1]


@pytest.mark.parametrize(
    "subspecialty",
    ["Hip", "Knee"],
)
def test_get_class_change_matrices_keeps_subspecialty_class_unchanged_at_activity_nodes(
    subspecialty,
):
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    obtained = des.get_class_change_matrices(
        nodes=nodes,
        subspecialties=subspecialties,
        subspec_probs_low=[0.7, 0.3],
        subspec_probs_medium=[0.4, 0.6],
        subspec_probs_high=[0.2, 0.8],
    )

    first_activity_node_changes = obtained[2]

    expected = {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
        "Hip": float(subspecialty == "Hip"),
        "Knee": float(subspecialty == "Knee"),
    }

    assert first_activity_node_changes[subspecialty] == expected


# Reneging distribution assignment tests
# --------------------------------------

def test_get_reneging_time_distributions_assigns_no_reneging_at_referral_nodes():
    nodes = ["*", "*", "A", "B", "C"]
    subspecialties = ["Hip"]
    reneging_distribution = object()

    obtained = des.get_reneging_time_distributions(
        nodes=nodes,
        subspecs=subspecialties,
        reneging_distribution=reneging_distribution,
    )

    for customer_class in ["Low", "Medium", "High", "Hip"]:
        assert obtained[customer_class][:2] == [None, None]


def test_get_reneging_time_distributions_assigns_distribution_to_activity_nodes():
    nodes = ["*", "*", "A", "B", "C"]
    subspecialties = ["Hip"]
    reneging_distribution = object()

    obtained = des.get_reneging_time_distributions(
        nodes=nodes,
        subspecs=subspecialties,
        reneging_distribution=reneging_distribution,
    )

    for customer_class in ["Low", "Medium", "High", "Hip"]:
        assert obtained[customer_class][2:] == [
            reneging_distribution,
            reneging_distribution,
            reneging_distribution,
        ]


def test_get_reneging_time_distributions_assigns_same_node_structure_to_all_customer_classes():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]
    reneging_distribution = object()

    obtained = des.get_reneging_time_distributions(
        nodes=nodes,
        subspecs=subspecialties,
        reneging_distribution=reneging_distribution,
    )

    expected = [
        None,
        None,
        reneging_distribution,
        reneging_distribution,
        reneging_distribution,
        reneging_distribution,
        reneging_distribution,
        reneging_distribution,
    ]

    assert obtained["Low"] == expected
    assert obtained["Medium"] == expected
    assert obtained["High"] == expected
    assert obtained["Hip"] == expected
    assert obtained["Knee"] == expected


# Network creation tests
# ----------------------

def make_minimal_network_components():
    alphabet = ["A", "B", "C"]
    alphabets = [
        alphabet,
        alphabet,
        alphabet,
    ]
    subspecialties = ["Hip"]

    activity_dict = {
        "A": 2,
        "B": 3,
        "C": 4,
    }
    subspec_dict = {
        "Hip": 0,
    }

    pdfa = make_deterministic_pdfa(
        alphabet=alphabet,
        activity_letter="A",
    )

    routing = des.PDFARouting(
        pdfa_matrices=[pdfa, pdfa, pdfa],
        alphabets=alphabets,
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    service_distributions = (
        des.make_deterministic_service_distributions(
            service_values=[
                [1.0, 2.0, 3.0],
            ],
        )
    )

    reneging_distribution = des.PreOpExpiryDist(
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    return (
        alphabets,
        subspecialties,
        service_distributions,
        routing,
        reneging_distribution,
    )


def test_get_network_constructs_valid_ciw_network():
    (
        alphabets,
        subspecialties,
        service_distributions,
        routing,
        reneging_distribution,
    ) = make_minimal_network_components()

    network = des.get_network(
        alphabets=alphabets,
        subspecialties=subspecialties,
        subspecialty_service_dists=service_distributions,
        emergency_nodes=[],
        subspecialty_class=routing,
        reneging_distribution=reneging_distribution,
        subspec_probs_low=[1.0],
        subspec_probs_medium=[1.0],
        subspec_probs_high=[1.0],
    )

    simulation = ciw.Simulation(network)

    assert simulation is not None


def test_get_network_constructs_expected_number_of_nodes():
    (
        alphabets,
        subspecialties,
        service_distributions,
        routing,
        reneging_distribution,
    ) = make_minimal_network_components()

    network = des.get_network(
        alphabets=alphabets,
        subspecialties=subspecialties,
        subspecialty_service_dists=service_distributions,
        emergency_nodes=[],
        subspecialty_class=routing,
        reneging_distribution=reneging_distribution,
        subspec_probs_low=[1.0],
        subspec_probs_medium=[1.0],
        subspec_probs_high=[1.0],
    )

    simulation = ciw.Simulation(network)

    expected_nodes = des.get_list_of_nodes(
        alphabets=alphabets,
        subspecialties=subspecialties,
    )

    assert len(simulation.nodes) - 2 == len(expected_nodes)


# Trial execution tests
# ---------------------

def make_trial_execution_test_network():
    return ciw.create_network(
        arrival_distributions=[
            ciw.dists.Deterministic(value=1.0),
        ],
        service_distributions=[
            ciw.dists.Deterministic(value=0.5),
        ],
        number_of_servers=[1],
        routing=[
            [0.0],
        ],
    )


@pytest.mark.parametrize(
    "run_time",
    [
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_run_des_trial_rejects_invalid_run_times(
    run_time,
):
    network = ciw.create_network(
        arrival_distributions=[
            ciw.dists.Deterministic(value=1.0),
        ],
        service_distributions=[
            ciw.dists.Deterministic(value=0.5),
        ],
        number_of_servers=[1],
        routing=[
            [0.0],
        ],
    )

    with pytest.raises(
        ValueError,
        match="run_time must be a finite number greater than zero",
    ):
        des.run_des_trial(
            network=network,
            run_time=run_time,
        )


def test_run_des_trial_returns_dataframe_of_simulation_records():
    network = make_trial_execution_test_network()

    obtained = des.run_des_trial(
        network=network,
        run_time=5.0,
    )

    assert isinstance(obtained, pd.DataFrame)
    assert not obtained.empty

    assert {
        "id_number",
        "node",
        "arrival_date",
        "service_start_date",
        "service_end_date",
        "exit_date",
    }.issubset(obtained.columns)


def test_run_des_trial_returns_records_with_exit_dates_within_run_time():
    network = make_trial_execution_test_network()
    run_time = 5.0

    obtained = des.run_des_trial(
        network=network,
        run_time=run_time,
    )

    completed_records = obtained.loc[
        obtained["exit_date"].notna()
    ]

    assert (completed_records["exit_date"] <= run_time).all()


# Custom ciw record tests
# -----------------------

def make_record_test_node(
    now=10.0,
    c=1,
    slotted=False,
    id_number=3,
    number_of_individuals=0,
):
    return SimpleNamespace(
        now=now,
        c=c,
        slotted=slotted,
        id_number=id_number,
        number_of_individuals=number_of_individuals,
    )


def make_record_test_individual():
    return SimpleNamespace(
        id_number=1,
        previous_class="Hip",
        original_class="Low",
        arrival_date=2.0,
        service_start_date=4.0,
        service_end_date=6.0,
        service_time=2.0,
        original_service_time=2.0,
        exit_date=7.0,
        destination=4,
        queue_size_at_arrival=3,
        queue_size_at_departure=1,
        server=SimpleNamespace(id_number=2),
        data_records=[],
        level="Low",
        referral_source="GP",
    )


def test_custom_write_individual_record_adds_completed_service_record():
    node = make_record_test_node()
    ind = make_record_test_individual()

    des.custom_write_individual_record(
        self=node,
        individual=ind,
    )

    assert len(ind.data_records) == 1

    record = ind.data_records[0]

    assert record.record_type == "service"
    assert record.id_number == 1
    assert record.customer_class == "Hip"
    assert record.original_customer_class == "Low"
    assert record.node == 3
    assert record.waiting_time == 2.0
    assert record.service_time == 2.0
    assert record.exit_date == 7.0
    assert record.server_id == 2
    assert record.level == "Low"
    assert record.referral_source == "GP"


def test_custom_write_individual_record_uses_false_server_id_for_infinite_server_node():
    node = make_record_test_node(
        c=float("inf"),
    )
    ind = make_record_test_individual()

    des.custom_write_individual_record(
        self=node,
        individual=ind,
    )

    record = ind.data_records[0]

    assert record.server_id is False


def test_custom_write_incomplete_record_returns_queue_record_for_patient_not_started_service():
    node = make_record_test_node(
        now=10.0,
    )
    ind = make_record_test_individual()
    ind.service_time = None

    record = des.custom_write_incomplete_record(
        self=node,
        individual=ind,
    )

    assert record.record_type == "incomplete"
    assert record.service_start_date is None
    assert record.waiting_time is None
    assert record.service_time is None
    assert record.service_end_date is None
    assert record.exit_date is None
    assert record.level == "Low"
    assert record.referral_source == "GP"


def test_custom_write_incomplete_record_returns_partial_service_record_for_patient_still_in_service():
    node = make_record_test_node(
        now=5.0,
    )
    ind = make_record_test_individual()
    ind.service_end_date = 6.0

    record = des.custom_write_incomplete_record(
        self=node,
        individual=ind,
    )

    assert record.record_type == "incomplete"
    assert record.service_start_date == 4.0
    assert record.waiting_time == 2.0
    assert record.service_time is None
    assert record.service_end_date is None
    assert record.exit_date is None


def test_custom_write_reneging_record_adds_renege_record():
    node = make_record_test_node()
    ind = make_record_test_individual()
    ind.exit_date = 8.0
    ind.destination = -1

    des.custom_write_reneging_record(
        self=node,
        individual=ind,
    )

    assert len(ind.data_records) == 1

    record = ind.data_records[0]

    assert record.record_type == "renege"
    assert record.id_number == 1
    assert record.node == 3
    assert record.waiting_time == 6.0
    assert record.exit_date == 8.0
    assert record.destination == -1
    assert record.level == "Low"
    assert record.referral_source == "GP"


# Minimal end-to-end tests
# ------------------------

def make_minimal_end_to_end_test_network():
    alphabet = ["A", "B", "C"]
    alphabets = [
        alphabet,
        alphabet,
        alphabet,
    ]
    subspecialties = ["Hip"]

    activity_dict = {
        "A": 3,
        "B": 4,
        "C": 5,
    }
    subspec_dict = {
        "Hip": 0,
    }

    pdfa = make_deterministic_pdfa(
        alphabet=alphabet,
        activity_letter="A",
        from_state=1,
        to_state=2,
    )

    routing = des.PDFARouting(
        pdfa_matrices=[pdfa, pdfa, pdfa],
        alphabets=alphabets,
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    service_distributions = des.make_deterministic_service_distributions(
        service_values=[
            [0.5, 0.5, 0.5],
        ],
    )

    reneging_distribution = des.PreOpExpiryDist(
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    gp_arrival_rates = [
        ciw.dists.Deterministic(value=1.0),
        None,
        None,
    ]

    return des.get_network(
        alphabets=alphabets,
        subspecialties=subspecialties,
        subspecialty_service_dists=service_distributions,
        emergency_nodes=[],
        subspecialty_class=routing,
        reneging_distribution=reneging_distribution,
        subspec_probs_low=[1.0],
        subspec_probs_medium=[1.0],
        subspec_probs_high=[1.0],
        gp_arrival_rates=gp_arrival_rates,
        other_arrival_rates=[None, None, None],
    )


def test_minimal_network_routes_gp_referral_to_activity_and_exit():
    des.apply_custom_record_changes()
    ciw.seed(1)

    network = make_minimal_end_to_end_test_network()

    records = des.run_des_trial(
        network=network,
        run_time=5.0,
    )

    assert not records.empty

    activity_records = records.loc[
        records["node"] == 3
    ]

    assert not activity_records.empty

    first_activity_record = activity_records.iloc[0]

    assert first_activity_record["customer_class"] == "Hip"
    assert first_activity_record["level"] == "Low"
    assert first_activity_record["node"] == 3
    assert first_activity_record["service_end_date"] == 1.5
    assert first_activity_record["exit_date"] == 1.5