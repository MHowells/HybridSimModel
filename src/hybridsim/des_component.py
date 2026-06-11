"""
Construct and run the orthopaedic discrete-event simulation.

This module contains Ciw record customisations, PDFA routing,
arrival and service distributions, reneging behaviour, and network
construction utilities.
"""

from pathlib import Path
import pickle
from collections import namedtuple
from math import isinf, nan

import ciw
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


PRE_OP_VALIDITY_DAYS = 182

DataRecord = namedtuple(
    "Record",
    [
        "id_number",
        "customer_class",
        "original_customer_class",
        "node",
        "arrival_date",
        "waiting_time",
        "service_start_date",
        "service_time",
        "service_end_date",
        "time_blocked",
        "exit_date",
        "destination",
        "queue_size_at_arrival",
        "queue_size_at_departure",
        "server_id",
        "record_type",
        "level",
        "referral_source",
    ],
)


def custom_write_individual_record(self, individual):
    """
    Write a data record for an individual when leaving a node.
    """
    if isinf(self.c) or self.slotted:
        server_id = False
    else:
        server_id = individual.server.id_number
    record = DataRecord(
        id_number=individual.id_number,
        customer_class=individual.previous_class,
        original_customer_class=individual.original_class,
        node=self.id_number,
        arrival_date=individual.arrival_date,
        waiting_time=individual.service_start_date - individual.arrival_date,
        service_start_date=individual.service_start_date,
        service_time=individual.service_end_date - individual.service_start_date,
        service_end_date=individual.service_end_date,
        time_blocked=individual.exit_date - individual.service_end_date,
        exit_date=individual.exit_date,
        destination=individual.destination,
        queue_size_at_arrival=individual.queue_size_at_arrival,
        queue_size_at_departure=individual.queue_size_at_departure,
        server_id=server_id,
        record_type="service",
        level=individual.level,
        referral_source=individual.referral_source,
    )
    individual.data_records.append(record)


def custom_write_incomplete_record(self, individual):
    """
    Write an incomplete data record for an individual
    at the end of the simulation run.
    """
    if not individual.service_time:  # Still in queue
        service_start_date = None
        waiting_time = None
        service_time = None
        service_end_date = None
    else:
        service_start_date = individual.service_start_date
        waiting_time = individual.service_start_date - individual.arrival_date
        if individual.service_end_date > self.now:  # Still in service
            service_time = None
            service_end_date = None
        else:  # Still blocked
            service_time = individual.service_time
            service_end_date = individual.service_end_date
    record = DataRecord(
        id_number=individual.id_number,
        customer_class=individual.previous_class,
        original_customer_class=individual.original_class,
        node=self.id_number,
        arrival_date=individual.arrival_date,
        waiting_time=waiting_time,
        service_start_date=service_start_date,
        service_time=service_time,
        service_end_date=service_end_date,
        time_blocked=None,
        exit_date=None,
        destination=None,
        queue_size_at_arrival=individual.queue_size_at_arrival,
        queue_size_at_departure=None,
        server_id=False,
        record_type="incomplete",
        level=individual.level,
        referral_source=individual.referral_source,
    )
    return record


def custom_write_interruption_record(self, individual, destination=nan):
    """
    Write a data record for an individual when being interrupted.
    """
    if self.slotted:
        server_id = False
    else:
        server_id = individual.server.id_number
    record = DataRecord(
        id_number=individual.id_number,
        customer_class=individual.previous_class,
        original_customer_class=individual.original_class,
        node=self.id_number,
        arrival_date=individual.arrival_date,
        waiting_time=individual.service_start_date - individual.arrival_date,
        service_start_date=individual.service_start_date,
        service_time=individual.original_service_time,
        service_end_date=nan,
        time_blocked=nan,
        exit_date=self.now,
        destination=destination,
        queue_size_at_arrival=individual.queue_size_at_arrival,
        queue_size_at_departure=individual.queue_size_at_departure,
        server_id=server_id,
        record_type="interrupted service",
        level=individual.level,
        referral_source=individual.referral_source,
    )
    individual.data_records.append(record)


def custom_write_reneging_record(self, individual):
    """
    Write a data record for an individual when reneging.
    """
    record = DataRecord(
        id_number=individual.id_number,
        customer_class=individual.previous_class,
        original_customer_class=individual.original_class,
        node=self.id_number,
        arrival_date=individual.arrival_date,
        waiting_time=individual.exit_date - individual.arrival_date,
        service_start_date=nan,
        service_time=nan,
        service_end_date=nan,
        time_blocked=nan,
        exit_date=individual.exit_date,
        destination=individual.destination,
        queue_size_at_arrival=individual.queue_size_at_arrival,
        queue_size_at_departure=individual.queue_size_at_departure,
        server_id=nan,
        record_type="renege",
        level=individual.level,
        referral_source=individual.referral_source,
    )
    individual.data_records.append(record)


def custom_write_baulking_or_rejection_record(self, individual, record_type):
    """
    Write a data record for an individual baulks.
    """
    record = DataRecord(
        id_number=individual.id_number,
        customer_class=individual.previous_class,
        original_customer_class=individual.original_class,
        node=self.id_number,
        arrival_date=self.now,
        waiting_time=nan,
        service_start_date=nan,
        service_time=nan,
        service_end_date=nan,
        time_blocked=nan,
        exit_date=self.now,
        destination=nan,
        queue_size_at_arrival=self.number_of_individuals,
        queue_size_at_departure=nan,
        server_id=nan,
        record_type=record_type,
        level=individual.level,
        referral_source=individual.referral_source,
    )
    individual.data_records.append(record)


def apply_custom_record_changes():
    """
    Apply custom Ciw record functions.

    This extends the default Ciw records so that the original customer 
    class and severity level are included.
    """
    ciw.node.Node.write_individual_record = custom_write_individual_record
    ciw.node.Node.write_incomplete_record = custom_write_incomplete_record
    ciw.node.Node.write_interruption_record = custom_write_interruption_record
    ciw.node.Node.write_reneging_record = custom_write_reneging_record
    ciw.node.Node.write_baulking_or_rejection_record = (
        custom_write_baulking_or_rejection_record
    )


def load_pdfa_and_alphabet(subspec, severity, pdfa_dir):
    """Load one PDFA matrix and its alphabet."""
    pdfa_dir = Path(pdfa_dir)
    stem = f"{subspec}_{severity}"

    pdfa_path = pdfa_dir / f"{stem}_pdfa.npy"
    alphabet_path = pdfa_dir / f"{stem}_alphabet.pkl"

    pdfa = np.load(pdfa_path)

    with alphabet_path.open("rb") as file:
        alphabet = pickle.load(file)

    return pdfa, alphabet


def load_pdfa_lookup(
    pdfa_subspec_names, 
    severity_levels, 
    pdfa_dir,
):
    """Load all PDFAs and alphabets into lookup dictionaries."""
    pdfa_lookup = {}
    alphabet_lookup = {}

    for subspec in pdfa_subspec_names:
        for severity in severity_levels:
            key = (subspec, severity)
            pdfa_lookup[key], alphabet_lookup[key] = (
                load_pdfa_and_alphabet(
                    subspec, 
                    severity, 
                    pdfa_dir,
                )
            )

    return pdfa_lookup, alphabet_lookup


def get_pdfa_lists(
    pdfa_lookup, 
    alphabet_lookup, 
    pdfa_subspec_names, 
    severity_levels,
):
    """Return PDFA and alphabet lists in the order expected by the DES routing code."""
    pdfas = [
        pdfa_lookup[(subspec, severity)]
        for subspec in pdfa_subspec_names
        for severity in severity_levels
    ]
    alphabets = [
        alphabet_lookup[(subspec, severity)]
        for subspec in pdfa_subspec_names
        for severity in severity_levels
    ]
    return pdfas, alphabets


def get_activity_dictionaries(alphabet, start_value=2):
    """
    Create forward and reverse mappings for activity letters.

    Parameters
    ----------
    alphabet : list
        Activity letters, such as ``["A", "B", "C"]``.
    start_value : int, default=2
        First integer assigned to an activity.

    Returns
    -------
    tuple[dict, dict]
        A letter-to-integer mapping and its reverse. Integer 1 is mapped
        to an empty string in the reverse mapping.
    """
    activity_dict = {
        letter: idx for idx, letter in enumerate(alphabet, start=start_value)
    }
    inverted_dict = {v: k for k, v in activity_dict.items()}
    inverted_dict[1] = ""
    return activity_dict, inverted_dict


def get_list_of_nodes(alphabets, subspecialties):
    """
    Return the network nodes identified by their PDFA letters.

    Parameters
    ----------
    alphabets : list[list]
        Activity alphabets for the subspecialties and severity levels.
    subspecialties : list
        Subspecialty names.

    Returns
    -------
    list
        Two referral nodes followed by the activity nodes for each
        subspecialty.
    """
    activity_nodes = sorted(set().union(*alphabets))
    return ["*", "*"] + activity_nodes * len(subspecialties)


def remove_activity_from_pdfa(pdfa, activity_index, tol=1e-12):
    """Return a PDFA copy with one activity removed.

    The remaining outgoing probabilities are renormalised. If removing
    the activity leaves no transitions from a state, its row remains
    empty so that its tau probability is one.
    """
    adjusted_pdfa = np.array(
        pdfa,
        dtype=float,
        copy=True,
    )
    n_states = adjusted_pdfa.shape[1]

    # Remove every transition associated with the selected activity.
    adjusted_pdfa[activity_index, :, :] = 0.0

    # Renormalise the remaining transitions from each state.
    for state in range(n_states):
        row_total = adjusted_pdfa[:, state, :].sum()

        if row_total > tol:
            adjusted_pdfa[:, state, :] /= row_total

    return adjusted_pdfa


def find_closed_pdfa_states(p_matrix, tol=1e-12):
    """
    Find closed recurrent state groups with no pathway exit.
    """
    n_states = p_matrix.shape[1]

    # Collapse over activities to get a state-to-state transition graph.
    # adjacency[i, j] = True means state i can move to state j somehow.
    adjacency = p_matrix.sum(axis=0) > tol
    graph = csr_matrix(adjacency)

    n_components, labels = connected_components(
        graph,
        directed=True,
        connection="strong"
    )

    closed_recurrents = []

    for component_id in range(n_components):
        group = np.where(labels == component_id)[0]

        # A state is only cyclic if it can transition to itself.
        if len(group) == 1:
            state = group[0]
            if not adjacency[state, state]:
                continue

        outside_group = np.setdiff1d(
            np.arange(n_states),
            group,
        )

        if len(outside_group) > 0:
            has_way_out = adjacency[np.ix_(group, outside_group)].any()
        else:
            has_way_out = False

        if has_way_out:
            continue

        # Check whether tau is possible from any state in the group.
        outgoing_probs = p_matrix[:, group, :].sum(axis=(0, 2))
        tau_probs = 1.0 - outgoing_probs

        has_tau = np.any(tau_probs > tol)

        if has_tau:
            continue

        closed_recurrents.append(group.tolist())

    return closed_recurrents


def add_tau_escape_to_states(
    pdfa, 
    states,
    tau_prob=0.05,
):
    """
    Adds an exit probability to selected PDFA states.
    """
    new_pdfa = np.array(
        pdfa, 
        dtype=float, 
        copy=True
    )

    for state in states:
        outgoing_prob = new_pdfa[:, state, :].sum()

        if outgoing_prob > 0:
            new_pdfa[:, state, :] *= (
                (1.0 - tau_prob) / outgoing_prob
            )

    return new_pdfa


class PDFARouting(ciw.routing.NodeRouting):
    """Implement PDFA-based routing in a Ciw simulation.

    The routing strategy selects the next activity and PDFA state from
    the transition probabilities associated with an individual's
    subspecialty and severity level.

    Attributes
    ----------
    p_matrices : list[numpy.ndarray]
        PDFA transition matrices, ordered by subspecialty and severity.
    alphabets : list[list]
        Activity alphabets corresponding to ``p_matrices``.
    activity_dict : dict
        Mapping from activity letters to Ciw node numbers.
    subspec_dict : dict
        Mapping from subspecialty names to integer offsets.
    pre_op_letter : str
        Activity letter for pre-operative assessment.
    elective_surgery_letter : str
        Activity letter for elective surgery.
    """

    def __init__(
        self,
        pdfa_matrices,
        alphabets,
        activity_dict,
        subspec_dict,
        pre_op_letter,
        elective_surgery_letter,
    ):
        """
        Initialise the PDFA routing strategy.

        Parameters
        ----------
        pdfa_matrices : list[numpy.ndarray]
            PDFA transition matrices.
        alphabets : list[list]
            Activity alphabets corresponding to the matrices.
        activity_dict : dict
            Mapping from activity letters to node numbers.
        subspec_dict : dict
            Mapping from subspecialties to integer offsets.
        pre_op_letter : str
            Activity letter for pre-operative assessment.
        elective_surgery_letter : str
            Activity letter for elective surgery.
        """
        super().__init__()
        self.p_matrices = pdfa_matrices
        self.alphabets = alphabets
        self.activity_dict = activity_dict
        self.subspec_dict = subspec_dict
        self.pre_op_letter = pre_op_letter
        self.elective_surgery_letter = elective_surgery_letter

    def next_node(self, ind):
        """
        Return the next node for an individual.

        Parameters
        ----------
        ind : ciw.Individual
            Individual whose route is being advanced.

        Returns
        -------
        ciw.Node
            Selected destination node, or the exit node when tau is
            selected.
        """
        if not hasattr(ind, "level"):
            if ind.original_class == "Low":
                ind.level = "Low"
            elif ind.original_class == "Medium":
                ind.level = "Medium"
            elif ind.original_class == "High":
                ind.level = "High"
            else:
                ind.level = ciw.dists.Pmf(
                    values=["Low", "Medium", "High"], 
                    probs=[0.5, 0.25, 0.25],
                ).sample()

        if not hasattr(ind, "referral_source"):
            if ind.node == 1:
                ind.referral_source = "GP"
            elif ind.node == 2:
                ind.referral_source = "Other"
            else:
                ind.referral_source = None

        if not hasattr(ind, "route_position"):
            ind.route_position = 1  # Or initial state if different

        if not hasattr(ind, "pre_op"):
            ind.pre_op = False

        subspec_index = self.subspec_dict[ind.customer_class]
        node_offset = len(self.activity_dict) * subspec_index

        pre_op_node = (
            self.activity_dict[self.pre_op_letter] + node_offset
        )
        surgery_node = (
            self.activity_dict[self.elective_surgery_letter] 
            + node_offset
        )

        if ind.node == pre_op_node:
            ind.pre_op = True

        if ind.node == surgery_node:
            ind.pre_op = False

        if ind.level == "Low":
            severity_offset = 0
        elif ind.level == "Medium":
            severity_offset = 1
        else:
            severity_offset = 2

        matrix_index = severity_offset + (3 * subspec_index)
        p_matrix = self.p_matrices[matrix_index]
        alphabet = self.alphabets[matrix_index]

        leaving_row = ind.route_position
        p_values = []
        possible_next_state = []
        possible_next_activity = []

        for letter in range(len(alphabet)):
            trans_probs = p_matrix[
                letter, 
                leaving_row, 
                :,
            ]
            if trans_probs.sum() > 0:
                p_values.append(trans_probs.sum())
                possible_next_state.append(
                    np.where(trans_probs > 0)[0][0]
                )
                possible_next_activity.append(letter)

        pre_op_index = alphabet.index(self.pre_op_letter)

        if (
            ind.pre_op 
            and pre_op_index in possible_next_activity
        ):
            transitions = list(
                zip(
                    p_values, 
                    possible_next_state, 
                    possible_next_activity,
                )
            )
            filtered = [
                (probability, state, activity)
                for probability, state, activity in transitions
                if activity != alphabet.index(self.pre_op_letter)
            ]

            if len(filtered) > 0:
                (
                    p_values, 
                    possible_next_state, 
                    possible_next_activity
                ) = zip(*filtered)

                total_probability = sum(p_values)
                p_values = [
                    probability / total_probability
                    for probability in p_values
                ]

                adjusted_pdfa = remove_activity_from_pdfa(
                    p_matrix, 
                    pre_op_index,
                )
                closed_states_check = find_closed_pdfa_states(
                    adjusted_pdfa
                )

                if len(closed_states_check) > 0:
                    for group in closed_states_check:
                        adjusted_pdfa = add_tau_escape_to_states(
                            adjusted_pdfa, 
                            group,
                        )

                    p_values = []
                    possible_next_state = []
                    possible_next_activity = []

                    for letter in range(len(alphabet)):
                        trans_probs = adjusted_pdfa[
                            letter, 
                            leaving_row, 
                            :,
                        ]
                        if trans_probs.sum() > 0:
                            p_values.append(trans_probs.sum())
                            possible_next_state.append(
                                np.where(trans_probs > 0)[0][0]
                            )
                            possible_next_activity.append(letter)

                possible_next_state = list(
                    possible_next_state
                )
                possible_next_activity = list(
                    possible_next_activity
                )
            else:
                p_values = []
                possible_next_state = []
                possible_next_activity = []

        if len(p_values) > 0:
            final_prob = max(0, 1 - sum(p_values))
        else:
            final_prob = 1

        if final_prob > 0:
            p_values.append(final_prob)
            possible_next_state.append("tau")
            possible_next_activity.append(-1)

        next_activity = ciw.rng.choice(
            a=possible_next_activity, 
            p=p_values,
        )
        activity_position = possible_next_activity.index(next_activity)
        next_state = possible_next_state[activity_position]

        if next_activity == -1:
            ind.route_position = -1
            return self.simulation.nodes[-1]
        else:
            next_node = self.activity_dict[alphabet[next_activity]]
            ind.route_position = next_state
            return self.simulation.nodes[next_node + node_offset]
        

def make_deterministic_service_distributions(service_values):
    """
    Convert numeric service-time values into Ciw Deterministic distributions.

    Parameters
    ----------
    service_values : list[list[float]]
        Nested list where each row corresponds to one subspecialty and each
        column corresponds to one activity.

    Returns
    -------
    list[list[ciw.dists.Deterministic]]
        Service distributions in the format expected by des.get_network().
    """
    return [
        [
            ciw.dists.Deterministic(value=float(value))
            for value in subspecialty_values
        ]
        for subspecialty_values in service_values
    ]


def make_gp_arrival_rates(
    lambdas, 
    t, 
    max_sample_date
):
    """
    Create Poisson interval distributions for GP arrivals.

    Parameters
    ----------
    lambdas : array-like
        Rate array with shape ``(T, 3)``. Columns represent low,
        medium, and high severity.
    t : array-like
        Time points corresponding to rows in ``lambdas``.
    max_sample_date : float
        Maximum sampling date used by Ciw.

    Returns
    -------
    list[ciw.dists.PoissonIntervals]
        One arrival distribution for each severity level.

    Raises
    ------
    ValueError
        If ``lambdas`` does not have shape ``(len(t), 3)``.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    t = np.asarray(t, dtype=float)

    if lambdas.ndim != 2:
        raise ValueError(
            "Expected lambdas to be two-dimensional with shape "
            f"(T, 3), but received {lambdas.shape}."
        )

    if lambdas.shape[0] != len(t):
        raise ValueError(
            f"Expected lambdas.shape[0] to match len(t), but got "
            f"lambdas.shape={lambdas.shape} and len(t)={len(t)}."
        )

    if lambdas.shape[1] != 3:
        raise ValueError(
            "Expected three severity columns, but received "
            f"{lambdas.shape[1]}."
        )

    return [
        ciw.dists.PoissonIntervals(
            rates=list(lambdas[:-1, severity_index]),
            endpoints=list(t[1:]),
            max_sample_date=max_sample_date,
        )
        for severity_index in range(3)
    ]


def make_other_referral_arrival_rates(
    weekday_rates,
    endpoints,
    max_sample_date,
    severity_proportions=(0.5, 0.3, 0.2),
):
    """
    Create Poisson interval distributions for other referrals.

    Parameters
    ----------
    weekday_rates : array-like
        Arrival rates for the seven weekdays.
    endpoints : array-like
        End time for each weekday interval.
    max_sample_date : float
        Maximum sampling date used by Ciw.
    severity_proportions : tuple, default=(0.5, 0.3, 0.2)
        Low-, medium-, and high-severity proportions.

    Returns
    -------
    list[ciw.dists.PoissonIntervals]
        One arrival distribution for each severity level.
    """
    return [
        ciw.dists.PoissonIntervals(
            rates=[
                rate * severity_proportion
                for rate in weekday_rates
            ],
            endpoints=list(endpoints),
            max_sample_date=max_sample_date,
        )
        for severity_proportion in severity_proportions
    ]


def get_arrival_distributions_for_nodes(
    nodes,
    subspecialties,
    gp_arrival_rates=None,
    other_arrival_rates=None,
):
    """
    Construct arrival distributions for every customer class.

    Parameters
    ----------
    nodes : list
        Network node names.
    subspecialties : list
        Subspecialty names.
    gp_arrival_rates : list, optional
        GP arrival distributions for low, medium, and high severity.
    other_arrival_rates : list, optional
        Other-referral distributions for low, medium, and high
        severity.

    Returns
    -------
    dict
        Customer classes mapped to node-specific arrival
        distributions.
    """
    if gp_arrival_rates is None:
        gp_arrival_rates = [None, None, None]
    if other_arrival_rates is None:
        other_arrival_rates = [None, None, None]

    n_nodes = len(nodes)
    inactive_nodes = [None] * (n_nodes - 2)

    arrival_dict = {
        "Low": [
            gp_arrival_rates[0],
            other_arrival_rates[0],
        ]
        + inactive_nodes.copy(),
        "Medium": [
            gp_arrival_rates[1],
            other_arrival_rates[1],
        ]
        + inactive_nodes.copy(),
        "High": [
            gp_arrival_rates[2],
            other_arrival_rates[2],
        ]
        + inactive_nodes.copy(),
    }

    for subspec in subspecialties:
        arrival_dict[subspec] = [None] * n_nodes

    return arrival_dict


def get_service_distributions_for_nodes(nodes, subspecialties, subspecialty_services):
    """
    Construct service distributions for every customer class.

    Parameters
    ----------
    nodes : list
        Network node names.
    subspecialties : list
        Subspecialty names.
    subspecialty_services : list[list]
        Service distributions for the activities in each subspecialty.

    Returns
    -------
    dict
        Customer classes mapped to node-specific service
        distributions.
    """
    service_dict = {
        severity: [
            ciw.dists.Deterministic(value=0)
            for _ in nodes
        ]
        for severity in ("Low", "Medium", "High")
    }

    for index, subspec in enumerate(subspecialties):
        services = [
            ciw.dists.Deterministic(value=0) 
            for _ in nodes
        ]
        activities = len(subspecialty_services[index])
        start = 2 + (activities * index)
        stop = start + activities

        services[start:stop] = subspecialty_services[index]
        service_dict[subspec] = services

    return service_dict


def get_servers(nodes, emergency_nodes=None):
    """
    Return the number of servers assigned to each node.

    Referral and emergency nodes receive infinitely many servers.
    Every other node receives one server.

    Parameters
    ----------
    nodes : list
        Network node names.
    emergency_nodes : list, optional
        Activity names that should have infinitely many servers.

    Returns
    -------
    list
        Number of servers at each node.
    """
    if emergency_nodes is None:
        emergency_nodes = []

    servers = []

    for node in nodes:
        if node == "*" or node in emergency_nodes:
            servers.append(float("inf"))
        else:
            servers.append(1)

    return servers


def get_routing(nodes, subspecialties, subspecialty_class):
    """
    Construct routing strategies for every customer class.

    Parameters
    ----------
    nodes : list
        Network node names.
    subspecialties : list
        Subspecialty names.
    subspecialty_class : ciw.routing.NodeRouting
        Routing strategy used by subspecialty classes.

    Returns
    -------
    dict
        Customer classes mapped to network routing strategies.
    """
    low_routes = [ciw.routing.Leave() for _ in nodes]
    medium_routes = [ciw.routing.Leave() for _ in nodes]
    high_routes = [ciw.routing.Leave() for _ in nodes]

    routing_dict = {
        "Low": ciw.routing.NetworkRouting(routers=low_routes),
        "Medium": ciw.routing.NetworkRouting(routers=medium_routes),
        "High": ciw.routing.NetworkRouting(routers=high_routes),
    }

    for subspec in subspecialties:
        subspec_routes = [
            subspecialty_class 
            for _ in nodes
        ]
        routing_dict[subspec] = ciw.routing.NetworkRouting(
            routers=subspec_routes
        )

    return routing_dict


def get_class_change_matrices(
    nodes, 
    subspecialties, 
    subspec_probs_low, 
    subspec_probs_medium, 
    subspec_probs_high,
):
    """Construct the class-change matrix for each network node.

    Parameters
    ----------
    nodes : list
        Network node names.
    subspecialties : list
        Subspecialty names.
    subspec_probs_low : list
        Subspecialty probabilities for low-severity arrivals.
    subspec_probs_medium : list
        Subspecialty probabilities for medium-severity arrivals.
    subspec_probs_high : list
        Subspecialty probabilities for high-severity arrivals.

    Returns
    -------
    list
        Class-change dictionaries for all network nodes.
    """
    class_matrices = []

    def zero_dict():
        return {
            key: 0.0 
            for key in ["Low", "Medium", "High"] + subspecialties
        }

    severity_levels = ["Low", "Medium", "High"]
    severity_probs = {
        "Low": subspec_probs_low,
        "Medium": subspec_probs_medium,
        "High": subspec_probs_high,
    }

    referral_class_changes = {}

    for level in severity_levels:
        dist = zero_dict()
        for i, spec in enumerate(subspecialties):
            dist[spec] = severity_probs[level][i]
        referral_class_changes[level] = dist

    for subspec in subspecialties:
        referral_class_changes[subspec] = {
            key: float(key == subspec) 
            for key in zero_dict()
        }

    class_matrices.append(referral_class_changes)
    class_matrices.append(referral_class_changes)

    identity_node = {
        key: {
            k: float(k == key) 
            for k in zero_dict()
        } 
        for key in zero_dict()
    }

    for _ in range(len(nodes) - 2):
        class_matrices.append(identity_node.copy())

    return class_matrices


class PreOpExpiryDist(ciw.dists.Distribution):
    """Model expiry of a pre-operative assessment.

    A patient waiting for elective surgery reneges when their most
    recent pre-operative assessment reaches its validity limit. A
    patient without a current pre-operative assessment does not renege.

    Attributes
    ----------
    activity_dict : dict
        Mapping from activity letters to node numbers.
    subspec_dict : dict
        Mapping from subspecialties to integer offsets.
    pre_op_letter : str
        Activity letter for pre-operative assessment.
    elective_surgery_letter : str
        Activity letter for elective surgery.
    """

    def __init__(
        self, 
        activity_dict, 
        subspec_dict, 
        pre_op_letter, 
        elective_surgery_letter,
    ):
        """Initialise the pre-operative expiry distribution."""
        self.activity_dict = activity_dict
        self.subspec_dict = subspec_dict
        self.pre_op_letter = pre_op_letter
        self.elective_surgery_letter = elective_surgery_letter

    def sample(self, t, ind=None):
        """Return the remaining validity time for an assessment.

        Parameters
        ----------
        t : float
            Current simulation time.
        ind : ciw.Individual
            Individual whose history is inspected.

        Returns
        -------
        float
            Time until the assessment expires, or infinity when the
            individual has no current assessment.
        """
        subspec_offset = (
            len(self.activity_dict)
            * self.subspec_dict[ind.customer_class]
        )
        pre_op_node = (
            self.activity_dict[self.pre_op_letter]
            + subspec_offset
        )
        surgery_node = (
            self.activity_dict[self.elective_surgery_letter]
            + subspec_offset
        )

        pre_op_appts = [
            i.service_end_date 
            for i in ind.data_records 
            if i.node == pre_op_node
        ]
        surgical_appts = [
            i.service_end_date
            for i in ind.data_records
            if (
                i.node in [surgery_node] 
                and str(i.service_end_date) != "nan"
            )
        ]
        
        if len(pre_op_appts) > 0:
            last_pre_op = pre_op_appts[-1]
            if len(surgical_appts) > 0:
                last_surgical_op = surgical_appts[-1]
                if last_pre_op > last_surgical_op:
                    return max(
                        0.0, 
                        PRE_OP_VALIDITY_DAYS - (t - last_pre_op)
                    )
                else:
                    return float("inf")
            else:
                return max(
                    0.0, 
                    PRE_OP_VALIDITY_DAYS - (t - last_pre_op)
                )
        else:
            return float("inf")


class JockeyRouting(PDFARouting):
    """
    Extend PDFA routing with support for queue jockeying.

    The inherited :meth:`next_node` method provides normal routing.
    :meth:`next_node_for_jockeying` sends an individual back to their
    subspecialty's pre-operative assessment node.
    """

    def __init__(
        self,
        pdfa_matrices,
        alphabets,
        activity_dict,
        subspec_dict,
        pre_op_letter,
        elective_surgery_letter,
    ):
        """
        Initializes the JockeyRouting instance.
        """
        self.pre_op_letter = pre_op_letter
        super().__init__(
            pdfa_matrices,
            alphabets,
            activity_dict,
            subspec_dict,
            pre_op_letter,
            elective_surgery_letter,
        )

    def next_node_for_jockeying(self, ind):
        """Return the individual's pre-operative assessment node.

        Parameters
        ----------
        ind : ciw.Individual
            Individual that is moving between queues.

        Returns
        -------
        ciw.Node
            Pre-operative assessment node for the individual's
            subspecialty.
        """
        pre_op_node = self.activity_dict[self.pre_op_letter] + (
            len(self.activity_dict) 
            * self.subspec_dict[ind.customer_class]
        )
        return self.simulation.nodes[pre_op_node]


def get_reneging_time_distributions(
    nodes, 
    subspecs, 
    reneging_distribution,
):
    """Construct reneging distributions for every customer class.

    Parameters
    ----------
    nodes : list
        Network node names.
    subspecs : list
        Subspecialty names.
    reneging_distribution : ciw.dists.Distribution
        Distribution used at activity nodes.

    Returns
    -------
    dict
        Customer classes mapped to node-specific reneging
        distributions.
    """
    reneging_distributions = (
        [None, None] 
        + [reneging_distribution] * (len(nodes) - 2)
    )
    reneging_dict = {
        "Low": reneging_distributions,
        "Medium": reneging_distributions,
        "High": reneging_distributions,
    }

    for subspec in subspecs:
        reneging_dict[subspec] = reneging_distributions

    return reneging_dict


def get_network(
    alphabets,
    subspecialties,
    subspecialty_service_dists,
    emergency_nodes,
    subspecialty_class,
    reneging_class,
    subspec_probs_low,
    subspec_probs_medium,
    subspec_probs_high,
    gp_arrival_rates=[None, None, None],
    other_arrival_rates=[None, None, None],
):
    """
    Constructs a Ciw network object based on the provided parameters.
    Parameters
    ----------
    alphabets : list of lists
        A list where each element is a list of activity letters for the corresponding
        subspecialty and severity level.
    subspecialties : list
        A list of subspecialty names.
    subspecialty_service_dists : list of lists
        A list where each element is a list of service time distributions for the
        corresponding subspecialty.
    emergency_nodes : list
        A list of nodes that should have infinite servers (these correspond to the
        emergency activities in the PDFA alphabet).
    subspecialty_class : ciw.routing.NodeRouting
        A routing strategy for the subspecialty customer classes.
    reneging_class : ciw.dists.Distribution
        A custom distribution class for modeling reneging times.
    subspec_probs_low : list
        A list of probabilities for each subspecialty given a Low severity level.
    subspec_probs_medium : list
        A list of probabilities for each subspecialty given a Medium severity level.
    subspec_probs_high : list
        A list of probabilities for each subspecialty given a High severity level.
    gp_arrival_rates : list, optional
        A list of arrival rates for the GP node for Low, Medium, and High severity
        levels (default is [None, None, None]).
    other_arrival_rates : list, optional
        A list of arrival rates for the other node for Low, Medium, and High severity
        levels (default is [None, None, None]).
    Returns
    -------
    ciw.Network
        A Ciw network object representing the discrete-event simulation model.
    """
    nodes = get_list_of_nodes(alphabets, subspecialties)
    arrivals = get_arrival_distributions(
        nodes, subspecialties, gp_arrival_rates, other_arrival_rates
    )
    services = get_service_distributions(
        nodes, subspecialties, subspecialty_service_dists
    )
    servers = get_servers(nodes, emergency_nodes)
    routes = get_routing(nodes, subspecialties, subspecialty_class)
    class_changes = get_class_change_matrices(
        nodes,
        subspecialties,
        subspec_probs_low,
        subspec_probs_medium,
        subspec_probs_high,
    )
    reneging_dists = get_reneging_time_distributions(
        nodes, subspecialties, reneging_class
    )
    N = ciw.create_network(
        arrival_distributions=arrivals,
        service_distributions=services,
        number_of_servers=servers,
        routing=routes,
        class_change_matrices=class_changes,
        reneging_time_distributions=reneging_dists,
    )
    return N
