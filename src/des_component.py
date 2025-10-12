import ciw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_activity_dictionaries(alphabet, start_value=2):
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
    activity_dict = {
        letter: idx for idx, letter in enumerate(alphabet, start=start_value)
    }
    inverted_dict = {v: k for k, v in activity_dict.items()}
    inverted_dict[1] = ""
    return activity_dict, inverted_dict


def get_list_of_nodes(alphabets, subspecialties):
    """
    Generates a list of all nodes in the DES component by their PDFA letter.
    Parameters
    ----------
    alphabets : list of lists
        A list containing the alphabets for each subspecialty and severity level.
    subspecialties : list
        A list of subspecialties (strings).
    Returns
    -------
    list
        A list of all nodes in the DES component by their PDFA letter.
    """
    nodes = ["*", "*"] + sorted(set().union(*alphabets)) * len(subspecialties)
    return nodes


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
    subspec_dict : dict
        A dictionary mapping subspecialties to their corresponding indices.
    """

    def __init__(self, pdfa_matrices, alphabets, activity_dict, subspec_dict):
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
        subspec_dict : dict
            A dictionary mapping subspecialties to their corresponding indices.
        """
        super().__init__()
        self.p_matrices = pdfa_matrices
        self.alphabets = alphabets
        self.activity_dict = activity_dict
        self.subspec_dict = subspec_dict

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
        if not hasattr(ind, "level"):
            if ind.original_class == "Low":
                ind.level = "Low"
            elif ind.original_class == "Medium":
                ind.level = "Medium"
            elif ind.original_class == "High":
                ind.level = "High"
            else:
                ind.level = ciw.dists.Pmf(
                    values=["Low", "Medium", "High"], probs=[0.5, 0.25, 0.25]
                ).sample()

        if not hasattr(ind, "route_position"):
            ind.route_position = 1  # Or initial state if different

        if ind.level == "Low":
            p_matrix = self.p_matrices[0 + (3 * self.subspec_dict[ind.customer_class])]
            alphabet = self.alphabets[0 + (3 * self.subspec_dict[ind.customer_class])]
        elif ind.level == "Medium":
            p_matrix = self.p_matrices[1 + (3 * self.subspec_dict[ind.customer_class])]
            alphabet = self.alphabets[1 + (3 * self.subspec_dict[ind.customer_class])]
        else:
            p_matrix = self.p_matrices[2 + (3 * self.subspec_dict[ind.customer_class])]
            alphabet = self.alphabets[2 + (3 * self.subspec_dict[ind.customer_class])]

        leaving_row = ind.route_position
        p_values = []
        possible_next_state = []
        possible_next_activity = []

        for letter in range(len(alphabet)):
            trans_probs = p_matrix[letter, leaving_row, :]
            if trans_probs.sum() > 0:
                p_values.append(trans_probs.sum())
                possible_next_state.append(np.where(trans_probs > 0)[0][0])
                possible_next_activity.append(letter)

        if len(p_values) > 0:
            final_prob = max(0, 1 - sum(p_values))
        else:
            final_prob = 1
        if final_prob > 0:
            p_values.append(final_prob)
            possible_next_state.append("tau")
            possible_next_activity.append(-1)

        rng = np.random.default_rng()
        next_activity = rng.choice(a=possible_next_activity, p=p_values)
        next_state = possible_next_state[possible_next_activity.index(next_activity)]

        if next_activity == -1:
            ind.route_position = -1
            return self.simulation.nodes[-1]
        else:
            next_node = self.activity_dict[alphabet[next_activity]]
            ind.route_position = next_state
            return self.simulation.nodes[
                next_node
                + (len(self.activity_dict) * self.subspec_dict[ind.customer_class])
            ]
        

def get_arrival_distributions(
    nodes,
    subspecialties,
    gp_arrival_rates=[None, None, None],
    other_arrival_rates=[None, None, None],
):
    """
    Constructs a dictionary of arrival distributions for each customer class.
    Parameters
    ----------
    nodes : list
        List of node names in the network.
    subspecialties : list
        List of subspecialty names.
    gp_arrival_rates : list, optional
        A list of arrival rates for the GP node for Low, Medium, and High severity
        levels (default is [None, None, None]).
    other_arrival_rates : list, optional
        A list of arrival rates for the other node for Low, Medium, and High severity
        levels (default is [None, None, None]).
    Returns
    -------
    dict
        A dictionary mapping each customer class to its list of arrival distributions.
    """
    n_nodes = len(nodes)
    low_arrivals = [gp_arrival_rates[0], other_arrival_rates[0]] + [None] * (
        n_nodes - 2
    )
    medium_arrivals = [gp_arrival_rates[1], other_arrival_rates[1]] + [None] * (
        n_nodes - 2
    )
    high_arrivals = [gp_arrival_rates[2], other_arrival_rates[2]] + [None] * (
        n_nodes - 2
    )

    arrival_dict = {
        "Low": low_arrivals,
        "Medium": medium_arrivals,
        "High": high_arrivals,
    }

    for subspec in subspecialties:
        arrival_dict[subspec] = [None] * n_nodes

    return arrival_dict


def get_service_distributions(nodes, subspecialties, subspecialty_services):
    """
    Constructs a dictionary of service time distributions for each customer class.
    Parameters
    ----------
    nodes : list
        List of node names in the network.
    subspecialties : list
        List of subspecialty names.
    subspecialty_services : list of lists
        A list where each element is a list of service time distributions for the
        corresponding subspecialty.
    Returns
    -------
    dict
        A dictionary mapping each customer class to its list of service time distributions.
    """
    low_services = [ciw.dists.Deterministic(value=0) for _ in nodes]
    medium_services = [ciw.dists.Deterministic(value=0) for _ in nodes]
    high_services = [ciw.dists.Deterministic(value=0) for _ in nodes]

    service_dict = {
        "Low": low_services,
        "Medium": medium_services,
        "High": high_services,
    }

    for i, subspec in enumerate(subspecialties):
        services = [ciw.dists.Deterministic(value=0) for _ in nodes]
        activities = len(subspecialty_services[i])
        services[2 + (activities * i) : 2 + activities + (activities * i)] = (
            subspecialty_services[i]
        )
        service_dict[subspec] = services

    return service_dict


def get_servers(nodes, emergency_nodes=[]):
    """
    Constructs a list of the number of servers at each node.
    Parameters
    ----------
    nodes : list
        List of node names in the network.
    emergency_nodes : list, optional
        List of nodes that should have infinite servers (default is []). These
        correspond to the emergency activities in the PDFA alphabet.
    Returns
    -------
    list
        A list containing the number of servers at each node.
    """
    servers = []
    for node in nodes:
        if node == "*":
            servers.append(float("inf"))
        elif node in emergency_nodes:
            servers.append(float("inf"))
        else:
            servers.append(1)
    return servers


def get_routing(nodes, subspecialties, subspecialty_class):
    """
    Constructs a dictionary of routing strategies for each customer class.
    Parameters
    ----------
    nodes : list
        List of node names in the network.
    subspecialties : list
        List of subspecialty names.
    subspecialty_class : ciw.routing.NodeRouting
        A routing strategy for the subspecialty customer classes.
    Returns
    -------
    dict
        A dictionary mapping each customer class to its routing strategy.
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
        subspec_routes = [subspecialty_class for _ in nodes]
        routing_dict[subspec] = ciw.routing.NetworkRouting(routers=subspec_routes)

    return routing_dict


def get_class_change_matrices(
    nodes, subspecialties, subspec_probs_low, subspec_probs_medium, subspec_probs_high
):
    """
    Constructs a list of class change matrices for each node in the network.
    Parameters
    ----------
    nodes : list
        List of node names in the network.
    subspecialties : list
        List of subspecialty names.
    subspec_probs_low : list
        A list of probabilities for each subspecialty given a Low severity level.
    subspec_probs_medium : list
        A list of probabilities for each subspecialty given a Medium severity level.
    subspec_probs_high : list
        A list of probabilities for each subspecialty given a High severity level.
    Returns
    -------
    list
        A list of class change matrices for each node in the network.
    """
    class_mat = []

    def zero_dict():
        return {key: 0.0 for key in ["Low", "Medium", "High"] + subspecialties}

    severity_levels = ["Low", "Medium", "High"]
    severity_probs = {
        "Low": subspec_probs_low,
        "Medium": subspec_probs_medium,
        "High": subspec_probs_high,
    }

    dummy = {}
    for level in severity_levels:
        dist = zero_dict()
        for i, spec in enumerate(subspecialties):
            dist[spec] = severity_probs[level][i]
        dummy[level] = dist
    for spec in subspecialties:
        dummy[spec] = {key: float(key == spec) for key in zero_dict()}
    class_mat.append(dummy)
    class_mat.append(dummy)

    identity_node = {
        key: {k: float(k == key) for k in zero_dict()} for key in zero_dict()
    }
    for _ in range(len(nodes) - 2):
        class_mat.append(identity_node.copy())

    return class_mat


def get_network(
    alphabets,
    subspecialties,
    subspecialty_service_dists,
    emergency_nodes,
    subspecialty_class,
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
    N = ciw.create_network(
        arrival_distributions=arrivals,
        service_distributions=services,
        number_of_servers=servers,
        routing=routes,
        class_change_matrices=class_changes,
    )
    return N


class PreOpExpiryDist(ciw.dists.Distribution):
    """
    A custom distribution to model the reneging time for patients waiting for
    elective surgery after a pre-operative assessment. The time is calculated
    based on the time since the last pre-operative assessment and the time of
    the last scheduled surgery. If the patient has not had a pre-operative
    assessment, they will not renege.

    Attributes
    ----------
    activity_dict : dict
        A dictionary mapping activity letters to their corresponding indices.
    subspec_dict : dict
        A dictionary mapping subspecialties to their corresponding indices.
    pre_op_letter : str
        The letter representing the pre-operative assessment activity.
    elective_surgery_letter : str
        The letter representing the elective surgery activity.
    """
    def __init__(self, activity_dict, subspec_dict, pre_op_letter, elective_surgery_letter):
        """
        Initializes the PreOpExpiryDist instance with the activity dictionary,
        subspecialty dictionary, pre-operative assessment letter, and elective
        surgery letter.
        Parameters
        ----------
        activity_dict : dict
            A dictionary mapping activity letters to their corresponding indices.
        subspec_dict : dict
            A dictionary mapping subspecialties to their corresponding indices.
        pre_op_letter : str
            The letter representing the pre-operative assessment activity.
        elective_surgery_letter : str
            The letter representing the elective surgery activity.
        """
        self.activity_dict = activity_dict
        self.subspec_dict = subspec_dict
        self.pre_op_letter = pre_op_letter
        self.elective_surgery_letter = elective_surgery_letter

    def sample(self, t, ind=None):
        """
        Samples the reneging time for an individual based on their history of
        pre-operative assessments and scheduled surgeries.
        Parameters
        ----------
        t : float
            The current time in the simulation.
        ind : ciw.Individual
            The individual for whom the reneging time is to be calculated.
        Returns
        -------
        float
            The reneging time for the individual. If the individual has not had
            a pre-operative assessment, returns infinity (indicating no reneging).
        """
        pre_op_node = self.activity_dict[self.pre_op_letter] + (len(self.activity_dict) * self.subspec_dict[ind.customer_class])
        surgery_node = self.activity_dict[self.elective_surgery_letter] + (len(self.activity_dict) * self.subspec_dict[ind.customer_class])
        pre_op_appts = [i.service_end_date for i in ind.data_records if i.node == pre_op_node]
        print(ind, pre_op_appts)
        surgical_appts = [i.service_end_date for i in ind.data_records if i.node in [surgery_node] and str(i.service_end_date) != 'nan']
        print(surgical_appts)
        if len(pre_op_appts) > 0:
            last_pre_op = pre_op_appts[-1]
            if len(surgical_appts) > 0:
                print("Yes")
                last_surgical_op = surgical_appts[-1]
                if last_pre_op > last_surgical_op:
                    return 182 - (t - last_pre_op)
                else:
                    return float("inf")
            else:
                return 182 - (t - last_pre_op)
        else:
            return float("inf")
        

class JockeyRouting(PDFARouting):
    """
    A subclass of PDFARouting to implement jockeying behaviour in a Ciw
    discrete-event simulation. This class overrides the `next_node` method to
    provide custom routing logic for individuals.
    Attributes
    ----------
    pre_op_letter : str
        The letter representing the pre-operative assessment activity.
    """
    def __init__(self, pdfa_matrix, alphabet, activity_dict, subspec_dict, pre_op_letter):
        """
        Initializes the JockeyRouting instance with a PDFA matrix, an alphabet,
        an activity dictionary, a subspecialty dictionary, and the pre-operative
        assessment letter.
        Parameters
        ----------
        pdfa_matrix : np.ndarray
            A 3D numpy array representing the PDFA transition probabilities.
        alphabet : list
            A list of activity letters corresponding to the PDFA transitions.
        activity_dict : dict
            A dictionary mapping activity letters to their corresponding indices.
        subspec_dict : dict
            A dictionary mapping subspecialties to their corresponding indices.
        pre_op_letter : str
            The letter representing the pre-operative assessment activity.
        """
        self.pre_op_letter = pre_op_letter
        super().__init__(pdfa_matrix, alphabet, activity_dict, subspec_dict)
        
    def next_node_for_jockeying(self, ind):
        """
        Determines the pre-operative assessment node for an individual.
        Parameters
        ----------
        ind : ciw.Individual
            The individual for whom the pre-operative assessment node is to be determined.
        Returns
        -------
        ciw.Node
            The pre-operative assessment node in the simulation for the individual.
        """
        pre_op_node = self.activity_dict[self.pre_op_letter] + (len(self.activity_dict) * self.subspec_dict[ind.customer_class])
        return self.simulation.nodes[pre_op_node]