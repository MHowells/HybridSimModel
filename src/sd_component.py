import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def strict_priority_gatekeeping(threshold):
    """
    Strict severity-priority gatekeeping where referral capacity is a 
    proportion of the total presenting demand.

    Patients are referred in severity order (e.g. high, medium, low),
    where highest-priority demand is referred first, up to the total 
    referral capacity. Lower-priority demand is only referred if there
    is remaining capacity after referring all higher-priority demand.

    Parameters
    ----------
    threshold : float in [0, 1]
        Proportion of the total presenting demand that can be referred.

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculate referral flows under a strict severity-priority policy,
        where the total referral capacity is a proportion of total presenting
        demand.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework, but unused.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input (unused here, but included for compatibility).

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (n_groups,) for scalar input
            - shape (n_groups, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)

        if stocks.ndim == 1:
            demand = presenting_proportion * stocks
            total_capacity = threshold * demand.sum()
            remaining_capacity = total_capacity
            lambdas = np.zeros(len(stocks))

            for i, d in enumerate(demand):
                allowed = min(d, remaining_capacity)
                lambdas[i] = allowed
                remaining_capacity = max(remaining_capacity - allowed, 0.0)

            return lambdas

        elif stocks.ndim == 2:
            demand = presenting_proportion * stocks
            n_groups, n_times = demand.shape
            lambdas = np.zeros((n_groups, n_times))
            total_capacity = threshold * demand.sum(axis=0)
            remaining_capacity = total_capacity.copy()

            for i in range(n_groups):
                allowed = np.minimum(demand[i], remaining_capacity)
                lambdas[i] = allowed
                remaining_capacity = np.maximum(remaining_capacity - allowed, 0.0)

            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def fixed_capacity_strict_gatekeeping(capacity):
    """
    Strict severity-priority gatekeeping with a fixed referral capacity
    per time step.

    Patients are referred in severity order (e.g. high, medium, low),
    where highest-priority demand is referred first, up to the total 
    referral capacity. Lower-priority demand is only referred if there
    is remaining capacity after referring all higher-priority demand.

    Parameters
    ----------
    capacity : float or int
        Maximum number of presenting patients who can be referred
        per time step.

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculate referral flows under a fixed-capacity strict
        severity-priority policy.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework, but unused.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input (unused here, but included for compatibility).

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (3,) for scalar input
            - shape (3, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)

        if stocks.ndim == 1:
            remaining_capacity = capacity
            lambdas = np.zeros(len(stocks))

            for i, stock in enumerate(stocks):
                demand = presenting_proportion * stock
                allowed = min(demand, remaining_capacity)
                lambdas[i] = allowed
                remaining_capacity = max(remaining_capacity - allowed, 0.0)

            return lambdas

        elif stocks.ndim == 2:
            n_groups, n_times = stocks.shape
            lambdas = np.zeros((n_groups, n_times))
            remaining_capacity = np.full(n_times, capacity, dtype=float)

            for i in range(n_groups):
                demand = presenting_proportion * stocks[i]
                allowed = np.minimum(demand, remaining_capacity)
                lambdas[i] = allowed
                remaining_capacity = np.maximum(remaining_capacity - allowed, 0.0)

            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def fixed_capacity_proportional_gatekeeping(capacity):
    """
    Fixed-capacity gatekeeping where referral capacity is allocated 
    proportionally across severity groups. Each severity group receives 
    a share of the available capacity based on its share of total 
    presenting demand.

    Parameters
    ----------
    capacity : float or int
        Maximum number of presenting patients who can be referred per time step.

    Returns
    -------
    function
        Function to compute lambda values for each stock.
    """

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculates lambda values by allocating a fixed referral capacity
        proportionally across severity groups based on presenting demand.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework, but unused.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input (unused here, but included for compatibility).

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (3,) for scalar input
            - shape (3, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)

        if stocks.ndim == 1:
            demand = presenting_proportion * stocks
            total_demand = demand.sum()

            if total_demand == 0:
                lambdas = np.zeros(len(stocks))
            else:
                lambdas = capacity * demand / total_demand
                lambdas = np.minimum(lambdas, demand)

            return lambdas

        elif stocks.ndim == 2:
            demand = presenting_proportion * stocks
            total_demand = demand.sum(axis=0)
            lambdas = np.zeros_like(demand)

            positive_demand = total_demand > 0
            lambdas[:, positive_demand] = (
                capacity * demand[:, positive_demand] / total_demand[positive_demand]
            )

            lambdas = np.minimum(lambdas, demand)

            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def seasonal_capacity_gatekeeping(baseline=8, amplitude=2, period=365, phase_shift=0):
    """
    Strict severity-priority gatekeeping with a referral capacity that
    varies seasonally according to a sinusoidal function.

    Patients are referred in severity order (e.g. high, medium, low),
    where highest-priority demand is referred first, up to the total 
    referral capacity. Lower-priority demand is only referred if there
    is remaining capacity after referring all higher-priority demand.

    Parameters
    ----------
    baseline : float
        Average referral capacity per time step.
    amplitude : float
        Amplitude of the seasonal variation in referral capacity.
    period : float
        Length of the seasonal cycle (e.g. 365 for yearly seasonality).
    phase_shift : float
        Horizontal shift of the seasonal cycle.

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculate referral flows under a seasonally varying fixed-capacity
        strict severity-priority policy.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input used to compute seasonal capacity.

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (n_groups,) for scalar input
            - shape (n_groups, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)
        t = np.asarray(t)

        if stocks.ndim == 1:
            capacity = max(
                0.0,
                baseline + amplitude * np.sin(2 * np.pi * (t + phase_shift) / period),
            )

            lambdas = np.zeros(len(stocks))

            if capacity == 0:
                return lambdas

            remaining_capacity = capacity
            for i, stock in enumerate(stocks):
                demand = presenting_proportion * stock
                allowed = min(demand, remaining_capacity)
                lambdas[i] = allowed
                remaining_capacity = max(remaining_capacity - allowed, 0.0)

            return lambdas

        elif stocks.ndim == 2:
            n_groups, n_times = stocks.shape
            lambdas = np.zeros((n_groups, n_times))

            capacities = np.maximum(
                0.0,
                baseline + amplitude * np.sin(2 * np.pi * (t + phase_shift) / period),
            )

            for i in range(n_times):
                remaining_capacity = capacities[i]

                if remaining_capacity == 0:
                    continue

                for j in range(n_groups):
                    demand = presenting_proportion * stocks[j, i]
                    allowed = min(demand, remaining_capacity)
                    lambdas[j, i] = allowed
                    remaining_capacity = max(remaining_capacity - allowed, 0.0)

            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def equal_access_proportion_gatekeeping(access_proportion):
    """
    Gatekeeping policy where the same referral proportion is applied to 
    each severity group.

    Each stock is treated independently, so there is no shared capacity 
    limit and no priority ordering. This represents equal proportional 
    access across all severity groups.

    Parameters
    ----------
    access_proportion : float in [0, 1]
        Proportion of the total presenting demand that can be referred at each
        time step.
    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculate referral flows under a proportional access policy, where the
        same proportion of each severity stock that presents to primary care is
        referred, up to a maximum total referral capacity.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework, but unused.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input (unused here, but included for compatibility).
        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (n_groups,) for scalar input
            - shape (n_groups, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)

        if stocks.ndim == 1:
            lambdas = presenting_proportion * access_proportion * stocks
            return lambdas

        elif stocks.ndim == 2:
            lambdas = presenting_proportion * access_proportion * stocks
            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def severity_specific_access_gatekeeping(proportions):
    """
    Gatekeeping function where each severity group has its own referral
    proportion. There is no shared capacity limit and no sequential 
    priority allocation.

    Parameters
    ----------
    proportions : list of floats
        Proportions for each stock, e.g. for high, medium, low severity.

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """
    proportions = np.array(proportions)

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculates lambda values based on fixed proportions for each stock.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework, but unused.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float
            Time input (unused here, but included for compatibility).

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (n_groups,) for scalar input
            - shape (n_groups, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)

        if stocks.ndim == 1:
            lambdas = presenting_proportion * proportions * stocks
            return lambdas

        elif stocks.ndim == 2:
            lambdas = presenting_proportion * proportions[:, None] * stocks
            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def partial_priority_gatekeeping(capacity, priority_relaxation):
    """
    Fixed-capacity gatekeeping that blends strict-severity priority and 
    proportional allocation.

    A priority_relaxation value of 0 gives fully strict-priority 
    allocation, while a value of 1 gives fully proportional allocation. 
    Intermediate values give softer priority rules.

    Parameters
    ----------
    capacity : float or int
        Maximum number of presenting patients who can be referred
        per time step.
    priority_relaxation : float in [0, 1]
        Degree of relaxation in priority-based allocation.

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculate referral flows under a blended-allocation gatekeeping policy.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework, but unused.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input (unused here, but included for compatibility).

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (n_groups,) for scalar input
            - shape (n_groups, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)

        if stocks.ndim == 1:
            demand = presenting_proportion * stocks
            n_groups = len(stocks)
            strict_lambdas = np.zeros(n_groups)
            remaining_capacity = capacity

            for i in range(n_groups):
                allowed = min(demand[i], remaining_capacity)
                strict_lambdas[i] = allowed
                remaining_capacity = max(remaining_capacity - allowed, 0.0)

            total_demand = demand.sum()
            if total_demand == 0:
                proportional_lambdas = np.zeros(n_groups)
            else:
                proportional_lambdas = capacity * demand / total_demand
                proportional_lambdas = np.minimum(proportional_lambdas, demand)

            lambdas = (
                1 - priority_relaxation
            ) * strict_lambdas + priority_relaxation * proportional_lambdas

            return lambdas

        elif stocks.ndim == 2:
            demand = presenting_proportion * stocks
            n_groups, n_times = demand.shape
            lambdas = np.zeros((n_groups, n_times))

            for k in range(n_times):
                d = demand[:, k]
                strict_lambdas = np.zeros(n_groups)
                remaining_capacity = capacity

                for i in range(n_groups):
                    allowed = min(d[i], remaining_capacity)
                    strict_lambdas[i] = allowed
                    remaining_capacity = max(remaining_capacity - allowed, 0.0)

                total_demand = d.sum()
                if total_demand == 0:
                    proportional_lambdas = np.zeros(n_groups)
                else:
                    proportional_lambdas = capacity * d / total_demand
                    proportional_lambdas = np.minimum(proportional_lambdas, d)

                lambdas[:, k] = (
                    1 - priority_relaxation
                ) * strict_lambdas + priority_relaxation * proportional_lambdas

            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def severity_responsive_gatekeeping(
    severity_threshold,
    low_severity_capacity,
    high_severity_capacity,
):
    """
    Strict severity-priority gatekeeping with referral capacity that 
    changes based on the share of presenting demand from the 
    highest-severity group.

    If the high-severity share is above the specified threshold, the 
    higher capacity is used. Otherwise, the lower capacity is used.

    Parameters
    ----------
    severity_threshold : float in [0, 1]
        Threshold for the proportion of total presenting demand attributable
        to the highest-severity group. If the high-severity proportion is
        at or above this threshold, the policy switches to the
        high_severity_capacity.
    low_severity_capacity : float
        Referral capacity per time step when the high-severity proportion
        is below the threshold.
    high_severity_capacity : float
        Referral capacity per time step when the high-severity proportion
        is at or above the threshold.

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculate referral flows under a severity-responsive strict
        severity-priority policy.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group, ordered by priority
            (e.g. high, medium, low).
        population : float or array
            Included for compatibility with the SD framework, but unused.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input (unused here, but included for compatibility).

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (n_groups,) for scalar input
            - shape (n_groups, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)

        if stocks.ndim == 1:
            demand = presenting_proportion * stocks
            total_demand = demand.sum()

            if total_demand == 0:
                return np.zeros(len(stocks))

            high_severity_proportion = demand[0] / total_demand

            if high_severity_proportion >= severity_threshold:
                capacity = high_severity_capacity
            else:
                capacity = low_severity_capacity

            lambdas = np.zeros(len(stocks))
            remaining_capacity = capacity

            for i in range(len(stocks)):
                allowed = min(demand[i], remaining_capacity)
                lambdas[i] = allowed
                remaining_capacity = max(remaining_capacity - allowed, 0.0)

            return lambdas

        elif stocks.ndim == 2:
            demand = presenting_proportion * stocks
            n_groups, n_times = demand.shape
            lambdas = np.zeros((n_groups, n_times))

            for k in range(n_times):
                d = demand[:, k]
                total_demand = d.sum()

                if total_demand == 0:
                    continue

                high_severity_proportion = d[0] / total_demand

                if high_severity_proportion >= severity_threshold:
                    capacity = high_severity_capacity
                else:
                    capacity = low_severity_capacity

                remaining_capacity = capacity
                for i in range(n_groups):
                    allowed = min(d[i], remaining_capacity)
                    lambdas[i, k] = allowed
                    remaining_capacity = max(remaining_capacity - allowed, 0.0)

            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def time_phased_gatekeeping(change_times, gatekeeping_policies):
    """
    Switches between multiple gatekeeping policies at specified change 
    times.

    Parameters
    ----------
    change_times : list or array of floats
        Sorted times at which the gatekeeping policy changes.
        If there are n policies, there must be n - 1 change times.
    gatekeeping_policies : list of functions
        Gatekeeping policies to apply in each phase.

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """
    change_times = np.asarray(change_times, dtype=float)

    if len(gatekeeping_policies) != len(change_times) + 1:
        raise ValueError(
            "There must be exactly one more gatekeeping policy than change times."
        )

    if np.any(np.diff(change_times) < 0):
        raise ValueError("change_times must be sorted in non-decreasing order.")

    def gatekeeping_function(stocks, population, presenting_proportion, t):
        """
        Calculate referral flows under a multi-phase gatekeeping policy.

        Parameters
        ----------
        stocks : list/array of scalars or arrays
            Stock levels for each severity group.
        population : float or array
            Total population at time t.
        presenting_proportion : float in [0, 1]
            Proportion of each stock presenting to primary care.
        t : float or array
            Time input used to determine which policy phase applies.

        Returns
        -------
        np.ndarray
            Referral flows for each stock, either as:
            - shape (n_groups,) for scalar input
            - shape (n_groups, T) for time-series input
        """
        stocks = np.array(stocks, dtype=float)
        t = np.asarray(t)

        def get_policy_index(time_point):
            policy_idx = 0
            for change_time in change_times:
                if time_point < change_time:
                    break
                policy_idx += 1
            return policy_idx

        if stocks.ndim == 1:
            policy_idx = get_policy_index(t)
            selected_policy = gatekeeping_policies[policy_idx]

            return np.asarray(
                selected_policy(
                    stocks=stocks,
                    population=population,
                    presenting_proportion=presenting_proportion,
                    t=t,
                ),
                dtype=float,
            )

        elif stocks.ndim == 2:
            n_groups, n_times = stocks.shape
            lambdas = np.zeros((n_groups, n_times))

            for k in range(n_times):
                stocks_k = stocks[:, k]
                population_k = population[k] if np.ndim(population) > 0 else population
                t_k = t[k]

                policy_idx = get_policy_index(t_k)
                selected_policy = gatekeeping_policies[policy_idx]

                lambdas[:, k] = np.asarray(
                    selected_policy(
                        stocks=stocks_k,
                        population=population_k,
                        presenting_proportion=presenting_proportion,
                        t=t_k,
                    ),
                    dtype=float,
                )

            return lambdas

        else:
            raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


class SD:
    """
    A class to hold the SD component.
    """

    def __init__(
        self,
        population_function,
        initial_unwell_proportion,
        unwell_splits,
        gatekeeping_function,
        presenting_proportion,
        deterioration_function,
        incidence_function,
        recovery_function,
    ):
        """
        Initialised the parameters for the SD component

        Parameters
        ----------
        population_function : function
            A function that returns the population size at a given time.
        initial_unwell_proportion : a positive float <= 1
            proportion of the initial population that is unwell
        unwell_splits : a tuple of three floats that sum to 1
            representing the proportions of the unwell population in each stock
        gatekeeping_function : a function
            function to calculate lambda values for each stock
        presenting_proportion : a positive float <= 1
            proportion of patients presenting for treatment
        deterioration_function : a function
            function to calculate the rate at which patients deteriorate
        incidence_function : a function
            function to calculate the rate at which new patients enter the system
        recovery_function : a function
            function to calculate the rate at which patients recover
        """
        w = unwell_splits
        self.initial_population = population_function
        unwell_pop = self.initial_population(t=0) * initial_unwell_proportion
        self.P = [unwell_pop * w[0], unwell_pop * w[1], unwell_pop * w[2]]
        self.presenting_proportion = presenting_proportion
        self.gatekeeping_function = gatekeeping_function
        self.deterioration_rate = deterioration_function
        self.incidence_rate = incidence_function
        self.recovery_rate = recovery_function
        self.time = np.array([0])
        self.lambdas = None

    def differential_equations(
        self,
        y,
        time_domain,
    ):
        """
        Defines the system of differential equations that describe the
        population model.

        Parameters
        ----------
        y : a tuple of three integers
            representing the populations in each stock
        time_domain : a float
            representing the time domain for the simulation

        Returns
        -------
        tuple
            dP_onedt, dP_twodt, dP_threedt : floats
        """
        P_one, P_two, P_three = y
        N_current = P_one + P_two + P_three
        all_stocks = [P_one, P_two, P_three]

        if N_current == 0:
            return 0, 0, 0

        current_population = max(self.initial_population(t=time_domain), 0)

        lambdas = self.gatekeeping_function(
            stocks=all_stocks,
            population=N_current,
            presenting_proportion=self.presenting_proportion,
            t=time_domain,
        )

        dP_onedt = (self.deterioration_rate(t=time_domain) * P_two) - lambdas[0]
        dP_twodt = (
            (self.deterioration_rate(t=time_domain) * P_three)
            - (self.deterioration_rate(t=time_domain) * P_two)
            - lambdas[1]
        )
        dP_threedt = (
            -lambdas[2]
            - (self.deterioration_rate(t=time_domain) * P_three)
            - (self.recovery_rate(t=time_domain, stock_size=N_current))
            + (self.incidence_rate(t=time_domain, population_size=current_population))
        )
        return dP_onedt, dP_twodt, dP_threedt

    def solve(
        self,
        t,
    ):
        """
        Solves the differential equations from the time of the previous event
        to time t.
        """
        # Solve the SD over the relevant time domain
        y = self.P
        results = odeint(
            self.differential_equations,
            y,
            t,
        )

        P1, P2, P3 = results.T
        self.P[0] = np.append(self.P[0], P1[1:])
        self.P[1] = np.append(self.P[1], P2[1:])
        self.P[2] = np.append(self.P[2], P3[1:])

        # Extract the lambdas from the results
        self.lambdas = self.gatekeeping_function(
            stocks=[P1, P2, P3],
            population=P1 + P2 + P3,
            presenting_proportion=self.presenting_proportion,
            t=t,
        )


def get_time_dependent_population_size(population_sizes, durations=np.NaN):
    """
    Returns a function that provides a time-dependent population size based on
    custom durations and population sizes.

    Parameters
    ----------
    population_sizes : list of int/float
        A list of population sizes.
    durations : list of int/float
        A list of time durations (in days, for example) that each population size lasts.

    Returns
    -------
    function
        A function that takes time t and returns the corresponding population size.
    """
    if isinstance(population_sizes, (int, float)):
        population_sizes = [population_sizes]
    if isinstance(durations, (int, float)):
        durations = [durations]

    if len(population_sizes) != len(durations):
        raise ValueError("The lengths of population_sizes and durations must match.")

    change_points = [0]
    for d in durations:
        change_points.append(change_points[-1] + d)

    def population_function(t):
        """
        Returns the population size at time t.

        Parameters
        ----------
        t : float
            The time at which to get the population size.

        Returns
        -------
        int or float
            The population size at time t.
        """
        for i in range(len(durations)):
            if change_points[i] <= t < change_points[i + 1]:
                return population_sizes[i]
        return population_sizes[-1]

    return population_function


def get_time_dependent_incidence_rate(incidence_proportions, durations=np.NaN):
    """
    Returns a function that calculates the incidence rate based on time
    and population size, using a list of incidence proportions and durations.

    Parameters
    ----------
    incidence_proportions : list of float
        Incidence proportions (e.g., cases per person per period).
    durations : list of int/float
        Durations for which each incidence proportion applies.

    Returns
    -------
    function
        A function that takes time t and population size, and returns
        the incidence rate at time t.
    """
    if isinstance(incidence_proportions, (int, float)):
        incidence_proportions = [incidence_proportions]
    if isinstance(durations, (int, float)):
        durations = [durations]

    if len(incidence_proportions) != len(durations):
        raise ValueError(
            "The lengths of incidence_proportions and durations must match."
        )

    change_points = [0]
    for d in durations:
        change_points.append(change_points[-1] + d)

    def incidence_function(t, population_size):
        """
        Calculates the incidence rate at time t given a population size.

        Parameters
        ----------
        t : float
            The time at which to calculate the incidence.
        population_size : int or float
            The population size at time t.

        Returns
        -------
        float
            The incidence rate at time t.
        """
        if population_size == 0:
            return 0.0
        for i in range(len(durations)):
            if change_points[i] <= t < change_points[i + 1]:
                return incidence_proportions[i] * population_size
        return incidence_proportions[-1] * population_size

    return incidence_function


def get_time_dependent_recovery_rate(recovery_proportions, durations=np.NaN):
    """
    Returns a function that calculates the recovery rate based on time
    and stock size, using a list of recovery proportions and durations.

    Parameters
    ----------
    recovery_proportions : list of float
        Recovery proportions (e.g., proportion recovered per period).
    durations : list of int/float
        Durations for which each recovery proportion applies.

    Returns
    -------
    function
        A function that takes time t and stock size, and returns
        the recovery rate at time t.
    """
    if isinstance(recovery_proportions, (int, float)):
        recovery_proportions = [recovery_proportions]
    if isinstance(durations, (int, float)):
        durations = [durations]

    if len(recovery_proportions) != len(durations):
        raise ValueError(
            "The lengths of recovery_proportions and durations must match."
        )

    change_points = [0]
    for d in durations:
        change_points.append(change_points[-1] + d)

    def recovery_function(t, stock_size):
        """
        Calculates the recovery rate based on time and stock size.

        Parameters
        ----------
        t : float
            The time at which to calculate the recovery rate.
        stock_size : int or float
            The stock size at time t.

        Returns
        -------
        float
            The recovery rate at time t.
        """
        if stock_size == 0:
            return 0.0
        for i in range(len(durations)):
            if change_points[i] <= t < change_points[i + 1]:
                return recovery_proportions[i] * stock_size
        return recovery_proportions[-1] * stock_size

    return recovery_function


def add_constant_lambda_warmup(
    lambdas, ts, warmup_days=365 * 2, value="initial", shift_time=True
):
    """
    Adds a warm-up period to the lambda values, where the lambda values
    are constant and equal to the initial lambda values (or a specified
    value) during the warm-up period.

    Parameters
    ----------
    lambdas : array-like
        Array of lambda values with shape (n_groups, T).
    ts : array-like
        Time vector corresponding to the lambda values.
    warmup_days : int or float
        Duration of the warm-up period in the same time units as ts
        (default is 2 years).
    value : "initial" or array-like
        If "initial", the warm-up lambda values will be set to the
        initial lambda values (i.e., lambdas[:, 0]).
        If an array-like is provided, it should contain the lambda
        values to use during the warm-up period for each group (shape
        should be (n_groups,)).
    shift_time : bool
        If True, the time vector will be shifted so that the warm-up
        period starts at t=0. If False, the time vector will include
        negative values for the warm-up period (default is True).

    Returns
    -------
    lambdas_with_warmup : np.ndarray
        Lambda values with the warm-up period added, shape
        (n_groups, T + warmup_points).
    ts_with_warmup : np.ndarray
        Time vector corresponding to the lambda values with warm-up,
        shape (T + warmup_points,).
    """
    lambdas = np.asarray(lambdas)
    ts = np.asarray(ts)
    dt = ts[1] - ts[0]
    warmup_points = int(round(warmup_days / dt))

    if value == "initial":
        warmup_values = lambdas[:, [0]]
    else:
        warmup_values = np.asarray(value, dtype=float).reshape(-1, 1)
        if warmup_values.shape[0] != lambdas.shape[0]:
            raise ValueError(
                f"Expected {lambdas.shape[0]} warm-up values, "
                f"but received {warmup_values.shape[0]}."
            )

    warmup_lambdas = np.repeat(warmup_values, warmup_points, axis=1)
    lambdas_with_warmup = np.concatenate([warmup_lambdas, lambdas], axis=1)

    if shift_time:
        ts_with_warmup = np.arange(lambdas_with_warmup.shape[1]) * dt
    else:
        ts_with_warmup = np.concatenate(
            [np.linspace(-warmup_days, -dt, warmup_points), ts]
        )

    return lambdas_with_warmup, ts_with_warmup


def plot_stocks_over_time(
    stocks,
    t,
    ylim=None,
    title="Stock Size Over Time",
    filename=None,
    show=True,
):
    """
    Plots the stock sizes over time.

    Parameters
    ----------
    stocks : list of arrays
        List of 3 arrays representing the stock sizes P1, P2, P3.
    t : array-like
        Time vector.
    ylim : tuple, optional
        Y-axis limits for the plot (default is None).
    title : str
        Title of the plot.
    filename : str, optional
        Filename to save the plot (default is None, which does not save the plot).
    show : bool, optional
        Whether to display the plot (default is True).

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
            The figure and axes objects for further customization if needed.
    """
    if len(stocks) != 3:
        raise ValueError("stocks must contain exactly 3 arrays.")
    if any(len(stock) != len(t) for stock in stocks):
        raise ValueError("Each stock array must have the same length as t.")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#FFC107", "#1E88E5", "#D81B60"]
    labels = ["$P_1$ (High)", "$P_2$ (Medium)", "$P_3$ (Low)"]
    for stock, color, label in zip(stocks, colors, labels):
        ax.plot(t, stock, label=label, color=color)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (days)", fontsize=14)
    ax.set_ylabel("Population in stock", fontsize=14)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    if show:
        plt.show()

    return fig, ax


def plot_stacked_stocks_over_time(
    stocks,
    t,
    overlay_values=None,
    overlay_label="Overlay",
    overlay_color="black",
    overlay_linestyle="--",
    ylim=None,
    title="Stacked Chart of SD Stocks Over Time",
    filename=None,
    show=True,
):
    """
    Plot a stacked area chart of the three SD stocks over time, with an
    optional overlay line.

    Parameters
    ----------
    stocks : list of arrays
        List of 3 arrays representing the stock sizes P1, P2, P3.
    t : array-like
        Time vector.
    overlay_values : array-like, optional
        Optional line to overlay on the plot. This should be on a scale
        comparable to the stock sizes.
    overlay_label : str, optional
        Label for the overlay line.
    overlay_color : str, optional
        Colour for the overlay line.
    overlay_linestyle : str, optional
        Line style for the overlay line.
    ylim : tuple, optional
        Y-axis limits for the plot (default is None).
    title : str
        Title of the plot.
    filename : str, optional
        Filename to save the plot (default is None, which does not save the plot).
    show : bool, optional
        Whether to display the plot (default is True).

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        The figure and axes objects for further customization if needed.
    """
    if len(stocks) != 3:
        raise ValueError("stocks must contain exactly 3 arrays.")
    if any(len(stock) != len(t) for stock in stocks):
        raise ValueError("Each stock array must have the same length as t.")
    if overlay_values is not None and len(overlay_values) != len(t):
        raise ValueError("overlay_values must have the same length as t.")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#FFC107", "#1E88E5", "#D81B60"]

    P0, P1, P2 = stocks[0], stocks[1], stocks[2]

    ax.fill_between(t, P0, 0, color=colors[0], label="$P_1$ (High)")
    ax.fill_between(t, P1 + P0, P0, color=colors[1], label="$P_2$ (Medium)")
    ax.fill_between(t, P2 + P1 + P0, P1 + P0, color=colors[2], label="$P_3$ (Low)")

    if overlay_values is not None:
        ax.plot(
            t,
            overlay_values,
            color=overlay_color,
            linestyle=overlay_linestyle,
            label=overlay_label,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Population in stock", fontsize=12)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    fig.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    if show:
        plt.show()

    return fig, ax


def plot_referral_numbers_over_time(
    referral_numbers,
    t,
    ylim=None,
    title="Referral Numbers Over Time",
    filename=None,
    show=True,
):
    """
    Plot referral numbers over time for the three SD stocks.

    Parameters
    ----------
    referral_numbers : list of arrays
        List of 3 arrays representing referral numbers for P1, P2, and P3.
    t : array-like
        Time vector.
    ylim : tuple, optional
        Y-axis limits for the plot (default is None).
    title : str
        Title of the plot.
    filename : str, optional
        Filename to save the plot (default is None, which does not save the plot).
    show : bool, optional
        Whether to display the plot (default is True).

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        The figure and axes objects for further customization if needed.
    """
    if len(referral_numbers) != 3:
        raise ValueError(
            "referral_numbers must contain exactly 3 arrays for P1, P2, and P3."
        )
    if any(len(referral) != len(t) for referral in referral_numbers):
        raise ValueError("Each array must have the same length as t.")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#FFC107", "#1E88E5", "#D81B60"]
    labels = ["$Λ_1(t)$ (High)", "$Λ_2(t)$ (Medium)", "$Λ_3(t)$ (Low)"]

    for i in range(len(referral_numbers)):
        ax.plot(t, referral_numbers[i], label=labels[i], color=colors[i])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (days)", fontsize=14)
    ax.set_ylabel("Referrals", fontsize=14)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    fig.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    if show:
        plt.show()

    return fig, ax
