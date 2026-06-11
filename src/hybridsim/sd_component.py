"""
Define the system-dynamics component of the orthopaedic model.

This module provides the system-dynamics stock model, time-dependent
population, incidence and recovery functions, deterioration rates, and
utilities for adding a simulation warm-up period.
"""

import numpy as np
from scipy.integrate import odeint

LOW = 0
MEDIUM = 1
HIGH = 2

PRIORITY_ORDER = [HIGH, MEDIUM, LOW]


def validate_initial_state_inputs(
    initial_unwell_proportion,
    unwell_splits,
):
    """Validate and normalise the initial population proportions.

    Parameters
    ----------
    initial_unwell_proportion : float
        Proportion of the initial population that is unwell.
    unwell_splits : sequence of three floats
        Proportions assigned to the low-, medium-, and high-severity
        stocks.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Validated unwell proportion and severity-stock proportions.

    Raises
    ------
    TypeError
        If either argument cannot be interpreted as numeric data.
    ValueError
        If the unwell proportion is outside the interval ``[0, 1]``,
        or if the severity splits are invalid.
    """
    if isinstance(
        initial_unwell_proportion,
        (bool, np.bool_),
    ) or not np.isscalar(initial_unwell_proportion):
        raise TypeError(
            "initial_unwell_proportion must be a numeric scalar."
        )

    try:
        initial_unwell_proportion = float(
            initial_unwell_proportion
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "initial_unwell_proportion must be a numeric scalar."
        ) from error

    if not np.isfinite(initial_unwell_proportion):
        raise ValueError(
            "initial_unwell_proportion must be finite."
        )

    if not 0.0 <= initial_unwell_proportion <= 1.0:
        raise ValueError(
            "initial_unwell_proportion must be between 0 and 1 "
            "inclusive."
        )

    try:
        unwell_splits = np.asarray(
            list(unwell_splits),
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "unwell_splits must be a sequence of three numeric values."
        ) from error

    if unwell_splits.ndim != 1 or unwell_splits.size != 3:
        raise ValueError(
            "unwell_splits must contain exactly three values for the "
            "low-, medium-, and high-severity stocks."
        )

    if not np.all(np.isfinite(unwell_splits)):
        raise ValueError(
            "Every value in unwell_splits must be finite."
        )

    if np.any(unwell_splits < 0):
        raise ValueError(
            "Every value in unwell_splits must be non-negative."
        )

    if not np.isclose(unwell_splits.sum(), 1.0):
        raise ValueError(
            "The values in unwell_splits must sum to 1. "
            f"Received a total of {unwell_splits.sum():.12g}."
        )

    return initial_unwell_proportion, unwell_splits


def normalise_piecewise_inputs(values, durations, value_name):
    """
    Convert piecewise values and durations to lists of equal length.

    Parameters
    ----------
    values : scalar or iterable
        Values applied over successive time intervals.
    durations : scalar or iterable
        Durations of the successive time intervals.
    value_name : str
        Name used to construct an informative error message.

    Returns
    -------
    tuple[list, list]
        Normalised values and durations.

    Raises
    ------
    ValueError
        If the values and durations have different lengths.
    """
    if np.isscalar(values):
        values = [values]
    else:
        values = list(values)

    if np.isscalar(durations):
        durations = [durations]
    else:
        durations = list(durations)

    if len(values) != len(durations):
        raise ValueError(
            f"The lengths of {value_name} and durations must match. "
            f"Received {len(values)} values and {len(durations)} "
            "durations."
        )

    return values, durations


def get_change_points(durations):
    """Return cumulative interval boundaries beginning at zero."""
    change_points = [0]

    for duration in durations:
        change_points.append(change_points[-1] + duration)

    return change_points


class SD:
    """
    Represent the system dynamics component of the model.

    The model contains low-, medium-, and high-severity population
    stocks. Population movements are determined by presentation,
    deterioration, incidence, and recovery functions.

    Attributes
    ----------
    P : list[numpy.ndarray]
        Historical values for the low-, medium-, and high-severity
        population stocks.
    time : numpy.ndarray
        Time points represented by the stored population histories.
    lambdas : numpy.ndarray or None
        Most recently calculated presentation rates.
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
        Initialise the system dynamics component.

        Parameters
        ----------
        population_function : callable
            Function accepting ``t`` and returning the total population
            at that time.
        initial_unwell_proportion : float
            Proportion of the initial population that is unwell.
        unwell_splits : sequence of three floats
            Proportions of the unwell population assigned to the low-,
            medium-, and high-severity stocks.
        gatekeeping_function : callable
            Function calculating presentation rates for each stock.
        presenting_proportion : float
            Proportion of eligible patients who present for treatment.
        deterioration_function : callable
            Function returning the low-to-medium and medium-to-high
            deterioration rates.
        incidence_function : callable
            Function calculating the number of new unwell patients.
        recovery_function : callable
            Function calculating the number of recovering patients.
        """
        (
            initial_unwell_proportion,
            unwell_splits,
        ) = validate_initial_state_inputs(
            initial_unwell_proportion,
            unwell_splits,
        )

        w = unwell_splits
        self.population_size = population_function
        unwell_pop = self.population_size(t=0) * initial_unwell_proportion
        self.P = [
            unwell_pop * w[0], 
            unwell_pop * w[1], 
            unwell_pop * w[2]
        ]
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
        Calculate the rates of change for the severity stocks.

        Parameters
        ----------
        y : sequence of three floats
            Current low-, medium-, and high-severity populations.
        time_domain : float
            Current simulation time.

        Returns
        -------
        tuple[float, float, float]
            Rates of change for the low-, medium-, and high-severity
            stocks.
        """
        P_low, P_medium, P_high = y
        N_current = P_low + P_medium + P_high
        all_stocks = [P_low, P_medium, P_high]

        if N_current == 0:
            return 0, 0, 0

        current_population = max(
            self.population_size(t=time_domain), 
            0,
        )
        
        lambdas = self.gatekeeping_function(
            stocks=all_stocks,
            population=N_current,
            presenting_proportion=self.presenting_proportion,
            t=time_domain,
        )

        deterioration = self.deterioration_rate(t=time_domain)

        # Continue supporting functions that return one shared rate.
        if np.isscalar(deterioration):
            low_to_medium_rate = deterioration
            medium_to_high_rate = deterioration
        else:
            low_to_medium_rate, medium_to_high_rate = deterioration

        dP_lowdt = (
            -lambdas[LOW]
            - (low_to_medium_rate * P_low)
            - self.recovery_rate(
                t=time_domain, 
                stock_size=N_current,
            )
            + self.incidence_rate(
                t=time_domain, 
                population_size=current_population
            )
        )

        dP_mediumdt = (
            (low_to_medium_rate * P_low)
            - (medium_to_high_rate * P_medium)
            - lambdas[MEDIUM]
        )

        dP_highdt = (
            (medium_to_high_rate * P_medium)
            - lambdas[HIGH]
        )
        return dP_lowdt, dP_mediumdt, dP_highdt

    def solve(
        self,
        t,
    ):
        """
        Solve the differential equations over the supplied times.

        The first time point is treated as the start of the interval.
        Results after that point are appended to the stored histories.

        Parameters
        ----------
        t : array-like
            Increasing time points over which to solve the equations.
        """
        # Solve the SD over the relevant time domain
        y = self.P
        results = odeint(
            self.differential_equations,
            y,
            t,
        )

        P_low, P_medium, P_high = results.T
        self.P[LOW] = np.append(self.P[LOW], P_low[1:])
        self.P[MEDIUM] = np.append(self.P[MEDIUM], P_medium[1:])
        self.P[HIGH] = np.append(self.P[HIGH], P_high[1:])

        # Extract the lambdas from the results
        self.lambdas = self.gatekeeping_function(
            stocks=[P_low, P_medium, P_high],
            population=P_low + P_medium + P_high,
            presenting_proportion=self.presenting_proportion,
            t=t,
        )


def get_time_dependent_population_size(
    population_sizes, 
    durations=np.NaN,
):
    """
    Return a time-dependent population-size function.

    Parameters
    ----------
    population_sizes : list of int/float
        A list of population sizes.
    durations : list of int/float, default=np.NaN
        Duration for which each population size applies. A scalar
        population and the default duration produce a constant
        population function.

    Returns
    -------
    function
        Function accepting time ``t`` and returning the corresponding
        population size.
    """
    population_sizes, durations = normalise_piecewise_inputs(
        population_sizes,
        durations,
        "population_sizes",
    )
    change_points = get_change_points(durations)

    def population_function(t):
        """Return the population size at time ``t``."""
        for index, population_size in enumerate(population_sizes):
            interval_start = change_points[index]
            interval_end = change_points[index + 1]

            if interval_start <= t < interval_end:
                return population_size

        return population_sizes[-1]

    return population_function


def get_time_dependent_incidence_rate(incidence_proportions, durations=np.NaN):
    """
    Return a time-dependent incidence-rate function.

    Parameters
    ----------
    incidence_proportions : list of float
        Incidence proportions (e.g., cases per person per period).
    durations : list of int/float
        Durations for which each incidence proportion applies.

    Returns
    -------
    function
        Function accepting ``t`` and ``population_size`` and returning
        the incidence flow at that time.
    """
    incidence_proportions, durations = normalise_piecewise_inputs(
        incidence_proportions,
        durations,
        "incidence_proportions",
    )
    change_points = get_change_points(durations)

    def incidence_function(t, population_size):
        """Return the incidence flow at time ``t``."""
        if population_size == 0:
            return 0.0
        
        for index, incidence_proportion in enumerate(
            incidence_proportions
        ):
            interval_start = change_points[index]
            interval_end = change_points[index + 1]

            if interval_start <= t < interval_end:
                return (incidence_proportion * population_size)

        return incidence_proportions[-1] * population_size

    return incidence_function


def get_time_dependent_recovery_rate(
    recovery_proportions, 
    durations=np.NaN,
):
    """
    Return a time-dependent recovery-rate function.

    Parameters
    ----------
    recovery_proportions : list of float
        Recovery proportions (e.g., proportion recovered per period).
    durations : list of int/float
        Durations for which each recovery proportion applies.

    Returns
    -------
    function
        Function accepting ``t`` and ``stock_size`` and returning the
        recovery flow at that time.
    """
    recovery_proportions, durations = normalise_piecewise_inputs(
        recovery_proportions,
        durations,
        "recovery_proportions",
    )
    change_points = get_change_points(durations)

    def recovery_function(t, stock_size):
        """Return the recovery flow at time ``t``."""
        if stock_size == 0:
            return 0.0
        
        for index, recovery_proportion in enumerate(
            recovery_proportions
        ):
            interval_start = change_points[index]
            interval_end = change_points[index + 1]

            if interval_start <= t < interval_end:
                return recovery_proportion * stock_size

        return recovery_proportions[-1] * stock_size

    return recovery_function


def get_deterioration_rates(
    category_widths,
    shift_proportion,
    shift_interval_days,
):
    """
    Return deterioration rates for discretised severity categories.

    The underlying severity scale is divided into low-, medium-, and
    high-severity regions. During each shift interval, patients move by
    ``shift_proportion`` along that scale. Patients crossing a category
    boundary move into the next stock.

    Parameters
    ----------
    category_widths : sequence of three floats 
        Widths of the low-, medium-, and high-severity regions. The
        widths must be positive and sum to one.
    shift_proportion : float
        Proportion of the severity scale moved over each interval.
    shift_interval_days : float
        Time interval over which the underlying shift occurs.

    Returns
    -------
    function
        Function returning the Low-to-Medium and Medium-to-High 
        transition rates.

    Raises
    ------
    ValueError
        If the widths or shift parameters are invalid.
    """
    low_width, med_width, high_width = category_widths

    if not np.isclose(
        low_width + med_width + high_width, 
        1.0,
    ):
        raise ValueError("category_widths must sum to 1.")

    if (
        low_width <= 0 
        or med_width <= 0 
        or high_width <= 0
    ):
        raise ValueError("all category widths must be positive.")

    if shift_proportion < 0:
        raise ValueError("shift_proportion must be non-negative.")

    if shift_interval_days <= 0:
        raise ValueError("shift_interval_days must be positive.")

    if shift_proportion > min(low_width, med_width):
        raise ValueError(
            "shift_proportion must be less than or equal to the smallest transition category width."
        )

    low_to_medium_rate = (
        shift_proportion
        / (low_width * shift_interval_days)
    )
    medium_to_high_rate = (
        shift_proportion
        / (med_width * shift_interval_days)
    )

    def deterioration_function(t):
        return low_to_medium_rate, medium_to_high_rate

    return deterioration_function


def add_constant_lambda_warmup(
    lambdas, 
    ts, 
    warmup_days=365, 
    value="initial", 
    shift_time=True,
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
        Regularly spaced time values corresponding to the columns in
        ``lambdas``.
    warmup_days : int or float, default=365
        Length of the warm-up period in the same units as ``ts``.
    value : {"initial"} or array-like, default="initial"
        Lambda value for each group during warm-up. ``"initial"`` uses
        the first value of each group.
    shift_time : bool, default=True
        When true, shift the complete time vector so that warm-up starts
        at zero. Otherwise, warm-up uses negative time values.

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

    if isinstance(value, str):
        if value != "initial":
            raise ValueError(
                'value must be "initial" or an array-like object.'
            )

        warmup_values = lambdas[:, [0]]
    else:
        warmup_values = np.asarray(
            value, 
            dtype=float,
        ).reshape(-1, 1)

        if warmup_values.shape[0] != lambdas.shape[0]:
            raise ValueError(
                f"Expected {lambdas.shape[0]} warm-up values, "
                f"but received {warmup_values.shape[0]}."
            )

    warmup_lambdas = np.repeat(
        warmup_values, 
        warmup_points, 
        axis=1,
    )
    lambdas_with_warmup = np.concatenate(
        [warmup_lambdas, lambdas], 
        axis=1,
    )

    if shift_time:
        ts_with_warmup = np.arange(lambdas_with_warmup.shape[1]) * dt
    else:
        ts_with_warmup = np.concatenate(
            [np.linspace(-warmup_days, -dt, warmup_points), ts]
        )

    return lambdas_with_warmup, ts_with_warmup
