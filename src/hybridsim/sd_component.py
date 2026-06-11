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
        self.initial_population = population_function
        unwell_pop = self.initial_population(t=0) * initial_unwell_proportion
        self.P = [unwell_pop * w[0], unwell_pop * w[1], unwell_pop * w[2]]
        self.presenting_proportion = presenting_proportion
        self.gatekeeping_function = gatekeeping_function
        self.deterioration_rate = deterioration_function
        self.incidence_rate = incidence_function
        self.recovery_rate = recovery_function
        self.time = np.array([0.0])
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
        N_current = (
            P_low 
            + P_medium 
            + P_high
        )
        all_stocks = [
            P_low, 
            P_medium, 
            P_high,
        ]

        if N_current == 0:
            return 0.0, 0.0, 0.0

        current_population = max(
            self.initial_population(t=time_domain), 
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
                population_size=current_population,
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

    def solve(self, t):
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
            stocks=[
                P_low, 
                P_medium, 
                P_high,
            ],
            population=(
                P_low 
                + P_medium 
                + P_high
            ),
            presenting_proportion=self.presenting_proportion,
            t=t,
        )


def get_time_dependent_population_size(
    population_sizes, 
    durations=np.NaN
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


def get_deterioration_rates(
    category_widths,
    shift_proportion,
    shift_interval_days,
):
    """
    Discretised deterioration on an underlying severity scale, 
    discretised into Low, Medium, and High regions.

    Every shift_interval_days, patients are assumed to shift by
    shift_proportion along the underlying scale. Only those near the
    boundary of a category move into the next category.

    Parameters
    ----------
    category_widths : tuple of floats 
        Widths of the Low, Medium, and High regions on the underlying 
        severity scale. Must sum to 1.
    shift_proportion : float
        Proportion of the severity scale moved over each interval.
    shift_interval_days : float
        Time interval over which the underlying shift occurs.

    Returns
    -------
    function
        Function returning the Low-to-Medium and Medium-to-High 
        transition rates.
    """
    low_width, med_width, high_width = category_widths

    if not np.isclose(low_width + med_width + high_width, 1.0):
        raise ValueError("category_widths must sum to 1.")

    if low_width <= 0 or med_width <= 0 or high_width <= 0:
        raise ValueError("all category widths must be positive.")

    if shift_proportion < 0:
        raise ValueError("shift_proportion must be non-negative.")

    if shift_interval_days <= 0:
        raise ValueError("shift_interval_days must be positive.")

    if shift_proportion > min(low_width, med_width):
        raise ValueError(
            "shift_proportion must be less than or equal to the smallest transition category width."
        )

    r_lm = shift_proportion / (low_width * shift_interval_days)
    r_mh = shift_proportion / (med_width * shift_interval_days)

    def deterioration_function(t):
        return r_lm, r_mh

    return deterioration_function


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
    colors = ["#D81B60", "#1E88E5", "#FFC107"]
    labels = ["$P_1$ (Low)", "$P_2$ (Medium)", "$P_3$ (High)"]
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

    ax.fill_between(t, P2, 0, color=colors[0], label="$P_3$ (High)")
    ax.fill_between(t, P1 + P2, P2, color=colors[1], label="$P_2$ (Medium)")
    ax.fill_between(t, P2 + P1 + P0, P2 + P1, color=colors[2], label="$P_1$ (Low)")

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
    colors = ["#D81B60", "#1E88E5", "#FFC107"]
    labels = ["$Λ_1(t)$ (Low)", "$Λ_2(t)$ (Medium)", "$Λ_3(t)$ (High)"]

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
