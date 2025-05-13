import numpy as np
from scipy.integrate import odeint


def proportional_gatekeeping(threshold):
    """
    Proportional gatekeeping function.

    Parameters
    ----------
    threshold : float in [0, 1]
        Proportion threshold for gatekeeping

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """
    def gatekeeping_function(stocks, population, presenting_rate, t):
        """
        Gatekeeping function to calculate lambda values for each stock.
        Parameters
        ----------
        stocks : list of floats or arrays
            Stock levels at current time step t or over time
        population : float or array
            Total population at time t (scalar or array)
        presenting_rate : float in [0, 1]
            Rate of patients presenting from each stock
        Returns
        -------
        list
            list of arrays or scalars representing the lambda values for each stock.
        """
        stocks = np.array(stocks)
        lambdas = []

        for i in range(len(stocks)):
            if i == 0:
                subtracted = 0
            else:
                subtracted = sum(stocks[:i])
            stock = stocks[i]
            ratio = np.zeros_like(stock)
            positives = stock > 0
            if np.isscalar(subtracted):
                ratio[positives] = (
                    (threshold * population) - subtracted
                ) / stock
            else:
                ratio[positives] = (
                    (threshold * population[positives])
                    - subtracted[positives]
                ) / stock[positives]
            lambdas.append(presenting_rate * np.clip(ratio, 0, 1) * stock)
        return lambdas
    return gatekeeping_function


def fixed_gatekeeping(threshold):
    """
    Fixed gatekeeping function.

    Parameters
    ----------
    threshold : float or int
        Fixed capacity threshold for gatekeeping per time step

    Returns
    -------
    function
        Function to calculate lambda values for each stock.
    """
    def gatekeeping_function(stocks, population, presenting_rate, t):
        """
        Gatekeeping function to calculate lambda values for each stock.
        Parameters
        ----------
        stocks : list of scalars or arrays
            Stock levels at current time step t or over time
        population : float or array
            Total population at time t (scalar or array)
        presenting_rate : float in [0, 1]
            Rate of patients presenting from each stock
        Returns
        -------
        list
            list of arrays or scalars representing the lambda values for each stock.
        """
        if np.isscalar(stocks[0]):
            remaining_capacity = threshold
            lambdas = []
            for stock in stocks:
                demand = presenting_rate * stock
                allowed = min(demand, remaining_capacity)
                lambdas.append(allowed)
                remaining_capacity -= allowed
                remaining_capacity = max(remaining_capacity, 0)
            return lambdas
        else:
            stocks = [np.asarray(s) for s in stocks]
            time_steps = len(stocks[0])
            lambdas = [np.zeros(time_steps) for _ in stocks]
            remaining_capacity = np.full(time_steps, threshold, dtype=float)
            for i, stock in enumerate(stocks):
                demand = presenting_rate * stock
                allowed = np.minimum(demand, remaining_capacity)
                lambdas[i] = allowed
                remaining_capacity = np.maximum(remaining_capacity - allowed, 0)
            return lambdas
    return gatekeeping_function


def seasonal_gatekeeping(baseline=8, amplitude=2, period=365, phase_shift=0):
    """
    Returns a gatekeeping function that varies seasonally based on a sine
    function. The returned function computes the lambda values for each stock.

    Parameters
    ----------
    baseline : float
        average value of the seasonal variation
    amplitude : float
        amplitude of the seasonal variation
    period : int
        period of the seasonal variation (in days)
    phase_shift : float
        phase shift of the seasonal variation (in days)

    Returns
    -------
    function
        Gatekeeping function to calculate lambda values for each stock.
    """
    def gatekeeping_function(stocks, population, presenting_rate, t):
        t = np.asarray(t)
        is_scalar = np.isscalar(t) or t.shape == ()
        stocks = np.array(stocks)

        if is_scalar:
            threshold = np.clip(
                baseline + amplitude * np.sin(2 * np.pi * (t + phase_shift) / period),
                0, None
            )

            if population == 0 or threshold == 0:
                lambdas = np.zeros(3)
            else:
                remaining_capacity = threshold
                lambdas = []
                for stock in stocks:
                    demand = presenting_rate * stock
                    allowed = min(demand, remaining_capacity)
                    lambdas.append(allowed)
                    remaining_capacity -= allowed
                    remaining_capacity = max(remaining_capacity, 0)
        else:
            thresholds = np.maximum(0, baseline + amplitude * np.sin(2 * np.pi * (t + phase_shift) / period))

            population = np.array(population)
            time_steps = len(t)
            lambdas = np.zeros((3, time_steps))

            for i in range(time_steps):
                if population[i] == 0 or thresholds[i] == 0:
                    lambdas[:, i] = 0
                else:
                    remaining_capacity = thresholds[i]
                    for j in range(3):
                        demand = presenting_rate * stocks[j, i]
                        allowed = min(demand, remaining_capacity)
                        lambdas[j, i] = allowed
                        remaining_capacity -= allowed
                        remaining_capacity = max(remaining_capacity, 0)
        return lambdas
    
    return gatekeeping_function

class SD:
    """
    A class to hold the SD component.
    """

    def __init__(
        self,
        initial_population,
        unwell_proportion,
        unwell_splits,
        gatekeeping_function,
        presenting_rate,
        deterioration_rate,
        incidence_rate,
    ):
        """
        Initialised the parameters for the SD component

        Parameters
        ----------
        initial_population : a positive integer
            initial population of the system
        unwell_proportion : a positive float <= 1
            proportion of the initial population that is unwell
        unwell_splits : a tuple of three floats that sum to 1
            representing the proportions of the unwell population in each stock
        gatekeeping_function : a function
            function to calculate lambda values for each stock
        presenting_rate : a positive float <= 1
            rate at which patients present for treatment
        deterioration_rate : a positive float <= 1
            rate at which patients deteriorate
        incidence_rate : a positive float <= 1
            rate at which new patients enter the system
        """
        w = unwell_splits
        self.initial_population = initial_population
        unwell_pop = initial_population * unwell_proportion
        self.P = [
            unwell_pop * w[0], 
            unwell_pop * w[1], 
            unwell_pop * w[2]
        ]
        self.presenting_rate = presenting_rate
        self.gatekeeping_function = gatekeeping_function
        self.deterioration_rate = deterioration_rate
        self.incidence_rate = incidence_rate
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

        susceptible_population = max(self.initial_population - N_current, 0)
        lambdas = self.gatekeeping_function(
            stocks=all_stocks,
            population=N_current,
            presenting_rate=self.presenting_rate,
            t=time_domain,
        )

        dP_onedt = -lambdas[0] + self.deterioration_rate * P_two
        dP_twodt = (
            -lambdas[1]
            - (self.deterioration_rate * P_two)
            + (self.deterioration_rate * P_three)
        )
        dP_threedt = (
            -lambdas[2]
            - (self.deterioration_rate * P_three)
            + (self.incidence_rate * susceptible_population)
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
            presenting_rate=self.presenting_rate,
            t=t,
        )


