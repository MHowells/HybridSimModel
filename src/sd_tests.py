import numpy as np
import pytest
import sd_component as sd


def test_get_time_dependent_population_size_returns_expected_size_within_each_period():
    population_fn = sd.get_time_dependent_population_size(
        population_sizes=[1000, 2000, 3000],
        durations=[10, 20, 30],
    )

    assert population_fn(0) == 1000
    assert population_fn(9.9) == 1000
    assert population_fn(10) == 2000
    assert population_fn(29.9) == 2000
    assert population_fn(30) == 3000
    assert population_fn(59.9) == 3000


def test_get_time_dependent_population_size_returns_final_size_after_all_periods():
    population_fn = sd.get_time_dependent_population_size(
        population_sizes=[1000, 2000, 3000],
        durations=[10, 20, 30],
    )

    assert population_fn(60) == 3000
    assert population_fn(100) == 3000


def test_get_time_dependent_population_size_accepts_scalar_population_size_and_duration():
    population_fn = sd.get_time_dependent_population_size(
        population_sizes=1000,
        durations=10,
    )

    assert population_fn(0) == 1000
    assert population_fn(9.9) == 1000
    assert population_fn(10) == 1000
    assert population_fn(100) == 1000


def test_get_time_dependent_population_size_uses_single_size_indefinitely_when_duration_not_provided():
    population_fn = sd.get_time_dependent_population_size(
        population_sizes=1000,
    )

    assert population_fn(0) == 1000
    assert population_fn(10) == 1000
    assert population_fn(1000) == 1000


def test_get_time_dependent_population_size_raises_value_error_for_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="The lengths of population_sizes and durations must match.",
    ):
        sd.get_time_dependent_population_size(
            population_sizes=[1000, 2000],
            durations=[10],
        )


def test_get_time_dependent_incidence_rate_returns_expected_rate_within_each_period():
    incidence_fn = sd.get_time_dependent_incidence_rate(
        incidence_proportions=[0.01, 0.02, 0.03],
        durations=[10, 20, 30],
    )

    population_size = 10000

    assert incidence_fn(0, population_size) == 100
    assert incidence_fn(9.9, population_size) == 100
    assert incidence_fn(10, population_size) == 200
    assert incidence_fn(29.9, population_size) == 200
    assert incidence_fn(30, population_size) == 300
    assert incidence_fn(59.9, population_size) == 300


def test_get_time_dependent_incidence_rate_returns_final_rate_after_all_periods():
    incidence_fn = sd.get_time_dependent_incidence_rate(
        incidence_proportions=[0.01, 0.02, 0.03],
        durations=[10, 20, 30],
    )

    population_size = 10000

    assert incidence_fn(60, population_size) == 300
    assert incidence_fn(100, population_size) == 300


def test_get_time_dependent_incidence_rate_returns_zero_when_population_zero():
    incidence_fn = sd.get_time_dependent_incidence_rate(
        incidence_proportions=[0.01, 0.02, 0.03],
        durations=[10, 20, 30],
    )

    assert incidence_fn(0, 0) == 0.0
    assert incidence_fn(15, 0) == 0.0
    assert incidence_fn(100, 0) == 0.0


def test_get_time_dependent_incidence_rate_accepts_scalar_incidence_proportion_and_duration():
    incidence_fn = sd.get_time_dependent_incidence_rate(
        incidence_proportions=0.01,
        durations=10,
    )

    population_size = 10000

    assert incidence_fn(0, population_size) == 100
    assert incidence_fn(9.9, population_size) == 100
    assert incidence_fn(10, population_size) == 100
    assert incidence_fn(100, population_size) == 100


def test_get_time_dependent_incidence_rate_uses_single_rate_indefinitely_when_duration_not_provided():
    incidence_fn = sd.get_time_dependent_incidence_rate(incidence_proportions=0.01)

    population_size = 10000

    assert incidence_fn(0, population_size) == 100
    assert incidence_fn(10, population_size) == 100
    assert incidence_fn(1000, population_size) == 100


def test_get_time_dependent_incidence_rate_raises_value_error_for_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="The lengths of incidence_proportions and durations must match.",
    ):
        sd.get_time_dependent_incidence_rate(
            incidence_proportions=[0.01, 0.02],
            durations=[10],
        )


def test_get_time_dependent_recovery_rate_returns_expected_rate_within_each_period():
    recovery_fn = sd.get_time_dependent_recovery_rate(
        recovery_proportions=[0.01, 0.02, 0.03],
        durations=[10, 20, 30],
    )

    stock_size = 10000

    assert recovery_fn(0, stock_size) == 100
    assert recovery_fn(9.9, stock_size) == 100
    assert recovery_fn(10, stock_size) == 200
    assert recovery_fn(29.9, stock_size) == 200
    assert recovery_fn(30, stock_size) == 300
    assert recovery_fn(59.9, stock_size) == 300


def test_get_time_dependent_recovery_rate_returns_final_rate_after_all_periods():
    recovery_fn = sd.get_time_dependent_recovery_rate(
        recovery_proportions=[0.01, 0.02, 0.03],
        durations=[10, 20, 30],
    )

    stock_size = 10000

    assert recovery_fn(60, stock_size) == 300
    assert recovery_fn(100, stock_size) == 300


def test_get_time_dependent_recovery_rate_returns_zero_when_stock_zero():
    recovery_fn = sd.get_time_dependent_recovery_rate(
        recovery_proportions=[0.01, 0.02, 0.03],
        durations=[10, 20, 30],
    )

    assert recovery_fn(0, 0) == 0.0
    assert recovery_fn(15, 0) == 0.0
    assert recovery_fn(100, 0) == 0.0


def test_get_time_dependent_recovery_rate_accepts_scalar_recovery_proportion_and_duration():
    recovery_fn = sd.get_time_dependent_recovery_rate(
        recovery_proportions=0.01,
        durations=10,
    )

    stock_size = 10000

    assert recovery_fn(0, stock_size) == 100
    assert recovery_fn(9.9, stock_size) == 100
    assert recovery_fn(10, stock_size) == 100
    assert recovery_fn(100, stock_size) == 100


def test_get_time_dependent_recovery_rate_uses_single_rate_indefinitely_when_duration_not_provided():
    recovery_fn = sd.get_time_dependent_recovery_rate(recovery_proportions=0.01)

    stock_size = 10000

    assert recovery_fn(0, stock_size) == 100
    assert recovery_fn(10, stock_size) == 100
    assert recovery_fn(1000, stock_size) == 100


def test_deterioration_returns_expected_transition_rates():
    deterioration_fn = sd.get_deterioration_rates(
        category_widths=(0.50, 0.25, 0.25),
        shift_proportion=0.05,
        shift_interval_days=182.5,
    )

    obtained = deterioration_fn(t=0.0)

    expected_r_lm = 0.05 / (0.50 * 182.5)
    expected_r_mh = 0.05 / (0.25 * 182.5)

    np.testing.assert_allclose(obtained, (expected_r_lm, expected_r_mh))


def test_deterioration_raises_value_error_when_category_widths_do_not_sum_to_one():
    with pytest.raises(ValueError, match="category_widths must sum to 1."):
        sd.get_deterioration_rates(
            category_widths=(0.40, 0.25, 0.25),
            shift_proportion=0.05,
            shift_interval_days=182.5,
        )

def test_deterioration_raises_value_error_when_any_category_width_is_non_positive():
    with pytest.raises(ValueError, match="all category widths must be positive."):
        sd.get_deterioration_rates(
            category_widths=(0.50, 0.00, 0.50),
            shift_proportion=0.05,
            shift_interval_days=182.5,
        )


def test_deterioration_raises_value_error_when_shift_proportion_is_negative():
    with pytest.raises(ValueError, match="shift_proportion must be non-negative."):
        sd.get_deterioration_rates(
            category_widths=(0.50, 0.25, 0.25),
            shift_proportion=-0.01,
            shift_interval_days=182.5,
        )


def test_deterioration_raises_value_error_when_shift_interval_days_is_non_positive():
    with pytest.raises(ValueError, match="shift_interval_days must be positive."):
        sd.get_deterioration_rates(
            category_widths=(0.50, 0.25, 0.25),
            shift_proportion=0.05,
            shift_interval_days=-1.0,
        )


def test_deterioration_raises_value_error_when_shift_proportion_exceeds_smallest_transition_width():
    with pytest.raises(
        ValueError,
        match="shift_proportion must be less than or equal to the smallest transition category width.",
    ):
        sd.get_deterioration_rates(
            category_widths=(0.50, 0.25, 0.25),
            shift_proportion=0.30,
            shift_interval_days=182.5,
        )


def test_get_time_dependent_recovery_rate_raises_value_error_for_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="The lengths of recovery_proportions and durations must match.",
    ):
        sd.get_time_dependent_recovery_rate(
            recovery_proportions=[0.01, 0.02],
            durations=[10],
        )


def get_simple_sd_model(
    population_function=None,
    initial_unwell_proportion=0.1,
    unwell_splits=[0.5, 0.3, 0.2],
    gatekeeping_function=None,
    presenting_proportion=0.4,
    deterioration_function=None,
    incidence_function=None,
    recovery_function=None,
):
    if population_function is None:

        def population_function(t):
            return 1000

    if gatekeeping_function is None:

        def gatekeeping_function(stocks, population, presenting_proportion, t):
            return [0.0, 0.0, 0.0]

    if deterioration_function is None:

        def deterioration_function(t):
            return 0.0

    if incidence_function is None:

        def incidence_function(t, population_size):
            return 0.0

    if recovery_function is None:

        def recovery_function(t, stock_size):
            return 0.0

    return sd.SD(
        population_function=population_function,
        initial_unwell_proportion=initial_unwell_proportion,
        unwell_splits=unwell_splits,
        gatekeeping_function=gatekeeping_function,
        presenting_proportion=presenting_proportion,
        deterioration_function=deterioration_function,
        incidence_function=incidence_function,
        recovery_function=recovery_function,
    )


def test_sd_initialises_stock_sizes_from_initial_population_unwell_proportion_and_splits():
    model = get_simple_sd_model(
        initial_unwell_proportion=0.2,
        unwell_splits=(0.2, 0.3, 0.5),
    )

    expected_unwell_population = 1000 * 0.2

    assert model.P[0] == expected_unwell_population * 0.2
    assert model.P[1] == expected_unwell_population * 0.3
    assert model.P[2] == expected_unwell_population * 0.5


def test_sd_initial_stock_sizes_sum_to_initial_unwell_population():
    model = get_simple_sd_model(
        initial_unwell_proportion=0.1,
        unwell_splits=(0.5, 0.3, 0.2),
    )

    expected_unwell_population = 1000 * 0.1

    assert sum(model.P) == expected_unwell_population


def test_sd_initialises_time_and_lambdas_attributes():
    model = get_simple_sd_model(
        initial_unwell_proportion=0.1,
        unwell_splits=(0.5, 0.3, 0.2),
        presenting_proportion=0.25,
    )

    assert model.presenting_proportion == 0.25
    assert np.array_equal(model.time, np.array([0]))
    assert model.lambdas is None


def test_sd_differential_equations_returns_zero_when_total_population_is_zero():
    model = get_simple_sd_model()

    obtained = model.differential_equations(
        y=(0.0, 0.0, 0.0),
        time_domain=0.0,
    )

    assert obtained == (0, 0, 0)


def test_sd_differential_equations_returns_zero_when_all_flows_are_zero():
    model = get_simple_sd_model()

    obtained = model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    assert obtained == (0.0, 0.0, 0.0)


def test_sd_differential_equations_returns_decrease_under_only_referrals():
    model = get_simple_sd_model(
        gatekeeping_function=lambda stocks, population, presenting_proportion, t: [
            3.0,
            2.0,
            1.0,
        ],
    )

    obtained = model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    expected = (-3.0, -2.0, -1.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_flow_under_only_deterioration():
    model = get_simple_sd_model(
        deterioration_function=lambda t: 0.1,
    )

    obtained = model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    expected = (-3.0, 1.0, 2.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_increase_under_only_incidence():
    model = get_simple_sd_model(
        incidence_function=lambda t, population_size: 5.0,
    )

    obtained = model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    expected = (5.0, 0.0, 0.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_decrease_under_only_recovery():
    model = get_simple_sd_model(
        recovery_function=lambda t, stock_size: 0.1 * stock_size,
    )

    obtained = model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    expected = (-6.0, 0.0, 0.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_values():
    model = get_simple_sd_model(
        gatekeeping_function=lambda stocks, population, presenting_proportion, t: [
            3.0,
            2.0,
            1.0,
        ],
        deterioration_function=lambda t: 0.1,
        incidence_function=lambda t, population_size: 5.0,
        recovery_function=lambda t, stock_size: 6.0,
    )

    obtained = model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    expected = (-7.0, -1.0, 1.0)

    assert obtained == expected


def test_sd_differential_equations_passes_current_population_to_incidence_function():
    population_sizes = []

    def population_function(t):
        return 500.0 + t

    def incidence_function(t, population_size):
        population_sizes.append(population_size)
        return 0.0

    model = get_simple_sd_model(
        population_function=population_function,
        incidence_function=incidence_function,
    )

    model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    assert population_sizes == [505.0]


def test_sd_differential_equations_passes_current_stock_size_to_recovery_function():
    stock_sizes = []

    def recovery_function(t, stock_size):
        stock_sizes.append(stock_size)
        return 0.0

    model = get_simple_sd_model(
        recovery_function=recovery_function,
    )

    model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    assert stock_sizes == [60.0]


def test_sd_differential_equations_negative_population_floored_to_zero():
    population_sizes = []

    def population_function(t):
        return -100.0

    def incidence_function(t, population_size):
        population_sizes.append(population_size)
        return 0.0

    model = get_simple_sd_model(
        population_function=population_function,
        gatekeeping_function=lambda stocks, population, presenting_proportion, t: [
            0.0,
            0.0,
            0.0,
        ],
        deterioration_function=lambda t: 0.0,
        incidence_function=incidence_function,
        recovery_function=lambda t, stock_size: 0.0,
    )

    model.differential_equations(
        y=(30.0, 20.0, 10.0),
        time_domain=5.0,
    )

    assert population_sizes == [0]


def test_sd_solve_stores_stocks_with_expected_length():
    model = get_simple_sd_model()

    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    model.solve(t=t)

    assert len(model.P[0]) == len(t)
    assert len(model.P[1]) == len(t)
    assert len(model.P[2]) == len(t)


def test_sd_solve_stores_initial_stock_values_at_start():
    model = get_simple_sd_model(
        initial_unwell_proportion=0.1,
        unwell_splits=(0.5, 0.3, 0.2),
    )

    initial_P = model.P.copy()

    t = np.array([0.0, 1.0, 2.0])
    model.solve(t=t)

    assert model.P[0][0] == initial_P[0]
    assert model.P[1][0] == initial_P[1]
    assert model.P[2][0] == initial_P[2]


def test_sd_solve_returns_constant_stocks_when_all_flows_zero():
    def gatekeeping_function(stocks, population, presenting_proportion, t):
        if np.isscalar(t):
            return [0.0, 0.0, 0.0]
        return [
            np.zeros_like(t, dtype=float),
            np.zeros_like(t, dtype=float),
            np.zeros_like(t, dtype=float),
        ]

    model = get_simple_sd_model(
        gatekeeping_function=gatekeeping_function,
    )

    initial_P = model.P.copy()
    t = np.linspace(0.0, 10.0, 6)

    model.solve(t=t)

    np.testing.assert_allclose(model.P[0], np.full_like(t, initial_P[0], dtype=float))
    np.testing.assert_allclose(model.P[1], np.full_like(t, initial_P[1], dtype=float))
    np.testing.assert_allclose(model.P[2], np.full_like(t, initial_P[2], dtype=float))


def test_sd_solve_computes_lambdas_with_expected_length():
    def gatekeeping_function(stocks, population, presenting_proportion, t):
        if np.isscalar(t):
            return [0.0, 0.0, 0.0]
        return [
            np.zeros_like(t, dtype=float),
            np.zeros_like(t, dtype=float),
            np.zeros_like(t, dtype=float),
        ]

    model = get_simple_sd_model(
        gatekeeping_function=gatekeeping_function,
    )

    t = np.linspace(0.0, 10.0, 6)
    model.solve(t=t)

    assert len(model.lambdas) == 3
    assert len(model.lambdas[0]) == len(t)
    assert len(model.lambdas[1]) == len(t)
    assert len(model.lambdas[2]) == len(t)


def test_sd_solve_stores_lambdas_from_gatekeeping_function():
    def gatekeeping_function(stocks, population, presenting_proportion, t):
        if np.isscalar(t):
            return [1.0, 2.0, 3.0]
        return [
            np.full_like(t, 1.0, dtype=float),
            np.full_like(t, 2.0, dtype=float),
            np.full_like(t, 3.0, dtype=float),
        ]

    model = get_simple_sd_model(
        gatekeeping_function=gatekeeping_function,
    )

    t = np.linspace(0.0, 10.0, 6)
    model.solve(t=t)

    np.testing.assert_allclose(model.lambdas[0], np.full_like(t, 1.0, dtype=float))
    np.testing.assert_allclose(model.lambdas[1], np.full_like(t, 2.0, dtype=float))
    np.testing.assert_allclose(model.lambdas[2], np.full_like(t, 3.0, dtype=float))


def test_sd_solve_accepts_gatekeeping_function_compatible_with_scalar_and_vector_inputs():
    def gatekeeping_function(stocks, population, presenting_proportion, t):
        if np.isscalar(t):
            return [0.5, 1.0, 1.5]
        return [
            np.full_like(t, 0.5, dtype=float),
            np.full_like(t, 1.0, dtype=float),
            np.full_like(t, 1.5, dtype=float),
        ]

    model = get_simple_sd_model(
        gatekeeping_function=gatekeeping_function,
    )

    t = np.linspace(0.0, 5.0, 6)
    model.solve(t=t)

    assert len(model.lambdas) == 3
    np.testing.assert_allclose(model.lambdas[0], np.full_like(t, 0.5, dtype=float))
    np.testing.assert_allclose(model.lambdas[1], np.full_like(t, 1.0, dtype=float))
    np.testing.assert_allclose(model.lambdas[2], np.full_like(t, 1.5, dtype=float))
