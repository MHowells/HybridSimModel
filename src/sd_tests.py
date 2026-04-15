import numpy as np
from scipy.integrate import odeint
import pytest
import sd_component as sd

proportional_threshold = 0.4
fixed_threshold = 16
sample_stocks = [
    np.array([1000, 1000.14235114, 1000.28470455, 1000.42706025, 1000.56941821]),
    np.array([3000, 3000.09854025, 3000.19706101, 3000.29556228, 3000.39404405]),
    np.array([6000, 5999.67150899, 5999.34303597, 5999.01458093, 5998.68614387]),
]
presenting_proportion = 0.002
ts_sample = np.array([0, 1, 2, 3, 4])


def test_strict_priority_gatekeeping_returns_callable():
    gatekeeping = sd.strict_priority_gatekeeping(threshold=0.5)
    assert callable(gatekeeping)


def test_strict_priority_gatekeeping_scalar_exact_fill():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = sd.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_partial_medium():
    stocks = np.array([40.0, 40.0, 20.0])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = sd.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([16.0, 4.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_zero_threshold():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    threshold = 0.0

    gatekeeping = sd.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = sd.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_full_threshold():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    threshold = 1.0

    gatekeeping = sd.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_all_capacity_to_first_group():
    stocks = np.array([100.0, 20.0, 10.0])
    presenting_proportion = 0.5
    threshold = 0.2

    gatekeeping = sd.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([13.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4

    gatekeeping = sd.strict_priority_gatekeeping(threshold=0.5)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [8.0, 8.0, 8.0],
        [12.0, 10.0, 8.0],
        [0.0, 2.0, 4.0],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.strict_priority_gatekeeping(threshold=0.5)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_fixed_capacity_strict_gatekeeping_returns_callable():
    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    assert callable(gatekeeping)


def test_fixed_capacity_strict_gatekeeping_scalar_exact_fill():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 20.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_partial_medium():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 7.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_no_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 0.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_full_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 40.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_all_capacity_to_first_group():
    stocks = np.array([100.0, 20.0, 10.0])
    presenting_proportion = 0.5
    capacity = 13.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([13.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [8.0, 8.0, 8.0],
        [7.0, 7.0, 7.0],
        [0.0, 0.0, 0.0],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_fixed_gatekeeping():
    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity=fixed_threshold)
    assert callable(gatekeeping)

    obtained_referrals_time_point = gatekeeping(
        stocks=[sample_stocks[0][0], sample_stocks[1][0], sample_stocks[2][0]],
        population=sample_stocks[0][0] + sample_stocks[1][0] + sample_stocks[2][0],
        presenting_proportion=presenting_proportion,
        t=0,
    )
    expected_referrals_time_point = [2, 6, 8]
    assert np.allclose(obtained_referrals_time_point, expected_referrals_time_point)

    obtained_referrals_time_series = gatekeeping(
        stocks=[sample_stocks[0], sample_stocks[1], sample_stocks[2]],
        population=sample_stocks[0] + sample_stocks[1] + sample_stocks[2],
        presenting_proportion=presenting_proportion,
        t=ts_sample,
    )
    expected_referrals_time_series = [
        np.array([2, 2.0002847 , 2.00056941, 2.00085412, 2.00113884]),
        np.array([6, 6.00019708, 6.00039412, 6.00059112, 6.00078809]),
        np.array([8, 7.99951822, 7.99903647, 7.99855475, 7.99807308]),
    ]
    assert np.allclose(
        obtained_referrals_time_series[0], expected_referrals_time_series[0]
    )
    assert np.allclose(
        obtained_referrals_time_series[1], expected_referrals_time_series[1]
    )
    assert np.allclose(
        obtained_referrals_time_series[2], expected_referrals_time_series[2]
    )


def test_seasonal_gatekeeping():
    gatekeeping = sd.seasonal_gatekeeping(baseline=8, amplitude=4, phase_shift=0)
    assert callable(gatekeeping)

    obtained_referrals_time_point = gatekeeping(
        stocks=[sample_stocks[0][0], sample_stocks[1][0], sample_stocks[2][0]],
        population=sample_stocks[0][0] + sample_stocks[1][0] + sample_stocks[2][0],
        presenting_proportion=presenting_proportion,
        t=0,
    )
    expected_referrals_time_point = [2, 6, 0]
    assert np.allclose(obtained_referrals_time_point, expected_referrals_time_point)

    obtained_referrals_time_series = gatekeeping(
        stocks=[sample_stocks[0], sample_stocks[1], sample_stocks[2]],
        population=sample_stocks[0] + sample_stocks[1] + sample_stocks[2],
        presenting_proportion=presenting_proportion,
        t=ts_sample,
    )
    expected_referrals_time_series = [
        np.array([2, 2.0002847 , 2.00056941, 2.00085412, 2.00113884]),
        np.array([6, 6.00019708, 6.00039412, 6.00059112, 6.00078809]),
        np.array([0, 0.06837164, 0.13672292, 0.20503342, 0.27328278]),
    ]
    assert np.allclose(
        obtained_referrals_time_series[0], expected_referrals_time_series[0]
    )
    assert np.allclose(
        obtained_referrals_time_series[1], expected_referrals_time_series[1]
    )
    assert np.allclose(
        obtained_referrals_time_series[2], expected_referrals_time_series[2]
    )


def test_get_time_dependent_population_size():
    population_sizes = [1000, 2000, 3000]
    durations = [10, 20, 30]

    population_fn = sd.get_time_dependent_population_size(
        population_sizes=population_sizes,
        durations=durations
    )

    assert population_fn(0) == 1000
    assert population_fn(9.9) == 1000
    assert population_fn(10) == 2000
    assert population_fn(20) == 2000
    assert population_fn(30) == 3000
    assert population_fn(100) == 3000

    single_value_population_fn = sd.get_time_dependent_population_size(
        population_sizes=1000
    )

    assert single_value_population_fn(0) == 1000
    assert single_value_population_fn(9.9) == 1000
    assert single_value_population_fn(10) == 1000

    with pytest.raises(ValueError):
        sd.get_time_dependent_population_size(
            population_sizes=[1000, 2000],
            durations=[10]
        )


def test_get_time_dependent_incidence_rate():
    incidence_proportions = [0.01, 0.02, 0.03]
    durations = [10, 20, 30]

    incidence_fn = sd.get_time_dependent_incidence_rate(
        incidence_proportions=incidence_proportions,
        durations=durations
    )

    population_size = 10000
    assert incidence_fn(0, population_size) == 100
    assert incidence_fn(9.9, population_size) == 100
    assert incidence_fn(10, population_size) == 200
    assert incidence_fn(20, population_size) == 200
    assert incidence_fn(30, population_size) == 300
    assert incidence_fn(100, population_size) == 300

    single_value_incidence_fn = sd.get_time_dependent_incidence_rate(
        incidence_proportions=0.01
    )

    assert single_value_incidence_fn(0, population_size) == 100
    assert single_value_incidence_fn(9.9, population_size) == 100
    assert single_value_incidence_fn(10, population_size) == 100

    with pytest.raises(ValueError):
        sd.get_time_dependent_incidence_rate(
            incidence_proportions=[0.01, 0.02],
            durations=[10]
        )


def test_get_time_dependent_recovery_rate():
    recovery_proportions = [0.01, 0.02, 0.03]
    durations = [10, 20, 30]

    recovery_fn = sd.get_time_dependent_recovery_rate(
        recovery_proportions=recovery_proportions,
        durations=durations
    )

    stock_size = 10000
    assert recovery_fn(0, stock_size) == 100
    assert recovery_fn(9.9, stock_size) == 100
    assert recovery_fn(10, stock_size) == 200
    assert recovery_fn(20, stock_size) == 200
    assert recovery_fn(30, stock_size) == 300
    assert recovery_fn(100, stock_size) == 300

    single_value_recovery_fn = sd.get_time_dependent_recovery_rate(
        recovery_proportions=0.01
    )

    assert single_value_recovery_fn(0, stock_size) == 100
    assert single_value_recovery_fn(9.9, stock_size) == 100
    assert single_value_recovery_fn(10, stock_size) == 100

    with pytest.raises(ValueError):
        sd.get_time_dependent_recovery_rate(
            recovery_proportions=[0.01, 0.02],
            durations=[10]
        )
