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


def test_proportional_gatekeeping():
    gatekeeping = sd.proportional_gatekeeping(threshold=proportional_threshold)
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
        np.array([2, 2.0002847, 2.00056941, 2.00085412, 2.00113884]),
        np.array([6, 5.99964522, 5.99929043, 5.99893564, 5.99858085]),
        np.array([0, 0, 0, 0, 0]),
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


def test_fixed_gatekeeping():
    gatekeeping = sd.fixed_gatekeeping(threshold=fixed_threshold)
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
