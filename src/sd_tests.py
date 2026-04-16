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


def test_strict_priority_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([10.0, 10.0, 100.0])
    presenting_proportion = 0.5
    threshold = 0.2

    gatekeeping = sd.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([5.0, 5.0, 2.0])
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


def test_fixed_capacity_strict_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([10.0, 10.0, 100.0])
    presenting_proportion = 0.5
    capacity = 12.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([5.0, 5.0, 2.0])
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


def test_fixed_capacity_proportional_gatekeeping_returns_callable():
    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    assert callable(gatekeeping)


def test_fixed_capacity_proportional_gatekeeping_scalar_exact_fill():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 40.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_partial_allocation():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([3.0, 4.5, 7.5])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_no_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 0.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_full_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 50.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_order_invariant():
    stocks_a = np.array([20.0, 30.0, 50.0])
    stocks_b = np.array([50.0, 20.0, 30.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)

    obtained_a = gatekeeping(
        stocks=stocks_a,
        population=stocks_a.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    obtained_b = gatekeeping(
        stocks=stocks_b,
        population=stocks_b.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected_a = np.array([3.0, 4.5, 7.5])
    expected_b = np.array([7.5, 3.0, 4.5])

    np.testing.assert_allclose(obtained_a, expected_a)
    np.testing.assert_allclose(obtained_b, expected_b)


def test_fixed_capacity_proportional_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [3.0, 3.0, 3.0],
        [4.5, 3.75, 3.0],
        [7.5, 8.25, 9.0],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_seasonal_gatekeeping_returns_callable():
    gatekeeping = sd.seasonal_gatekeeping(baseline=8, amplitude=2, period=365, phase_shift=0)
    assert callable(gatekeeping)


def test_seasonal_gatekeeping_scalar_at_baseline_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=10.0,
        amplitude=2.0,
        period=365.0,
        phase_shift=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 2.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_scalar_partial_medium():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=15.0,
        amplitude=0.0,
        period=365.0,
        phase_shift=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 7.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([10.0, 10.0, 100.0])
    presenting_proportion = 0.5

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=12.0,
        amplitude=0.0,
        period=365.0,
        phase_shift=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([5.0, 5.0, 2.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_scalar_no_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=0.0,
        amplitude=0.0,
        period=365.0,
        phase_shift=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=10.0,
        amplitude=5.0,
        period=365.0,
        phase_shift=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=0.0,
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_scalar_full_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=50.0,
        amplitude=0.0,
        period=365.0,
        phase_shift=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    t = np.array([0.0, 1.0, 2.0])

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=10.0,
        amplitude=5.0,
        period=4.0,
        phase_shift=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=t,
    )

    expected = np.array([
        [8.0, 8.0, 8.0],
        [2.0, 7.0, 2.0],
        [0.0, 0.0, 0.0],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_phase_shift_changes_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping_no_shift = sd.seasonal_gatekeeping(
        baseline=10.0,
        amplitude=5.0,
        period=4.0,
        phase_shift=0.0,
    )
    gatekeeping_shifted = sd.seasonal_gatekeeping(
        baseline=10.0,
        amplitude=5.0,
        period=4.0,
        phase_shift=1.0,
    )

    obtained_no_shift = gatekeeping_no_shift(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )
    obtained_shifted = gatekeeping_shifted(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected_no_shift = np.array([8.0, 2.0, 0.0])
    expected_shifted = np.array([8.0, 7.0, 0.0])

    np.testing.assert_allclose(obtained_no_shift, expected_no_shift)
    np.testing.assert_allclose(obtained_shifted, expected_shifted)


def test_seasonal_gatekeeping_negative_capacity_clipped_to_zero():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_gatekeeping(
        baseline=1.0,
        amplitude=5.0,
        period=4.0,
        phase_shift=3.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.seasonal_gatekeeping(
        baseline=8.0,
        amplitude=2.0,
        period=365.0,
        phase_shift=0.0,
    )
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_proportional_access_gatekeeping_returns_callable():
    gatekeeping = sd.proportional_access_gatekeeping(threshold=0.5)
    assert callable(gatekeeping)


def test_proportional_access_gatekeeping_scalar():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = sd.proportional_access_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([4.0, 6.0, 10.0])
    np.testing.assert_allclose(obtained, expected)


def test_proportional_access_gatekeeping_scalar_zero_threshold():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    threshold = 0.0

    gatekeeping = sd.proportional_access_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_proportional_access_gatekeeping_scalar_full_threshold():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    threshold = 1.0

    gatekeeping = sd.proportional_access_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_proportional_access_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = sd.proportional_access_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_proportional_access_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = sd.proportional_access_gatekeeping(threshold)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [4.0, 4.0, 4.0],
        [6.0, 5.0, 4.0],
        [10.0, 11.0, 12.0],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_proportional_access_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.proportional_access_gatekeeping(threshold=0.5)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_severity_specific_gatekeeping_returns_callable():
    gatekeeping = sd.severity_specific_gatekeeping(proportions=[0.5, 0.3, 0.1])
    assert callable(gatekeeping)


def test_severity_specific_gatekeeping_scalar():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    proportions = np.array([0.5, 0.3, 0.1])

    gatekeeping = sd.severity_specific_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([4.0, 3.6, 2.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_gatekeeping_scalar_zero_proportions():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    proportions = np.array([0.0, 0.0, 0.0])

    gatekeeping = sd.severity_specific_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_gatekeeping_scalar_full_proportions():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    proportions = np.array([1.0, 1.0, 1.0])

    gatekeeping = sd.severity_specific_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    proportions = np.array([0.5, 0.3, 0.1])

    gatekeeping = sd.severity_specific_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    proportions = np.array([0.5, 0.3, 0.1])

    gatekeeping = sd.severity_specific_gatekeeping(proportions)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [4.0, 4.0, 4.0],
        [3.6, 3.0, 2.4],
        [2.0, 2.2, 2.4],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.severity_specific_gatekeeping(proportions=[0.5, 0.3, 0.1])
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_partial_priority_gatekeeping_returns_callable():
    gatekeeping = sd.partial_priority_gatekeeping(capacity=15.0, priority_relaxation=0.5)
    assert callable(gatekeeping)


def test_partial_priority_gatekeeping_scalar_full_strict_priority():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.0

    gatekeeping = sd.partial_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 7.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_scalar_full_proportional():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 1.0

    gatekeeping = sd.partial_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([3.0, 4.5, 7.5])
    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_scalar_halfway_blend():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.5

    gatekeeping = sd.partial_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([5.5, 5.75, 3.75])
    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_scalar_no_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 0.0
    priority_relaxation = 0.5

    gatekeeping = sd.partial_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.5

    gatekeeping = sd.partial_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_scalar_full_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 40.0
    priority_relaxation = 0.5

    gatekeeping = sd.partial_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.5

    gatekeeping = sd.partial_priority_gatekeeping(capacity, priority_relaxation)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [5.5, 5.5, 5.5],
        [5.75, 5.375, 5.0],
        [3.75, 4.125, 4.5],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.partial_priority_gatekeeping(capacity=15.0, priority_relaxation=0.5)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_partial_priority_gatekeeping_zero_relaxation_matches_fixed_capacity_strict():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 15.0

    blended = sd.partial_priority_gatekeeping(capacity=capacity, priority_relaxation=0.0)
    strict = sd.fixed_capacity_strict_gatekeeping(capacity=capacity)

    obtained_blended = blended(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )
    obtained_strict = strict(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    np.testing.assert_allclose(obtained_blended, obtained_strict)


def test_partial_priority_gatekeeping_full_relaxation_matches_fixed_capacity_proportional():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    capacity = 15.0

    blended = sd.partial_priority_gatekeeping(capacity=capacity, priority_relaxation=1.0)
    proportional = sd.fixed_capacity_proportional_gatekeeping(capacity=capacity)

    obtained_blended = blended(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )
    obtained_proportional = proportional(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    np.testing.assert_allclose(obtained_blended, obtained_proportional)


def test_severity_responsive_gatekeeping_returns_callable():
    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )
    assert callable(gatekeeping)


def test_severity_responsive_gatekeeping_scalar_below_threshold_uses_low_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 2.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_above_threshold_uses_high_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_at_threshold_uses_high_capacity():
    stocks = np.array([30.0, 30.0, 40.0])
    presenting_proportion = 0.4

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([12.0, 8.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_no_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=0.0,
        high_severity_capacity=0.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_full_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=50.0,
        high_severity_capacity=50.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([10.0, 10.0, 100.0])
    presenting_proportion = 0.5

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.05,
        low_severity_capacity=12.0,
        high_severity_capacity=12.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([5.0, 5.0, 2.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_time_series_case():
    stocks = np.array([
        [20.0, 50.0, 30.0],
        [30.0, 30.0, 30.0],
        [50.0, 20.0, 40.0],
    ])
    presenting_proportion = 0.4

    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [8.0, 20.0, 12.0],
        [2.0, 0.0, 8.0],
        [0.0, 0.0, 0.0],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_equal_capacities_matches_fixed_capacity_strict():
    stocks = np.array([
        [20.0, 50.0, 30.0], 
        [30.0, 30.0, 30.0], 
        [50.0, 20.0, 40.0], 
    ])
    presenting_proportion = 0.4
    capacity = 15.0

    severity_responsive = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=capacity,
        high_severity_capacity=capacity,
    )
    fixed_strict = sd.fixed_capacity_strict_gatekeeping(capacity=capacity)

    obtained_severity_responsive = severity_responsive(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )
    obtained_fixed_strict = fixed_strict(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array([
        [8.0, 15.0, 12.0],
        [7.0,  0.0,  3.0],
        [0.0,  0.0,  0.0],
    ])

    np.testing.assert_allclose(obtained_severity_responsive, obtained_fixed_strict)
    np.testing.assert_allclose(obtained_severity_responsive, expected)


def test_severity_responsive_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_time_phased_gatekeeping_returns_callable():
    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0],
        gatekeeping_policies=[
            sd.fixed_capacity_strict_gatekeeping(capacity=15.0),
            sd.fixed_capacity_proportional_gatekeeping(capacity=15.0),
        ],
    )
    assert callable(gatekeeping)


def test_time_phased_gatekeeping_raises_for_wrong_number_of_policies():
    with pytest.raises(
        ValueError,
        match="There must be exactly one more gatekeeping policy than change times.",
    ):
        sd.time_phased_gatekeeping(
            change_times=[10.0, 20.0],
            gatekeeping_policies=[
                sd.fixed_capacity_strict_gatekeeping(capacity=15.0),
                sd.fixed_capacity_proportional_gatekeeping(capacity=15.0),
            ],
        )


def test_time_phased_gatekeeping_raises_for_unsorted_change_times():
    with pytest.raises(
        ValueError,
        match="change_times must be sorted in non-decreasing order.",
    ):
        sd.time_phased_gatekeeping(
            change_times=[20.0, 10.0],
            gatekeeping_policies=[
                sd.fixed_capacity_strict_gatekeeping(capacity=15.0),
                sd.fixed_capacity_proportional_gatekeeping(capacity=15.0),
                sd.proportional_access_gatekeeping(threshold=0.5),
            ],
        )


def test_time_phased_gatekeeping_scalar_before_first_change_uses_first_policy():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    phase_one = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)

    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0],
        gatekeeping_policies=[phase_one, phase_two],
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=5.0,
    )

    expected = phase_one(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=5.0,
    )

    np.testing.assert_allclose(obtained, expected)


def test_time_phased_gatekeeping_scalar_at_change_uses_next_policy():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    phase_one = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)

    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0],
        gatekeeping_policies=[phase_one, phase_two],
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=10.0,
    )

    expected = phase_two(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=10.0,
    )

    np.testing.assert_allclose(obtained, expected)


def test_time_phased_gatekeeping_scalar_after_last_change_uses_final_policy():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    phase_one = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    phase_three = sd.proportional_access_gatekeeping(threshold=0.5)

    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0, 20.0],
        gatekeeping_policies=[phase_one, phase_two, phase_three],
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=25.0,
    )

    expected = phase_three(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=25.0,
    )

    np.testing.assert_allclose(obtained, expected)


def test_time_phased_gatekeeping_scalar_between_changes_uses_middle_policy():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    phase_one = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    phase_three = sd.proportional_access_gatekeeping(threshold=0.5)

    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0, 20.0],
        gatekeeping_policies=[phase_one, phase_two, phase_three],
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=15.0,
    )

    expected = phase_two(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=15.0,
    )

    np.testing.assert_allclose(obtained, expected)


def test_time_phased_gatekeeping_time_series_crosses_multiple_phases():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    t = np.array([5.0, 15.0, 25.0])

    phase_one = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    phase_three = sd.proportional_access_gatekeeping(threshold=0.5)

    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0, 20.0],
        gatekeeping_policies=[phase_one, phase_two, phase_three],
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=t,
    )

    expected = np.column_stack([
        phase_one(
            stocks=stocks[:, 0],
            population=stocks[:, 0].sum(),
            presenting_proportion=presenting_proportion,
            t=t[0],
        ),
        phase_two(
            stocks=stocks[:, 1],
            population=stocks[:, 1].sum(),
            presenting_proportion=presenting_proportion,
            t=t[1],
        ),
        phase_three(
            stocks=stocks[:, 2],
            population=stocks[:, 2].sum(),
            presenting_proportion=presenting_proportion,
            t=t[2],
        ),
    ])

    np.testing.assert_allclose(obtained, expected)


def test_time_phased_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4

    phase_one = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)

    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0],
        gatekeeping_policies=[phase_one, phase_two],
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=0.0,
        presenting_proportion=presenting_proportion,
        t=5.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_time_phased_gatekeeping_single_policy_no_changes_matches_base_policy():
    stocks = np.array([
        [20.0, 20.0, 20.0],
        [30.0, 25.0, 20.0],
        [50.0, 55.0, 60.0],
    ])
    presenting_proportion = 0.4
    t = np.array([0.0, 10.0, 20.0])

    base_policy = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)

    wrapped_policy = sd.time_phased_gatekeeping(
        change_times=[],
        gatekeeping_policies=[base_policy],
    )

    obtained_wrapped = wrapped_policy(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=t,
    )

    obtained_base = base_policy(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=t,
    )

    np.testing.assert_allclose(obtained_wrapped, obtained_base)


def test_time_phased_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.time_phased_gatekeeping(
        change_times=[10.0],
        gatekeeping_policies=[
            sd.fixed_capacity_strict_gatekeeping(capacity=15.0),
            sd.fixed_capacity_proportional_gatekeeping(capacity=15.0),
        ],
    )
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(ValueError, match="stocks must be a 1D or 2D array-like structure."):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=5.0,
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
