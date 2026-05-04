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
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
    presenting_proportion = 0.4

    gatekeeping = sd.strict_priority_gatekeeping(threshold=0.5)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [8.0, 8.0, 8.0],
            [12.0, 10.0, 8.0],
            [0.0, 2.0, 4.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.strict_priority_gatekeeping(threshold=0.5)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
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
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [8.0, 8.0, 8.0],
            [7.0, 7.0, 7.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
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
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [3.0, 3.0, 3.0],
            [4.5, 3.75, 3.0],
            [7.5, 8.25, 9.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_seasonal_capacity_gatekeeping_returns_callable():
    gatekeeping = sd.seasonal_capacity_gatekeeping(
        baseline=8, amplitude=2, period=365, phase_shift=0
    )
    assert callable(gatekeeping)


def test_seasonal_capacity_gatekeeping_scalar_at_baseline_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_scalar_partial_medium():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([10.0, 10.0, 100.0])
    presenting_proportion = 0.5

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_scalar_no_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_scalar_full_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
    presenting_proportion = 0.4
    t = np.array([0.0, 1.0, 2.0])

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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

    expected = np.array(
        [
            [8.0, 8.0, 8.0],
            [2.0, 7.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_phase_shift_changes_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping_no_shift = sd.seasonal_capacity_gatekeeping(
        baseline=10.0,
        amplitude=5.0,
        period=4.0,
        phase_shift=0.0,
    )
    gatekeeping_shifted = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_negative_capacity_clipped_to_zero():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_capacity_gatekeeping(
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


def test_seasonal_capacity_gatekeeping_time_series_zero_capacity_continue_branch():
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
            [50.0, 50.0, 50.0],
        ]
    )
    presenting_proportion = 0.4

    gatekeeping = sd.seasonal_capacity_gatekeeping(
        baseline=1.0,
        amplitude=1.0,
        period=4.0,
        phase_shift=3.0,
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.seasonal_capacity_gatekeeping(
        baseline=8.0,
        amplitude=2.0,
        period=365.0,
        phase_shift=0.0,
    )
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_equal_access_proportion_gatekeeping_returns_callable():
    gatekeeping = sd.equal_access_proportion_gatekeeping(access_proportion=0.5)
    assert callable(gatekeeping)


def test_equal_access_proportion_gatekeeping_scalar():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    access_proportion = 0.5

    gatekeeping = sd.equal_access_proportion_gatekeeping(access_proportion)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([4.0, 6.0, 10.0])
    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_scalar_zero_threshold():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    access_proportion = 0.0

    gatekeeping = sd.equal_access_proportion_gatekeeping(access_proportion)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_scalar_full_threshold():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    access_proportion = 1.0

    gatekeeping = sd.equal_access_proportion_gatekeeping(access_proportion)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    access_proportion = 0.5

    gatekeeping = sd.equal_access_proportion_gatekeeping(access_proportion)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
    presenting_proportion = 0.4
    access_proportion = 0.5

    gatekeeping = sd.equal_access_proportion_gatekeeping(access_proportion)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [4.0, 4.0, 4.0],
            [6.0, 5.0, 4.0],
            [10.0, 11.0, 12.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.equal_access_proportion_gatekeeping(access_proportion=0.5)
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_severity_specific_access_gatekeeping_returns_callable():
    gatekeeping = sd.severity_specific_access_gatekeeping(proportions=[0.5, 0.3, 0.1])
    assert callable(gatekeeping)


def test_severity_specific_access_gatekeeping_scalar():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    proportions = np.array([0.5, 0.3, 0.1])

    gatekeeping = sd.severity_specific_access_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([4.0, 3.6, 2.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_scalar_zero_proportions():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    proportions = np.array([0.0, 0.0, 0.0])

    gatekeeping = sd.severity_specific_access_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_scalar_full_proportions():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4
    proportions = np.array([1.0, 1.0, 1.0])

    gatekeeping = sd.severity_specific_access_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([8.0, 12.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    proportions = np.array([0.5, 0.3, 0.1])

    gatekeeping = sd.severity_specific_access_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
    presenting_proportion = 0.4
    proportions = np.array([0.5, 0.3, 0.1])

    gatekeeping = sd.severity_specific_access_gatekeeping(proportions)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [4.0, 4.0, 4.0],
            [3.6, 3.0, 2.4],
            [2.0, 2.2, 2.4],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.severity_specific_access_gatekeeping(proportions=[0.5, 0.3, 0.1])
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=0.0,
        )


def test_partial_priority_gatekeeping_returns_callable():
    gatekeeping = sd.partial_priority_gatekeeping(
        capacity=15.0, priority_relaxation=0.5
    )
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
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
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

    expected = np.array(
        [
            [5.5, 5.5, 5.5],
            [5.75, 5.375, 5.0],
            [3.75, 4.125, 4.5],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_partial_priority_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.partial_priority_gatekeeping(
        capacity=15.0, priority_relaxation=0.5
    )
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
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

    blended = sd.partial_priority_gatekeeping(
        capacity=capacity, priority_relaxation=0.0
    )
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

    blended = sd.partial_priority_gatekeeping(
        capacity=capacity, priority_relaxation=1.0
    )
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


def test_partial_priority_gatekeeping_time_series_zero_demand_branch():
    stocks = np.array(
        [
            [0.0, 20.0, 20.0],
            [0.0, 30.0, 25.0],
            [0.0, 50.0, 55.0],
        ]
    )
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

    expected = np.array(
        [
            [0.0, 5.5, 5.5],
            [0.0, 5.75, 5.375],
            [0.0, 3.75, 4.125],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


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
    stocks = np.array(
        [
            [20.0, 50.0, 30.0],
            [30.0, 30.0, 30.0],
            [50.0, 20.0, 40.0],
        ]
    )
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

    expected = np.array(
        [
            [8.0, 20.0, 12.0],
            [2.0, 0.0, 8.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_equal_capacities_matches_fixed_capacity_strict():
    stocks = np.array(
        [
            [20.0, 50.0, 30.0],
            [30.0, 30.0, 30.0],
            [50.0, 20.0, 40.0],
        ]
    )
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

    expected = np.array(
        [
            [8.0, 15.0, 12.0],
            [7.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(obtained_severity_responsive, obtained_fixed_strict)
    np.testing.assert_allclose(obtained_severity_responsive, expected)


def test_severity_responsive_gatekeeping_time_series_zero_demand_continue_branch():
    stocks = np.array(
        [
            [0.0, 20.0, 50.0],
            [0.0, 30.0, 30.0],
            [0.0, 50.0, 20.0],
        ]
    )
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

    expected = np.array(
        [
            [0.0, 8.0, 20.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = sd.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )
    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
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
                sd.equal_access_proportion_gatekeeping(access_proportion=0.5),
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
    phase_three = sd.equal_access_proportion_gatekeeping(access_proportion=0.5)

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
    phase_three = sd.equal_access_proportion_gatekeeping(access_proportion=0.5)

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
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
    presenting_proportion = 0.4
    t = np.array([5.0, 15.0, 25.0])

    phase_one = sd.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = sd.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    phase_three = sd.equal_access_proportion_gatekeeping(access_proportion=0.5)

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

    expected = np.column_stack(
        [
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
        ]
    )

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
    stocks = np.array(
        [
            [20.0, 20.0, 20.0],
            [30.0, 25.0, 20.0],
            [50.0, 55.0, 60.0],
        ]
    )
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

    with pytest.raises(
        ValueError, match="stocks must be a 1D or 2D array-like structure."
    ):
        gatekeeping(
            stocks=stocks,
            population=1.0,
            presenting_proportion=0.4,
            t=5.0,
        )


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


def test_boundary_based_deterioration_returns_expected_transition_rates():
    deterioration_fn = sd.get_boundary_based_deterioration_rates(
        category_widths=(0.50, 0.25, 0.25),
        shift_proportion=0.05,
        shift_interval_days=182.5,
    )

    obtained = deterioration_fn(t=0.0)

    expected_r_lm = 0.05 / (0.50 * 182.5)
    expected_r_mh = 0.05 / (0.25 * 182.5)

    np.testing.assert_allclose(obtained, (expected_r_lm, expected_r_mh))


def test_boundary_based_deterioration_raises_value_error_when_category_widths_do_not_sum_to_one():
    with pytest.raises(ValueError, match="category_widths must sum to 1."):
        sd.get_boundary_based_deterioration_rates(
            category_widths=(0.40, 0.25, 0.25),
            shift_proportion=0.05,
            shift_interval_days=182.5,
        )

def test_boundary_based_deterioration_raises_value_error_when_any_category_width_is_non_positive():
    with pytest.raises(ValueError, match="all category widths must be positive."):
        sd.get_boundary_based_deterioration_rates(
            category_widths=(0.50, 0.00, 0.50),
            shift_proportion=0.05,
            shift_interval_days=182.5,
        )


def test_boundary_based_deterioration_raises_value_error_when_shift_proportion_is_negative():
    with pytest.raises(ValueError, match="shift_proportion must be non-negative."):
        sd.get_boundary_based_deterioration_rates(
            category_widths=(0.50, 0.25, 0.25),
            shift_proportion=-0.01,
            shift_interval_days=182.5,
        )


def test_boundary_based_deterioration_raises_value_error_when_shift_interval_days_is_non_positive():
    with pytest.raises(ValueError, match="shift_interval_days must be positive."):
        sd.get_boundary_based_deterioration_rates(
            category_widths=(0.50, 0.25, 0.25),
            shift_proportion=0.05,
            shift_interval_days=-1.0,
        )


def test_boundary_based_deterioration_raises_value_error_when_shift_proportion_exceeds_smallest_transition_width():
    with pytest.raises(
        ValueError,
        match="shift_proportion must be less than or equal to the smallest transition category width.",
    ):
        sd.get_boundary_based_deterioration_rates(
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
    unwell_splits=(0.2, 0.3, 0.5),
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
        unwell_splits=(0.5, 0.3, 0.2),
    )

    expected_unwell_population = 1000 * 0.2

    assert model.P[0] == expected_unwell_population * 0.5
    assert model.P[1] == expected_unwell_population * 0.3
    assert model.P[2] == expected_unwell_population * 0.2


def test_sd_initial_stock_sizes_sum_to_initial_unwell_population():
    model = get_simple_sd_model(
        initial_unwell_proportion=0.1,
        unwell_splits=(0.2, 0.3, 0.5),
    )

    expected_unwell_population = 1000 * 0.1

    assert sum(model.P) == expected_unwell_population


def test_sd_initialises_time_and_lambdas_attributes():
    model = get_simple_sd_model(
        initial_unwell_proportion=0.1,
        unwell_splits=(0.2, 0.3, 0.5),
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
        y=(10.0, 20.0, 30.0),
        time_domain=5.0,
    )

    assert obtained == (0.0, 0.0, 0.0)


def test_sd_differential_equations_returns_decrease_under_only_referrals():
    model = get_simple_sd_model(
        gatekeeping_function=lambda stocks, population, presenting_proportion, t: [
            1.0,
            2.0,
            3.0,
        ],
    )

    obtained = model.differential_equations(
        y=(10.0, 20.0, 30.0),
        time_domain=5.0,
    )

    expected = (-1.0, -2.0, -3.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_flow_under_only_deterioration():
    model = get_simple_sd_model(
        deterioration_function=lambda t: 0.1,
    )

    obtained = model.differential_equations(
        y=(10.0, 20.0, 30.0),
        time_domain=5.0,
    )

    expected = (2.0, 1.0, -3.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_increase_under_only_incidence():
    model = get_simple_sd_model(
        incidence_function=lambda t, population_size: 5.0,
    )

    obtained = model.differential_equations(
        y=(10.0, 20.0, 30.0),
        time_domain=5.0,
    )

    expected = (0.0, 0.0, 5.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_decrease_under_only_recovery():
    model = get_simple_sd_model(
        recovery_function=lambda t, stock_size: 0.1 * stock_size,
    )

    obtained = model.differential_equations(
        y=(10.0, 20.0, 30.0),
        time_domain=5.0,
    )

    expected = (0.0, 0.0, -6.0)

    assert obtained == expected


def test_sd_differential_equations_returns_expected_values():
    model = get_simple_sd_model(
        gatekeeping_function=lambda stocks, population, presenting_proportion, t: [
            1.0,
            2.0,
            3.0,
        ],
        deterioration_function=lambda t: 0.1,
        incidence_function=lambda t, population_size: 5.0,
        recovery_function=lambda t, stock_size: 6.0,
    )

    obtained = model.differential_equations(
        y=(10.0, 20.0, 30.0),
        time_domain=5.0,
    )

    expected = (1.0, -1.0, -7.0)

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
        y=(10.0, 20.0, 30.0),
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
        y=(10.0, 20.0, 30.0),
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
        y=(10.0, 20.0, 30.0),
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
        unwell_splits=(0.2, 0.3, 0.5),
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
