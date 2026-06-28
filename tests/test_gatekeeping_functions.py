import numpy as np
import pytest
import hybridsim.gatekeeping_functions as gf


# Helper function
# ---------------

def assert_referral_flows_are_feasible(
    obtained,
    stocks,
    presenting_proportion,
):
    obtained = np.asarray(obtained, dtype=float)
    stocks = np.asarray(stocks, dtype=float)
    demand = presenting_proportion * stocks

    assert np.all(obtained >= 0.0)
    assert np.all(obtained <= demand + 1e-12)


# Tests for valid gatekeeping referral and capacity constraints

@pytest.mark.parametrize(
    "gatekeeping",
    [
        gf.strict_priority_gatekeeping(threshold=0.5),
        gf.fixed_capacity_strict_gatekeeping(capacity=15.0),
        gf.fixed_capacity_proportional_gatekeeping(capacity=15.0),
        gf.weighted_priority_gatekeeping(
            threshold=0.5,
            weights=(1.0, 2.0, 5.0),
        ),
        gf.seasonal_capacity_gatekeeping(
            baseline=10.0,
            amplitude=2.0,
            period=365.0,
            phase_shift=0.0,
        ),
        gf.equal_access_proportion_gatekeeping(access_proportion=0.5),
        gf.severity_specific_access_gatekeeping(
            proportions=(0.1, 0.3, 0.5),
        ),
        gf.split_capacity_priority_gatekeeping(
            capacity=15.0,
            priority_relaxation=0.5,
        ),
        gf.severity_responsive_gatekeeping(
            severity_threshold=0.3,
            low_severity_capacity=10.0,
            high_severity_capacity=20.0,
        ),
    ],
)
def test_gatekeeping_policies_return_non_negative_referrals_not_exceeding_demand(
    gatekeeping,
):
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    assert_referral_flows_are_feasible(
        obtained=obtained,
        stocks=stocks,
        presenting_proportion=presenting_proportion,
    )


@pytest.mark.parametrize(
    "gatekeeping, maximum_capacity",
    [
        (gf.strict_priority_gatekeeping(threshold=0.5), 20.0),
        (gf.fixed_capacity_strict_gatekeeping(capacity=15.0), 15.0),
        (gf.fixed_capacity_proportional_gatekeeping(capacity=15.0), 15.0),
        (
            gf.weighted_priority_gatekeeping(
                threshold=0.5,
                weights=(1.0, 2.0, 5.0),
            ),
            20.0,
        ),
        (
            gf.seasonal_capacity_gatekeeping(
                baseline=10.0,
                amplitude=0.0,
                period=365.0,
                phase_shift=0.0,
            ),
            10.0,
        ),
        (
            gf.split_capacity_priority_gatekeeping(
                capacity=15.0,
                priority_relaxation=0.5,
            ),
            15.0,
        ),
        (
            gf.severity_responsive_gatekeeping(
                severity_threshold=0.3,
                low_severity_capacity=10.0,
                high_severity_capacity=20.0,
            ),
            20.0,
        ),
    ],
)
def test_capacity_constrained_gatekeeping_policies_do_not_exceed_capacity(
    gatekeeping,
    maximum_capacity,
):
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    assert obtained.sum() <= maximum_capacity + 1e-12


# Strict-priority gatekeeping tests
# ---------------------------------

def test_strict_priority_gatekeeping_returns_callable():
    gatekeeping = gf.strict_priority_gatekeeping(threshold=0.5)
    assert callable(gatekeeping)


def test_strict_priority_gatekeeping_scalar_exact_fill():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = gf.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_partial_medium():
    stocks = np.array([20.0, 40.0, 40.0])
    presenting_proportion = 0.4
    threshold = 0.5

    gatekeeping = gf.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 4.0, 16.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([100.0, 10.0, 10.0])
    presenting_proportion = 0.5
    threshold = 0.2

    gatekeeping = gf.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([2.0, 5.0, 5.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_zero_threshold():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    threshold = 0.0

    gatekeeping = gf.strict_priority_gatekeeping(threshold)
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

    gatekeeping = gf.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_full_threshold():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    threshold = 1.0

    gatekeeping = gf.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_scalar_all_capacity_to_first_group():
    stocks = np.array([10.0, 20.0, 100.0])
    presenting_proportion = 0.5
    threshold = 0.2

    gatekeeping = gf.strict_priority_gatekeeping(threshold)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 13.0])
    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4

    gatekeeping = gf.strict_priority_gatekeeping(threshold=0.5)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [0.0, 2.0, 4.0],
            [12.0, 10.0, 8.0],
            [8.0, 8.0, 8.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_strict_priority_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.strict_priority_gatekeeping(threshold=0.5)
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


# Fixed-capacity strict gatekeeping tests
# ---------------------------------------

def test_fixed_capacity_strict_gatekeeping_returns_callable():
    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
    assert callable(gatekeeping)


def test_fixed_capacity_strict_gatekeeping_scalar_exact_fill():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 20.0

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_partial_medium():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 7.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([100.0, 10.0, 10.0])
    presenting_proportion = 0.5
    capacity = 12.0

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([2.0, 5.0, 5.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_no_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 0.0

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)
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

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_full_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 40.0

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_scalar_all_capacity_to_first_group():
    stocks = np.array([10.0, 20.0, 100.0])
    presenting_proportion = 0.5
    capacity = 13.0

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 13.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_strict_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
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


# Fixed-capacity proportional gatekeeping tests
# ---------------------------------------------

def test_fixed_capacity_proportional_gatekeeping_returns_callable():
    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    assert callable(gatekeeping)


def test_fixed_capacity_proportional_gatekeeping_scalar_exact_fill():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 40.0

    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_partial_allocation():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([7.5, 4.5, 3.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_no_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 0.0

    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity)
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

    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_full_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 50.0

    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_scalar_order_invariant():
    stocks_a = np.array([20.0, 30.0, 50.0])
    stocks_b = np.array([50.0, 20.0, 30.0])
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity)

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
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    capacity = 15.0

    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [7.5, 8.25, 9.0],
            [4.5, 3.75, 3.0],
            [3.0, 3.0, 3.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_fixed_capacity_proportional_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)
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


# Weighted-priority gatekeeping tests
# -----------------------------------

def test_weighted_priority_gatekeeping_scalar_returns_expected_values():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.5,
        weights=(1.0, 2.0, 5.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([5.45454545, 6.54545455, 8.0])

    np.testing.assert_allclose(obtained, expected)


def test_weighted_priority_gatekeeping_scalar_redistributes_remaining_capacity():
    stocks = np.array([10.0, 10.0, 1.0])
    presenting_proportion = 1.0

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.8,
        weights=(1.0, 1.0, 100.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([7.9, 7.9, 1.0])

    np.testing.assert_allclose(obtained, expected)


def test_weighted_priority_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.5,
        weights=(1.0, 2.0, 5.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=0.0,
        presenting_proportion=0.4,
        t=0.0,
    )

    np.testing.assert_allclose(obtained, np.zeros_like(stocks))


def test_weighted_priority_gatekeeping_returns_equal_allocation_with_equal_weights():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.5,
        weights=(1.0, 1.0, 1.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    demand = presenting_proportion * stocks
    expected = 0.5 * demand

    np.testing.assert_allclose(obtained, expected)


def test_weighted_priority_gatekeeping_gives_higher_priority_groups_larger_allocations_when_demands_are_equal():
    stocks = np.array([10.0, 10.0, 10.0])
    presenting_proportion = 1.0

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.6,
        weights=(1.0, 2.0, 3.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    assert obtained[2] > obtained[1] > obtained[0]
    np.testing.assert_allclose(obtained.sum(), 0.6 * stocks.sum())


def test_weighted_priority_gatekeeping_zero_threshold():
    stocks = np.array([50.0, 30.0, 20.0])

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.0,
        weights=(1.0, 2.0, 5.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=0.4,
        t=0.0,
    )

    np.testing.assert_allclose(obtained, np.zeros_like(stocks))


def test_weighted_priority_gatekeeping_full_threshold():
    stocks = np.array([50.0, 30.0, 20.0])

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=1.0,
        weights=(1.0, 2.0, 3.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=0.4,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])

    np.testing.assert_allclose(obtained, expected)


def test_weighted_priority_gatekeeping_time_series_returns_expected_values():
    stocks = np.array([
        [50.0, 40.0, 60.0],
        [30.0, 20.0, 10.0],
        [20.0, 10.0, 30.0],
    ])
    presenting_proportion = 0.4
    t = np.array([0.0, 1.0, 2.0])

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.5,
        weights=(1.0, 2.0, 5.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=t,
    )

    expected = np.array([
        [5.45454545, 5.0, 6.0],
        [6.54545455, 5.0, 2.0],
        [8.0, 4.0, 12.0],
    ])

    np.testing.assert_allclose(obtained, expected)


def test_weighted_priority_gatekeeping_breaks_when_only_zero_weight_eligible_groups_remain():
    stocks = np.array([10.0, 10.0, 1.0])
    presenting_proportion = 1.0

    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.8,
        weights=(0.0, 0.0, 1.0),
    )

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(obtained, expected)


def test_weighted_priority_gatekeeping_raises_value_error_when_weights_contain_negative_values():
    with pytest.raises(ValueError, match="weights must be non-negative."):
        gf.weighted_priority_gatekeeping(
            threshold=0.5,
            weights=(1.0, -2.0, 5.0),
        )


def test_weighted_priority_gatekeeping_raises_value_error_when_all_weights_are_zero():
    with pytest.raises(ValueError, match="at least one weight must be positive."):
        gf.weighted_priority_gatekeeping(
            threshold=0.5,
            weights=(0.0, 0.0, 0.0),
        )


def test_weighted_priority_gatekeeping_raises_value_error_for_invalid_stock_dimension():
    gatekeeping = gf.weighted_priority_gatekeeping(
        threshold=0.5,
        weights=(1.0, 2.0, 5.0),
    )

    stocks = np.zeros((3, 2, 2))

    with pytest.raises(
        ValueError,
        match="stocks must be a 1D or 2D array-like structure.",
    ):
        gatekeeping(
            stocks=stocks,
            population=0.0,
            presenting_proportion=0.4,
            t=0.0,
        )


# Seasonal-capacity gatekeeping tests
# -----------------------------------

def test_seasonal_capacity_gatekeeping_returns_callable():
    gatekeeping = gf.seasonal_capacity_gatekeeping(
        baseline=8, amplitude=2, period=365, phase_shift=0
    )
    assert callable(gatekeeping)


def test_seasonal_capacity_gatekeeping_scalar_at_baseline_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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

    expected = np.array([0.0, 2.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_scalar_partial_medium():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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

    expected = np.array([0.0, 7.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([100.0, 10.0, 10.0])
    presenting_proportion = 0.5

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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

    expected = np.array([2.0, 5.0, 5.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_scalar_no_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    t = np.array([0.0, 1.0, 2.0])

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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
            [0.0, 0.0, 0.0],
            [2.0, 7.0, 2.0],
            [8.0, 8.0, 8.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_phase_shift_changes_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping_no_shift = gf.seasonal_capacity_gatekeeping(
        baseline=10.0,
        amplitude=5.0,
        period=4.0,
        phase_shift=0.0,
    )
    gatekeeping_shifted = gf.seasonal_capacity_gatekeeping(
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

    expected_no_shift = np.array([0.0, 2.0, 8.0])
    expected_shifted = np.array([0.0, 7.0, 8.0])

    np.testing.assert_allclose(obtained_no_shift, expected_no_shift)
    np.testing.assert_allclose(obtained_shifted, expected_shifted)


def test_seasonal_capacity_gatekeeping_negative_capacity_clipped_to_zero():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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
            [50.0, 50.0, 50.0],
            [30.0, 30.0, 30.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4

    gatekeeping = gf.seasonal_capacity_gatekeeping(
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
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_seasonal_capacity_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.seasonal_capacity_gatekeeping(
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


# Equal-access-proportion gatekeeping tests
# -----------------------------------------

def test_equal_access_proportion_gatekeeping_returns_callable():
    gatekeeping = gf.equal_access_proportion_gatekeeping(access_proportion=0.5)
    assert callable(gatekeeping)


def test_equal_access_proportion_gatekeeping_scalar():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    access_proportion = 0.5

    gatekeeping = gf.equal_access_proportion_gatekeeping(access_proportion)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([10.0, 6.0, 4.0])
    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_scalar_zero_threshold():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    access_proportion = 0.0

    gatekeeping = gf.equal_access_proportion_gatekeeping(access_proportion)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_scalar_full_threshold():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    access_proportion = 1.0

    gatekeeping = gf.equal_access_proportion_gatekeeping(access_proportion)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    access_proportion = 0.5

    gatekeeping = gf.equal_access_proportion_gatekeeping(access_proportion)
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
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    access_proportion = 0.5

    gatekeeping = gf.equal_access_proportion_gatekeeping(access_proportion)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [10.0, 11.0, 12.0],
            [6.0, 5.0, 4.0],
            [4.0, 4.0, 4.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_equal_access_proportion_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.equal_access_proportion_gatekeeping(access_proportion=0.5)
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


# Severity-specific access gatekeeping tests
# ------------------------------------------

def test_severity_specific_access_gatekeeping_returns_callable():
    gatekeeping = gf.severity_specific_access_gatekeeping(proportions=[0.1, 0.3, 0.5])
    assert callable(gatekeeping)


def test_severity_specific_access_gatekeeping_scalar():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    proportions = np.array([0.1, 0.3, 0.5])

    gatekeeping = gf.severity_specific_access_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([2.0, 3.6, 4.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_scalar_zero_proportions():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    proportions = np.array([0.0, 0.0, 0.0])

    gatekeeping = gf.severity_specific_access_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_scalar_full_proportions():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    proportions = np.array([1.0, 1.0, 1.0])

    gatekeeping = gf.severity_specific_access_gatekeeping(proportions)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    proportions = np.array([0.1, 0.3, 0.5])

    gatekeeping = gf.severity_specific_access_gatekeeping(proportions)
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
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    proportions = np.array([0.1, 0.3, 0.5])

    gatekeeping = gf.severity_specific_access_gatekeeping(proportions)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [2.0, 2.2, 2.4],
            [3.6, 3.0, 2.4],
            [4.0, 4.0, 4.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_severity_specific_access_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.severity_specific_access_gatekeeping(proportions=[0.1, 0.3, 0.5])
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


# Split-capacity-priority gatekeeping tests
# -----------------------------------------

def test_split_capacity_priority_gatekeeping_returns_callable():
    gatekeeping = gf.split_capacity_priority_gatekeeping(
        capacity=15.0, priority_relaxation=0.5
    )
    assert callable(gatekeeping)


def test_split_capacity_priority_gatekeeping_scalar_full_strict_priority():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.0

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 7.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_scalar_full_proportional():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 1.0

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([7.5, 4.5, 3.0])
    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_scalar_halfway_blend():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.5

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([4.61538462, 2.76923077, 7.61538462])
    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_scalar_no_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 0.0
    priority_relaxation = 0.5

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_scalar_empty_stocks():
    stocks = np.array([0.0, 0.0, 0.0])
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.5

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_scalar_full_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 40.0
    priority_relaxation = 0.5

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)
    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(),
        presenting_proportion=presenting_proportion,
        t=0.0,
    )

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.5

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [4.61538462, 5.07692308, 5.53846154],
            [2.76923077, 2.30769231, 1.84615385],
            [7.61538462, 7.61538462, 7.61538462],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_zero_relaxation_matches_fixed_capacity_strict():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 15.0

    blended = gf.split_capacity_priority_gatekeeping(
        capacity=capacity, priority_relaxation=0.0
    )
    strict = gf.fixed_capacity_strict_gatekeeping(capacity=capacity)

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


def test_split_capacity_priority_gatekeeping_full_relaxation_matches_fixed_capacity_proportional():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4
    capacity = 15.0

    blended = gf.split_capacity_priority_gatekeeping(
        capacity=capacity, priority_relaxation=1.0
    )
    proportional = gf.fixed_capacity_proportional_gatekeeping(capacity=capacity)

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


def test_split_capacity_priority_gatekeeping_time_series_zero_demand_branch():
    stocks = np.array(
        [
            [0.0, 50.0, 55.0],
            [0.0, 30.0, 25.0],
            [0.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    capacity = 15.0
    priority_relaxation = 0.5

    gatekeeping = gf.split_capacity_priority_gatekeeping(capacity, priority_relaxation)

    obtained = gatekeeping(
        stocks=stocks,
        population=stocks.sum(axis=0),
        presenting_proportion=presenting_proportion,
        t=np.array([0.0, 1.0, 2.0]),
    )

    expected = np.array(
        [
            [0.0, 4.61538462, 5.07692308],
            [0.0, 2.76923077, 2.30769231],
            [0.0, 7.61538462, 7.61538462],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_split_capacity_priority_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.split_capacity_priority_gatekeeping(
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


# Severity-responsive gatekeeping tests
# -------------------------------------

def test_severity_responsive_gatekeeping_returns_callable():
    gatekeeping = gf.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=10.0,
        high_severity_capacity=20.0,
    )
    assert callable(gatekeeping)


def test_severity_responsive_gatekeeping_scalar_below_threshold_uses_low_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.severity_responsive_gatekeeping(
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

    expected = np.array([0.0, 2.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_above_threshold_uses_high_capacity():
    stocks = np.array([20.0, 30.0, 50.0])
    presenting_proportion = 0.4

    gatekeeping = gf.severity_responsive_gatekeeping(
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

    expected = np.array([0.0, 0.0, 20.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_at_threshold_uses_high_capacity():
    stocks = np.array([40.0, 30.0, 30.0])
    presenting_proportion = 0.4

    gatekeeping = gf.severity_responsive_gatekeeping(
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

    expected = np.array([0.0, 8.0, 12.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_no_capacity():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.severity_responsive_gatekeeping(
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

    gatekeeping = gf.severity_responsive_gatekeeping(
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
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    gatekeeping = gf.severity_responsive_gatekeeping(
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

    expected = np.array([20.0, 12.0, 8.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_scalar_leftovers_to_final_group():
    stocks = np.array([100.0, 10.0, 10.0])
    presenting_proportion = 0.5

    gatekeeping = gf.severity_responsive_gatekeeping(
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

    expected = np.array([2.0, 5.0, 5.0])
    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_time_series_case():
    stocks = np.array(
        [
            [50.0, 20.0, 40.0],
            [30.0, 30.0, 30.0],
            [20.0, 50.0, 30.0],
        ]
    )
    presenting_proportion = 0.4

    gatekeeping = gf.severity_responsive_gatekeeping(
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
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 8.0],
            [8.0, 20.0, 12.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_severity_responsive_gatekeeping_equal_capacities_matches_fixed_capacity_strict():
    stocks = np.array(
        [
            [50.0, 20.0, 40.0],
            [30.0, 30.0, 30.0],
            [20.0, 50.0, 30.0],
        ]
    )
    presenting_proportion = 0.4
    capacity = 15.0

    severity_responsive = gf.severity_responsive_gatekeeping(
        severity_threshold=0.3,
        low_severity_capacity=capacity,
        high_severity_capacity=capacity,
    )
    fixed_strict = gf.fixed_capacity_strict_gatekeeping(capacity=capacity)

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
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 3.0],
            [8.0, 15.0, 12.0],
        ]
    )

    np.testing.assert_allclose(obtained_severity_responsive, obtained_fixed_strict)
    np.testing.assert_allclose(obtained_severity_responsive, expected)


def test_severity_responsive_gatekeeping_time_series_zero_demand_continue_branch():
    stocks = np.array(
        [
            [0.0, 50.0, 20.0],
            [0.0, 30.0, 30.0],
            [0.0, 20.0, 50.0],
        ]
    )
    presenting_proportion = 0.4

    gatekeeping = gf.severity_responsive_gatekeeping(
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
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 8.0, 20.0],
        ]
    )

    np.testing.assert_allclose(obtained, expected)


def test_time_phased_gatekeeping_returns_callable():
    gatekeeping = gf.time_phased_gatekeeping(
        change_times=[10.0],
        gatekeeping_policies=[
            gf.fixed_capacity_strict_gatekeeping(capacity=15.0),
            gf.fixed_capacity_proportional_gatekeeping(capacity=15.0),
        ],
    )
    assert callable(gatekeeping)


def test_severity_responsive_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.severity_responsive_gatekeeping(
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


# Time-phased gatekeeping tests
# -----------------------------

def test_time_phased_gatekeeping_scalar_before_first_change_uses_first_policy():
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    phase_one = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)

    gatekeeping = gf.time_phased_gatekeeping(
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
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    phase_one = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)

    gatekeeping = gf.time_phased_gatekeeping(
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
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    phase_one = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    phase_three = gf.equal_access_proportion_gatekeeping(access_proportion=0.5)

    gatekeeping = gf.time_phased_gatekeeping(
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
    stocks = np.array([50.0, 30.0, 20.0])
    presenting_proportion = 0.4

    phase_one = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    phase_three = gf.equal_access_proportion_gatekeeping(access_proportion=0.5)

    gatekeeping = gf.time_phased_gatekeeping(
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
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    t = np.array([5.0, 15.0, 25.0])

    phase_one = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)
    phase_three = gf.equal_access_proportion_gatekeeping(access_proportion=0.5)

    gatekeeping = gf.time_phased_gatekeeping(
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

    phase_one = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)
    phase_two = gf.fixed_capacity_proportional_gatekeeping(capacity=15.0)

    gatekeeping = gf.time_phased_gatekeeping(
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
            [50.0, 55.0, 60.0],
            [30.0, 25.0, 20.0],
            [20.0, 20.0, 20.0],
        ]
    )
    presenting_proportion = 0.4
    t = np.array([0.0, 10.0, 20.0])

    base_policy = gf.fixed_capacity_strict_gatekeeping(capacity=15.0)

    wrapped_policy = gf.time_phased_gatekeeping(
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


def test_time_phased_gatekeeping_raises_for_wrong_number_of_policies():
    with pytest.raises(
        ValueError,
        match="There must be exactly one more gatekeeping policy than change times.",
    ):
        gf.time_phased_gatekeeping(
            change_times=[10.0, 20.0],
            gatekeeping_policies=[
                gf.fixed_capacity_strict_gatekeeping(capacity=15.0),
                gf.fixed_capacity_proportional_gatekeeping(capacity=15.0),
            ],
        )


def test_time_phased_gatekeeping_raises_for_unsorted_change_times():
    with pytest.raises(
        ValueError,
        match="change_times must be sorted in non-decreasing order.",
    ):
        gf.time_phased_gatekeeping(
            change_times=[20.0, 10.0],
            gatekeeping_policies=[
                gf.fixed_capacity_strict_gatekeeping(capacity=15.0),
                gf.fixed_capacity_proportional_gatekeeping(capacity=15.0),
                gf.equal_access_proportion_gatekeeping(access_proportion=0.5),
            ],
        )


def test_time_phased_gatekeeping_raises_for_invalid_dimension():
    gatekeeping = gf.time_phased_gatekeeping(
        change_times=[10.0],
        gatekeeping_policies=[
            gf.fixed_capacity_strict_gatekeeping(capacity=15.0),
            gf.fixed_capacity_proportional_gatekeeping(capacity=15.0),
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