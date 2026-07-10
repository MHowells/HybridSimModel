"""
Tests for the linkage between the system-dynamics and discrete-event
simulation components.

Covers warm-up construction for SD-derived referral flows, conversion of
lambda trajectories into DES arrival-rate intervals, preservation of
Low/Medium/High ordering at the DES entry nodes, severity-specific
subspecialty allocation, warm-up removal, and severity reporting.
"""

import numpy as np
import pandas as pd
import pytest
import ciw

import hybridsim.des_component as des
import hybridsim.results as res
import hybridsim.sd_component as sd


# Helper functions for SD-to-DES linkage tests
# --------------------------------------------

def make_simple_sd_model(
    gatekeeping_function,
    presenting_proportion=0.1,
):
    return sd.SD(
        population_function=lambda t: 1000.0,
        initial_unwell_proportion=0.1,
        unwell_splits=(0.5, 0.3, 0.2),
        gatekeeping_function=gatekeeping_function,
        presenting_proportion=presenting_proportion,
        deterioration_function=lambda t: 0.0,
        incidence_function=lambda t, population_size: 0.0,
        recovery_function=lambda t, stock_size: 0.0,
    )


def proportional_gatekeeping(
    stocks,
    population,
    presenting_proportion,
    t,
):
    stocks = np.asarray(stocks, dtype=float)
    return presenting_proportion * stocks


def zero_referral_gatekeeping(
    stocks,
    population,
    presenting_proportion,
    t,
):
    stocks = np.asarray(stocks, dtype=float)
    return np.zeros_like(stocks)


def constant_referral_gatekeeping(
    stocks,
    population,
    presenting_proportion,
    t,
):
    stocks = np.asarray(stocks, dtype=float)

    if stocks.ndim == 1:
        return np.array([1.0, 2.0, 3.0])

    return np.repeat(
        np.array(
            [
                [1.0],
                [2.0],
                [3.0],
            ]
        ),
        stocks.shape[1],
        axis=1,
    )


def run_simple_sd(
    gatekeeping_function,
    t=None,
):
    if t is None:
        t = np.array([0.0, 1.0, 2.0, 3.0])

    model = make_simple_sd_model(
        gatekeeping_function=gatekeeping_function,
    )
    model.solve(t=t)

    lambdas_sd_order = np.asarray(model.lambdas)
    lambdas_des_order = lambdas_sd_order.T

    return model, t, lambdas_sd_order, lambdas_des_order


# SD lambda warm-up tests
# ----------------------

def test_add_constant_lambda_warmup_preserves_initial_rates():
    lambdas = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ]
    )
    ts = np.array([0.0, 1.0, 2.0])

    obtained_lambdas, obtained_ts = sd.add_constant_lambda_warmup(
        lambdas=lambdas,
        ts=ts,
        warmup_days=2.0,
        value="initial",
        shift_time=True,
    )

    expected_warmup_lambdas = np.array(
        [
            [1.0, 1.0],
            [10.0, 10.0],
            [100.0, 100.0],
        ]
    )

    np.testing.assert_allclose(
        obtained_lambdas[:, :2],
        expected_warmup_lambdas,
    )
    np.testing.assert_allclose(
        obtained_lambdas[:, 2:],
        lambdas,
    )

    assert obtained_lambdas.shape == (3, 5)
    assert obtained_ts.shape == (5,)


def test_add_constant_lambda_warmup_returns_expected_time_alignment():
    lambdas = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ]
    )
    ts = np.array([0.0, 1.0, 2.0])

    _, shifted_ts = sd.add_constant_lambda_warmup(
        lambdas=lambdas,
        ts=ts,
        warmup_days=2.0,
        value="initial",
        shift_time=True,
    )

    _, unshifted_ts = sd.add_constant_lambda_warmup(
        lambdas=lambdas,
        ts=ts,
        warmup_days=2.0,
        value="initial",
        shift_time=False,
    )

    expected_shifted_ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    expected_unshifted_ts = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    np.testing.assert_allclose(shifted_ts, expected_shifted_ts)
    np.testing.assert_allclose(unshifted_ts, expected_unshifted_ts)


# SD-to-DES GP arrival-rate conversion tests
# ------------------------------------------

@pytest.mark.parametrize(
    "lambdas, t, expected_message",
    [
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            "Expected lambdas to be two-dimensional",
        ),
        (
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            np.array([0.0, 1.0, 2.0]),
            "lambdas.shape\\[0\\] to match len\\(t\\)",
        ),
        (
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ),
            np.array([0.0, 1.0, 2.0]),
            "Expected three severity columns",
        ),
    ],
)
def test_make_gp_arrival_rates_rejects_wrong_shape(
    lambdas,
    t,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )


def test_make_gp_arrival_rates_uses_low_medium_high_columns():
    lambdas = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    t = np.array([0.0, 1.0, 2.0])

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )
    )

    assert low_arrivals.rates == [1.0, 4.0]
    assert medium_arrivals.rates == [2.0, 5.0]
    assert high_arrivals.rates == [3.0, 6.0]


def test_zero_sd_referral_flows_create_zero_des_gp_arrival_rates():
    _, t, _, lambdas = run_simple_sd(
        gatekeeping_function=zero_referral_gatekeeping,
    )

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )
    )

    np.testing.assert_allclose(lambdas, np.zeros_like(lambdas))

    assert low_arrivals.rates == [0.0, 0.0, 0.0]
    assert medium_arrivals.rates == [0.0, 0.0, 0.0]
    assert high_arrivals.rates == [0.0, 0.0, 0.0]


def test_constant_sd_referral_flows_create_constant_des_gp_arrival_rates():
    _, t, _, lambdas = run_simple_sd(
        gatekeeping_function=constant_referral_gatekeeping,
    )

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )
    )

    expected_lambdas = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )

    np.testing.assert_allclose(lambdas, expected_lambdas)

    assert low_arrivals.rates == [1.0, 1.0, 1.0]
    assert medium_arrivals.rates == [2.0, 2.0, 2.0]
    assert high_arrivals.rates == [3.0, 3.0, 3.0]


# SD-to-DES linkage tests using SD-generated lambdas
# --------------------------------------------------

def test_sd_generated_lambdas_are_converted_to_des_arrival_intervals():
    _, t, _, lambdas = run_simple_sd(
        gatekeeping_function=proportional_gatekeeping,
    )

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_gp_arrival_rates(
            lambdas=lambdas,
            t=t,
            max_sample_date=10.0,
        )
    )

    assert lambdas.shape == (len(t), 3)

    np.testing.assert_allclose(
        low_arrivals.rates,
        lambdas[:-1, 0],
    )
    np.testing.assert_allclose(
        medium_arrivals.rates,
        lambdas[:-1, 1],
    )
    np.testing.assert_allclose(
        high_arrivals.rates,
        lambdas[:-1, 2],
    )

    assert low_arrivals.endpoints == list(t[1:])
    assert medium_arrivals.endpoints == list(t[1:])
    assert high_arrivals.endpoints == list(t[1:])


def test_sd_generated_lambdas_with_warmup_are_converted_to_des_arrival_intervals():
    _, t, lambdas_sd_order, _ = run_simple_sd(
        gatekeeping_function=proportional_gatekeeping,
    )

    lambdas_with_warmup, t_with_warmup = sd.add_constant_lambda_warmup(
        lambdas=lambdas_sd_order,
        ts=t,
        warmup_days=2.0,
        value="initial",
        shift_time=True,
    )

    lambdas_for_des = lambdas_with_warmup.T

    low_arrivals, medium_arrivals, high_arrivals = (
        des.make_gp_arrival_rates(
            lambdas=lambdas_for_des,
            t=t_with_warmup,
            max_sample_date=10.0,
        )
    )

    assert lambdas_for_des.shape == (len(t_with_warmup), 3)

    np.testing.assert_allclose(
        lambdas_for_des[:3],
        np.repeat(
            lambdas_for_des[[0]],
            3,
            axis=0,
        ),
    )

    np.testing.assert_allclose(
        low_arrivals.rates,
        lambdas_for_des[:-1, 0],
    )
    np.testing.assert_allclose(
        medium_arrivals.rates,
        lambdas_for_des[:-1, 1],
    )
    np.testing.assert_allclose(
        high_arrivals.rates,
        lambdas_for_des[:-1, 2],
    )

    assert low_arrivals.endpoints == list(t_with_warmup[1:])
    assert medium_arrivals.endpoints == list(t_with_warmup[1:])
    assert high_arrivals.endpoints == list(t_with_warmup[1:])


# DES entry-node linkage tests
# ----------------------------

def test_gp_arrivals_are_assigned_to_correct_entry_nodes():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    gp_arrival_rates = [
        "gp_low",
        "gp_medium",
        "gp_high",
    ]
    other_arrival_rates = [
        "other_low",
        "other_medium",
        "other_high",
    ]

    obtained = des.get_arrival_distributions_for_nodes(
        nodes=nodes,
        subspecialties=subspecialties,
        gp_arrival_rates=gp_arrival_rates,
        other_arrival_rates=other_arrival_rates,
    )

    assert obtained["Low"][0] == "gp_low"
    assert obtained["Medium"][0] == "gp_medium"
    assert obtained["High"][0] == "gp_high"

    assert obtained["Low"][1] == "other_low"
    assert obtained["Medium"][1] == "other_medium"
    assert obtained["High"][1] == "other_high"

    assert obtained["Low"][2:] == [None] * 6
    assert obtained["Medium"][2:] == [None] * 6
    assert obtained["High"][2:] == [None] * 6


def test_class_change_matrices_preserve_severity_specific_subspecialty_probabilities():
    nodes = ["*", "*", "A", "B", "C", "A", "B", "C"]
    subspecialties = ["Hip", "Knee"]

    obtained = des.get_class_change_matrices(
        nodes=nodes,
        subspecialties=subspecialties,
        subspec_probs_low=[0.7, 0.3],
        subspec_probs_medium=[0.4, 0.6],
        subspec_probs_high=[0.2, 0.8],
    )

    referral_class_changes = obtained[0]

    assert referral_class_changes["Low"] == {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
        "Hip": 0.7,
        "Knee": 0.3,
    }

    assert referral_class_changes["Medium"] == {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
        "Hip": 0.4,
        "Knee": 0.6,
    }

    assert referral_class_changes["High"] == {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0,
        "Hip": 0.2,
        "Knee": 0.8,
    }

    assert obtained[0] == obtained[1]


# Helper functions for DES initialisation linkage tests
# -----------------------------------------------------

def make_single_severity_sd_gatekeeping(
    severity,
    referral_rate,
):
    severity_index_lookup = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    severity_index = severity_index_lookup[severity]

    def gatekeeping_function(
        stocks,
        population,
        presenting_proportion,
        t,
    ):
        stocks = np.asarray(stocks, dtype=float)

        if stocks.ndim == 1:
            lambdas = np.zeros(3)
            lambdas[severity_index] = referral_rate
            return lambdas

        if stocks.ndim == 2:
            lambdas = np.zeros_like(stocks, dtype=float)
            lambdas[severity_index, :] = referral_rate
            return lambdas

        raise ValueError("stocks must be a 1D or 2D array-like structure.")

    return gatekeeping_function


def make_sd_generated_gp_arrival_rates_for_severity(
    severity,
    referral_rate=20.0,
):
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    model = sd.SD(
        population_function=lambda t: 1000.0,
        initial_unwell_proportion=0.1,
        unwell_splits=(0.5, 0.3, 0.2),
        gatekeeping_function=make_single_severity_sd_gatekeeping(
            severity=severity,
            referral_rate=referral_rate,
        ),
        presenting_proportion=0.1,
        deterioration_function=lambda t: 0.0,
        incidence_function=lambda t, population_size: 0.0,
        recovery_function=lambda t, stock_size: 0.0,
    )

    model.solve(t=t)

    lambdas = np.asarray(model.lambdas).T

    gp_arrival_rates = des.make_gp_arrival_rates(
        lambdas=lambdas,
        t=t,
        max_sample_date=t[-1],
    )

    return gp_arrival_rates, lambdas, t


def make_linkage_test_pdfa(
    alphabet,
    activity_letter,
    from_state=1,
    to_state=2,
    n_states=3,
):
    pdfa = np.zeros((len(alphabet), n_states, n_states))
    activity_index = alphabet.index(activity_letter)
    pdfa[activity_index, from_state, to_state] = 1.0
    return pdfa


def make_severity_initialisation_test_network(
    gp_arrival_rates,
):
    alphabet = ["A", "B", "C"]

    alphabets = [
        alphabet,
        alphabet,
        alphabet,
    ]

    subspecialties = ["Hip"]

    activity_dict = {
        "A": 3,
        "B": 4,
        "C": 5,
    }

    subspec_dict = {
        "Hip": 0,
    }

    low_pdfa = make_linkage_test_pdfa(
        alphabet=alphabet,
        activity_letter="A",
    )

    medium_pdfa = make_linkage_test_pdfa(
        alphabet=alphabet,
        activity_letter="B",
    )

    high_pdfa = make_linkage_test_pdfa(
        alphabet=alphabet,
        activity_letter="C",
    )

    routing = des.JockeyRouting(
        pdfa_matrix=[
            low_pdfa,
            medium_pdfa,
            high_pdfa,
        ],
        alphabet=alphabets,
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    service_distributions = des.make_deterministic_service_distributions(
        service_values=[
            [0.5, 0.5, 0.5],
        ],
    )

    reneging_distribution = des.PreOpExpiryDist(
        activity_dict=activity_dict,
        subspec_dict=subspec_dict,
        pre_op_letter="B",
        elective_surgery_letter="C",
    )

    return des.get_network(
        alphabets=alphabets,
        subspecialties=subspecialties,
        subspecialty_service_dists=service_distributions,
        emergency_nodes=[],
        subspecialty_class=routing,
        reneging_distribution=reneging_distribution,
        subspec_probs_low=[1.0],
        subspec_probs_medium=[1.0],
        subspec_probs_high=[1.0],
        gp_arrival_rates=gp_arrival_rates,
        other_arrival_rates=[
            None,
            None,
            None,
        ],
    )


# DES severity initialisation tests
# ---------------------------------

@pytest.mark.parametrize(
    "severity, expected_activity_node",
    [
        ("Low", 3),
        ("Medium", 4),
        ("High", 5),
    ],
)
def test_sd_generated_gp_patients_retain_severity_attributes_when_initialised_in_des(
    severity,
    expected_activity_node,
):
    des.apply_custom_record_changes()
    ciw.seed(1)

    gp_arrival_rates, lambdas, t = (
        make_sd_generated_gp_arrival_rates_for_severity(
            severity=severity,
            referral_rate=20.0,
        )
    )

    severity_index_lookup = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    severity_index = severity_index_lookup[severity]

    assert lambdas.shape == (len(t), 3)
    np.testing.assert_allclose(
        lambdas[:, severity_index],
        np.repeat(20.0, len(t)),
    )

    other_severity_indices = [
        index
        for index in range(3)
        if index != severity_index
    ]

    np.testing.assert_allclose(
        lambdas[:, other_severity_indices],
        0.0,
    )

    network = make_severity_initialisation_test_network(
        gp_arrival_rates=gp_arrival_rates,
    )

    records = des.run_des_trial(
        network=network,
        run_time=5.0,
    )

    activity_records = records.loc[
        (
            (records["customer_class"] == "Hip")
            & (records["level"] == severity)
            & (records["referral_source"] == "GP")
            & (records["node"] == expected_activity_node)
            & (records["record_type"] == "service")
        )
    ]

    assert not activity_records.empty

    first_activity_record = activity_records.iloc[0]

    assert first_activity_record["customer_class"] == "Hip"
    assert first_activity_record["level"] == severity
    assert first_activity_record["referral_source"] == "GP"
    assert first_activity_record["node"] == expected_activity_node


# Warm-up removal tests
# ---------------------

def test_remove_warmup_patients_removes_whole_patient_histories():
    records_df = pd.DataFrame(
        {
            "id_number": [1, 1, 2, 2, 3],
            "arrival_date": [5.0, 15.0, 10.0, 12.0, 14.0],
            "service_start_date": [5.0, 15.0, 10.0, 12.0, 14.0],
            "service_end_date": [6.0, 16.0, 11.0, 13.0, 15.0],
            "exit_date": [6.0, 16.0, 11.0, 13.0, 15.0],
        }
    )

    obtained = res.remove_warmup_patients(
        records_df=records_df,
        warmup_days=10.0,
        reset_time=False,
    )

    assert set(obtained["id_number"]) == {2, 3}
    assert 1 not in obtained["id_number"].values
    assert len(obtained) == 3


def test_remove_warmup_activity_records_removes_only_warmup_records():
    records_df = pd.DataFrame(
        {
            "id_number": [1, 1, 2, 2, 3],
            "arrival_date": [5.0, 12.0, 8.0, 15.0, 20.0],
            "service_start_date": [5.0, 12.0, 8.0, 15.0, 20.0],
            "service_end_date": [6.0, 13.0, 9.0, 16.0, 21.0],
            "exit_date": [6.0, 13.0, 9.0, 16.0, 21.0],
        }
    )

    obtained = res.remove_warmup_activity_records(
        records_df=records_df,
        warmup_days=10.0,
        reset_time=False,
    )

    assert list(obtained["id_number"]) == [1, 2, 3]
    assert list(obtained["arrival_date"]) == [12.0, 15.0, 20.0]
    assert len(obtained) == 3