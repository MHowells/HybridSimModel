import numpy as np
from scipy.integrate import odeint
import sd_component as sd

gk_threshold = 0.4
sample_stocks = [
    np.array([1000, 1000.14235114, 1000.28470455, 1000.42706025, 1000.56941821]),
    np.array([3000, 3000.09854025, 3000.19706101, 3000.29556228, 3000.39404405]),
    np.array([6000, 5999.67150899, 5999.34303597, 5999.01458093, 5998.68614387]),
]
presenting_rate = 0.002
ts = np.linspace(0, 365 * 3, 100000 + 1)


def test_proportional_gatekeeping():
    gatekeeping = sd.proportional_gatekeeping(threshold=gk_threshold)
    assert callable(gatekeeping)

    obtained_referrals_time_point = gatekeeping(
        stocks=[sample_stocks[0][0], sample_stocks[1][0], sample_stocks[2][0]],
        population=sample_stocks[0][0] + sample_stocks[1][0] + sample_stocks[2][0],
        presenting_rate=presenting_rate,
        t=0,
    )
    expected_referrals_time_point = [2, 6, 0]
    assert np.allclose(obtained_referrals_time_point, expected_referrals_time_point)

    obtained_referrals_time_series = gatekeeping(
        stocks=[sample_stocks[0], sample_stocks[1], sample_stocks[2]],
        population=sample_stocks[0] + sample_stocks[1] + sample_stocks[2],
        presenting_rate=presenting_rate,
        t=ts,
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
