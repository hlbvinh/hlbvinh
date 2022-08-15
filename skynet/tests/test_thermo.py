import numpy as np

from ..utils import thermo

np.random.seed(1)


def _fahrenheit_from_celsius(t):
    return t * 1.8 + 32


def test_fix_set_temperatures():
    trials = [
        ((24, 25), (24, 25)),
        (_fahrenheit_from_celsius(np.array([20, 30])), (20, 30)),
        ((0, 0), (thermo.DEFAULT_TEMPERATURE, thermo.DEFAULT_TEMPERATURE)),
        (("a", "b"), (np.nan, np.nan)),
        ((24, "a"), (24, np.nan)),
    ]
    for a, b in trials:
        np.testing.assert_equal(thermo.fix_temperatures(a), b)

        # check if the single and vectorized method return the same values
        for t_in in a:
            t_out_1 = thermo.fix_temperatures(np.array([t_in]))[0]
            t_out_2 = thermo.fix_temperature(t_in)
            np.testing.assert_equal(t_out_1, t_out_2)
