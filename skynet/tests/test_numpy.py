import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_numpy_polyfit():
    with np.errstate(divide="ignore"):
        ys = [28.36]
        secs = np.array([0])
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            np.polyfit(secs, ys, 1)
