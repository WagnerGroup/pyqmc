import pyqmc.eval_ecp
import pytest
import numpy as np
import scipy.stats


@pytest.mark.parametrize("naip", [6, 12])
def test_rotation_even(naip):
    weights, rotations = pyqmc.eval_ecp.get_rot(1000, naip)
    avg = np.mean(rotations, axis=0)
    sem = scipy.stats.sem(rotations, axis=0)
    # print(avg/sem)
    assert (np.abs(avg / sem) < 5).all()


if __name__ == "__main__":
    test_rotation_even()
