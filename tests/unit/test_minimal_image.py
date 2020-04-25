# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pytest
from pyqmc.distance import MinimalImageDistance


def test():
    import time

    configs = np.random.rand(400, 80, 3) * 2
    latvecs = np.diag([2, 3, 4])
    s = np.zeros((3, 3))
    s[0, 1] = 1
    mid = MinimalImageDistance(latvecs)
    vec = np.dot(np.random.random((len(configs), 3)), latvecs)
    d1 = vec[:, np.newaxis, :] - configs
    d1norm = np.linalg.norm(d1, axis=-1)
    start = time.time()
    gd = mid.general_dist_i(configs, vec)
    print("general_dist_i,", "\ttime={0:.5f}".format(time.time() - start))
    start = time.time()
    od = mid.orthogonal_dist_i(configs, vec)
    print("orthogonal_dist_i,", "\ttime={0:.5f}".format(time.time() - start))
    diff = gd - od
    print("matrix norm |gen-orth| = {0}".format(np.linalg.norm(diff)))
    print("matrix shape is (nconf,nelec,3) = {0}".format(gd.shape))
    gnorm = np.linalg.norm(gd, axis=-1)
    onorm = np.linalg.norm(od, axis=-1)
    dist_diff = gnorm - onorm
    if np.linalg.norm(diff) > 1e-12:
        print("number", np.count_nonzero(diff > 1e-8))
        print("general\n", gd[dist_diff > 1e-8], "\ndist", gnorm[dist_diff > 1e-8])
        print("orthogonal\n", od[dist_diff > 1e-8], "\ndist", onorm[dist_diff > 1e-8])
        print(
            np.round(diff[dist_diff > 1e-8], 2),
            "\ndist",
            np.linalg.norm(diff, axis=-1)[dist_diff > 1e-8],
        )
    assert (
        np.linalg.norm(diff) < 1e-12
    ), "general_dist_i and orthogonal_dist_i don't give the same answer for the lattice vectors {0}; at least one of them has an error".format(
        latvecs
    )


if __name__ == "__main__":
    test()
