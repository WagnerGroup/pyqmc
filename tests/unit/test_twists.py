import numpy as np
import pyqmc.twists


def run_tests(cell, mf, S, n):
    avai = pyqmc.twists.available_twists(cell, mf, S)
    assert (
        avai.shape[0] == n
    ), f"Found {avai.shape[0]} available twists but should have found {n}"


def test_H_pbc_sto3g_krks(H_pbc_sto3g_krks):
    cell, mf = H_pbc_sto3g_krks
    run_tests(cell, mf, 1 * np.eye(3), 8)
    run_tests(cell, mf, 2 * np.eye(3), 1)
    run_tests(cell, mf, 3 * np.eye(3), 0)


def test_li_cubic_ccecp(li_cubic_ccecp):
    cell, mf = li_cubic_ccecp
    run_tests(cell, mf, 1 * np.eye(3), 8)
    run_tests(cell, mf, 2 * np.eye(3), 1)
    run_tests(cell, mf, 3 * np.eye(3), 0)


def test_diamond_primitive(diamond_primitive):
    cell, mf = diamond_primitive
    run_tests(cell, mf, 1 * np.eye(3), 8)
    run_tests(cell, mf, 2 * np.eye(3), 1)
    run_tests(cell, mf, 3 * np.eye(3), 0)


def test_h_noncubic_sto3g_triplet(h_noncubic_sto3g_triplet):
    cell, mf = h_noncubic_sto3g_triplet
    run_tests(cell, mf, 1 * np.eye(3), 1)
    run_tests(cell, mf, 2 * np.eye(3), 0)
    run_tests(cell, mf, 3 * np.eye(3), 0)
