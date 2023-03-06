import numpy as np
import pyqmc.twists
import pyqmc.supercell

def run_tests(cell, mf, S, n):
    cell = pyqmc.supercell.get_supercell(cell, S)
    twists = pyqmc.twists.create_supercell_twists(cell, mf)
    print(twists)
    assert (
        twists['twists'].shape[0] == n
    ), f"Found {twists['twists'].shape[0]} available twists but should have found {n}"

    assert(twists['counts'][0]==cell.scale)
    assert(twists['primitive_ks'][0].shape[0] == cell.scale)

def test_H_pbc_sto3g_krks(H_pbc_sto3g_krks):
    cell, mf = H_pbc_sto3g_krks
    run_tests(cell, mf, 1 * np.eye(3), 8)
    run_tests(cell, mf, 2 * np.eye(3), 1)


def test_h_noncubic_sto3g_triplet(h_noncubic_sto3g_triplet):
     cell, mf = h_noncubic_sto3g_triplet
     run_tests(cell, mf, 1 * np.eye(3), 1)
