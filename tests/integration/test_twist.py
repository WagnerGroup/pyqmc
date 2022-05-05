import numpy as np
import pyqmc.api as pyq
from pyqmc.slater import Slater
from pyqmc.pbc import enforce_pbc
from pyqmc.coord import PeriodicConfigs


def test_cubic_with_ecp(li_cubic_ccecp, kind=1):
    cell, mf = li_cubic_ccecp
    runtest(cell, mf, kind=kind)


def test_noncubic(diamond_primitive, kind=1):
    cell, mf = diamond_primitive
    runtest(cell, mf, kind=kind)


def runtest(mol, mf, kind=0):
    kpt = mf.kpts[kind]
    twist = np.dot(kpt, mol.lattice_vectors().T / (2 * np.pi))

    wf0 = Slater(mol, mf)
    wft = Slater(mol, mf, twist=twist)

    #####################################
    ## compare values across boundary
    ## psi, KE, ecp,
    #####################################
    nconfig = 50
    coords = pyq.initial_guess(mol, nconfig, 1)
    epos, wrap = enforce_pbc(coords.lvecs, coords.configs)
    coords = PeriodicConfigs(epos, coords.lvecs)

    shift_ = np.random.randint(10, size=coords.configs.shape) - 5
    phase = np.exp(2j * np.pi * np.einsum("ijk,k->ij", shift_, twist))

    shift = np.dot(shift_, mol.lattice_vectors())
    epos, wrap = enforce_pbc(coords.lvecs, epos + shift)
    newcoords = PeriodicConfigs(epos, coords.lvecs, wrap=wrap)

    assert np.linalg.norm(newcoords.configs - coords.configs) < 1e-12

    ph0, val0 = wf0.recompute(coords)
    pht, valt = wft.recompute(coords)
    enacc = pyq.EnergyAccumulator(mol, threshold=np.inf)
    np.random.seed(0)
    en0 = enacc(coords, wf0)
    np.random.seed(0)
    ent = enacc(coords, wft)

    e = 0
    rat0 = wf0.testvalue(e, newcoords.electron(e))[0]
    assert np.linalg.norm(rat0 - 1) < 1e-9, rat0 - 1
    ratt = wft.testvalue(e, newcoords.electron(e))[0]
    rattdiff = ratt - phase[:, e]
    print("phase", phase[:, e])
    assert np.linalg.norm(rattdiff) < 1e-9, [
        np.round(rattdiff, 10),
        np.amax(np.abs(rattdiff)),
    ]

    ph0new, val0new = wf0.recompute(newcoords)
    phtnew, valtnew = wft.recompute(newcoords)
    np.random.seed(0)
    en0new = enacc(newcoords, wf0)
    np.random.seed(0)
    entnew = enacc(newcoords, wft)

    assert np.linalg.norm(ph0 - ph0new) < 1e-11
    assert np.linalg.norm(pht * phase.prod(axis=1) - phtnew) < 1e-11, (
        pht * phase.prod(axis=1) - phtnew
    )
    assert np.linalg.norm(val0 - val0new) < 1e-11, np.linalg.norm(val0 - val0new)
    assert np.linalg.norm(valt - valtnew) < 1e-11, np.linalg.norm(valt - valtnew)

    for k in en0.keys():
        diff0 = en0[k] - en0new[k]
        difft = ent[k] - entnew[k]
        if k == "ecp":
            for l, diff in [("0", diff0), ("t", difft)]:
                mad = np.mean(np.abs(diff))
                if True:  # mad > 1e-12:
                    print("ecp%s diff" % l, mad, np.linalg.norm(diff))
                    assert mad < 1e-3, diff
        else:
            assert np.mean(np.abs(diff0)) < 1e-6, diff0
            assert np.mean(np.abs(difft)) < 1e-6, difft
