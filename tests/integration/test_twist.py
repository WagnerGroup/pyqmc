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
    supercell = pyq.get_supercell(mol, np.identity(3))
    twists = pyq.create_supercell_twists(supercell, mf, 12)
    kpt = twists['twists'][kind]
    wft = Slater(mol, mf, twist=kind)

    #####################################
    ## compare values across boundary
    ## psi, KE, ecp,
    #####################################
    nconfig = 5
    coords = pyq.initial_guess(mol, nconfig, 1)
    epos, wrap = enforce_pbc(coords.lvecs, coords.configs)
    coords = PeriodicConfigs(epos, coords.lvecs)

    # Move the atoms by random lattice constants.
    # The value should change by phase and the 
    # local energy should not change.
    L = np.random.randint(10, size=coords.configs.shape) - 5
    shift = np.dot(L, mol.lattice_vectors())
    phase = np.exp(1.0j*np.einsum("ijk,k->ij", shift, kpt))
    epos, wrap = enforce_pbc(coords.lvecs, epos + shift)
    newcoords = PeriodicConfigs(epos, coords.lvecs, wrap=wrap)

    assert np.linalg.norm(newcoords.configs - coords.configs) < 1e-12

    pht, valt = wft.recompute(coords)
    enacc = pyq.EnergyAccumulator(mol, threshold=np.inf)
    np.random.seed(0)
    ent = enacc(coords, wft)

    e = 0
    ratt = wft.testvalue(e, newcoords.electron(e))[0]
    rattdiff = ratt - phase[:, e]
    print("phase", phase[:, e])
    assert np.linalg.norm(rattdiff)/nconfig < 1e-9, [
        np.round(rattdiff, 10),
        np.amax(np.abs(rattdiff)),
    ]

    phtnew, valtnew = wft.recompute(newcoords)
    np.random.seed(0)
    entnew = enacc(newcoords, wft)

    assert np.linalg.norm(pht * phase.prod(axis=1) - phtnew) < 1e-10, (
        pht * phase.prod(axis=1) - phtnew
    )
    assert np.linalg.norm(valt - valtnew) < 1e-10, np.linalg.norm(valt - valtnew)

    for k in ent.keys():
        difft = ent[k] - entnew[k]
        if k == "ecp":
            mad = np.mean(np.abs(difft))
            print("ecp diff", mad, np.linalg.norm(difft))
            assert mad < 1e-3, difft
        else:
            assert np.mean(np.abs(difft)) < 1e-6, difft


    