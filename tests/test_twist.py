import numpy as np
import pyqmc
import pandas as pd
from pyqmc.reblock import reblock
from pyqmc.pbc import enforce_pbc
from pyqmc.coord import PeriodicConfigs
from pyscf.pbc import gto, scf
from pyscf.pbc.dft.multigrid import multigrid
from pyscf.scf.addons import remove_linear_dep_


def test_cubic_with_ecp(kind=1, nk=(2, 2, 2)):
    from pyscf.pbc.dft.multigrid import multigrid

    L = 6.63 * 2
    cell = gto.Cell(
        atom="""Li     {0}      {0}      {0}                
                  Li     {1}      {1}      {1}""".format(
            0.0, L / 4
        ),
        basis="bfd-vdz",
        ecp={"Li": "bfd"},
        spin=0,
        unit="bohr",
    )
    cell.exp_to_discard = 0.1
    cell.build(a=np.eye(3) * L)
    kpts = cell.make_kpts(nk)
    mf = scf.KRKS(cell, kpts)
    mf.xc = "pbe"
    mf = mf.density_fit()
    mf = multigrid(mf)
    mf = mf.run()
    runtest(cell, mf, kind=kind)


def test_RKS(kind=1, nk=(2, 2, 2)):
    L = 2
    mol = gto.M(
        atom="""He     {0}      {0}      {0}""".format(0.0),
        basis="sto-3g",
        a=np.eye(3) * L,
        unit="bohr",
    )
    kpts = mol.make_kpts(nk)
    mf = scf.KRKS(mol, kpts)
    mf.xc = "pbe"
    # mf = mf.density_fit()
    mf = mf.run()

    runtest(mol, mf, kind=kind)


def test_noncubic(kind=1, nk=(2, 2, 2)):
    L = 3
    mol = gto.M(
        atom="""H     {0}      {0}      {0}                
                  H     {1}      {1}      {1}""".format(
            0.0, L / 4
        ),
        basis="sto-3g",
        a=(np.ones((3, 3)) - np.eye(3)) * L / 2,
        spin=0,
        unit="bohr",
    )
    kpts = mol.make_kpts(nk)
    mf = scf.KRKS(mol, kpts)
    mf.xc = "pbe"
    # mf = mf.density_fit()
    mf = mf.run()
    runtest(mol, mf, kind=kind)


def runtest(mol, mf, kind=0):
    for k, occ in enumerate(mf.mo_occ):
        print(k, occ)
    kpt = mf.kpts[kind]
    twist = np.dot(kpt, mol.lattice_vectors().T / (2 * np.pi))
    print("kpt", kpt)
    print("twist", twist)
    wf0 = pyqmc.PySCFSlaterPBC(mol, mf)
    wft = pyqmc.PySCFSlaterPBC(mol, mf, twist=twist)

    #####################################
    ## compare values across boundary
    ## psi, KE, ecp,
    #####################################
    nconfig = 100
    coords = pyqmc.initial_guess(mol, nconfig, 1)
    nelec = coords.configs.shape[1]
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
    enacc = pyqmc.accumulators.EnergyAccumulator(mol, threshold=np.inf)
    np.random.seed(0)
    en0 = enacc(coords, wf0)
    np.random.seed(0)
    ent = enacc(coords, wft)

    e = 0
    rat0 = wf0.testvalue(e, newcoords.electron(e))
    assert np.linalg.norm(rat0 - 1) < 1e-10, rat0 - 1
    ratt = wft.testvalue(e, newcoords.electron(e))
    rattdiff = ratt - phase[:, e]
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
        print(k)
        diff0 = en0[k] - en0new[k]
        difft = ent[k] - entnew[k]
        if k == "ecp":
            for l, diff in [("0", diff0), ("t", difft)]:
                mad = np.mean(np.abs(diff))
                if True:  # mad > 1e-12:
                    print("ecp%s diff" % l, mad, np.linalg.norm(diff))
                    assert mad < 1e-3, diff
        else:
            assert np.linalg.norm(diff0) < 1e-10, diff0
            assert np.linalg.norm(difft) < 1e-10, difft


if __name__ == "__main__":
    kind = 1
    nk = [2, 2, 2]
    test_cubic_with_ecp(kind, nk)
    test_RKS(kind, nk)
    test_noncubic(kind, nk)
