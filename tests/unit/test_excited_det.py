import pyqmc
import pyscf
import numpy as np


def test_mol():
    from pyscf import gto, scf, mcscf

    mol = gto.M(atom="H 0. 0. 0.; H .9 .9 .9", unit="bohr", basis="ccpvdz")
    mf = scf.RHF(mol).run()
    mf.mo_occ[0] -= 1
    mf.mo_occ[2] += 1
    mc = mcscf.CASCI(mf, ncas=1, nelecas=(1, 1))
    mc.ci = np.array([[1]])

    wf, to_opt = pyqmc.default_slater(mol, mf)
    wfref, to_opt_ref = pyqmc.default_multislater(mol, mf, mc)
    for k, p in wf.parameters.items():
        wfref.parameters[k] = p

    configs = pyqmc.initial_guess(mol, 100)
    # Make sure it agrees with multislater implementation
    ph, val = wf.recompute(configs)
    phr, valr = wfref.recompute(configs)
    assert np.linalg.norm(ph - phr) < 1e-12, ph - phr
    assert np.linalg.norm(val - valr) < 1e-12, val - valr

    # Make sure it runs vmc steps without errors
    df, configs = pyqmc.mc.vmc_worker(wf, configs, 0.5, 5, {})


def test_pbc():
    from pyscf.pbc import gto, scf
    from pyqmc.multislaterpbc import MultiSlaterPBC

    mol = gto.M(
        atom="H 0. 0. 0.; H .9 .9 .9", unit="bohr", basis="ccpvdz", a=np.eye(3) * 4
    )
    mf = scf.KRKS(mol, kpts=mol.make_kpts((2, 2, 2)))
    mf.run()
    mf.mo_occ[0][0] -= 1
    mf.mo_occ[1][1] += 1

    mol = pyqmc.supercell.get_supercell(mol, np.eye(3) * 2)

    # occup: list (spin, det, dict{kind: occupation list})
    # map_dets: list (spin, ndet) to identify which determinant of each spin to use (e.g. may use the same up-determinant in multiple products)
    detwt = (1.0,)
    kinds = np.arange(8)
    mo_occ = np.asarray([mf.mo_occ[kind] for kind in kinds])
    double = mo_occ >= 1.1
    single_inds = np.where((mo_occ > 0.9) & (mo_occ < 1.1))
    occs = [double, double.copy()]
    for s in (0, 1):
        occs[s][(single_inds[0][s::2], single_inds[1][s::2])] = True
    occup = [[{kind: np.where(occ[kind])[0] for kind in kinds}] for occ in occs]
    map_dets = [[0], [0]]

    wfref = MultiSlaterPBC(mol, mf, detwt=detwt, occup=occup, map_dets=map_dets)
    wf, to_opt = pyqmc.default_slater(mol, mf)

    configs = pyqmc.initial_guess(mol, 100)
    # Make sure it agrees with multislaterpbc implementation
    ph, val = wf.recompute(configs)
    phr, valr = wfref.recompute(configs)
    assert np.linalg.norm(ph - phr) < 1e-12, ph - phr
    assert np.linalg.norm(val - valr) < 1e-12, val - valr

    # Make sure it runs vmc steps without errors
    df, configs = pyqmc.mc.vmc_worker(wf, configs, 0.5, 5, {})
