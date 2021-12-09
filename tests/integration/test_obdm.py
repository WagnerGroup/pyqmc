import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from numpy.linalg import solve

np.random.seed(12534234)
from pyscf import gto, scf, lo
from pyqmc.slater import Slater
from pyqmc.mc import initial_guess, vmc
from pyqmc.obdm import OBDMAccumulator, normalize_obdm
import pytest


def test():

    mol = gto.M(
        atom="Li 0. 0. 0.; Li 0. 0. 1.5", basis="sto-3g", unit="bohr", verbose=0
    )
    mf = scf.RHF(mol).run()

    # Lowdin orthogonalized AO basis.
    lowdin = lo.orth_ao(mol, "lowdin")

    # MOs in the Lowdin basis.
    mo = solve(lowdin, mf.mo_coeff)

    # make AO to localized orbital coefficients.
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)

    ### Test OBDM calculation.
    nconf = 500
    nsteps = 400
    warmup = 15
    wf = Slater(mol, mf)
    configs = initial_guess(mol, nconf)
    obdm_dict = dict(mol=mol, orb_coeff=lowdin, nsweeps=5, warmup=15)
    obdm = OBDMAccumulator(**obdm_dict)
    obdm_up = OBDMAccumulator(**obdm_dict, spin=0)
    obdm_down = OBDMAccumulator(**obdm_dict, spin=1)

    df, coords = vmc(
        wf,
        configs,
        nsteps=nsteps,
        accumulators={"obdm": obdm, "obdm_up": obdm_up, "obdm_down": obdm_down},
    )
    obdm_est = {}
    for k in ["obdm", "obdm_up", "obdm_down"]:
        avg_norm = np.mean(df[k + "norm"][warmup:], axis=0)
        avg_obdm = np.mean(df[k + "value"][warmup:], axis=0)
        obdm_est[k] = normalize_obdm(avg_obdm, avg_norm)

    assert np.mean(np.abs(obdm_est["obdm_up"] + obdm_est["obdm_down"] - mfobdm)) < 0.05


@pytest.mark.slow
def test_pbc(li_cubic_ccecp):
    from pyqmc import supercell
    import scipy

    mol, mf = li_cubic_ccecp

    # S = np.ones((3, 3)) - np.eye(3)
    S = np.identity(3)
    mol = supercell.get_supercell(mol, S)
    kpts = supercell.get_supercell_kpts(mol)[:2]
    kdiffs = mf.kpts[np.newaxis] - kpts[:, np.newaxis]
    kinds = np.nonzero(np.linalg.norm(kdiffs, axis=-1) < 1e-12)[1]

    # Lowdin orthogonalized AO basis.
    # lowdin = lo.orth_ao(mol, "lowdin")
    loiao = lo.iao.iao(mol.original_cell, mf.mo_coeff, kpts=kpts)
    occs = [mf.mo_occ[k] for k in kinds]
    coefs = [mf.mo_coeff[k] for k in kinds]
    ovlp = mf.get_ovlp()[kinds]
    lowdin = [lo.vec_lowdin(l, o) for l, o in zip(loiao, ovlp)]
    lreps = [np.linalg.multi_dot([l.T, o, c]) for l, o, c in zip(lowdin, ovlp, coefs)]

    # make AO to localized orbital coefficients.
    mfobdm = [np.einsum("ij,j,kj->ik", l.conj(), o, l) for l, o in zip(lreps, occs)]

    ### Test OBDM calculation.
    nconf = 500
    nsteps = 100
    warmup = 6
    wf = Slater(mol, mf)
    configs = initial_guess(mol, nconf)
    obdm_dict = dict(mol=mol, orb_coeff=lowdin, kpts=kpts, nsweeps=4, warmup=10)
    obdm = OBDMAccumulator(**obdm_dict)

    df, coords = vmc(
        wf,
        configs,
        nsteps=nsteps,
        accumulators={"obdm": obdm},  # , "obdm_up": obdm_up, "obdm_down": obdm_down},
        verbose=True,
    )

    obdm_est = {}
    for k in ["obdm"]:  # , "obdm_up", "obdm_down"]:
        avg_norm = np.mean(df[k + "norm"][warmup:], axis=0)
        avg_obdm = np.mean(df[k + "value"][warmup:], axis=0)
        obdm_est[k] = normalize_obdm(avg_obdm, avg_norm)

    mfobdm = scipy.linalg.block_diag(*mfobdm)

    mae = np.mean(np.abs(obdm_est["obdm"] - mfobdm))
    assert mae < 0.05, f"mae {mae}"


if __name__ == "__main__":
    test()
    test_pbc()
