import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from pyscf import gto, scf, lo
from numpy.linalg import solve
from pyqmc import PySCFSlaterUHF
from pyqmc.mc import initial_guess, vmc
from pandas import DataFrame
from pyqmc.obdm import OBDMAccumulator, normalize_obdm


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
    wf = PySCFSlaterUHF(mol, mf)
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
    df = DataFrame(df)

    obdm_est = {}
    for k in ["obdm", "obdm_up", "obdm_down"]:
        avg_norm = np.array(df.loc[warmup:, k + "norm"].values.tolist()).mean(axis=0)
        avg_obdm = np.array(df.loc[warmup:, k + "value"].values.tolist()).mean(axis=0)
        obdm_est[k] = normalize_obdm(avg_obdm, avg_norm)

    print("Average OBDM(orb,orb)", obdm_est["obdm"].diagonal().round(3))
    print("mf obdm", mfobdm.diagonal().round(3))
    assert np.max(np.abs(obdm_est["obdm"] - mfobdm)) < 0.05
    print(obdm_est["obdm_up"].diagonal().round(3))
    print(obdm_est["obdm_down"].diagonal().round(3))
    assert np.mean(np.abs(obdm_est["obdm_up"] + obdm_est["obdm_down"] - mfobdm)) < 0.05


def test_pbc():
    from pyscf.pbc import gto, scf
    from pyqmc import PySCFSlaterUHF, PySCFSlaterPBC
    from pyqmc import slaterpbc
    import scipy

    lvecs = (np.ones((3, 3)) - np.eye(3)) * 2.0
    mol = gto.M(
        atom="H 0. 0. -{0}; H 0. 0. {0}".format(0.7),
        basis="sto-3g",
        unit="bohr",
        verbose=0,
        a=lvecs,
    )
    mf = scf.KRHF(mol, kpts=mol.make_kpts((2, 2, 2)))
    mf = mf.run()

    S = np.ones((3, 3)) - 2 * np.eye(3)
    mol = slaterpbc.get_supercell(mol, S)
    kpts = slaterpbc.get_supercell_kpts(mol)[:2]
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
    nconf = 800
    nsteps = 50
    warmup = 6
    wf = PySCFSlaterPBC(mol, mf)
    configs = initial_guess(mol, nconf)
    obdm_dict = dict(mol=mol, orb_coeff=lowdin, kpts=kpts, nsweeps=4, warmup=10)
    obdm = OBDMAccumulator(**obdm_dict)
    obdm_up = OBDMAccumulator(**obdm_dict, spin=0)
    obdm_down = OBDMAccumulator(**obdm_dict, spin=1)

    df, coords = vmc(
        wf,
        configs,
        nsteps=nsteps,
        accumulators={"obdm": obdm, "obdm_up": obdm_up, "obdm_down": obdm_down},
        verbose=True,
    )
    df = DataFrame(df)

    obdm_est = {}
    for k in ["obdm", "obdm_up", "obdm_down"]:
        avg_norm = np.array(df.loc[warmup:, k + "norm"].values.tolist()).mean(axis=0)
        avg_obdm = np.array(df.loc[warmup:, k + "value"].values.tolist()).mean(axis=0)
        obdm_est[k] = normalize_obdm(avg_obdm, avg_norm)

    print("Average OBDM(orb,orb)", obdm_est["obdm"].round(3))
    mfobdm = scipy.linalg.block_diag(*mfobdm)
    print("mf obdm", mfobdm.round(3))
    max_abs_err = np.max(np.abs(obdm_est["obdm"] - mfobdm))
    assert max_abs_err < 0.05, "max abs err {0}".format(max_abs_err)
    print(obdm_est["obdm_up"].diagonal().round(3))
    print(obdm_est["obdm_down"].diagonal().round(3))
    mae = np.mean(np.abs(obdm_est["obdm_up"] + obdm_est["obdm_down"] - mfobdm))
    maup = np.mean(np.abs(obdm_est["obdm_up"]))
    madn = np.mean(np.abs(obdm_est["obdm_down"]))
    mamf = np.mean(np.abs(mfobdm))
    assert mae < 0.05, "mae {0}\n maup {1}\n madn {2}\n mamf {3}".format(
        mae, maup, madn, mamf
    )


if __name__ == "__main__":
    test()
    test_pbc()
