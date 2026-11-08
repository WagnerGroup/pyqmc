# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import h5py
import numpy as np
import pytest
from pyscf import gto, scf, mcscf
import pyqmc.api as pyq
from pyqmc.recipes import initialize_qmc_objects


@pytest.fixture(scope="module")
def checkfiles(tmp_path_factory):
    """H2 SCF and two-root CASCI checkfiles, plus a ground-state parameter file
    with a nonzero Jastrow."""
    d = tmp_path_factory.mktemp("target_root")
    scf_file, ci_file, gs_file = (str(d / n) for n in ("scf.hdf5", "ci.hdf5", "gs.hdf5"))
    mol = gto.M(atom="H 0. 0. 0.; H 0. 0. 2.4", basis="6-31g", unit="bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.chkfile = scf_file
    mf.kernel()
    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 2
    mc.kernel()
    with h5py.File(ci_file, "a") as f:
        f["ci/ncas"] = mc.ncas
        f["ci/nelecas"] = list(mc.nelecas)
        f["ci/ci"] = np.asarray(mc.ci)
        f["ci/mo_coeff"] = mc.mo_coeff

    mc.ci = mc.ci[0]
    wf, _ = pyq.generate_wf(mol, mf, mc=mc)
    rng = np.random.default_rng(0)
    with h5py.File(gs_file, "w") as f:
        for k, v in wf.parameters.items():
            f["wf/" + k] = 0.1 * rng.random(np.shape(v))
    return scf_file, ci_file, gs_file


def _params(fname, group):
    with h5py.File(fname, "r") as f:
        return {k: f[group][k][()] for k in f[group]}


def test_target_root_selects_ci_root(checkfiles):
    scf_file, ci_file, _ = checkfiles
    kws = dict(ci_checkfile=ci_file, nconfig=2, slater_kws=dict(tol=-1))
    wf0, _, _ = initialize_qmc_objects(scf_file, target_root=0, **kws)
    wf1, _, _ = initialize_qmc_objects(scf_file, target_root=1, **kws)
    assert not np.allclose(wf0.parameters["wf1det_coeff"], wf1.parameters["wf1det_coeff"])
    with pytest.raises(ValueError):
        initialize_qmc_objects(scf_file, target_root=2, **kws)
    with pytest.raises(ValueError):
        initialize_qmc_objects(scf_file, target_root=1, nconfig=2)


def test_load_jastrow_only(checkfiles):
    """The ensemble recipe takes the Jastrow from a ground-state file and keeps
    the determinant coefficients of its own CI root."""
    scf_file, ci_file, gs_file = checkfiles
    kws = dict(ci_checkfile=ci_file, nconfig=2, target_root=1)
    fresh, _, _ = initialize_qmc_objects(scf_file, **kws)
    wf, _, _ = initialize_qmc_objects(
        scf_file, load_parameters=gs_file, load_jastrow_only=True, **kws
    )
    for k, v in _params(gs_file, "wf").items():
        expected = fresh.parameters[k] if k.startswith("wf1") else v
        assert np.array_equal(wf.parameters[k], expected)


def test_ensemble_optimize_then_vmc(checkfiles, tmp_path):
    scf_file, ci_file, gs_file = checkfiles
    output = str(tmp_path / "ensemble.hdf5")
    short = {"nblocks": 1, "nsteps_per_block": 2}
    pyq.ENSEMBLE_OPTIMIZE(
        scf_file,
        output,
        ci_file,
        nstates=2,
        nconfig=50,
        load_parameters=gs_file,
        max_iterations=1,
        verbose=False,
        vmc_kwargs=short,
        overlap_kwargs=short,
        initial_vmc_warmup_kwargs=short,
        initial_overlap_warmup_kwargs=short,
        rng=0,
    )
    with h5py.File(output, "r") as f:
        assert sorted(f["wf"].keys()) == ["0", "1"]

    # VMC/DMC rebuild state 1 from root 1 and read wf/1.
    wf, _, _ = initialize_qmc_objects(
        scf_file,
        ci_checkfile=ci_file,
        nconfig=2,
        load_parameters=output,
        target_root=1,
    )
    for k, v in _params(output, "wf/1").items():
        assert np.array_equal(wf.parameters[k], v)

    vmc_out = str(tmp_path / "vmc.hdf5")
    pyq.VMC(
        scf_file, vmc_out, nconfig=20, ci_checkfile=ci_file, load_parameters=output, target_root=1,
        nblocks=1, nsteps_per_block=1, rng=0,
    )
    assert np.isfinite(pyq.read_mc_output(vmc_out, warmup=0)["energytotal"])
