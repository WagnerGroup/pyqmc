import pyqmc.obdm as obdm
import pyqmc.wftools as wftools
import pyqmc.pyscftools as pyscftools
import pyqmc.supercell as supercell
import pyqmc.linemin as linemin
import pyqmc.optimize_ortho as optimize_ortho
import pyqmc.dmc as dmc
import pyqmc.mc
import pyqmc.reblock
import numpy as np
import h5py
import scipy.stats
import pandas as pd
import copy
import pyqmc.accumulators


def OPTIMIZE(
    dft_checkfile,
    output,
    anchors=None,
    nconfig=1000,
    ci_checkfile=None,
    start_from=None,
    S=None,
    jastrow_kws=None,
    slater_kws=None,
    **linemin_kws,
):
    linemin_kws["hdf_file"] = output
    wf, configs, acc = initialize_qmc_objects(
        dft_checkfile,
        opt_wf=True,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        start_from=start_from,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
    )
    if anchors is None:
        linemin.line_minimization(wf, configs, acc, **linemin_kws)
    else:
        wfs = [wftools.read_wf(copy.deepcopy(wf), a) for a in anchors]
        wfs.append(wf)
        optimize_ortho.optimize_orthogonal(wfs, configs, acc, **linemin_kws)


def generate_accumulators(mol, mf, energy=True, rdm1=False, extra_accumulators=None):
    acc = {} if extra_accumulators is None else extra_accumulators

    if hasattr(mf, "kpts") and len(mf.mo_coeff[0][0].shape) < 2:
        mo_coeff = [mf.mo_coeff, mf.mo_coeff]
    elif hasattr(mf.mo_coeff, "shape") and len(mf.mo_coeff.shape) == 2:
        mo_coeff = [mf.mo_coeff, mf.mo_coeff]
    else:
        mo_coeff = mf.mo_coeff

    if energy:
        if "energy" in acc.keys():
            raise Exception("Found energy in extra_accumulators and energy is True")
        acc["energy"] = pyqmc.accumulators.EnergyAccumulator(mol)
    if rdm1:
        if "rdm1_up" in acc.keys() or "rdm1_down" in acc.keys():
            raise Exception(
                "Found rdm1_up or rdm1_down in extra_accumulators and rdm1 is True"
            )
        acc["rdm1_up"] = obdm.OBDMAccumulator(mol, orb_coeff=mo_coeff[0], spin=0)
        acc["rdm1_down"] = obdm.OBDMAccumulator(mol, orb_coeff=mo_coeff[1], spin=1)

    return acc


def VMC(
    dft_checkfile,
    output,
    nconfig=1000,
    ci_checkfile=None,
    start_from=None,
    S=None,
    jastrow_kws=None,
    slater_kws=None,
    accumulators=None,
    **vmc_kws,
):
    vmc_kws["hdf_file"] = output
    wf, configs, acc = initialize_qmc_objects(
        dft_checkfile,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        start_from=start_from,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        accumulators=accumulators
    )

    pyqmc.mc.vmc(wf, configs, accumulators=acc, **vmc_kws)


def DMC(
    dft_checkfile,
    output,
    nconfig=1000,
    ci_checkfile=None,
    start_from=None,
    S=None,
    jastrow_kws=None,
    slater_kws=None,
    accumulators=None,
    **dmc_kws,
):
    dmc_kws["hdf_file"] = output
    wf, configs, acc = initialize_qmc_objects(
        dft_checkfile,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        start_from=start_from,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        accumulators=accumulators
    )

    dmc.rundmc(wf, configs, accumulators=acc, **dmc_kws)


def initialize_qmc_objects(
    dft_checkfile,
    nconfig=1000,
    start_from=None,
    ci_checkfile=None,
    S=None,
    jastrow_kws=None,
    slater_kws=None,
    accumulators=None,
    opt_wf=False,
):
    target_root = 0
    if ci_checkfile is None:
        mol, mf = pyscftools.recover_pyscf(dft_checkfile)
        mc = None
    else:
        mol, mf, mc = pyscftools.recover_pyscf(dft_checkfile, ci_checkfile=ci_checkfile)
        mc.ci = mc.ci[target_root]

    if S is not None:
        mol = supercell.get_supercell(mol, np.asarray(S))

    wf, to_opt = wftools.generate_wf(
        mol, mf, mc=mc, jastrow_kws=jastrow_kws, slater_kws=slater_kws
    )
    if start_from is not None:
        wftools.read_wf(wf, start_from)

    configs = pyqmc.mc.initial_guess(mol, nconfig)
    if opt_wf:
        acc = pyqmc.accumulators.gradient_generator(mol, wf, to_opt)
    else:
        if accumulators == None:
            accumulators = {}
        acc = generate_accumulators(mol, mf, **accumulators)

    return wf, configs, acc


def read_opt(fname):
    with h5py.File(fname) as f:
        return pd.DataFrame(
            {
                "energy": f["energy"][...],
                "iteration": f["iteration"][...],
                "error": f["energy_error"][...],
                "fname": [fname] * len(f["energy"]),
            }
        )


def read_mc_output(fname, warmup=5, reblock=16):
    ret = {"fname": fname, "warmup": warmup, "reblock": reblock}
    with h5py.File(fname) as f:
        for k in f.keys():
            if "energy" in k:
                vals = pyqmc.reblock.avg_reblock(f[k][warmup:], reblock)
                ret[k] = np.mean(vals)
                ret[k + "_err"] = scipy.stats.sem(vals)
    return ret
