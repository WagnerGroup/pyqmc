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

import pyqmc.observables.obdm as obdm
import pyqmc.wftools as wftools
import pyqmc.pyscftools as pyscftools
import pyqmc.pbc.supercell as supercell
import pyqmc.method.linemin as linemin
import pyqmc.method.optimize_ortho as optimize_ortho
import pyqmc.method.dmc as dmc
import pyqmc.method.mc
import pyqmc.reblock
import numpy as np
import h5py
import scipy.stats
import pandas as pd
import copy
import pyqmc.observables.accumulators
import os


def OPTIMIZE(
    dft_checkfile,
    output,
    anchors=None,
    nconfig=1000,
    ci_checkfile=None,
    load_parameters=None,
    S=None,
    jastrow_kws=None,
    slater_kws=None,
    target_root=None,
    nodal_cutoff=1e-3,
    **linemin_kws,
):
    linemin_kws["hdf_file"] = output
    if load_parameters is not None and output is not None and os.path.isfile(output):
        raise RuntimeError(
            "load_parameters is not None and output={0} already exists! Delete or rename {0} and try again.".format(
                output
            )
        )
    if target_root is None and anchors is not None:
        target_root = len(anchors)
    else:
        target_root = 0

    wf, configs, acc = initialize_qmc_objects(
        dft_checkfile,
        opt_wf=True,
        nconfig=nconfig,
        ci_checkfile=ci_checkfile,
        load_parameters=load_parameters,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        target_root=target_root,
        nodal_cutoff=nodal_cutoff,
    )
    if anchors is None:
        linemin.line_minimization(wf, configs, acc, **linemin_kws)
    else:
        wfs = []
        for i, a in enumerate(anchors):
            wfs.append(
                initialize_qmc_objects(
                    dft_checkfile,
                    ci_checkfile=ci_checkfile,
                    load_parameters=a,
                    S=S,
                    jastrow_kws=jastrow_kws,
                    slater_kws=slater_kws,
                    target_root=i,
                )[0]
            )
        # wfs = [wftools.read_wf(copy.deepcopy(wf), a) for a in anchors]
        wfs.append(wf)
        optimize_ortho.optimize_orthogonal(wfs, configs, acc, **linemin_kws)


def generate_accumulators(
    mol, mf, energy=True, rdm1=False, extra_accumulators=None, twist=0
):
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
        acc["energy"] = pyqmc.observables.accumulators.EnergyAccumulator(mol)
    if rdm1:
        if hasattr(mol, "a"):
            from pyqmc.pbc.twists import create_supercell_twists

            kinds = create_supercell_twists(mol, mf)["primitive_ks"][twist]
            kpts = mf.kpts[kinds]
            mo_coeff = [
                [mo_coeff[0][k] for k in kinds],
                [mo_coeff[1][k] for k in kinds],
            ]
        else:
            kpts = None

        if "rdm1_up" in acc.keys() or "rdm1_down" in acc.keys():
            raise Exception(
                "Found rdm1_up or rdm1_down in extra_accumulators and rdm1 is True"
            )
        acc["rdm1_up"] = obdm.OBDMAccumulator(
            mol, orb_coeff=mo_coeff[0], spin=0, kpts=kpts
        )
        acc["rdm1_down"] = obdm.OBDMAccumulator(
            mol, orb_coeff=mo_coeff[1], spin=1, kpts=kpts
        )

    return acc


def VMC(
    dft_checkfile,
    output,
    nconfig=1000,
    ci_checkfile=None,
    load_parameters=None,
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
        load_parameters=load_parameters,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        accumulators=accumulators,
    )

    pyqmc.method.mc.vmc(wf, configs, accumulators=acc, **vmc_kws)


def DMC(
    dft_checkfile,
    output,
    nconfig=1000,
    ci_checkfile=None,
    load_parameters=None,
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
        load_parameters=load_parameters,
        S=S,
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        accumulators=accumulators,
    )

    dmc.rundmc(wf, configs, accumulators=acc, **dmc_kws)


def initialize_qmc_objects(
    dft_checkfile,
    nconfig=1000,
    load_parameters=None,
    ci_checkfile=None,
    S=None,
    jastrow_kws=None,
    slater_kws=None,
    accumulators=None,
    opt_wf=False,
    target_root=0,
    nodal_cutoff=1e-3,
):
    if ci_checkfile is None:
        mol, mf = pyscftools.recover_pyscf(dft_checkfile)
        mc = None
    else:
        mol, mf, mc = pyscftools.recover_pyscf(dft_checkfile, ci_checkfile=ci_checkfile)
        if not hasattr(mc.ci, "shape") or len(mc.ci.shape) == 3:
            mc.ci = mc.ci[target_root]

    if S is not None:
        mol = supercell.get_supercell(mol, np.asarray(S))

    wf, to_opt = wftools.generate_wf(
        mol, mf, mc=mc, jastrow_kws=jastrow_kws, slater_kws=slater_kws
    )
    if load_parameters is not None:
        wftools.read_wf(wf, load_parameters)

    configs = pyqmc.method.mc.initial_guess(mol, nconfig)
    if opt_wf:
        acc = pyqmc.observables.accumulators.gradient_generator(
            mol, wf, to_opt, nodal_cutoff=nodal_cutoff
        )
    else:
        if accumulators is None:
            accumulators = {}
        if slater_kws is not None and "twist" in slater_kws.keys():
            twist = slater_kws["twist"]
        else:
            twist = 0
        acc = generate_accumulators(mol, mf, twist=twist, **accumulators)

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


def read_mc_output(
    fname,
    warmup=1,
    reblock=None,
    exclude_keys=("configs", "weights", "block", "nconfig", "wrap"),
):
    ret = {"fname": fname, "warmup": warmup, "reblock": reblock}
    with h5py.File(fname, "r") as f:
        for k in f.keys():
            if k not in exclude_keys:
                vals = f[k][warmup:]
                if reblock is not None:
                    vals = pyqmc.reblock.reblock(vals, reblock)
                ret[k] = np.mean(vals, axis=0)
                ret[k + "_err"] = scipy.stats.sem(vals, axis=0)
    return ret
