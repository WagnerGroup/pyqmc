import pyqmc
import pyqmc.obdm
import numpy as np
import h5py
import pyqmc.reblock
import scipy.stats
import pandas as pd
import copy



def OPTIMIZE(
    dft_checkfile,
    output,
    anchors=None,
    nconfig=1000,
    ci_checkfile=None,
    start_from=None,
    S=None,
    client=None,
    npartitions=None,
    jastrow_kws=None,
    slater_kws=None,
    linemin_kws=None,
):
    if linemin_kws is None:
        linemin_kws = {}

    target_root = 0
    if ci_checkfile is None:
        mol, mf = pyqmc.recover_pyscf(dft_checkfile)
        mc = None
    else:
        mol, mf, mc = pyqmc.recover_pyscf(dft_checkfile, ci_checkfile=ci_checkfile)
        mc.ci = mc.ci[target_root]

    if S is not None:
        mol = pyqmc.get_supercell(mol, np.asarray(S))

    wf, to_opt = pyqmc.generate_wf(
        mol, mf, mc=mc, jastrow_kws=jastrow_kws, slater_kws=slater_kws
    )
    if start_from is not None:
        pyqmc.read_wf(wf, start_from)

    configs = pyqmc.initial_guess(mol, nconfig)
    acc = pyqmc.gradient_generator(mol, wf, to_opt)
    if anchors is None:
        pyqmc.line_minimization(
            wf,
            configs,
            acc,
            verbose=True,
            hdf_file=output,
            client=client,
            npartitions=npartitions,
            **linemin_kws
        )
    else:
        wfs = [pyqmc.read_wf(copy.deepcopy(wf), a) for a in anchors]
        wfs.append(wf)
        pyqmc.optimize_orthogonal(
            wfs,
            configs,
            acc,
            # verbose=True,
            hdf_file=output,
            client=client,
            npartitions=npartitions,
            **linemin_kws
        )


def generate_accumulators(mol, mf, energy=True, rdm1=False, extra_accumulators=None):
    acc = {} if extra_accumulators is None else extra_accumulators

    if len(mf.mo_coeff.shape) == 2:
        mo_coeff = [mf.mo_coeff, mf.mo_coeff]
    else:
        mo_coeff = mf.mo_coeff

    if energy:
        if "energy" in acc.keys():
            raise Exception("Found energy in extra_accumulators and energy is True")
        acc["energy"] = pyqmc.EnergyAccumulator(mol)
    if rdm1:
        if "rdm1_up" in acc.keys() or "rdm1_down" in acc.keys():
            raise Exception(
                "Found rdm1_up or rdm1_down in extra_accumulators and rdm1 is True"
            )
        acc["rdm1_up"] = (
            pyqmc.obdm.OBDMAccumulator(mol, orb_coeff=mo_coeff[0], spin=0),
        )
        acc["rdm1_down"] = (
            pyqmc.obdm.OBDMAccumulator(mol, orb_coeff=mo_coeff[1], spin=1),
        )

    return acc


def VMC(
    dft_checkfile,
    output,
    nconfig=1000,
    ci_checkfile=None,
    start_from=None,
    S=None,
    client=None,
    npartitions=None,
    jastrow_kws=None,
    slater_kws=None,
    vmc_kws=None,
    accumulators=None,
):
    if vmc_kws is None:
        vmc_kws = {}

    target_root = 0
    if ci_checkfile is None:
        mol, mf = pyqmc.recover_pyscf(dft_checkfile)
        mc = None
    else:
        mol, mf, mc = pyqmc.recover_pyscf(dft_checkfile, ci_checkfile=ci_checkfile)
        mc.ci = mc.ci[target_root]

    if S is not None:
        mol = pyqmc.get_supercell(mol, np.asarray(S))

    if accumulators is None:
        accumulators = {}

    wf, _ = pyqmc.generate_wf(
        mol, mf, mc=mc, jastrow_kws=jastrow_kws, slater_kws=slater_kws
    )

    if start_from is not None:
        pyqmc.read_wf(wf, start_from)

    configs = pyqmc.initial_guess(mol, nconfig)

    pyqmc.vmc(
        wf,
        configs,
        accumulators=generate_accumulators(mol, mf, **accumulators),
        verbose=True,
        hdf_file=output,
        client=client,
        npartitions=npartitions,
        **vmc_kws
    )


def DMC(
    dft_checkfile,
    output,
    nconfig=1000,
    start_from=None,
    ci_checkfile=None,
    S=None,
    client=None,
    npartitions=None,
    jastrow_kws=None,
    slater_kws=None,
    dmc_kws=None,
    accumulators=None,
):
    if dmc_kws is None:
        dmc_kws = {}

    target_root = 0
    if ci_checkfile is None:
        mol, mf = pyqmc.recover_pyscf(dft_checkfile)
        mc = None
    else:
        mol, mf, mc = pyqmc.recover_pyscf(dft_checkfile, ci_checkfile=ci_checkfile)
        mc.ci = mc.ci[target_root]

    if S is not None:
        mol = pyqmc.get_supercell(mol, np.asarray(S))
    if accumulators is None:
        accumulators = {}

    wf, _ = pyqmc.generate_wf(mol, mf, jastrow_kws=jastrow_kws, slater_kws=slater_kws)

    if start_from is not None:
        pyqmc.read_wf(wf, start_from)

    configs = pyqmc.initial_guess(mol, nconfig)

    pyqmc.rundmc(
        wf,
        configs,
        accumulators=generate_accumulators(mol, mf, **accumulators),
        verbose=True,
        hdf_file=output,
        client=client,
        npartitions=npartitions,
        **dmc_kws
    )


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
                vals = pyqmc.reblock.reblock(f[k][warmup:], reblock)
                ret[k] = np.mean(vals)
                ret[k + "_err"] = scipy.stats.sem(vals)
    return ret
