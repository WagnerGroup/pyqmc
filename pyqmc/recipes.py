import pyqmc
import pyqmc.obdm
import numpy as np


def OPTIMIZE(
    dft_checkfile,
    output,
    nconfig=1000,
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
    mol, mf = pyqmc.recover_pyscf(dft_checkfile)
    if S is not None:
        mol = pyqmc.get_supercell(mol, np.asarray(S))

    wf, to_opt = pyqmc.generate_wf(
        mol, mf, jastrow_kws=jastrow_kws, slater_kws=slater_kws
    )
    if start_from is not None:
        pyqmc.read_wf(wf, start_from)

    configs = pyqmc.initial_guess(mol, nconfig)
    acc = pyqmc.gradient_generator(mol, wf, to_opt)
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


def generate_accumulators(mol, mf, energy=True, rdm1=False, rdm1_options=None):
    acc = {}
    if energy:
        acc["energy"] = pyqmc.EnergyAccumulator(mol)
    if rdm1:
        if rdm1_options is None:
            rdm1_options = dict(orb_coeff=mf.mo_coeff)
        acc["rdm1"] = pyqmc.obdm.OBDMAccumulator(mol, **rdm1_options)
    return acc


def VMC(
    dft_checkfile,
    output,
    nconfig=1000,
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
    mol, mf = pyqmc.recover_pyscf(dft_checkfile)
    if S is not None:
        mol = pyqmc.get_supercell(mol, np.asarray(S))

    wf, _ = pyqmc.generate_wf(mol, mf, jastrow_kws=jastrow_kws, slater_kws=slater_kws)

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
    mol, mf = pyqmc.recover_pyscf(dft_checkfile)
    if S is not None:
        mol = pyqmc.get_supercell(mol, np.asarray(S))

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
