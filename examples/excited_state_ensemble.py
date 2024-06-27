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

from pyscf import lib, gto, scf
import pyscf.pbc
import numpy as np
import pyqmc.api as pyq
import pyqmc.accumulators
from rich import print

# from pyqmc.optimize_excited_states import optimize
from pyqmc.ensemble_optimization import optimize_ensemble
from pyqmc.stochastic_reconfiguration import StochasticReconfigurationMultipleWF


def H2_casci():
    mol = gto.M(
        atom="H 0. 0. 0.0; H 0. 0. 1.4",
        basis=f"ccpvtz",
        unit="bohr",
        charge=0,
        spin=0,
        verbose=1,
    )
    mf = scf.ROHF(mol).run()
    mc = pyscf.mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 4
    mc.kernel()
    return mol, mf, mc


def run_optimization_best_practice_3states(
    hdf_file,
    max_iterations,
    client=None,
    npartitions=None,
):
    """
    First optimize the ground state and then optimize the excited
    states while fixing the
    """

    mol, mf, mc = H2_casci()
    import copy

    mf.output = None
    mol.output = None
    mc.output = None
    mc.stdout = None
    mol.stdout = None
    mc.stdout = None
    nstates = 3
    mcs = [copy.copy(mc) for i in range(nstates)]
    for i in range(nstates):
        mcs[i].ci = mc.ci[i]

    wfs = []
    to_opts = []
    for i in range(nstates):
        wf, to_opt = pyq.generate_wf(
            mol, mf, mc=mcs[i], slater_kws=dict(optimize_determinants=True)
        )
        wfs.append(wf)
        to_opts.append(to_opt)
    configs = pyq.initial_guess(mol, 2000)

    pgrad1 = pyq.gradient_generator(mol, wfs[0], to_opt=to_opts[0])
    wfs[0], _ = pyq.line_minimization(
        wfs[0],
        configs,
        pgrad1,
        verbose=True,
        max_iterations=10,
        client=client,
        npartitions=npartitions,
    )

    for k in to_opts[0]:
        to_opts[0][k] = np.zeros_like(to_opts[0][k])
    to_opts[0]["wf1det_coeff"][0] = True  # Bug workaround for linear transform
    for to_opt in to_opts[1:]:
        to_opt["wf1det_coeff"] = np.ones_like(to_opt["wf1det_coeff"])

    transforms = [
        pyqmc.accumulators.LinearTransform(wf.parameters, to_opt)
        for wf, to_opt in zip(wfs, to_opts)
    ]
    for wf in wfs[1:]:
        for k in wf.parameters.keys():
            if "wf2" in k:
                wf.parameters[k] = wfs[0].parameters[k].copy()
    _, configs = pyq.vmc(wfs[0], configs, client=client, npartitions=npartitions)
    energy = pyq.EnergyAccumulator(mol)
    sr_accumulator = StochasticReconfigurationMultipleWF(energy, transforms)

    return optimize_ensemble(
        wfs,
        configs,
        sr_accumulator,
        hdf_file=hdf_file,
        max_iterations=max_iterations,
        client=client,
        npartitions=npartitions,
    )


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2) as client:
        if True:
            run_optimization_best_practice_3states(
                hdf_file=f"optimize.hdf5",
                max_iterations=200,
                client=client,
                npartitions=2,
            )
