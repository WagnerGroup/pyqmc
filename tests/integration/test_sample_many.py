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

# from pyscf import lib, gto, scf
# import pyscf.pbc
import numpy as np
from pyscf.fci.addons import overlap
import pyqmc.api as pyq
import pyqmc.observables.accumulators
#from pyqmc.method.optimize_excited_states import average, collect_terms
from pyqmc.observables.accumulators_multiwf import EnergyAccumulatorMultipleWF
from pyqmc.method.sample_many import sample_overlap
import copy
import scipy

def average(weighted, unweighted):
    """
    (more or less) correctly average the output from sample_overlap
    Returns the average and error as dictionaries.

    TODO: use a more accurate formula for weighted uncertainties
    """
    avg = {}
    error = {}
    for k, it in unweighted.items():
        avg[k] = np.mean(it, axis=0)
        error[k] = scipy.stats.sem(it, axis=0)

    N = np.abs(avg["overlap"].diagonal())
    Nij = np.sqrt(np.outer(N, N))

    for k, it in weighted.items():
        avg[k] = np.mean(it, axis=0) / Nij
        error[k] = scipy.stats.sem(it, axis=0) / Nij
    return avg, error

def test_sampler(H2_casci):
    mol, mf, mc = H2_casci

    ci_energies = mc.e_tot
    mc1 = copy.copy(mc)
    mc2 = copy.copy(mc)
    mc1.ci = mc.ci[0]
    mc2.ci = (mc.ci[0] + mc.ci[1]) / np.sqrt(2)

    wf1, _ = pyq.generate_slater(mol, mf, mc=mc1, optimize_determinants=True)
    wf2, _ = pyq.generate_slater(mol, mf, mc=mc2, optimize_determinants=True)

    configs = pyq.initial_guess(mol, 2000)
    _, configs = pyq.vmc(wf1, configs)
    energy = EnergyAccumulatorMultipleWF(pyq.EnergyAccumulator(mol))
    data_weighted, data_unweighted, configs = sample_overlap(
        [wf1, wf2], configs, energy, nsteps=40, nblocks=20
    )
    avg, error = average(data_weighted, data_unweighted)
    print(avg, error)

    ref_energy1 = 0.5 * (ci_energies[0] + ci_energies[1])
    assert abs(avg["total"][1, 1] - ref_energy1) < 3 * error["total"][1][1]

    ref_energy01 = ci_energies[0] / np.sqrt(2)
    assert abs(avg["total"][0, 1] - ref_energy01) < 3 * error["total"][0, 1]

    overlap_tolerance = 0.2  # magic number..be careful.
    N = np.abs(avg["overlap"].diagonal())
    Nij = np.sqrt(np.outer(N, N))
    terms = dict(norm=N, energy=avg["total"], overlap=avg["overlap"] / Nij)

    norm = [np.sum(np.abs(m.ci) ** 2) for m in [mc1, mc2]]
    norm_ref = norm
    assert np.all(np.abs(norm_ref - terms["norm"]) < overlap_tolerance)

    overlap_ref = np.sum(mc1.ci * mc2.ci)
    assert abs(overlap_ref - terms["overlap"][0, 1]) < overlap_tolerance


def test_correlated_sampling(H2_casci):
    mol, mf, mc = H2_casci
    mol.stdout = None
    mol.output = None

    ci_energies = mc.e_tot
    import copy

    mc1 = copy.copy(mc)
    mc2 = copy.copy(mc)
    mc1.ci = mc.ci[0]
    mc2.ci = mc.ci[1]

    wfs = []
    parms = []
    transforms = []
    for mc_ in [mc1, mc2]:
        wf, to_opt = pyq.generate_slater(mol, mf, mc=mc_, optimize_determinants=True)
        to_opt["det_coeff"] = np.ones_like(to_opt["det_coeff"], dtype=bool)
        transform = pyqmc.observables.accumulators.LinearTransform(wf.parameters, to_opt)
        wfs.append(wf)
        transforms.append(transform)
        parms.append(transform.serialize_parameters(wf.parameters))

    configs = pyq.initial_guess(mol, 1000)
    _, configs = pyq.vmc(wfs[0], configs)
    energy = EnergyAccumulatorMultipleWF(pyq.EnergyAccumulator(mol))
    data_weighted, data_unweighted, configs = sample_overlap(
        wfs, configs, energy, nsteps=10, nblocks=10
    )

    sample_parameters = []
    energies_reference = []
    overlap_reference = []
    for theta in np.linspace(0, np.pi / 8, 4):
        A = [np.cos(theta), np.sin(theta)]
        a = np.cos(theta)
        b = np.sin(theta)
        sample_parameters.append([np.dot(A, parms), np.dot(A[::-1], parms)])
        AB = np.outer(A, A)
        energies_reference.append(AB * ci_energies[0] + AB[::-1, ::-1] * ci_energies[1])
        overlap_reference.append(AB + AB[::-1, ::-1])
    energies_reference = np.asarray(energies_reference)[:, 0, 0]
    overlap_reference = np.asarray(overlap_reference)[:, 0, 0]
    sample_parameters = np.asarray(sample_parameters)[:, 0]
    wfparms = []
    for i, s in enumerate(sample_parameters):
        wf_ = copy.deepcopy(wfs[0])
        for k, it in transform.deserialize(wf_, s).items():
            wf_.parameters[k] = it
        wfparms.append(wf_)
    weighted, unweighted, _ = sample_overlap(wfparms, configs, energy)
    avg, error = average(weighted, unweighted)
    correlated_results = dict(energy=avg["total"], overlap=avg["overlap"])
    print(correlated_results)

    energy_sample = correlated_results["energy"]
    print("energy reference", energies_reference)
    print("energy sample", energy_sample)

    assert np.all(np.abs(energy_sample - energies_reference) < 0.1)

    print("overlap sample", correlated_results["overlap"])

    print("overlap reference", overlap_reference)
    assert np.all(np.abs(correlated_results["overlap"] - overlap_reference) < 0.1)
