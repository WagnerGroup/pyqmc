# from pyscf import lib, gto, scf
# import pyscf.pbc
import numpy as np
from pyscf.fci.addons import overlap
import pyqmc.api as pyq
import pyqmc.accumulators
from pyqmc.optimize_excited_states import (
    sample_overlap_worker,
    average,
    collect_terms,
    objective_function_derivative,
    correlated_sampling,
)
import copy


def take_derivative_casci_energy(mc, civec, delta=1e-4):
    h1e = mc.get_h1cas()[0]
    eri = mc.ao2mo()
    enbase = mc.fcisolver.energy(h1e=h1e, eri=eri, fcivec=civec, norb=2, nelec=(1, 1))
    en_derivative = []
    for i in range(civec.shape[0]):
        for j in range(civec.shape[1]):
            citest = civec.copy()
            citest[i, j] += delta
            # citest /= np.linalg.norm(citest)
            entest = mc.fcisolver.energy(
                h1e=h1e, eri=eri, fcivec=citest, norb=2, nelec=(1, 1)
            ) / np.sum(citest ** 2)
            derivative = (entest - enbase) / delta
            en_derivative.append(derivative)
    return np.asarray(en_derivative).reshape(civec.shape)


def test_sampler(H2_casci):

    mol, mf, mc = H2_casci

    ci_energies = mc.e_tot
    mc1 = copy.copy(mc)
    mc2 = copy.copy(mc)
    mc1.ci = mc.ci[0]
    mc2.ci = (mc.ci[0] + mc.ci[1]) / np.sqrt(2)

    wf1, to_opt1 = pyq.generate_slater(mol, mf, mc=mc1, optimize_determinants=True)
    wf2, to_opt2 = pyq.generate_slater(mol, mf, mc=mc2, optimize_determinants=True)
    for to_opt in [to_opt1, to_opt2]:
        to_opt["det_coeff"] = np.ones_like(to_opt["det_coeff"], dtype=bool)

    transform1 = pyqmc.accumulators.LinearTransform(wf1.parameters, to_opt1)
    transform2 = pyqmc.accumulators.LinearTransform(wf2.parameters, to_opt2)
    configs = pyq.initial_guess(mol, 2000)
    _, configs = pyq.vmc(wf1, configs)
    energy = pyq.EnergyAccumulator(mol)
    data_weighted, data_unweighted, configs = sample_overlap_worker(
        [wf1, wf2], configs, energy, [transform1, transform2], nsteps=40, nblocks=20
    )
    avg, error = average(data_weighted, data_unweighted)
    print(avg, error)

    ref_energy1 = 0.5 * (ci_energies[0] + ci_energies[1])
    assert abs(avg["total"][1, 1] - ref_energy1) < 3 * error["total"][1][1]

    ref_energy01 = ci_energies[0] / np.sqrt(2)
    assert abs(avg["total"][0, 1] - ref_energy01) < 3 * error["total"][0, 1]

    overlap_tolerance = 0.2  # magic number..be careful.
    terms = collect_terms(avg, error)

    norm = [np.sum(np.abs(m.ci) ** 2) for m in [mc1, mc2]]
    norm_ref = norm
    assert np.all(np.abs(norm_ref - terms["norm"]) < overlap_tolerance)

    norm_derivative_ref = 2 * np.real(mc2.ci).flatten()
    print(terms[("dp_norm", 1)].shape, norm_derivative_ref.shape)
    assert np.all(
        np.abs(norm_derivative_ref - terms[("dp_norm", 1)]) < overlap_tolerance
    )

    overlap_ref = np.sum(mc1.ci * mc2.ci)
    print("overlap test", overlap_ref, terms["overlap"][0, 1])
    assert abs(overlap_ref - terms["overlap"][0, 1]) < overlap_tolerance

    overlap_derivative_ref = mc1.ci.flatten() - 0.5 * overlap_ref * norm_derivative_ref
    assert np.all(
        np.abs(overlap_derivative_ref - terms[("dp_overlap", 1)][:, 0, 1])
        < overlap_tolerance
    )

    en_derivative = take_derivative_casci_energy(mc, mc2.ci)
    assert np.all(
        abs(terms[("dp_energy", 1)][:, 1, 1].reshape(mc2.ci.shape) - en_derivative)
        - overlap_tolerance
    )
    derivative = objective_function_derivative(
        terms, overlap_penalty=1.0, norm_penalty=1.0, offdiagonal_energy_penalty=0.1
    )


def test_correlated_sampling(H2_casci):

    mol, mf, mc = H2_casci

    ci_energies = mc.e_tot
    import copy

    mc1 = copy.copy(mc)
    mc2 = copy.copy(mc)
    mc1.ci = mc.ci[0]
    mc2.ci = mc.ci[1]

    wf1, to_opt1 = pyq.generate_slater(mol, mf, mc=mc1, optimize_determinants=True)
    wf2, to_opt2 = pyq.generate_slater(mol, mf, mc=mc2, optimize_determinants=True)
    for to_opt in [to_opt1, to_opt2]:
        to_opt["det_coeff"] = np.ones_like(to_opt["det_coeff"], dtype=bool)

    transform1 = pyqmc.accumulators.LinearTransform(wf1.parameters, to_opt1)
    transform2 = pyqmc.accumulators.LinearTransform(wf2.parameters, to_opt2)
    configs = pyq.initial_guess(mol, 1000)
    _, configs = pyq.vmc(wf1, configs)
    energy = pyq.EnergyAccumulator(mol)
    data_weighted, data_unweighted, configs = sample_overlap_worker(
        [wf1, wf2], configs, energy, [transform1, transform2], nsteps=10, nblocks=10
    )

    parameters1 = transform1.serialize_parameters(wf1.parameters)
    parameters2 = transform1.serialize_parameters(wf2.parameters)
    sample_parameters = []
    energies_reference = []
    overlap_reference = []
    for theta in np.linspace(0, np.pi / 8, 4):
        a = np.cos(theta)
        b = np.sin(theta)
        sample_parameters.append(
            [a * parameters1 + b * parameters2, b * parameters1 + a * parameters2]
        )
        energies_reference.append(
            [
                [
                    a * a * ci_energies[0] + b * b * ci_energies[1],
                    a * b * ci_energies[0] + b * a * ci_energies[1],
                ],
                [
                    a * b * ci_energies[0] + b * a * ci_energies[1],
                    b * b * ci_energies[0] + a * a * ci_energies[1],
                ],
            ]
        )
        overlap_reference.append([[1.0, a * b + b * a], [a * b + b * a, 1.0]])
    energies_reference = np.asarray(energies_reference)
    overlap_reference = np.asarray(overlap_reference)
    correlated_results = correlated_sampling(
        [wf1, wf2], configs, energy, [transform1, transform2], sample_parameters
    )
    print(correlated_results)
    N = np.abs(correlated_results["overlap"].diagonal(axis1=1, axis2=2))
    Nij = np.asarray([np.sqrt(np.outer(a, a)) for a in N])

    energy_sample = correlated_results["energy"] / Nij
    print("energy reference", energies_reference)
    print("energy sample", energy_sample)

    assert np.all(np.abs(energy_sample - energies_reference) < 0.1)

    print("overlap sample", correlated_results["overlap"])

    print("overlap reference", overlap_reference)
    assert np.all(np.abs(correlated_results["overlap"] - overlap_reference) < 0.1)
