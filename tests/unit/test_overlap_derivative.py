import numpy as np
from pyscf import gto, scf
import pyqmc
from pyqmc import optimize_ortho as oo


def numerical_gradient(wfs, configs, pgrad, deltas=(1e-5,)):
    wfparms = wfs[-1].parameters
    transform = pgrad.transform
    flat = transform.serialize_parameters(wfparms)
    numgrad = {"N": np.zeros((flat.size, len(deltas)))}
    numgrad["S"] = np.zeros((flat.size, len(deltas), len(wfs)))

    deltas = np.asarray(deltas)
    parameters = np.zeros((2 * len(deltas), flat.size))
    for i, c in enumerate(flat):
        # parameters = np.tile(flat, (2 * len(deltas), 1))
        parameters[:] = flat
        for d, delta in enumerate(deltas):
            parameters[d, i] += delta
            parameters[d + len(deltas), i] -= delta
        normalization = compute_normalization(wfs, parameters, transform, configs)
        N = normalization.reshape(2, len(deltas))
        numgrad["N"][i] = -np.diff(N, axis=0) / (2 * deltas)

        data = oo.correlated_sample(wfs, configs, parameters, pgrad)
        overlap = data["overlap"].reshape(2, len(deltas), len(wfs))
        numgrad["S"][i] = -np.diff(overlap, axis=0) / (2 * deltas[:, np.newaxis])

    return numgrad


def compute_normalization(wfs, parameters, transform, configs):
    p0 = transform.serialize_parameters(wfs[-1].parameters)
    log_values = np.real([wf.recompute(configs)[1] for wf in wfs])
    ref = log_values[0]
    denominator = np.sum(np.exp(2 * (log_values - ref)), axis=0)

    normalization = np.zeros(len(parameters))
    for p, param in enumerate(parameters):
        for k, it in transform.deserialize(param).items():
            wfs[-1].parameters[k][:] = it
        val = wfs[-1].recompute(configs)[1]
        normalized_value = np.exp(2 * (val - ref)) / denominator
        normalization[p] = np.mean(normalized_value)

    for k, it in transform.deserialize(p0).items():
        wfs[-1].parameters[k] = it
    return normalization


def get_data(wfs, configs, pgrad):
    wfs[-1].recompute(configs)
    data = oo.collect_overlap_data(wfs, configs, pgrad)
    for k, d in data.items():
        data[k] = d[np.newaxis]
    deriv_data = oo.evaluate(data, 0)
    deriv_data["N"] = deriv_data["N"][[-1]]
    return deriv_data


def test():
    print("running scf", flush=True)
    mol = gto.M(atom="H 0. 0. 0.; H 0. 0. 1.6", basis="ccpvdz", unit="bohr")
    mf = scf.UHF(mol).run()
    mf.stdout = None

    print("setting up wfs", flush=True)
    wf0 = pyqmc.MultiSlater(mol, mf)
    mf.mo_coeff[0][:, 0] = np.mean(mf.mo_coeff[0][:, :2], axis=1)
    wf1, to_opt = pyqmc.default_slater(mol, mf, optimize_orbitals=True)

    pgrad = pyqmc.gradient_generator(mol, wf1, to_opt)
    configs = pyqmc.initial_guess(mol, 2000)

    wf0.recompute(configs)
    wf1.recompute(configs)
    wfs = [wf0, wf1]

    print("warming up", flush=True)
    block_avg, configs = oo.sample_overlap_worker(wfs, configs, pgrad, 20, tstep=1.5)

    print("computing gradients and normalization", flush=True)
    data = get_data(wfs, configs, pgrad)
    parameters = pgrad.transform.serialize_parameters(wfs[-1].parameters)
    N = compute_normalization(wfs, [parameters], pgrad.transform, configs)
    print(np.stack([data["N"], N]))

    print("computing numerical gradients", flush=True)
    error = {"N": [], "S": []}
    deltas = [1e-4, 1e-5, 1e-6]
    numgrad = numerical_gradient(wfs, configs, pgrad, deltas)
    for k, ng in numgrad.items():
        pgerr = data[k + "_derivative"].T[:, np.newaxis] - ng
        error[k] = pgerr

    print("computing errors", flush=True)
    for k, er in error.items():
        error[k] = np.amin(er, axis=1)
        print(k)
        print(error[k])


if __name__ == "__main__":
    test()
