import copy
import time
import numpy as np


def test_mask(wf, e, epos, mask=None):
    # testvalue
    if mask is None:
        num_e = len(wf.value()[1])
        mask = np.random.randint(0, 2, num_e).astype(bool)
    ratio = wf.testvalue(e, epos, mask)
    ratio_ref = wf.testvalue(e, epos)[mask]
    assert np.sum(np.abs(ratio - ratio_ref)) < 1e-10
    print("testcase for test_value() with mask passed")
    # update internals


def test_updateinternals(wf, configs):
    """
    Parameters:
    wf: a wave function object to be tested
    configs: nconf x nelec x 3 position array

    Returns:
    tuple which 

    """

    nconf, ne, ndim = configs.configs.shape
    delta = 1e-2

    iscomplex = 1j if wf.iscomplex else 1
    updatevstest = np.zeros((ne, nconf)) * iscomplex
    recomputevstest = np.zeros((ne, nconf)) * iscomplex
    recomputevsupdate = np.zeros((ne, nconf)) * iscomplex
    wfcopy = copy.copy(wf)
    val1 = wf.recompute(configs)
    for e in range(ne):
        # val1 = wf.recompute(configs)
        epos = configs.make_irreducible(e, configs.configs[:, e, :] + delta)
        ratio = wf.testvalue(e, epos)
        wf.updateinternals(e, epos)
        update = wf.value()
        configs.move(e, epos, [True] * nconf)
        recompute = wfcopy.recompute(configs)
        updatevstest[e, :] = update[0] / val1[0] * np.exp(update[1] - val1[1]) - ratio
        recomputevsupdate[e, :] = update[0] / val1[0] * np.exp(
            update[1] - val1[1]
        ) - recompute[0] / val1[0] * np.exp(recompute[1] - val1[1])
        recomputevstest[e, :] = (
            recompute[0] / val1[0] * np.exp(recompute[1] - val1[1]) - ratio
        )
        val1 = recompute

    return {
        "updatevstest": np.max(np.abs(updatevstest)),
        "recomputevstest": np.max(np.abs(recomputevstest)),
        "recomputevsupdate": np.max(np.abs(recomputevsupdate)),
    }


def test_wf_gradient(wf, configs, delta=1e-5):
    """ 
    Parameters:
        wf: a wavefunction object with functions wf.recompute(configs), wf.testvalue(e,configs) and wf.gradient(e,configs)
        configs: nconf x nelec x 3 position array to set the wf object
        delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision
    Tests wf.gradient(e,configs) against numerical derivatives of wf.testvalue(e,configs)
    For gradient and testvalue:
        e is the electron index
        epos is nconf x 3 positions of electron e
    wf.testvalue(e,epos) should return a ratio: the wf value at the position where electron e is moved to epos divided by the current value
    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e
    
    """
    nconf, nelec = configs.configs.shape[0:2]
    iscomplex = 1j if wf.iscomplex else 1
    wf.recompute(configs)
    maxerror = 0
    grad = np.zeros(configs.configs.shape) * iscomplex
    numeric = np.zeros(configs.configs.shape) * iscomplex
    for e in range(nelec):
        grad[:, e, :] = wf.gradient(e, configs.electron(e)).T
        for d in range(0, 3):
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] + delta * np.eye(3)[d]
            )
            plusval = wf.testvalue(e, epos)
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] - delta * np.eye(3)[d]
            )
            minuval = wf.testvalue(e, epos)
            numeric[:, e, d] = (plusval - minuval) / (2 * delta)
    maxerror = np.amax(np.abs(grad - numeric))
    normerror = np.mean(np.abs(grad - numeric))

    # print('maxerror', maxerror, np.log10(maxerror))
    # print('normerror', normerror, np.log10(normerror))
    return (maxerror, normerror)


def test_wf_pgradient(wf, configs, delta=1e-5):
    iscomplex = 1j if wf.iscomplex else 1
    baseval = wf.recompute(configs)
    gradient = wf.pgradient()
    error = {}
    for k in gradient.keys():  # We only check the gradients that are exposed.
        flt = wf.parameters[k].reshape(-1)
        # print(flt.shape,wf.parameters[k].shape,gradient[k].shape)
        nparms = len(flt)
        numgrad = np.zeros((configs.configs.shape[0], nparms)) * iscomplex
        for i, c in enumerate(flt):
            flt[i] += delta
            wf.parameters[k] = flt.reshape(wf.parameters[k].shape)
            plusval = wf.recompute(configs)
            flt[i] -= 2 * delta
            wf.parameters[k] = flt.reshape(wf.parameters[k].shape)
            minuval = wf.recompute(configs)
            numgrad[:, i] = (
                plusval[0] * baseval[0] * np.exp(plusval[1] - baseval[1])
                - minuval[0] * baseval[0] * np.exp(minuval[1] - baseval[1])
            ) / (2 * delta)
            flt[i] += delta
            wf.parameters[k] = flt.reshape(wf.parameters[k].shape)

        pgerr = np.abs(gradient[k].reshape((-1, nparms)) - numgrad)
        error[k] = (np.amax(pgerr), np.mean(pgerr))
    if len(error) == 0:
        return (0, 0)
    return error[max(error)]  # Return maximum coefficient error


def test_wf_laplacian(wf, configs, delta=1e-5):
    """
    Parameters:
        wf: a wavefunction object with functions wf.recompute(configs), 
             wf.gradient(e,configs) and wf.laplacian(e,configs)
        configs: nconf x nelec x 3 position array to set the wf object
        delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision
    Tests wf.laplacian(e,epos) against numerical derivatives of wf.gradient(e,epos)
    For gradient and laplacian:
        e is the electron index
        epos is nconf x 3 positions of electron e
    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e
    wf.laplacian(e,epos) should behave the same as gradient, except lap(Psi(epos))/Psi(epos)
    """
    nconf, nelec = configs.configs.shape[0:2]
    iscomplex = 1j if wf.iscomplex else 1
    wf.recompute(configs)
    maxerror = 0
    lap = np.zeros(configs.configs.shape[:2]) * iscomplex
    numeric = np.zeros(configs.configs.shape[:2]) * iscomplex

    for e in range(nelec):
        lap[:, e] = wf.laplacian(e, configs.electron(e))

        for d in range(0, 3):
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] + delta * np.eye(3)[d]
            )
            plusval = wf.testvalue(e, epos)
            plusgrad = wf.gradient(e, epos)[d] * plusval
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] - delta * np.eye(3)[d]
            )
            minuval = wf.testvalue(e, epos)
            minugrad = wf.gradient(e, epos)[d] * minuval
            numeric[:, e] += np.real(plusgrad - minugrad) / (2 * delta)

    maxerror = np.amax(np.abs(lap - numeric))
    normerror = np.mean(np.abs((lap - numeric) / numeric))
    # print('maxerror', maxerror, np.log10(maxerror))
    # print('normerror', normerror, np.log10(normerror))
    return (maxerror, normerror)


def test_wf_gradient_laplacian(wf, configs):
    nconf, nelec = configs.configs.shape[0:2]
    iscomplex = 1j if wf.iscomplex else 1
    wf.recompute(configs)
    maxerror = 0
    lap = np.zeros(configs.configs.shape[:2]) * iscomplex
    grad = np.zeros(configs.configs.shape).transpose((1, 2, 0)) * iscomplex
    andlap = np.zeros(configs.configs.shape[:2]) * iscomplex
    andgrad = np.zeros(configs.configs.shape).transpose((1, 2, 0)) * iscomplex

    tsep = 0
    ttog = 0
    for e in range(nelec):
        ts0 = time.time()
        lap[:, e] = wf.laplacian(e, configs.electron(e))
        grad[e] = wf.gradient(e, configs.electron(e))
        ts1 = time.time()
        tt0 = time.time()
        andgrad[e], andlap[:, e] = wf.gradient_laplacian(e, configs.electron(e))
        tt1 = time.time()
        tsep += ts1 - ts0
        ttog += tt1 - tt0
        rmae_grad = np.mean(np.abs((andgrad - grad) / grad))
        rmae_lap = np.mean(np.abs((andlap - lap) / lap))
        norm_grad = np.linalg.norm((andgrad - grad) / grad)
        norm_lap = np.linalg.norm((andlap - lap) / lap)

    print("separate", tsep)
    print("together", ttog)

    d = []
    d.append({"error": rmae_grad, "deriv": "grad", "type": "mae"})
    d.append({"error": rmae_lap, "deriv": "lap", "type": "mae"})
    d.append({"error": norm_grad, "deriv": "grad", "type": "norm"})
    d.append({"error": norm_lap, "deriv": "lap", "type": "norm"})
    return d
