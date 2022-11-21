import copy
import time
import numpy as np
import pyqmc.mc as mc


def test_mask(wf, e, epos, mask=None, tolerance=1e-6):
    # testvalue
    if mask is None:
        num_e = len(wf.value()[1])
        mask = np.random.randint(0, 2, num_e).astype(bool)
    ratio, _ = wf.testvalue(e, epos, mask)
    ratio_ref = wf.testvalue(e, epos)[0][mask]
    error = np.abs((ratio - ratio_ref) / np.abs(np.max(ratio)))
    assert np.all(error < tolerance)
    print("testcase for test_value() with mask passed")

def test_testvalue_many(wf,configs,tol=1e-6):
    """
    :parameter wf: a wave function object to be tested
    :parameter configs: electron positions
    :type configs: (nconf, nelec, 3) array
    :returns: max abs errors
    :rtype: dictionary

    """
    nconf, ne, ndim = configs.configs.shape
    val1 = wf.recompute(configs)
    wfcopy = copy.copy(wf)

    delta=1e-2
    tval = np.zeros((nconf,ne))
    epos = configs.make_irreducible(0, configs.configs[:, 0, :] + delta)
    for e in range(ne):
        tval[:,e], savedvals = wf.testvalue(e, epos)
    
    e_all = np.arange(ne)

    tmany= wfcopy.testvalue_many(e_all,epos)
    terr=tmany-tval
    assert np.max(np.abs(terr))<tol

def test_updateinternals(wf, configs):
    """
    :parameter wf: a wave function object to be tested
    :parameter configs: electron positions
    :type configs: (nconf, nelec, 3) array
    :returns: max abs errors
    :rtype: dictionary

    """
    import pyqmc.mc as mc

    nconf, ne, ndim = configs.configs.shape
    delta = 1e-2

    iscomplex = 1j if wf.iscomplex else 1
    updatevstest = np.zeros((ne, nconf)) * iscomplex
    recomputevstest = np.zeros((ne, nconf)) * iscomplex
    recomputevsupdate = np.zeros((ne, nconf)) * iscomplex
    wfcopy = copy.copy(wf)
    val1 = wf.recompute(configs)
    for e in range(ne):
        print("#### Electron", e)
        # val1 = wf.recompute(configs)
        epos = configs.make_irreducible(e, configs.configs[:, e, :] + delta)
        ratio, savedvals = wf.testvalue(e, epos)
        print("*****updateinternals")
        wf.updateinternals(e, epos, configs, saved_values=savedvals)
        print("*****value")
        update = wf.value()
        configs.move(e, epos, [True] * nconf)
        print("*****copy recompute")
        recompute = wfcopy.recompute(configs)
        updatevstest[e, :] = update[0] / val1[0] * np.exp(update[1] - val1[1]) - ratio
        recomputevsupdate[e, :] = update[0] / val1[0] * np.exp(
            update[1] - val1[1]
        ) - recompute[0] / val1[0] * np.exp(recompute[1] - val1[1])
        recomputevstest[e, :] = (
            recompute[0] / val1[0] * np.exp(recompute[1] - val1[1]) - ratio
        )
        val1 = recompute

    # Test mask and pgrad
    # _, configs = mc.vmc(wf, configs, nblocks=1, nsteps_per_block=1, tstep=2)
    # pgradupdate = wf.pgradient()
    # wf.recompute(configs)
    # pgrad = wf.pgradient()
    # pgdict = {
    #    k: np.max(np.abs(pgu - pgrad[k]))
    #    for k, pgu in pgradupdate.items()
    #    if np.prod(pgu.shape) > 0
    # }
    return {
        "updatevstest": np.max(np.abs(updatevstest)),
        "recomputevstest": np.max(np.abs(recomputevstest)),
        "recomputevsupdate": np.max(np.abs(recomputevsupdate)),
        #    **pgdict,
    }


def test_wf_gradient(wf, configs, delta=1e-5):
    """Tests wf.gradient(e,configs) against numerical derivatives of wf.testvalue(e,configs)

    :parameter wf: a wavefunction object with functions wf.recompute(configs), wf.testvalue(e,configs) and wf.gradient(e,configs)
    :parameter configs: positions to set the wf object
    :type configs: (nconf, nelec, 3) array
    :parameter float delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision

    For gradient and testvalue:
        e is the electron index
        epos is nconf x 3 positions of electron e

    wf.testvalue(e,epos) should return a ratio: the wf value at the position where electron e is moved to epos divided by the current value
    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e

    :returns: max abs errors
    :rtype: dictionary
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
            plusval, _ = wf.testvalue(e, epos)
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] - delta * np.eye(3)[d]
            )
            minuval, _ = wf.testvalue(e, epos)
            numeric[:, e, d] = (plusval - minuval) / (2 * delta)
    maxerror = np.amax(np.abs(grad - numeric))
    return maxerror


def test_wf_pgradient(wf, configs, delta=1e-5):
    dtype = complex if wf.iscomplex else float
    baseval = wf.recompute(configs)
    gradient = wf.pgradient()
    error = {}
    for k in gradient.keys():  # We only check the gradients that are exposed.
        flt = wf.parameters[k].reshape(-1)
        # print(flt.shape,wf.parameters[k].shape,gradient[k].shape)
        nparms = len(flt)
        numgrad = np.zeros((configs.configs.shape[0], nparms), dtype=dtype)
        for i, c in enumerate(flt):
            flt[i] += delta
            wf.parameters[k] = flt.reshape(wf.parameters[k].shape)
            plusval = wf.recompute(configs)
            flt[i] -= 2 * delta
            wf.parameters[k] = flt.reshape(wf.parameters[k].shape)
            minuval = wf.recompute(configs)
            numgrad[:, i] = (
                plusval[0] / baseval[0] * np.exp(plusval[1] - baseval[1])
                - minuval[0] / baseval[0] * np.exp(minuval[1] - baseval[1])
            ) / (2 * delta)
            flt[i] += delta
            wf.parameters[k] = flt.reshape(wf.parameters[k].shape)

        pgerr = np.abs(gradient[k].reshape((-1, nparms)) - numgrad)
        error[k] = np.amax(pgerr)

    if len(error) == 0:
        return (0, 0)
    return error[max(error)]  # Return maximum coefficient error


def test_wf_laplacian(wf, configs, delta=1e-5):
    """Tests wf.laplacian(e,epos) against numerical derivatives of wf.gradient(e,epos)

    :parameter wf: a wavefunction object with functions wf.recompute(configs),
             wf.gradient(e,configs) and wf.laplacian(e,configs)
    :parameter configs: positions to set the wf object
    :type configs: (nconf, nelec, 3) array
    :parameter float delta: the finite difference step; 1e-5 to 1e-6 seem to be the best compromise between accuracy and machine precision

    For gradient and laplacian:
        e is the electron index
        epos is nconf x 3 positions of electron e

    wf.gradient(e,epos) should return grad ln Psi(epos), while keeping all the other electrons at current position. epos may be different from the current position of electron e
    wf.laplacian(e,epos) should behave the same as gradient, except lap(Psi(epos))/Psi(epos)

    :returns: max abs errors
    :rtype: dictionary
    """
    nconf, nelec = configs.configs.shape[0:2]
    iscomplex = 1j if wf.iscomplex else 1
    wf.recompute(configs)
    maxerror = 0
    lap = np.zeros(configs.configs.shape[:2]) * iscomplex
    numeric = np.zeros(configs.configs.shape[:2]) * iscomplex

    for e in range(nelec):
        lap[:, e] = wf.laplacian(e, configs.electron(e))

        for d in range(3):
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] + delta * np.eye(3)[d]
            )
            plusval, _ = wf.testvalue(e, epos)
            plusgrad = wf.gradient(e, epos)[d] * plusval
            epos = configs.make_irreducible(
                e, configs.configs[:, e, :] - delta * np.eye(3)[d]
            )
            minuval, _ = wf.testvalue(e, epos)
            minugrad = wf.gradient(e, epos)[d] * minuval
            numeric[:, e] += (plusgrad - minugrad) / (2 * delta)

    maxerror = np.amax(np.abs(lap - numeric))
    return maxerror


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
        ts0 = time.perf_counter()
        lap[:, e] = wf.laplacian(e, configs.electron(e))
        grad[e] = wf.gradient(e, configs.electron(e))
        ts1 = time.perf_counter()
        tt0 = time.perf_counter()
        andgrad[e], andlap[:, e] = wf.gradient_laplacian(e, configs.electron(e))
        tt1 = time.perf_counter()
        tsep += ts1 - ts0
        ttog += tt1 - tt0
    rel_grad = np.abs((andgrad - grad) / grad)
    rel_lap = np.abs((andlap - lap) / lap)
    rmax_grad = np.max(rel_grad)
    rmax_lap = np.max(rel_lap)

    print("time evaluated separately", tsep)
    print("time evaluated together", ttog)

    return {"grad": rmax_grad, "lap": rmax_lap}


def compare_nested_saved_vals(saved1, saved2):
    if hasattr(saved1, "shape"):
        return np.amax(np.abs(saved1 - saved2))
    else:
        a = [compare_nested_saved_vals(s1, s2) for s1, s2 in zip(saved1, saved2)]
        return np.amax(np.abs(a))


def test_wf_gradient_value(wf, configs):
    nconf, nelec = configs.configs.shape[0:2]
    iscomplex = 1j if wf.iscomplex else 1
    wf.recompute(configs)
    maxerror = 0
    val = np.zeros(configs.configs.shape[:2]) * iscomplex
    grad = np.zeros(configs.configs.shape).transpose((1, 2, 0)) * iscomplex
    andval = np.zeros(configs.configs.shape[:2]) * iscomplex
    andgrad = np.zeros(configs.configs.shape).transpose((1, 2, 0)) * iscomplex
    saved_diff = np.zeros(configs.configs.shape[0])

    tsep = 0
    ttog = 0
    for e in range(nelec):
        ts0 = time.perf_counter()
        val[:, e], savedv = wf.testvalue(e, configs.electron(e))
        grad[e] = wf.gradient(e, configs.electron(e))
        ts1 = time.perf_counter()
        tt0 = time.perf_counter()
        andgrad[e], andval[:, e], savedg = wf.gradient_value(e, configs.electron(e))
        saved_diff[e] = compare_nested_saved_vals(savedv, savedg)
        tt1 = time.perf_counter()
        tsep += ts1 - ts0
        ttog += tt1 - tt0
    rel_grad = np.abs((andgrad - grad) / grad)
    rel_val = np.abs((andval - val) / val)
    rmax_grad = np.max(rel_grad)
    rmax_val = np.max(rel_val)
    max_saved = np.max(saved_diff)

    print("separate", tsep)
    print("together", ttog)

    return {"grad": rmax_grad, "val": rmax_val, "saved": max_saved}
