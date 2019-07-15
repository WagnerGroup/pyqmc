import numpy as np
import pandas as pd
import scipy


def sr_update(pgrad, Sij, step, eps=0.1):
    invSij = np.linalg.inv(Sij + eps * np.eye(Sij.shape[0]))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step / np.linalg.norm(v)


def sd_update(pgrad, Sij, step, eps=0.1):
    return -pgrad * step / np.linalg.norm(pgrad)


def sr12_update(pgrad, Sij, step, eps=0.1):
    invSij = scipy.linalg.sqrtm(np.linalg.inv(Sij + eps * np.eye(Sij.shape[0])))
    v = np.einsum("ij,j->i", invSij, pgrad)
    return -v * step / np.linalg.norm(v)


def line_minimization(
    wf,
    coords,
    pgrad_acc,
    steprange=0.5,
    warmup=None,
    maxiters=50,
    vmc=None,
    vmcoptions=None,
    cslm=None,
    cslmoptions=None,
    dataprefix="",
    update=sr_update,
    update_kws=None,
    verbose=2,
    npts=5,
):
    """Optimizes energy by determining gradients with stochastic reconfiguration
        and minimizing the energy along gradient directions using correlated sampling.

    Args:

      wf: initial wave function

      coords: initial configurations

      pgrad_acc: A PGradAccumulator-like object

      steprange: How far to search in the line minimization

      warmup: number of steps to use for vmc warmup; if None, same as in vmcoptions

      maxiters: (maximum) number of steps in the gradient descent

      vmc: A function that works like mc.vmc()

      vmcoptions: a dictionary of options for the vmc method

      cslm: the correlated sampling line minimization function to use

      cslmoptions: a dictionary of options for the cslm method

      update: A function that generates a parameter change 

      update_kws: Any keywords 

      dataprefix: A base filename in which to save datafileline.json and datafilegrad.json, which contain information about the optimization

      npts: number of points to fit to in each line minimization

    Returns:

      wf: optimized wave function

      datagrad: dictionary with gradient descent data

      dataline: dictionary with line minimization data

    """
    if vmc is None:
        import pyqmc.mc 
        vmc = pyqmc.mc.vmc 
    if vmcoptions is None:
        vmcoptions = {}
    if cslm is None:
        cslm = cslm_sampler
    if cslmoptions is None:
        cslmoptions = {}
    if update_kws is None:
        update_kws = {}

    def gradient_energy_function(x, configs):
        newparms = pgrad_acc.transform.deserialize(x)
        for k in newparms:
            wf.parameters[k] = newparms[k]
        data, configs= vmc(
            wf, configs, accumulators={"pgrad": pgrad_acc}, **vmcoptions
        )
        df = pd.DataFrame(data)
        en = np.mean(df['pgradtotal'])
        en_err = np.std(df['pgradtotal']) / len(df)
        dpH = np.mean(df["pgraddpH"], axis=0)
        dp = np.mean(df["pgraddppsi"], axis=0)
        dpdp = np.mean(df["pgraddpidpj"], axis=0)
        grad = 2 * (dpH - en * dp)
        Sij = dpdp - np.einsum("i,j->ij", dp, dp)  # + eps*np.eye(dpdp.shape[0])
        return configs, df['pgradtotal'].values[-1], grad, Sij, en, en_err

    x0 = pgrad_acc.transform.serialize_parameters(wf.parameters)
    datagrad = []
    datatest = []

    # VMC warm up period
    print('starting warmup')
    warmupoptions = vmcoptions.copy()
    if warmup is not None:
        warmupoptions.update(nsteps=warmup)
    if warmup is None or if warmup>0:
        data, coords = vmc(wf, coords, accumulators={}, **warmupoptions)
    print('warmup finished, nsteps', len(data))

    # Gradient descent cycles
    for it in range(maxiters):
        # Calculate gradient accurately
        coords, last_en, pgrad, Sij, en, en_err = gradient_energy_function(x0, coords)
        datagrad.append(
            {
                "pgrad": pgrad,
                "S": Sij,
                "en": en,
                "en_err": en_err,
                "iter": it,
                "params": x0.copy(),
            }
        )

        print("descent en", en, en_err )
        #print("descent grad", pgrad)
        print("descent |grad|", np.linalg.norm(pgrad), flush=True)

        xfit = []
        yfit = []
        xfit.append(0.0)
        yfit.append(last_en)

        # Calculate samples to fit
        steps = np.linspace(0, steprange, npts)
        steps[0] = -steprange / npts
        params = [x0 + update(pgrad, Sij, step, **update_kws) for step in steps]
        stepsdata = cslm(wf, coords, params, pgrad_acc, **cslmoptions)
        dfs = []
        for data, p, step in zip(stepsdata, params, steps):
            en = np.mean(data['total'])
            dfs.append( {
                'en': en,
                'en_err': np.std(data['total']) / np.sqrt(data['total'].size),
                'pgrad': 2*(np.mean(data['dpH'], axis=0)-en*np.mean(data['dppsi'],axis=0)),
                'step': step,
                'params': p.copy(),
                'iter': it
            })
            print("descent step", step, dfs[-1]['en'], dfs[-1]['en_err'], flush=True)

        xfit.extend(steps)
        yfit.extend([df['en'] for df in dfs])
        datatest.extend(dfs)

        # Fit minimum
        p = np.polyfit(xfit, yfit, 2)
        print("polynomial fit", p)
        est_min = -p[1] / (2 * p[0])
        print("estimated minimum", est_min, flush=True)
        minstep=np.min(xfit)
        if est_min > steprange and p[0] > 0: #minimum past the search radius
            est_min=steprange
        if est_min < minstep and p[0] > 0:  #mimimum behind the search radius
            est_min=minstep
        if p[0] < 0:
            plin=np.polyfit(xfit,yfit,1)
            if plin[0] < 0:
                est_min=steprange
            if plin[0] > 0:
                est_min=minstep
        print("estimated minimum adjusted",est_min,flush=True)
        
        x0 += update(pgrad, Sij, est_min, **update_kws)

        pd.DataFrame(datagrad).to_json(dataprefix + "grad.json")
        pd.DataFrame(datatest).to_json(dataprefix + "line.json")

    newparms = pgrad_acc.transform.deserialize(x0)
    for k in newparms:
        wf.parameters[k] = newparms[k]

    return wf, datagrad, datatest

def cslm_sampler(wf, configs, params, pgrad_acc):
    """ 
    Evaluates accumulator on the same set of configs for correlated sampling of different wave function parameters

    Args:
        wf: wave function object
        configs: (nconf, nelec, 3) array
        params: (nsteps, nparams) array 
            list of arrays of parameters (serialized) at each step
        pgrad_acc: PGradAccumulator 

    Returns:
        data: list of dicts, one dict for each sample
            each dict contains arrays returned from pgrad_acc, weighted by psi**2/psi0**2
    """

    import copy
    import numpy as np
    data = []
    psi0 = wf.recompute(configs)[1] # recompute gives logdet 
    for p in params:  
        newparms = pgrad_acc.transform.deserialize(p)
        for k in newparms:
            wf.parameters[k] = newparms[k]
        psi = wf.recompute(configs)[1] # recompute gives logdet
        rawweights = np.exp(2*(psi-psi0)) # convert from log(|psi|) to |psi|**2
        weights = rawweights/np.mean(rawweights) 
        df = pgrad_acc(configs, wf) 
        for k in df: 
            df[k] = np.einsum('i,i...->i...',weights,df[k]) # reweight all averaged quantities by sampling probability
        data.append(df)
    return data

