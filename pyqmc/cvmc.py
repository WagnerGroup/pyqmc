import pyqmc
import numpy as np
import h5py


class DescriptorFromOBDM:
    """
    The reason that this has to be an object here is that parsl doesn't support 
    functions and objects that are defined in __main__.
    """

    def __init__(self, mapping, norm=1.0):
        """mapping should be a dictionary such that each descriptor has 
        nret lists of weights and indices to add together
        For example, 
        {'t': [ 
                 [ (1.0, (0,1)), 
                   (1.0, (1,0)) 
                   ], 
                 [ (1.0, (0,1)), 
                    (1.0,(1,0)) ] 
               ] 
        }
        """
        self.norm = norm
        self.mapping = mapping
        pass

    def __call__(self, rets):
        """
        Return a dictionary of descriptors
        """
        avgvals = {}
        for k, mapping in self.mapping.items():
            n = rets[0]["value"].shape[0]
            totsum = np.zeros(n)
            for ret, ellist in zip(rets, mapping):
                for w, ind in ellist:
                    totsum += self.norm * w * ret["value"][:, ind[0], ind[1]]
            avgvals[k] = totsum

        return avgvals


class DescriptorFromTBDM:
    """
    The reason that this has to be an object here is that parsl doesn't support 
    functions and objects that are defined in __main__. 
    """

    def __init__(self, mapping, norm=1.0):
        """mapping should be a dictionary such that each descriptor has 
        nret lists of weights and indices to add together
        For example, 
        {'U': [
                [], #up - up
                [(0.5, (0))], #up - dn
                [(0.5, (0))], #dn - up
                [], #up - up
        ]
        """
        self.norm = norm
        self.mapping = mapping
        pass

    def __call__(self, rets):
        """
        Return a dictionary of descriptors
        """
        avgvals = {}
        for k, mapping in self.mapping.items():
            n = rets[0]["value"].shape[0]
            totsum = np.zeros(n)
            for ret, ellist in zip(rets, mapping):
                for w, ind in ellist:
                    totsum += self.norm * w * ret["value"][:, ind]
            avgvals[k] = totsum

        return avgvals


class PGradDescriptor:
    """   """

    def __init__(self, enacc, transform, dm_evaluators, descriptors, nodal_cutoff=1e-3):
        """ 
        dm_evaluators: dictionary of density matrix accumulator objects   
        descriptors : dictionary of function-like objects that translate dm_evaluators[key] to 
                      a dictionary of descriptors
        """
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff
        self.dm_evaluators = dm_evaluators
        self.descriptors = descriptors

    def _node_regr(self, configs, wf):
        """ 
        Return true if a given configuration is within nodal_cutoff 
        of the node 
        Also return the regularization polynomial if true, 
        f = a * r ** 2 + b * r ** 4 + c * r ** 3
        """
        ne = configs.configs.shape[1]
        d2 = 0.0
        for e in range(ne):
            d2 += np.sum(wf.gradient(e, configs.electron(e)) ** 2, axis=0)
        r = 1.0 / d2
        mask = r < self.nodal_cutoff ** 2

        c = 7.0 / (self.nodal_cutoff ** 6)
        b = -15.0 / (self.nodal_cutoff ** 4)
        a = 9.0 / (self.nodal_cutoff ** 2)

        f = a * r + b * r ** 2 + c * r ** 3
        f[np.logical_not(mask)] = 1.0

        return mask, f

    def __call__(self, configs, wf):
        pgrad = wf.pgradient()
        d = self.enacc(configs, wf)
        energy = d["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut, f = self._node_regr(configs, wf)
        dp_regularized = dp * f[:, np.newaxis]

        d["dpH"] = np.einsum("i,ij->ij", energy, dp_regularized)
        d["dppsi"] = dp_regularized
        d["dpidpj"] = np.einsum("ij,ik->ijk", dp, dp_regularized)
        raise NotImplementedError("define __call__ for PGradOBDMTransform")
        return d

    def avg(self, configs, wf):
        nconf = configs.configs.shape[0]
        pgrad = wf.pgradient()
        den = self.enacc(configs, wf)
        energy = den["total"]
        dp = self.transform.serialize_gradients(pgrad)

        descript = {}
        for dm_type, evaluators in self.dm_evaluators.items():
            dms = [evaluate(configs, wf) for evaluate in evaluators]
            descript.update(self.descriptors[dm_type](dms))

        node_cut, f = self._node_regr(configs, wf)
        dp_regularized = dp * f[:, np.newaxis]

        d = {}
        for k, it in den.items():
            d[k] = np.mean(it, axis=0)
        d["dpH"] = np.einsum("i,ij->j", energy, dp_regularized) / nconf
        d["dppsi"] = np.mean(dp_regularized, axis=0)
        d["dpidpj"] = np.einsum("ij,ik->jk", dp, dp_regularized) / nconf

        for di, desc in descript.items():
            d["dp" + di] = np.einsum("ij,i->j", dp_regularized, desc) / nconf
            d["avg" + di] = np.sum(desc) / nconf
        d["nodal_cutoff"] = np.sum(node_cut)

        return d


def cvmc_optimize(
    wf,
    configs,
    acc,
    objective,
    forcing,
    iters=10,
    tstep=0.5,
    npts=10,
    datafile=None,
    vmc=None,
    vmcoptions=None,
    lm=None,
    lmoptions=None,
    update=pyqmc.linemin.sr_update,
    update_kws=None,
    hdf_file=None,
):
    """
    Args:

       wf : a wave function object

       configs : starting configurations for VMC. Ideally equilibrated with wf.

       acc : A PGradDescriptor object which generates descriptors

       objective : A dictionary which has one value for every descriptor returned by acc

       forcing : A dictionary which has one value for every descriptor returned by acc
    """
    import pandas as pd

    if vmc is None:
        vmc = pyqmc.vmc
    if vmcoptions is None:
        vmcoptions = {}
    if lm is None:
        lm = lm_cvmc
    if lmoptions is None:
        lmoptions = {}
    if update_kws is None:
        update_kws = {}

    # Restart
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "wf" in hdf.keys():
                grp = hdf["wf"]
                for k in grp.keys():
                    wf.parameters[k] = np.array(grp[k])

    # Attributes for cvmc solve
    attr = dict(iters=iters, npts=npts, tstep=tstep)
    for k, it in objective.items():
        attr["objective_" + k] = it
    for k, it in forcing.items():
        attr["forcing_" + k] = it

    def get_obj_deriv(x):
        nonlocal configs
        for k, p in acc.transform.deserialize(x).items():
            wf.parameters[k] = p
        df, configs = vmc(wf, configs, accumulators={"grad": acc}, **vmcoptions)

        df = pd.DataFrame(df)
        dpavg = np.mean(df["graddppsi"], axis=0)
        Sij = np.mean(df["graddpidpj"], axis=0) - np.einsum("i,j->ij", dpavg, dpavg)

        havg = np.mean(df["gradtotal"], axis=0)
        dpH = np.mean(df["graddpH"], axis=0)
        dEdp = dpH - dpavg * havg

        qavg = {}
        qdp = {}
        distfromobj = 0.0
        objderiv = dEdp
        objfunc = havg

        for k, force in forcing.items():
            qavg[k] = np.mean(df["gradavg" + k])
            qdp[k] = np.mean(df["graddp" + k]) - dpavg * qavg[k]
            distobj = qavg[k] - objective[k]
            objderiv += 2 * force * distobj * qdp[k]
            distfromobj += distobj
            objfunc += force * distobj ** 2

        dret = {
            "objderiv": objderiv,
            "energy": havg,
            "objfunc": objfunc,
            "dist": distfromobj,
            "dEdp": dEdp,
        }
        for k, avg in qavg.items():
            dret["avg" + k] = avg
        for k, avg in qdp.items():
            dret["dp" + k] = avg
        return dret, Sij

    x0 = acc.transform.serialize_parameters(wf.parameters)

    df = []
    for it in range(iters):
        grad, Sij = get_obj_deriv(x0)
        grad["iteration"] = it
        grad["parameters"] = x0.copy()
        for k, force in forcing.items():
            print(k, grad["avg" + k], grad["dp" + k], flush=True)
        xfit = []
        yfit = []

        taus = np.linspace(0, tstep, npts + 1)
        taus[0] = -tstep / (npts - 1)
        params = [x0 + update(grad["objderiv"], Sij, tau, **update_kws) for tau in taus]
        stepsdata = lm(wf, configs, params, acc, **lmoptions)

        for data, p, tau in zip(stepsdata, params, taus):
            en = np.mean(data["total"] * data["weight"]) / np.mean(data["weight"])

            qavg = {}
            distfromobj = 0.0
            objfunc = en
            for k, force in forcing.items():
                qavg[k] = np.mean(data[k] * data["weight"]) / np.mean(data["weight"])
                distobj = qavg[k] - objective[k]
                distfromobj += distobj
                objfunc += force * distobj ** 2

            xfit.append(tau)
            yfit.append(objfunc)

        est_min = pyqmc.linemin.stable_fit(xfit, yfit)
        x0 = x0 + update(grad["objderiv"], Sij, est_min, **update_kws)

        grad["yfit"] = yfit
        grad["taus"] = xfit
        pyqmc.linemin.opt_hdf(
            hdf_file, grad, attr, configs, acc.transform.deserialize(x0)
        )

        df.append(grad)
        if datafile is not None:
            pd.DataFrame(df).to_json(datafile)

    for k, p in acc.transform.deserialize(x0).items():
        wf.parameters[k] = p

    return wf, df


def lm_cvmc(wf, configs, params, acc):
    """ 
    Evaluates accumulator on the same set of configs for correlated sampling of different wave function parameters

    Args:
        wf: wave function object
        configs: (nconf, nelec, 3) array
        params: (nsteps, nparams) array 
            list of arrays of parameters (serialized) at each step
        acc: PGradDescriptor 

    Returns:
        data: list of dicts, one dict for each sample
            each dict contains arrays returned from PGradDescriptor, weighted by psi**2/psi0**2
    """

    import copy
    import numpy as np

    data = []
    psi0 = wf.recompute(configs)[1]  # recompute gives logdet

    # Aggregate evaluator configurations
    extra_configs = []
    auxassignments = []
    for dm_type, evaluators in acc.dm_evaluators.items():
        for evaluate in evaluators:
            res = evaluate.get_extra_configs(configs)
            extra_configs.append(res[0])
            auxassignments.append(res[1])

    # Run the correlated evaluation
    for p in params:
        newparms = acc.transform.deserialize(p)
        for k in newparms:
            wf.parameters[k] = newparms[k]
        psi = wf.recompute(configs)[1]  # recompute gives logdet
        rawweights = np.exp(2 * (psi - psi0))  # convert from log(|psi|) to |psi|**2

        df = acc.enacc(configs, wf)

        descript = {}
        i = 0
        for dm_type, evaluators in acc.dm_evaluators.items():
            dms = []
            for evaluate in evaluators:
                dms += [evaluate(configs, wf, extra_configs[i], auxassignments[i])]
                i += 1
            descript.update(acc.descriptors[dm_type](dms))

        for di, desc in descript.items():
            df[di] = desc
        df["weight"] = rawweights

        data.append(df)

    return data
