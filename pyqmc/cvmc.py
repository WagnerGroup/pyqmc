import pyqmc
import numpy as np


class DescriptorFromOBDM:
    """
    The reason that this has to be an object here is that parsl doesn't support 
    functions and objects that are defined in __main__. 
    """

    def __init__(self, mapping, norm=1.0):
        """mapping should be a dictionary such that each descriptor has nret lists of weights and indices to add t
        together
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


class PGradDescriptor:
    """   """

    def __init__(self, enacc, transform, dm_evaluators, descriptors, nodal_cutoff=1e-5):
        """ 
        
        descriptors : function-like object that translates an obdm_up and obdm_down return to a dictionary of descriptors
        """
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff
        self.dm_evaluators = dm_evaluators
        self.descriptors = descriptors

    def _node_cut(self, configs, wf):
        """ Return true if a given configuration is within nodal_cutoff 
        of the node """
        ne = configs.configs.shape[1]
        d2 = 0.0
        for e in range(ne):
            d2 += np.sum(wf.gradient(e, configs.electron(e)) ** 2, axis=0)
        r = 1.0 / (d2 * ne * ne)
        return r < self.nodal_cutoff ** 2

    def __call__(self, configs, wf):
        pgrad = wf.pgradient()
        d = self.enacc(configs, wf)
        energy = d["total"]
        dp = self.transform.serialize_gradients(pgrad)
        node_cut = self._node_cut(configs, wf)
        dp[node_cut, :] = 0.0
        # print('number cut off',np.sum(node_cut))

        d["dpH"] = np.einsum("i,ij->ij", energy, dp)
        d["dppsi"] = dp
        d["dpidpj"] = np.einsum("ij,ik->ijk", dp, dp)
        raise NotImplementedError("define __call__ for PGradOBDMTransform")
        return d

    def avg(self, configs, wf):
        nconf = configs.configs.shape[0]
        pgrad = wf.pgradient()
        den = self.enacc(configs, wf)
        energy = den["total"]
        dp = self.transform.serialize_gradients(pgrad)

        dms = [evaluate(configs, wf) for evaluate in self.dm_evaluators]
        descript = self.descriptors(dms)

        node_cut = self._node_cut(configs, wf)
        dp[node_cut, :] = 0.0
        # print('number cut off',np.sum(node_cut))

        d = {}
        for k, it in den.items():
            d[k] = np.mean(it, axis=0)
        d["dpH"] = np.einsum("i,ij->j", energy, dp) / nconf
        d["dppsi"] = np.mean(dp, axis=0)
        d["dpidpj"] = np.einsum("ij,ik->jk", dp, dp) / nconf

        for di, desc in descript.items():
            d["dp" + di] = np.einsum("ij,i->j", dp, desc) / nconf
            d["avg" + di] = np.sum(desc) / nconf
        d["nodal_cutoff"] = np.sum(node_cut)

        return d


def cvmc(
    wf,
    configs,
    acc,
    objective,
    forcing,
    iters=10,
    tstep=0.5,
    npts=10,
    vmc=None,
    vmcoptions=None,
    datafile=None,
):
    """ 
    Args: 

       wf : a wave function object

       configs : starting configurations for VMC. Ideally equilibrated with wf.

       acc : A PGradDescriptor object which generates descriptors

       objective : A dictionary which has one value for every descriptor returned by acc

       forcing : A dictionary which has one value for every descriptor returned by acc
    """
    if vmc is None:
        vmc = pyqmc.vmc
    if vmcoptions is None:
        vmcoptions = {}

    import pandas as pd

    def get_obj_deriv(x):
        nonlocal configs
        for k, p in acc.transform.deserialize(x).items():
            wf.parameters[k] = p
        df, configs = vmc(wf, configs, accumulators={"grad": acc}, **vmcoptions)

        df = pd.DataFrame(df)
        dpavg = np.mean(df["graddppsi"])

        havg = np.mean(df["gradtotal"])
        dpH = np.mean(df["graddpH"])
        dEdp = dpH - dpavg * havg

        # avgt=np.mean(df['gradavgt'])
        # dpt=np.mean(df['graddpt'])
        # dtdp=dpt-dpavg*avgt
        # distfromobj=avgt-tobj
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

        print("energy", havg, "avg", qavg, "objective function", objfunc, flush=True)
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
        return dret

    x0 = acc.transform.serialize_parameters(wf.parameters)
    df = []
    for it in range(iters):
        grad = get_obj_deriv(x0)
        grad["steptype"] = "gradient"
        grad["tau"] = 0.0
        grad["iteration"] = it
        grad["parameters"] = x0.copy()
        df.append(grad)
        taus = np.linspace(0, tstep, npts)
        taus[0] = -tstep / npts
        xfit = [0]
        yfit = [grad["objfunc"]]

        for tau in taus:
            print("+++++ tau ++++++", tau)
            x = x0 - tau * grad["objderiv"] / np.linalg.norm(grad["objderiv"])
            gradnew = get_obj_deriv(x)
            xfit.append(tau)
            yfit.append(gradnew["objfunc"])
            gradnew["steptype"] = "line"
            gradnew["tau"] = tau
            gradnew["iteration"] = it
            gradnew["parameters"] = x.copy()
            df.append(gradnew)

        p = np.polyfit(xfit, yfit, 2)
        print("fitting", xfit, yfit)
        print("polynomial fit", p)
        est_min = -p[1] / (2 * p[0])
        print("estimated minimum", est_min, flush=True)
        minstep = np.min(xfit)
        if est_min > tstep and p[0] > 0:  # minimum past the search radius
            est_min = tstep
        if est_min < minstep and p[0] > 0:  # mimimum behind the search radius
            est_min = minstep
        if p[0] < 0:
            plin = np.polyfit(xfit, yfit, 1)
            if plin[0] < 0:
                est_min = tstep
            if plin[0] > 0:
                est_min = minstep
        print("estimated minimum adjusted", est_min, flush=True)

        x0 = x0 - est_min * grad["objderiv"] / np.linalg.norm(grad["objderiv"])
        if not datafile is None:
            pd.DataFrame(df).to_json(datafile)

    for k, p in acc.transform.deserialize(x0).items():
        wf.parameters[k] = p

    return wf, df
