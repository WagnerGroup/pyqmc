import numpy as np

class StochasticReconfiguration:
    """ 
    This class works as an accumulator, but has an extra method that computes the change in parameters
    given the averages given by avg() and __call__. 
    """

    def __init__(self, enacc, transform, nodal_cutoff=1e-3):
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff

    def _node_regr(self, configs, grad2):
        """
        Return true if a given configuration is within nodal_cutoff
        of the node
        Also return the regularization polynomial if true,
        f = a * r ** 2 + b * r ** 4 + c * r ** 6
        """
        r = 1.0 / grad2
        mask = r < self.nodal_cutoff**2

        c = 7.0 / (self.nodal_cutoff**6)
        b = -15.0 / (self.nodal_cutoff**4)
        a = 9.0 / (self.nodal_cutoff**2)

        f = a * r + b * r**2 + c * r**3
        f[np.logical_not(mask)] = 1.0

        return mask, f

    def __call__(self, configs, wf):
        pgrad = wf.pgradient()
        d = self.enacc(configs, wf)
        energy = d["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut, f = self._node_regr(configs, d["grad2"])
        dp_regularized = dp * f[:, np.newaxis]

        d["dpH"] = np.einsum("i,ij->ij", energy, dp_regularized)
        d["dppsi"] = dp_regularized
        d["dpidpj"] = np.einsum("ij,ik->ijk", dp, dp_regularized)

        return d

    def avg(self, configs, wf, weights=None):
        """
        Compute (weighted) average
        """

        nconf = configs.configs.shape[0]
        if weights is None:
            weights = np.ones(nconf)
        weights = weights / np.sum(weights)

        pgrad = wf.pgradient()
        den = self.enacc(configs, wf)
        energy = den["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut, f = self._node_regr(configs, den["grad2"])

        dp_regularized = dp * f[:, np.newaxis]

        d = {k: np.average(it, weights=weights, axis=0) for k, it in den.items()}
        d["dpH"] = np.einsum("i,ij->j", energy, weights[:, np.newaxis] * dp_regularized)
        d["dppsi"] = np.average(dp_regularized, weights=weights, axis=0)
        d["dpidpj"] = np.einsum(
            "ij,ik->jk", dp, weights[:, np.newaxis] * dp_regularized, optimize=True
        )

        return d

    def keys(self):
        return self.enacc.keys().union(["dpH", "dppsi", "dpidpj"])

    def shapes(self):
        nparms = np.sum([np.sum(opt) for opt in self.transform.to_opt.values()])
        d = {"dpH": (nparms,), "dppsi": (nparms,), "dpidpj": (nparms, nparms)}
        d.update(self.enacc.shapes())
        return d

