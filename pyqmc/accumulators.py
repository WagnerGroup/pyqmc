import numpy as np
import pyqmc.energy as energy
from pyqmc.ewald import Ewald


class EnergyAccumulator:
    """returns energy of each configuration in a dictionary. 
  Keys and their meanings can be found in energy.energy """

    def __init__(self, mol, threshold=10, **kwargs):
        self.mol = mol
        self.threshold = threshold
        if hasattr(mol, "a"):
            ewald = Ewald(mol, **kwargs)
            self.compute_energy = ewald.compute_total_energy
        else:
            self.compute_energy = energy.energy

    def __call__(self, configs, wf):
        return self.compute_energy(self.mol, configs, wf, self.threshold)

    def avg(self, configs, wf):
        d = {}
        for k, it in self(configs, wf).items():
            d[k] = np.mean(it, axis=0)
        return d

    def keys(self):
        return set(["ke", "ee", "ei", "ecp", "total"])

    def shapes(self):
        return {"ke": (), "ee": (), "ei": (), "ecp": (), "total": ()}


class LinearTransform:
    """
    Linearize a dictionary of wf parameters. 
    to_opt is a dictionary with the keys to optimize, and its values are boolean arrays indicating which specific elements to optimize
    Note: to_opt[k] can't be boolean scalar; it has to be an array with the same dimension as parameters[k]. 
    to_opt doesn't have to have all the keys of parameters, but all keys of to_opt must be keys of parameters.
    """

    def __init__(self, parameters, to_opt=None):
        if to_opt is None:
            to_opt = {k: np.ones(p.shape, dtype=bool) for k, p in parameters.items()}
        self.to_opt = {k: o for k, o in to_opt.items() if np.any(o)}

        self.frozen_parms = {k: parameters[k][~opt] for k, opt in self.to_opt.items()}

        self.shapes = {k: parameters[k].shape for k in self.to_opt}
        self.slices = {k: np.prod(s) for k, s in self.shapes.items()}
        self.dtypes = {k: parameters[k].dtype for k in self.to_opt}
        self.complex = {k: d == complex for k, d in self.dtypes.items()}
        self.nimag = {k: to_opt[k].sum() if c else 0 for k, c in self.complex.items()}
        self.complex_inds = np.concatenate(
            [np.ones(to_opt[k].sum(), dtype=bool) * c for k, c in self.complex.items()]
        )
        self.nparams = np.sum([v.sum() for v in self.to_opt.values()])

    def serialize_parameters(self, parameters):
        """Convert the dictionary to a linear list
        of gradients
        """
        params = np.concatenate([parameters[k][opt] for k, opt in self.to_opt.items()])
        return np.concatenate((params.real, params[self.complex_inds].imag))

    def serialize_gradients(self, pgrad):
        """Convert the dictionary to a linear list
        of gradients, mask allows for certain fixed parameters
        """
        grads = []
        for k, opt in self.to_opt.items():
            mask = ~np.repeat(opt[np.newaxis, :], pgrad[k].shape[0], axis=0)
            mask_grads = np.ma.array(pgrad[k], mask=mask).reshape(pgrad[k].shape[0], -1)
            grads.append(np.ma.compress_cols(mask_grads))

        grads = np.concatenate(grads, axis=1)
        return np.concatenate((grads, grads[:, self.complex_inds] * 1j), axis=1)

    def deserialize(self, parameters):
        """Convert serialized parameters to dictionary
        """
        n = 0
        m = self.nparams
        d = {}
        for k, opt in self.to_opt.items():
            opt_ = opt.flatten()
            n_p = np.sum(opt_)

            flat_parms = np.zeros(self.slices[k], dtype=self.dtypes[k])
            flat_parms[~opt_] = self.frozen_parms[k]
            flat_parms[opt_] = parameters[n : n + n_p]
            if self.complex[k]:
                m_p = self.nimag[k]
                flat_parms[opt_] += parameters[m : m + m_p] * 1j
                m += m_p
            d[k] = flat_parms.reshape(self.shapes[k])
            n += n_p
        return d


class PGradTransform:
    """   """

    def __init__(self, enacc, transform, nodal_cutoff=1e-3):
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff

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
            d2 += np.sum(np.abs(wf.gradient(e, configs.electron(e))) ** 2, axis=0)
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

        node_cut, f = self._node_regr(configs, wf)

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


class SqAccumulator:
    r"""
    Accumulates structure factor 

    .. math:: S(\vec{q}) = \langle \rho_{\vec{q}} \rho_{-\vec{q}} \rangle
                         = \langle \left| \sum_{j=1}^{N_e} e^{i\vec{q}\cdot\vec{r}_j} \right| \rangle

    """

    def __init__(self, qlist=None, Lvecs=None, nq=4):
        """
        Inputs:
            qlist: (n, 3) array-like. If qlist is provided, Lvecs and nq are ignored
            Lvecs: (3, 3) array-like of lattice vectors. Required if qlist is None
            nq: int, if qlist is nonzero, use a uniform grid of shape (nq, nq, nq)
        """
        if qlist is not None:
            self.qlist = qlist
        else:
            assert (
                Lvecs is not None
            ), "need to provide either list of q vectors or lattice vectors"
            Gvecs = np.linalg.inv(Lvecs).T * 2 * np.pi
            qvecs = list(map(np.ravel, np.meshgrid(*[np.arange(nq)] * 3)))
            qvecs = np.stack(qvecs, axis=1)
            self.qlist = np.dot(qvecs, Gvecs)

    def __call__(self, configs, wf):
        nelec = configs.configs.shape[1]
        exp_iqr = np.exp(1j * np.inner(configs.configs, self.qlist))
        sum_exp_iqr = exp_iqr.sum(axis=1)
        return {"Sq": (sum_exp_iqr.real ** 2 + sum_exp_iqr.imag ** 2) / nelec}

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def keys(self):
        return set(["Sq"])

    def shapes(self):
        return {"Sq": (len(self.qlist),)}
