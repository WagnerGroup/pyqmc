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
            print("EnergyAccumulator using Ewald\n", kwargs)
            self.ewald = Ewald(mol, **kwargs)

            def compute_energy(mol, configs, wf, threshold):
                ee, ei, ii = self.ewald.energy(configs)
                ecp_val = energy.get_ecp(mol, configs, wf, threshold)
                ke = energy.kinetic(configs, wf)
                return {
                    "ke": ke,
                    "ee": ee,
                    "ei": ei + ecp_val,
                    "total": ke + ee + ei + ecp_val + ii,
                }

            self.compute_energy = compute_energy
        else:
            self.compute_energy = energy.energy

    def __call__(self, configs, wf):
        return self.compute_energy(self.mol, configs, wf, self.threshold)

    def avg(self, configs, wf):
        d = {}
        for k, it in self(configs, wf).items():
            d[k] = np.mean(it, axis=0)
        return d


class LinearTransform:
    """
    Linearize a dictionary of wf parameters, only to_opt
    terms optimized. A dict freeze can be used to freeze
    certain wf parameters.
    """

    def __init__(self, parameters, to_opt=None, freeze=None):
        if to_opt is None:
            self.to_opt = list(parameters.keys())
        else:
            self.to_opt = to_opt

        if freeze is None:
            freeze = {}
            for i, k in enumerate(parameters):
                freeze[k] = np.zeros(parameters[k].shape).astype(bool)
            self.frozen = freeze
        else:
            self.frozen = freeze

        self.frozen_parms = {}
        for k in self.to_opt:
            mask = self.frozen[k]
            self.frozen_parms[k] = np.ma.array(parameters[k], mask=~mask)

        self.shapes = np.array([parameters[k].shape for k in self.to_opt])
        self.slices = np.array([np.prod(s) for s in self.shapes])

    def serialize_parameters(self, parameters):
        """Convert the dictionary to a linear list
        of gradients
        """
        params = []
        for k in self.to_opt:
            mask = self.frozen[k]
            params.append(np.ma.array(parameters[k], mask=mask).compressed())
        return np.concatenate(params)

    def serialize_gradients(self, pgrad):
        """Convert the dictionary to a linear list
        of gradients, mask allows for certain fixed parameters
        """
        grads = []
        for k in self.to_opt:
            mask = np.repeat(self.frozen[k][np.newaxis, :], pgrad[k].shape[0], axis=0)
            mask_grads = np.ma.array(pgrad[k], mask=mask).reshape(pgrad[k].shape[0], -1)
            grads.append(np.ma.compress_cols(mask_grads))
        return np.concatenate(grads, axis=1)

    def deserialize(self, parameters):
        """Convert serialized parameters to dictionary
        """
        n = 0
        d = {}
        for i, k in enumerate(self.to_opt):
            mask = self.frozen[k].flatten()
            n_p = np.sum(~mask)

            flat_parms = np.zeros(self.slices[i])
            flat_parms[mask] = self.frozen_parms[k].compressed()
            flat_parms[~mask] = parameters[n : n + n_p]
            d[k] = flat_parms.reshape(self.shapes[i])
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

        return d

    def avg(self, configs, wf):
        nconf = configs.configs.shape[0]
        pgrad = wf.pgradient()
        den = self.enacc(configs, wf)
        energy = den["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut, f = self._node_regr(configs, wf)
        dp_regularized = dp * f[:, np.newaxis]

        d = {}
        for k, it in den.items():
            d[k] = np.mean(it, axis=0)
        d["dpH"] = np.einsum("i,ij->j", energy, dp_regularized) / nconf
        d["dppsi"] = np.mean(dp_regularized, axis=0)
        d["dpidpj"] = np.einsum("ij,ik->jk", dp, dp_regularized) / nconf

        return d
