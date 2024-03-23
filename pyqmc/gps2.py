import numpy as np
from pyqmc import mc
import copy
import pyqmc.distance as distance


class GPSJastrow:
    def __init__(self, mol, X_support):
        self.n_support = X_support.shape[0]
        self._mol = mol
        self.iscomplex = False
        self.dtype = float
        self.parameters = {}

        # Xsupport.shape is nsupport,2,3
        self.parameters["Xsupport"] = X_support
        self.parameters["alpha"] = np.zeros(self.n_support)
        self.parameters["f"] = np.array([100])

    def recompute(self, configs):
        self._configscurrent = configs
        self.nconfig, self.nelec = configs.configs.shape[:-1]

        Xsup = self.parameters["Xsupport"]
        # ds are distance from xsup to config
        d_cs = np.zeros((self.nconfig, self.n_support, self.nelec, 2, 3))
        d_cs = (
            configs.configs[:, np.newaxis, :, np.newaxis, :]
            - Xsup[np.newaxis, :, np.newaxis, :, :]
        )

        r2_cs = np.sum(d_cs**2, axis=-1)
        e_cs = np.exp(-self.parameters["f"] * r2_cs)
        self.e_cs = e_cs
        return self.value()

    def value_from_e_cs(self, e_cs):
        cs = np.einsum("csi,csj->cs", e_cs[:, :, :, 0], e_cs[:, :, :, 1])
        cs -= np.einsum("csi,csi->cs", e_cs[:, :, :, 0], e_cs[:, :, :, 1])
        val = np.einsum("s,cs->c", self.parameters["alpha"], cs)
        return val

    def value(self):
        """returns phase and log(value) tuple"""
        val = self.value_from_e_cs(self.e_cs)
        return np.ones(self.nconfig), val

    def _compute_e_ne(self, e, epos):
        # shape is nsup,nconf,2,3
        d_ne = epos[..., np.newaxis, np.newaxis, :] - self.parameters["Xsupport"]
        # d_ne = epos[np.newaxis,:,np.newaxis,...] - self.parameters['Xsupport'][:,np.newaxis,:,:]
        # should be ending in 4,3
        r2_ne = np.sum(d_ne**2, axis=-1)
        e_ne = np.exp(-self.parameters["f"] * r2_ne)
        return e_ne

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]

        self.e_cs[mask, :, e, :] = self._compute_e_ne(e, epos.configs[mask])
        self._configscurrent.move(e, epos, mask)

    def testvalue(self, e, epos, mask=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        is_epos_3 = len(epos.configs.shape) - 2
        new_e_ne = self._compute_e_ne(e, epos.configs[mask])
        tup = (slice(None),) + is_epos_3 * (None,) + (slice(None), e, slice(None))
        dif1 = new_e_ne - self.e_cs[mask][tup]
        summ = np.sum(self.e_cs[mask], axis=2)
        dif2 = summ - self.e_cs[mask, :, e, :]

        tup1 = (slice(None),) + is_epos_3 * (None,) + (slice(None), 1)
        tup0 = (slice(None),) + is_epos_3 * (None,) + (slice(None), 0)

        final = dif1[..., 0] * dif2[tup1] + dif1[..., 1] * dif2[tup0]
        val = np.einsum("s,c...s->c...", self.parameters["alpha"], final)
        return np.exp(val), np.array([1])

    def gradient_value(self, e, epos):
        new_e_ne = self._compute_e_ne(e, epos.configs)
        dif1 = new_e_ne - self.e_cs[:, :, e, :]
        dif2 = np.sum(self.e_cs, axis=2) - self.e_cs[:, :, e, :]

        final = dif1[..., 0] * dif2[:, :, 1] + dif1[..., 1] * dif2[:, :, 0]
        val = np.einsum("s,c...s->c...", self.parameters["alpha"], final)

        d = (
            self.parameters["Xsupport"][np.newaxis, :, :, :]
            - epos.configs[:, np.newaxis, np.newaxis, :]
        )
        d = np.moveaxis(d, [0, 1, 2, 3], [1, 2, 3, 0])
        final = (
            new_e_ne[:, :, 0] * d[:, :, :, 0] * 2 * self.parameters["f"] * dif2[:, :, 1]
            + new_e_ne[:, :, 1]
            * d[:, :, :, 1]
            * 2
            * self.parameters["f"]
            * dif2[:, :, 0]
        )
        grad = np.einsum("s,dcs->dc", self.parameters["alpha"], final)
        return grad, np.exp(val), np.array([1])

    def gradient(self, e, epos):
        d = (
            self.parameters["Xsupport"][np.newaxis, :, :, :]
            - epos.configs[:, np.newaxis, np.newaxis, :]
        )
        d = np.moveaxis(d, [0, 1, 2, 3], [1, 2, 3, 0])
        new_e_ne = self._compute_e_ne(e, epos.configs)
        summ = np.sum(self.e_cs, axis=2) - self.e_cs[:, :, e, :]
        # can simplify last 2 lines by switching axis order of summ, so this fits into 1 einsum. but that will make stuff less readable, so will do
        # later
        final = (
            new_e_ne[:, :, 0] * d[:, :, :, 0] * 2 * self.parameters["f"] * summ[:, :, 1]
            + new_e_ne[:, :, 1]
            * d[:, :, :, 1]
            * 2
            * self.parameters["f"]
            * summ[:, :, 0]
        )
        grad = np.einsum("s,dcs->dc", self.parameters["alpha"], final)
        return grad

    def gradient_laplacian(self, e, epos):
        d = (
            self.parameters["Xsupport"][np.newaxis, :, :, :]
            - epos.configs[:, np.newaxis, np.newaxis, :]
        )
        d = np.moveaxis(d, [0, 1, 2, 3], [1, 2, 3, 0])
        new_e_ne = self._compute_e_ne(e, epos.configs)
        summ = np.sum(self.e_cs, axis=2) - self.e_cs[:, :, e, :]
        final = (
            new_e_ne[:, :, 0] * d[:, :, :, 0] * 2 * self.parameters["f"] * summ[:, :, 1]
            + new_e_ne[:, :, 1]
            * d[:, :, :, 1]
            * 2
            * self.parameters["f"]
            * summ[:, :, 0]
        )

        grad = np.einsum("s,dcs->dc", self.parameters["alpha"], final)
        d = np.moveaxis(d, [0, 1, 2, 3], [3, 0, 1, 2])
        r = np.sum(d**2, axis=-1)

        term1 = 4 * self.parameters["f"] ** 2 * r - 6 * self.parameters["f"]
        final = (
            term1[:, :, 0] * new_e_ne[:, :, 0] * summ[:, :, 1]
            + term1[:, :, 1] * new_e_ne[:, :, 1] * summ[:, :, 0]
        )
        lap = np.einsum("s,cs->c", self.parameters["alpha"], final)
        lap += np.sum(grad**2, axis=0)
        return grad, lap

    def laplacian(self, e, epos):

        d = (
            self.parameters["Xsupport"][np.newaxis, :, :, :]
            - epos.configs[:, np.newaxis, np.newaxis, :]
        )

        r = np.sum(d**2, axis=-1)

        new_e_ne = self._compute_e_ne(e, epos.configs)
        summ = np.sum(self.e_cs, axis=2) - self.e_cs[:, :, e, :]

        term1 = 4 * self.parameters["f"] ** 2 * r - 6 * self.parameters["f"]
        final = (
            term1[:, :, 0] * new_e_ne[:, :, 0] * summ[:, :, 1]
            + term1[:, :, 1] * new_e_ne[:, :, 1] * summ[:, :, 0]
        )
        lap = np.einsum("s,cs->c", self.parameters["alpha"], final)

        grad = self.gradient(e, epos)
        lap += np.sum(grad**2, axis=0)
        return lap

    # configs shape nconfig,nelec,3
    def pgradient(self):
        configs = self._configscurrent.configs
        A = (self.e_cs - np.sum(self.e_cs, axis=2, keepdims=True))[:, :, :, ::-1]
        alphader = -np.einsum("csi,csi->cs", A[:, :, :, 0], A[:, :, :, 1])

        d = (
            self.parameters["Xsupport"][np.newaxis, :, np.newaxis, :, :]
            - configs[:, np.newaxis, :, np.newaxis, :]
        )
        r = np.sum(d**2, axis=-1)

        fder = np.einsum(
            "s,csit,csit,csit->c",
            self.parameters["alpha"],
            self.e_cs,
            A,
            r,
            optimize="greedy",
        )

        Xder = np.einsum(
            "s,csitd,csit,csit->cstd",
            2 * self.parameters["f"] * self.parameters["alpha"],
            d,
            self.e_cs,
            A,
            optimize="greedy",
        )

        return {"alpha": alphader, "Xsupport": Xder, "f": fder}
