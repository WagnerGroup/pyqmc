import numpy as np
from pyqmc.func3d import GaussianFunction
from pyqmc.distance import RawDistance


class JastrowSpin:
    """
    1 body and 2 body jastrow factor
    """

    def __init__(self, mol, a_basis=None, b_basis=None, dist=RawDistance()):
        """
        Args:

        mol : a pyscf molecule object

        a_basis : list of func3d objects that comprise the electron-ion basis

        b_basis : list of func3d objects that comprise the electron-electron basis

        dist: a distance calculator

        """
        if b_basis is None:
            nexpand = 5
            self.b_basis = [
                GaussianFunction(0.2 * 2 ** n) for n in range(1, nexpand + 1)
            ]
        else:
            nexpand = len(b_basis)
            self.b_basis = b_basis

        if a_basis is None:
            aexpand = 4
            self.a_basis = [
                GaussianFunction(0.2 * 2 ** n) for n in range(1, aexpand + 1)
            ]
        else:
            aexpand = len(a_basis)
            self.a_basis = a_basis

        self.parameters = {}
        self._nelec = np.sum(mol.nelec)
        self._mol = mol
        self._dist = dist
        self.parameters["bcoeff"] = np.zeros((nexpand, 3))
        self.parameters["acoeff"] = np.zeros((self._mol.natm, aexpand, 2))

    def recompute(self, configs):
        """ """
        u = 0.0
        configsc = configs.configs.copy()
        self._configscurrent = configsc
        elec = self._mol.nelec
        nconfig = configsc.shape[0]
        nexpand = len(self.b_basis)
        aexpand = len(self.a_basis)
        self._bvalues = np.zeros((nconfig, nexpand, 3))
        self._avalues = np.zeros((nconfig, self._mol.natm, aexpand, 2))

        nup = elec[0]
        d1, ij = self._dist.dist_matrix(configsc[:, :nup, :])
        d2, ij = self._dist.pairwise(configsc[:, :nup, :], configsc[:, nup:, :])
        d3, ij = self._dist.dist_matrix(configsc[:, nup:, :])

        d1 = d1.reshape((-1, 3))
        d2 = d2.reshape((-1, 3))
        d3 = d3.reshape((-1, 3))
        r1 = np.linalg.norm(d1, axis=1)
        r2 = np.linalg.norm(d2, axis=1)
        r3 = np.linalg.norm(d3, axis=1)

        # Package the electron-ion distances into a 1d array
        di1 = np.zeros((configsc.shape[0], self._mol.natm, nup, 3))
        di2 = np.zeros((configsc.shape[0], self._mol.natm, configsc.shape[1] - nup, 3))

        for e in range(nup):
            di1[:, :, e, :] = self._dist.dist_i(
                self._mol.atom_coords(), configsc[:, e, :]
            )
        for e in range(nup, configsc.shape[1]):
            di2[:, :, e - nup, :] = self._dist.dist_i(
                self._mol.atom_coords(), configsc[:, e, :]
            )

        # print(di1.shape)
        di1 = di1.reshape((-1, 3))
        di2 = di2.reshape((-1, 3))
        ri1 = np.linalg.norm(di1, axis=1)
        ri2 = np.linalg.norm(di2, axis=1)

        # Update bvalues according to spin case
        for i, b in enumerate(self.b_basis):
            self._bvalues[:, i, 0] = np.sum(
                b.value(d1, r1).reshape((configsc.shape[0], -1)), axis=1
            )
            self._bvalues[:, i, 1] = np.sum(
                b.value(d2, r2).reshape((configsc.shape[0], -1)), axis=1
            )
            self._bvalues[:, i, 2] = np.sum(
                b.value(d3, r3).reshape((configsc.shape[0], -1)), axis=1
            )

        # Update avalues according to spin case
        for i, a in enumerate(self.a_basis):
            self._avalues[:, :, i, 0] = np.sum(
                a.value(di1, ri1).reshape((configsc.shape[0], self._mol.natm, -1)),
                axis=2,
            )
            self._avalues[:, :, i, 1] = np.sum(
                a.value(di2, ri2).reshape((configsc.shape[0], self._mol.natm, -1)),
                axis=2,
            )

        u = np.sum(self._bvalues * self.parameters["bcoeff"], axis=(2, 1))
        u += np.einsum("ijkl,jkl->i", self._avalues, self.parameters["acoeff"])

        return (1, u)

    def updateinternals(self, e, epos, wrap=None, mask=None):
        """ Update a, b, and c sums. """
        if mask is None:
            mask = [True] * self._configscurrent.shape[0]
        self._bvalues[mask, :, :] += self._get_deltab(e, epos)[mask, :, :]
        self._avalues[mask, :, :, :] += self._get_deltaa(e, epos)[mask, :, :, :]
        self._configscurrent[mask, e, :] = epos.configs[mask, :]

    def value(self):
        """Compute the current log value of the wavefunction"""
        u = np.sum(self._bvalues * self.parameters["bcoeff"], axis=(2, 1))

        u += np.einsum("ijkl,jkl->i", self._avalues, self.parameters["acoeff"])
        return (1, u)

    def gradient(self, e, epos):
        """We compute the gradient for electron e as
        :math:`grad_e ln Psi_J = sum_k c_k sum_{j > e} grad_e b_k(r_{ej}) + sum_{i < e} grad_e b_k(r_{ie}) `
        So we need to compute the gradient of the b's for these indices.
        Note that we need to compute distances between electron position given and the current electron distances.
        We will need this for laplacian() as well"""
        nconf = epos.configs.shape[0]
        ne = self._configscurrent.shape[1]
        nup = self._mol.nelec[0]
        dnew = self._dist.dist_i(self._configscurrent, epos.configs)
        dinew = self._dist.dist_i(self._mol.atom_coords(), epos.configs)
        dinew = dinew.reshape(-1, 3)

        mask = [True] * ne
        mask[e] = False
        dnew = dnew[:, mask, :]

        delta = np.zeros((3, nconf))

        # Check if selected electron is spin up or down
        eup = int(e < nup)
        edown = int(e >= nup)

        dnewup = dnew[:, : nup - eup, :].reshape(-1, 3)  # Other electron is spin up
        dnewdown = dnew[:, nup - eup :, :].reshape(-1, 3)  # Other electron is spin down

        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            delta += (
                c[edown] * np.sum(b.gradient(dnewup).reshape(nconf, -1, 3), axis=1).T
            )
            delta += (
                c[1 + edown]
                * np.sum(b.gradient(dnewdown).reshape(nconf, -1, 3), axis=1).T
            )
            """
        for c,a in zip(self.parameters['acoeff'],self.a_basis):
            delta+=np.einsum('j,ijk->ki', c[:,edown], a.gradient(dinew).reshape(nconf,-1,3))

            """
        for i in range(self._mol.natm):
            for c, a in zip(self.parameters["acoeff"][i], self.a_basis):
                grad_all = a.gradient(dinew).reshape(nconf, -1, 3)
                grad_slice = grad_all[:, i, :]
                delta += c[edown] * grad_slice.T

        return delta

    def gradient_laplacian(self, e, epos):
        """ """
        nconf = epos.configs.shape[0]
        nup = self._mol.nelec[0]
        ne = self._configscurrent.shape[1]

        # Get and break up eedist_i
        dnew = self._dist.dist_i(self._configscurrent, epos.configs)
        mask = [True] * ne
        mask[e] = False
        dnew = dnew[:, mask, :]

        eup = int(e < nup)
        edown = int(e >= nup)
        dnewup = dnew[:, : nup - eup, :].reshape(-1, 3)  # Other electron is spin up
        dnewdown = dnew[:, nup - eup :, :].reshape(-1, 3)  # Other electron is spin down

        # Electron-ion distances
        dinew = self._dist.dist_i(self._mol.atom_coords(), epos.configs)
        dinew = dinew.reshape(-1, 3)

        delta = np.zeros(nconf)

        # b-value component
        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            delta += c[edown] * np.sum(b.laplacian(dnewup).reshape(nconf, -1), axis=1)
            delta += c[1 + edown] * np.sum(
                b.laplacian(dnewdown).reshape(nconf, -1), axis=1
            )

        for i in range(self._mol.natm):
            for c, a in zip(self.parameters["acoeff"][i], self.a_basis):
                lap_all = a.laplacian(dinew).reshape(nconf, -1, 3)
                lap_slice = lap_all[:, i, :]
                delta += np.sum(c[edown] * lap_slice, axis=1)

        g = self.gradient(e, epos)
        return g, delta + np.sum(g ** 2, axis=0)

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def _get_deltab(self, e, epos, mask_configs=None):
        """
        here we will evaluate the b's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf = epos.configs.shape[0]
        ne = self._configscurrent.shape[1]
        nup = self._mol.nelec[0]
        mask = [True] * ne
        mask[e] = False
        if mask_configs is None:
            mask_configs = [True] * nconf

        tmpconfigs = self._configscurrent[mask_configs, :, :]
        tmpconfigs = tmpconfigs[:, mask, :]

        dnew = self._dist.dist_i(tmpconfigs, epos.configs[mask_configs])
        dold = self._dist.dist_i(tmpconfigs, self._configscurrent[mask_configs, e, :])

        eup = int(e < nup)
        edown = int(e >= nup)
        # This is the point at which we switch between up and down
        # We subtract eup because we have removed e from the set
        sep = nup - eup
        dnewup = dnew[:, :sep, :].reshape((-1, 3))
        dnewdown = dnew[:, sep:, :].reshape((-1, 3))
        doldup = dold[:, :sep, :].reshape((-1, 3))
        dolddown = dold[:, sep:, :].reshape((-1, 3))

        rnewup = np.linalg.norm(dnewup, axis=1)
        rnewdown = np.linalg.norm(dnewdown, axis=1)
        roldup = np.linalg.norm(doldup, axis=1)
        rolddown = np.linalg.norm(dolddown, axis=1)

        nconf_mask = sum(mask_configs)

        delta = np.zeros((nconf_mask, len(self.b_basis), 3))
        for i, b in enumerate(self.b_basis):
            delta[:, i, edown] += np.sum(
                (b.value(dnewup, rnewup) - b.value(doldup, roldup)).reshape(
                    nconf_mask, -1
                ),
                axis=1,
            )
            delta[:, i, 1 + edown] += np.sum(
                (b.value(dnewdown, rnewdown) - b.value(dolddown, rolddown)).reshape(
                    nconf_mask, -1
                ),
                axis=1,
            )
        return delta

    def _get_deltaa(self, e, epos, mask=None):
        """
        here we will evaluate the a's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf = epos.configs.shape[0]
        ni = self._mol.natm
        nup = self._mol.nelec[0]
        if mask is None:
            mask = [True] * nconf

        dnew = self._dist.dist_i(self._mol.atom_coords(), epos.configs[mask]).reshape(
            (-1, 3)
        )
        dold = self._dist.dist_i(
            self._mol.atom_coords(), self._configscurrent[mask, e, :]
        ).reshape((-1, 3))

        nconf_mask = sum(mask)
        delta = np.zeros((nconf_mask, ni, len(self.a_basis), 2))

        rnew = np.linalg.norm(dnew, axis=1)
        rold = np.linalg.norm(dold, axis=1)

        for i, a in enumerate(self.a_basis):
            delta[:, :, i, int(e >= nup)] += (
                a.value(dnew, rnew) - a.value(dold, rold)
            ).reshape((nconf_mask, -1))

        return delta

    def testvalue(self, e, epos, mask=None):
        b_val = np.sum(
            self._get_deltab(e, epos, mask) * self.parameters["bcoeff"], axis=(2, 1)
        )
        a_val = np.einsum(
            "ijkl,jkl->i", self._get_deltaa(e, epos, mask), self.parameters["acoeff"]
        )
        return np.exp(b_val + a_val)

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative
        of all the b's and redo the sums, similar to recompute() """
        return {"bcoeff": self._bvalues, "acoeff": self._avalues}
