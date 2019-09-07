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
        nconfig, nelec = configs.configs.shape[:2]
        nexpand = len(self.b_basis)
        aexpand = len(self.a_basis)
        self._bvalues = np.zeros((nconfig, nexpand, 3))
        self._avalues = np.zeros((nconfig, self._mol.natm, aexpand, 2))
        self._a_partial = np.zeros((nelec, nconfig, self._mol.natm, aexpand))
        self._b_partial = np.zeros((nelec, nconfig, nexpand, 2)) 
        notmask = [True] * nconfig
        for e in range(nelec):
            epos = configs.electron(e)
            self._a_partial[e] = self._a_update(e, epos, notmask)
            self._b_partial[e] = self._b_update(e, epos, notmask)

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
        edown = int(e >= self._mol.nelec[0])
        aupdate = self._a_update(e, epos, mask)
        bupdate = self._b_update(e, epos, mask)
        self._avalues[mask, :, :, edown] += aupdate - self._a_partial[e, mask]
        self._bvalues[mask, :, edown:edown+2] += bupdate - self._b_partial[e, mask]
        self._a_partial[e, mask] = aupdate
        self._b_partial[e, mask] = bupdate
        self._configscurrent[mask, e, :] = epos.configs[mask, :]

    def _a_update(self, e, epos, mask):
        """
          Calculate a (e-ion) partial sums
        """
        d = epos.dist.dist_i(self._mol.atom_coords(), epos.configs[mask])
        r = np.linalg.norm(d, axis=-1)
        a_partial = np.zeros((np.sum(mask), *self._a_partial.shape[2:]))
        for i, a in enumerate(self.a_basis):
            a_partial[..., i] = a.value(d, r)
        return a_partial

    def _b_update(self, e, epos, mask):
        """
          Calculate b (e-e) partial sums
        """
        ne = np.sum(self._mol.nelec)
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        not_e = np.arange(ne) != e
        d = epos.dist.dist_i(
            self._configscurrent[mask][:, not_e], epos.configs[mask]
        )
        r = np.linalg.norm(d, axis=-1)
        b_partial = np.zeros((np.sum(mask), *self._b_partial.shape[2:]))
        for i, b in enumerate(self.b_basis):
            bval = b.value(d, r)
            b_partial[..., i, 0] = bval[:, : sep].sum(axis=1)
            b_partial[..., i, 1] = bval[:, sep :].sum(axis=1)
        return b_partial

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

    def testvalue(self, e, epos, mask=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        edown = int(e >= self._mol.nelec[0])
        deltaa = self._a_update(e, epos, mask) - self._a_partial[e, mask]
        a_val = np.einsum("ijk,jk->i", deltaa, self.parameters["acoeff"][..., edown])
        deltab = self._b_update(e, epos, mask) - self._b_partial[e, mask]
        b_val = np.einsum("ijk,jk->i",
            deltab, self.parameters["bcoeff"][..., edown:edown+2],
        )
        return np.exp(b_val + a_val)

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative
        of all the b's and redo the sums, similar to recompute() """
        return {"bcoeff": self._bvalues, "acoeff": self._avalues}
