import numpy as np
from pyqmc.func3d import GaussianFunction
from pyqmc.distance import RawDistance


class JastrowSpin:
    """
    1 body and 2 body jastrow factor
    """

    def __init__(self, mol, a_basis=None, b_basis=None):
        """
        Args:

        mol : a pyscf molecule object

        a_basis : list of func3d objects that comprise the electron-ion basis

        b_basis : list of func3d objects that comprise the electron-electron basis

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
        self.parameters["bcoeff"] = np.zeros((nexpand, 3))
        self.parameters["acoeff"] = np.zeros((self._mol.natm, aexpand, 2))

    def recompute(self, configs):
        """ """
        u = 0.0
        self._configscurrent = configs.copy()
        nconf, nelec = configs.configs.shape[0:2]
        nexpand = len(self.b_basis)
        aexpand = len(self.a_basis)
        self._bvalues = np.zeros((nconf, nexpand, 3))
        self._avalues = np.zeros((nconf, self._mol.natm, aexpand, 2))
        self._a_partial = np.zeros((nelec, nconf, self._mol.natm, aexpand))
        self._b_partial = np.zeros((nelec, nconf, nexpand, 2))
        for e in range(nelec):
            self._set_partial_sums(e)

        # electron-electron distances
        nup = self._mol.nelec[0]
        d_upup, ij = configs.dist.dist_matrix(configs.configs[:, :nup])
        d_updown, ij = configs.dist.pairwise(
            configs.configs[:, :nup], configs.configs[:, nup:]
        )
        d_downdown, ij = configs.dist.dist_matrix(configs.configs[:, nup:])

        # Update bvalues according to spin case
        for j, d in enumerate([d_upup, d_updown, d_downdown]):
            r = np.linalg.norm(d, axis=-1)
            for i, b in enumerate(self.b_basis):
                self._bvalues[:, i, j] = np.sum(b.value(d, r), axis=1)

        # Package the electron-ion distances into a 1d array
        di = np.zeros((nelec, nconf, self._mol.natm, 3))
        for e in range(nelec):
            di[e] = configs.dist.dist_i(
                self._mol.atom_coords(), configs.configs[:, e, :]
            )
        ri = np.linalg.norm(di, axis=-1)

        # Update avalues according to spin case
        for i, a in enumerate(self.a_basis):
            avals = a.value(di, ri)
            self._avalues[:, :, i, 0] = np.sum(avals[:nup], axis=0)
            self._avalues[:, :, i, 1] = np.sum(avals[nup:], axis=0)

        u = np.sum(self._bvalues * self.parameters["bcoeff"], axis=(2, 1))
        u += np.einsum("ijkl,jkl->i", self._avalues, self.parameters["acoeff"])

        return (1, u)

    def updateinternals(self, e, epos, mask=None):
        """ Update a, b, and c sums. """
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        self._bvalues[mask, :, :] += self._get_deltab(e, epos, mask)
        self._avalues[mask, :, :, :] += self._get_deltaa(e, epos, mask)
        self._configscurrent.move(e, epos, mask)
        self._set_partial_sums(e, mask)

    def _set_partial_sums(self, e, mask=None):
        if mask is None:
            mask = [True] * self._configscurrent.configs.shape[0]
        epos = self._configscurrent.electron(e)
        self._a_partial[e, mask] = self._get_a_partial(e, epos, mask)
        self._b_partial[e, mask] = self._get_b_partial(e, epos, mask)

    def _get_a_partial(self, e, epos, mask):
        """ 
        Set _a_partial and _b_partial
        """
        d = epos.dist.dist_i(self._mol.atom_coords(), epos.configs[mask])
        r = np.linalg.norm(d, axis=-1)
        a_partial = np.zeros((np.sum(mask), *self._a_partial.shape[2:]))
        for i, a in enumerate(self.a_basis):
            a_partial[..., i] = a.value(d, r)
        return a_partial

    def _get_b_partial(self, e, epos, mask):
        ne = np.sum(self._mol.nelec)
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        not_e = np.arange(ne) != e
        d = epos.dist.dist_i(
            self._configscurrent.configs[mask][:, not_e], epos.configs[mask]
        )
        r = np.linalg.norm(d, axis=-1)
        b_partial = np.zeros((np.sum(mask), *self._b_partial.shape[2:]))
        for i, b in enumerate(self.b_basis):
            bval = b.value(d, r)
            b_partial[..., i, 0] = bval[:, : sep].sum(axis=1)
            b_partial[..., i, 1] = bval[:, sep :].sum(axis=1)
        return b_partial

    def _get_deltaa(self, e, epos, mask):
        """
        here we will evaluate the a's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf = epos.configs.shape[0]
        ni = self._mol.natm
        nup = self._mol.nelec[0]
        delta = np.zeros((np.sum(mask), ni, len(self.a_basis), 2))
        deltaa = self._get_a_partial(e, epos, mask) - self._a_partial[e, mask]
        delta[:, :, :, int(e >= nup)] += deltaa

        return delta

    def _get_deltab(self, e, epos, mask):
        """
        here we will evaluate the b's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        ne = self._configscurrent.configs.shape[1]
        nup = self._mol.nelec[0]
        edown = int(e >= nup)

        delta = np.zeros((np.sum(mask), len(self.b_basis), 3))
        deltab = self._get_b_partial(e, epos, mask) - self._b_partial[e, mask]
        delta[:, :, edown : edown + 2] += deltab

        return delta

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
        nconf, ne = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        # Get e-e and e-ion distances
        not_e = np.arange(ne) != e
        dnew = epos.dist.dist_i(self._configscurrent.configs, epos.configs)[:, not_e]
        dinew = epos.dist.dist_i(self._mol.atom_coords(), epos.configs)

        grad = np.zeros((3, nconf))

        # Check if selected electron is spin up or down
        eup = int(e < nup)
        edown = int(e >= nup)
        sep = nup - eup

        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            bgrad = b.gradient(dnew)
            grad += c[edown] * np.sum(bgrad[:, :sep], axis=1).T
            grad += c[1 + edown] * np.sum(bgrad[:, sep:], axis=1).T

        for c, a in zip(self.parameters["acoeff"].transpose()[edown], self.a_basis):
            grad += np.einsum("j,ijk->ki", c, a.gradient(dinew))

        return grad

    def gradient_laplacian(self, e, epos):
        """ """
        nconf, ne = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        # Get e-e and e-ion distances
        not_e = np.arange(ne) != e
        dnew = epos.dist.dist_i(self._configscurrent.configs, epos.configs)[:, not_e]
        dinew = epos.dist.dist_i(self._mol.atom_coords(), epos.configs)

        eup = int(e < nup)
        edown = int(e >= nup)
        sep = nup - eup

        grad = np.zeros((3, nconf))
        lap = np.zeros(nconf)
        # a-value component
        for c, a in zip(self.parameters["acoeff"].transpose()[edown], self.a_basis):
            grad += np.einsum("j,ijk->ki", c, a.gradient(dinew))
            lap += np.einsum("j,ijk->i", c, a.laplacian(dinew))

        # b-value component
        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            bgrad = b.gradient(dnew)
            blap = b.laplacian(dnew)
            grad += c[edown] * np.sum(bgrad[:, :sep], axis=1).T
            grad += c[1 + edown] * np.sum(bgrad[:, sep:], axis=1).T
            lap += c[edown] * np.sum(blap[:, :sep], axis=(1, 2))
            lap += c[1 + edown] * np.sum(blap[:, sep:], axis=(1, 2))

        return grad, lap + np.sum(grad ** 2, axis=0)

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def testvalue(self, e, epos, mask=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]
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
