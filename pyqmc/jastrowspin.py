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
        d1, ij = configs.dist.dist_matrix(configs.configs[:, :nup])
        d2, ij = configs.dist.pairwise(
            configs.configs[:, :nup], configs.configs[:, nup:]
        )
        d3, ij = configs.dist.dist_matrix(configs.configs[:, nup:])

        r1 = np.linalg.norm(d1, axis=-1)
        r2 = np.linalg.norm(d2, axis=-1)
        r3 = np.linalg.norm(d3, axis=-1)

        # Update bvalues according to spin case
        for i, b in enumerate(self.b_basis):
            self._bvalues[:, i, 0] = np.sum(b.value(d1, r1), axis=1)
            self._bvalues[:, i, 1] = np.sum(b.value(d2, r2), axis=1)
            self._bvalues[:, i, 2] = np.sum(b.value(d3, r3), axis=1)

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
        self._bvalues[mask, :, :] += self._get_deltab(e, epos)[mask, :, :]
        self._avalues[mask, :, :, :] += self._get_deltaa(e, epos)[mask, :, :, :]
        self._configscurrent.move(e, epos, mask)
        self._set_partial_sums(e)

    def _set_partial_sums(self, e):
        self._a_partial[e] = self._get_a(e, self._configscurrent.electron(e))
        self._b_partial[e] = self._get_b(e, self._configscurrent.electron(e))

    def _get_a(self, e, epos, mask=None):
        """ 
        Set _a_partial and _b_partial
        """
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        d = epos.dist.dist_i( self._mol.atom_coords(), epos.configs[mask])
        r = np.linalg.norm(d, axis=-1)
        return np.stack([a.value(d, r) for a in self.a_basis], axis=-1)

    def _get_b(self, e, epos, mask=None):
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        ne = np.sum(self._mol.nelec)
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        not_e = np.arange(ne) != e 
        d = epos.dist.dist_i(
            self._configscurrent.configs[mask][:,not_e], epos.configs[mask]
        ) 
        r = np.linalg.norm(d, axis=-1)
        b_all_e = np.stack( [b.value(d, r) for b in self.b_basis], axis=-1)
        up, down = b_all_e[:,:sep].sum(axis=1), b_all_e[:,sep:].sum(axis=1)
        return np.stack([up, down], axis=-1)

    def _get_deltaa(self, e, epos):
        """
        here we will evaluate the a's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf = epos.configs.shape[0]
        ni = self._mol.natm
        nup = self._mol.nelec[0]
        delta = np.zeros((nconf, ni, len(self.a_basis), 2))
        deltaa = self._get_a(e, epos) - self._a_partial[e]
        delta[:, :, :, int(e >= nup)] += deltaa
        
        return delta

    def _get_deltab(self, e, epos):
        """
        here we will evaluate the b's for a given electron (both the old and new)
        and work out the updated value. This allows us to save a lot of memory
        """
        nconf, ne = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        eup = int(e < nup)
        edown = int(e >= nup)
        not_e = np.arange(ne) != e 
        sep = nup - eup

        delta = np.zeros((nconf, len(self.b_basis), 3))
        deltab = self._get_b(e, epos) - self._b_partial[e]
        delta[:, :, edown:edown+2] += deltab
            
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
            grad += np.einsum('j,ijk->ki', c, a.gradient(dinew))

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
            grad += np.einsum('j,ijk->ki', c, a.gradient(dinew))
            lap += np.einsum('j,ijk->i', c, a.laplacian(dinew))

        # b-value component
        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            bgrad = b.gradient(dnew)
            blap = b.laplacian(dnew)
            grad += c[edown] * np.sum(bgrad[:, :sep], axis=1).T
            grad += c[1 + edown] * np.sum(bgrad[:, sep:], axis=1).T
            lap += c[edown] * np.sum(blap[:, :sep], axis=(1,2))
            lap += c[1 + edown] * np.sum(blap[:, sep:], axis=(1,2))

        return grad, lap + np.sum(grad ** 2, axis=0)

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def testvalue(self, e, epos):
        b_val = np.sum(
            self._get_deltab(e, epos) * self.parameters["bcoeff"], axis=(2, 1)
        )
        a_val = np.einsum(
            "ijkl,jkl->i", self._get_deltaa(e, epos), self.parameters["acoeff"]
        )
        return np.exp(b_val + a_val)

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative
        of all the b's and redo the sums, similar to recompute() """
        return {"bcoeff": self._bvalues, "acoeff": self._avalues}


def test():
    from pyscf import lib, gto, scf

    np.random.seed(10)

    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr")
    l = dir(mol)
    nconf = 20
    configs = np.random.randn(nconf, np.sum(mol.nelec), 3)

    abasis = [GaussianFunction(0.2), GaussianFunction(0.4)]
    bbasis = [GaussianFunction(0.2), GaussianFunction(0.4)]
    jastrow = JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    jastrow.parameters["bcoeff"] = np.random.random(jastrow.parameters["bcoeff"].shape)
    jastrow.parameters["acoeff"] = np.random.random(jastrow.parameters["acoeff"].shape)
    import pyqmc.testwf as testwf

    for key, val in testwf.test_updateinternals(jastrow, configs).items():
        print(key, val)

    print()
    for delta in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        print(
            "delta",
            delta,
            "Testing gradient",
            testwf.test_wf_gradient(jastrow, configs, delta=delta),
        )
        print(
            "delta",
            delta,
            "Testing laplacian",
            testwf.test_wf_laplacian(jastrow, configs, delta=delta),
        )
        print(
            "delta",
            delta,
            "Testing pgradient",
            testwf.test_wf_pgradient(jastrow, configs, delta=delta),
        )
        print()


if __name__ == "__main__":
    test()
