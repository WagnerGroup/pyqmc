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
        r""" 
        Jastrow form is $e^{U(R)}, where 
        $$U(R) = 
        \sum_{I, \alpha, k} c^{a}_{Ik\uparrow } a_{k}(r_{I\alpha}) + 
        \sum_{I, \beta, k}  c^{a}_{Ik\downarrow } a_{k}(r_{I\beta}) +
        \sum_{\alpha_1 < \alpha_2, l} c^{b}_{l\uparrow\uparrow} b^{l}(r_{\alpha_1\alpha_2}) + 
        \sum_{\alpha, \beta, l} c^{b}_{l\uparrow\downarrow} b^{l}(r_{\beta_1\beta_2})
        \sum_{\beta_1 < \beta_2, l} c^{b}_{l\downarrow\downarrow} b^{l}(r_{\beta_1\beta_2}) + 
        $$
        the indices are $I$ for ions, $k$ for one-body (a) basis, $l$ for two-body (b) basis, $\alpha$ for up electrons, and $\beta$ for down electrons. $c^a, c^b$ are the coeffecient arrays. $r_{ij}$ denotes the distance between particles $i$ and $j$.
        _avalues is the array for current configurations $A_{Iks} = \sum_s a_{k}(r_{Is})$ where $s$ indexes over $\uparrow$ ($\alpha$) and $\downarrow$ ($\beta$) sums.
        _bvalues is the array for current configurations $B_{ls} = \sum_s b_{l}(r_{s})$ where $s$ indexes over $\uparrow\uparrow$ ($\alpha_1 < \alpha_2$), $\uparrow\downarrow$ ($\alpha, \beta$), and $\downarrow\downarrow$ ($\beta_1 < \beta_2$)  sums.
        the partial sums store values before summing over electrons
        _a_partial is the array $A^p_{eIk} = a_k(r_{Ie}$, where $e$ is any electron
        _b_partial is the array $B^p_{els} = \sum_s b_l(r_{es}$, where $e$ is any electron, $s$ indexes over $\uparrow$ ($\alpha$) and $\downarrow$ ($\beta$) sums, not including $e$.
        """
        u = 0.0
        self._configscurrent = configs.copy()
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
        d1, ij = configs.dist.dist_matrix(configs.configs[:, :nup, :])
        d2, ij = configs.dist.pairwise(configs.configs[:, :nup, :], configs.configs[:, nup:, :])
        d3, ij = configs.dist.dist_matrix(configs.configs[:, nup:, :])

        r1 = np.linalg.norm(d1, axis=-1)
        r2 = np.linalg.norm(d2, axis=-1)
        r3 = np.linalg.norm(d3, axis=-1)

        # Package the electron-ion distances into a 1d array
        di1 = np.zeros((nconfig, self._mol.natm, nup, 3))
        di2 = np.zeros((nconfig, self._mol.natm, nelec - nup, 3))

        for e in range(nup):
            di1[:, :, e, :] = configs.dist.dist_i(
                self._mol.atom_coords(), configs.configs[:, e, :]
            )
        for e in range(nup, nelec):
            di2[:, :, e - nup, :] = configs.dist.dist_i(
                self._mol.atom_coords(), configs.configs[:, e, :]
            )

        # print(di1.shape)
        ri1 = np.linalg.norm(di1, axis=-1)
        ri2 = np.linalg.norm(di2, axis=-1)

        # Update bvalues according to spin case
        for i, b in enumerate(self.b_basis):
            self._bvalues[:, i, 0] = np.sum( b.value(d1, r1), axis=1)
            self._bvalues[:, i, 1] = np.sum( b.value(d2, r2), axis=1)
            self._bvalues[:, i, 2] = np.sum( b.value(d3, r3), axis=1)

        # Update avalues according to spin case
        for i, a in enumerate(self.a_basis):
            self._avalues[:, :, i, 0] = np.sum( a.value(di1, ri1), axis=2,)
            self._avalues[:, :, i, 1] = np.sum( a.value(di2, ri2), axis=2,)

        u = np.sum(self._bvalues * self.parameters["bcoeff"], axis=(2, 1))
        u += np.einsum("ijkl,jkl->i", self._avalues, self.parameters["acoeff"])

        return (1, u)

    def updateinternals(self, e, epos, wrap=None, mask=None):
        r""" Update a and b sums. 
        _avalues is the array for current configurations $A_{Iks} = \sum_s a_{k}(r_{Is})$ where $s$ indexes over $\uparrow$ ($\alpha$) and $\downarrow$ ($\beta$) sums.
        _bvalues is the array for current configurations $B_{ls} = \sum_s b_{l}(r_{s})$ where $s$ indexes over $\uparrow\uparrow$ ($\alpha_1 < \alpha_2$), $\uparrow\downarrow$ ($\alpha, \beta$), and $\downarrow\downarrow$ ($\beta_1 < \beta_2$)  sums.
        The update for _avalues and _b_values from moving one electron only requires computing the new sum for that electron. The sums for the electron in the current configuration are stored in _a_partial and _b_partial.

"""
        if mask is None:
            mask = [True] * self._configscurrent.configs.shape[0]
        edown = int(e >= self._mol.nelec[0])
        aupdate = self._a_update(e, epos, mask)
        bupdate = self._b_update(e, epos, mask)
        self._avalues[mask, :, :, edown] += aupdate - self._a_partial[e, mask]
        self._bvalues[mask, :, edown:edown+2] += bupdate - self._b_partial[e, mask]
        self._a_partial[e, mask] = aupdate
        self._update_b_partial(e, epos, mask)
        self._configscurrent.move(e, epos, mask)

    def _a_update(self, e, epos, mask):
        r"""
          Calculate a (e-ion) partial sum for electron e
        _a_partial_e is the array $A^p_{iIk} = a_k(r^i_{Ie}$ with e fixed
        i is the configuration index
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. a_partial_e might have a smaller configs axis than epos, _configscurrent, and _a_partial because of the mask.
        """
        d = epos.dist.dist_i(self._mol.atom_coords(), epos.configs[mask])
        r = np.linalg.norm(d, axis=-1)
        a_partial_e = np.zeros((np.sum(mask), *self._a_partial.shape[2:]))
        for k, a in enumerate(self.a_basis):
            a_partial_e[..., k] = a.value(d, r)
        return a_partial_e

    def _b_update(self, e, epos, mask):
        r"""
          Calculate b (e-e) partial sums for electron e
        _b_partial_e is the array $B^p_{ils} = \sum_s b_l(r^i_{es}$, with e fixed; $s$ indexes over $\uparrow$ ($\alpha$) and $\downarrow$ ($\beta$) sums, not including electron e. 
          $i$ is the configuration index.
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. b_partial_e might have a smaller configs axis than epos, _configscurrent, and _b_partial because of the mask.
        """
        ne = np.sum(self._mol.nelec)
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        not_e = np.arange(ne) != e
        d = epos.dist.dist_i(
            self._configscurrent.configs[mask][:, not_e], epos.configs[mask]
        )
        r = np.linalg.norm(d, axis=-1)
        b_partial_e = np.zeros((np.sum(mask), *self._b_partial.shape[2:]))
        for l, b in enumerate(self.b_basis):
            bval = b.value(d, r)
            b_partial_e[:, l, 0] = bval[:, :sep].sum(axis=1)
            b_partial_e[:, l, 1] = bval[:, sep:].sum(axis=1)
        return b_partial_e

    def _update_b_partial(self, e, epos, mask):
        r"""
          Calculate b (e-e) partial sum contributions from electron e
        _b_partial_e is the array $B^p_{ils} = \sum_s b_l(r^i_{es}$, with e fixed; $s$ indexes over $\uparrow$ ($\alpha$) and $\downarrow$ ($\beta$) sums, not including electron e. 
          Since $B^p_{ils}$ is summed over other electrons, moving electron e will affect other partial sums. This function updates all the necessary partial sums instead of just evaluating the one for electron e.
          $i$ is the configuration index.
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. b_partial_e might have a smaller configs axis than epos, _configscurrent, and _b_partial because of the mask.
        """
        ne = np.sum(self._mol.nelec)
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        not_e = np.arange(ne) != e
        edown = int(e >= nup)
        d = epos.dist.dist_i(
            self._configscurrent.configs[mask][:, not_e], epos.configs[mask]
        )
        r = np.linalg.norm(d, axis=-1)
        dold = epos.dist.dist_i(
            self._configscurrent.configs[mask][:, not_e], 
            self._configscurrent.configs[mask, e], 
        )
        rold = np.linalg.norm(dold, axis=-1)
        b_partial_e = np.zeros((np.sum(mask), *self._b_partial.shape[2:]))
        eind, mind = np.ix_(not_e, mask)
        for l, b in enumerate(self.b_basis):
            bval = b.value(d, r)
            bdiff = bval - b.value(dold, rold)
            self._b_partial[eind, mind, l, edown] += bdiff.transpose((1,0))
            self._b_partial[e, mask, l, 0] = bval[:, :sep].sum(axis=1)
            self._b_partial[e, mask, l, 1] = bval[:, sep:].sum(axis=1)

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
        nconf, ne = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]
        dnew = epos.dist.dist_i(self._configscurrent.configs, epos.configs)
        dinew = epos.dist.dist_i(self._mol.atom_coords(), epos.configs)

        mask = [True] * ne
        mask[e] = False
        dnew = dnew[:, mask, :]

        delta = np.zeros((3, nconf))

        # Check if selected electron is spin up or down
        eup = int(e < nup)
        edown = int(e >= nup)

        dnewup = dnew[:, : nup - eup, :]  # Other electron is spin up
        dnewdown = dnew[:, nup - eup :, :]  # Other electron is spin down

        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            delta += (
                c[edown] * np.sum(b.gradient(dnewup), axis=1).T
            )
            delta += (
                c[1 + edown]
                * np.sum(b.gradient(dnewdown), axis=1).T
            )
            """
        for c,a in zip(self.parameters['acoeff'],self.a_basis):
            delta+=np.einsum('j,ijk->ki', c[:,edown], a.gradient(dinew).reshape(nconf,-1,3))

            """
        for i in range(self._mol.natm):
            for c, a in zip(self.parameters["acoeff"][i], self.a_basis):
                grad_all = a.gradient(dinew)
                grad_slice = grad_all[:, i, :]
                delta += c[edown] * grad_slice.T

        return delta

    def gradient_laplacian(self, e, epos):
        """ """
        nconf = epos.configs.shape[0]
        nup = self._mol.nelec[0]
        nconf, ne = self._configscurrent.configs.shape[:2]

        # Get and break up eedist_i
        dnew = epos.dist.dist_i(self._configscurrent.configs, epos.configs)
        mask = [True] * ne
        mask[e] = False
        dnew = dnew[:, mask, :]

        eup = int(e < nup)
        edown = int(e >= nup)
        dnewup = dnew[:, : nup - eup, :]  # Other electron is spin up
        dnewdown = dnew[:, nup - eup :, :]  # Other electron is spin down

        # Electron-ion distances
        dinew = epos.dist.dist_i(self._mol.atom_coords(), epos.configs)

        delta = np.zeros(nconf)

        # b-value component
        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            delta += c[edown] * np.sum(b.laplacian(dnewup).reshape(nconf, -1), axis=1)
            delta += c[1 + edown] * np.sum(
                b.laplacian(dnewdown).reshape(nconf, -1), axis=1
            )

        for i in range(self._mol.natm):
            for c, a in zip(self.parameters["acoeff"][i], self.a_basis):
                lap_all = a.laplacian(dinew)
                lap_slice = lap_all[:, i, :]
                delta += np.sum(c[edown] * lap_slice, axis=1)

        g = self.gradient(e, epos)
        return g, delta + np.sum(g ** 2, axis=0)

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def testvalue(self, e, epos, mask=None):
        r"""
        Compute the ratio $\Psi_{\rm new}/\Psi_{\rm old}$ for moving electron e to epos.
        _avalues is the array for current configurations $A_{Iks} = \sum_s a_{k}(r_{Is})$ where $s$ indexes over $\uparrow$ ($\alpha$) and $\downarrow$ ($\beta$) sums.
        _bvalues is the array for current configurations $B_{ls} = \sum_s b_{l}(r_{s})$ where $s$ indexes over $\uparrow\uparrow$ ($\alpha_1 < \alpha_2$), $\uparrow\downarrow$ ($\alpha, \beta$), and $\downarrow\downarrow$ ($\beta_1 < \beta_2$)  sums.
        The update for _avalues and _b_values from moving one electron only requires computing the new sum for that electron. The sums for the electron in the current configuration are stored in _a_partial and _b_partial.
        deltaa = $a_{k}(r_{Ie})$, indexing (atom, a_basis)
        deltab = $\sum_s b_{l}(r_{se})$, indexing (b_basis, spin s)
        """
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        edown = int(e >= self._mol.nelec[0])
        deltaa = self._a_update(e, epos, mask) - self._a_partial[e, mask]
        a_val = np.einsum("ijk,jk->i", deltaa, self.parameters["acoeff"][..., edown])
        deltab = self._b_update(e, epos, mask) - self._b_partial[e, mask]
        b_val = np.einsum("ijk,jk->i",
            deltab, self.parameters["bcoeff"][:, edown:edown+2],
        )
        return np.exp(b_val + a_val)

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative
        of all the b's and redo the sums, similar to recompute() """
        return {"bcoeff": self._bvalues, "acoeff": self._avalues}
