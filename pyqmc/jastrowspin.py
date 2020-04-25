import numpy as np
from pyqmc.func3d import GaussianFunction
from pyqmc.distance import RawDistance


class JastrowSpin:
    r"""
    1 body and 2 body jastrow factor

            The Jastrow form is :math:`e^{U(R)}`, where 

        .. math::  U(R) = \sum_{I, i, k} c^{a}_{Ik\sigma(i)} a_{k}(r_{Ii}) + \sum_{i,j,k} c^{b}_{k\sigma(i)\sigma(j)} b^{l}(r_{ij})
        

        """

    def __init__(self, mol, a_basis=None, b_basis=None):
        r"""
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
        self.iscomplex = False

    def recompute(self, configs):
        r"""

        _avalues is the array for current configurations :math:`A_{Iks} = \sum_s a_{k}(r_{Is})` where :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums.
        _bvalues is the array for current configurations :math:`B_{ls} = \sum_s b_{l}(r_{s})` where :math:`s` indexes over :math:`\uparrow\uparrow` (:math:`\alpha_1 < \alpha_2`), :math:`\uparrow\downarrow` (:math:`\alpha, \beta`), and :math:`\downarrow\downarrow` (:math:`\beta_1 < \beta_2`)  sums.
 
         the partial sums store values before summing over electrons
        _a_partial is the array :math:`A^p_{eIk} = a_k(r_{Ie}`, where :math:`e` is any electron
        _b_partial is the array :math:`B^p_{els} = \sum_s b_l(r_{es}`, where :math:`e` is any electron, :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums, not including :math:`e`.
        """
        self._configscurrent = configs.copy()
        nconf, nelec = configs.configs.shape[:2]
        nexpand = len(self.b_basis)
        aexpand = len(self.a_basis)
        self._bvalues = np.zeros((nconf, nexpand, 3))
        self._avalues = np.zeros((nconf, self._mol.natm, aexpand, 2))
        self._a_partial = np.zeros((nelec, nconf, self._mol.natm, aexpand))
        self._b_partial = np.zeros((nelec, nconf, nexpand, 2))
        notmask = [True] * nconf
        for e in range(nelec):
            epos = configs.electron(e)
            self._a_partial[e] = self._a_update(e, epos, notmask)
            self._b_partial[e] = self._b_update(e, epos, notmask)

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

        # electron-ion distances
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
        return (np.ones(len(u)), u)

    def updateinternals(self, e, epos, wrap=None, mask=None):
        r""" Update a and b sums. 
        _avalues is the array for current configurations :math:`A_{Iks} = \sum_s a_{k}(r_{Is})` where :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums.
        _bvalues is the array for current configurations :math:`B_{ls} = \sum_s b_{l}(r_{s})` where :math:`s` indexes over :math:`\uparrow\uparrow` (:math:`\alpha_1 < \alpha_2`), :math:`\uparrow\downarrow` (:math:`\alpha, \beta`), and :math:`\downarrow\downarrow` (:math:`\beta_1 < \beta_2`)  sums.
        The update for _avalues and _b_values from moving one electron only requires computing the new sum for that electron. The sums for the electron in the current configuration are stored in _a_partial and _b_partial """
        if mask is None:
            mask = [True] * self._configscurrent.configs.shape[0]
        edown = int(e >= self._mol.nelec[0])
        aupdate = self._a_update(e, epos, mask)
        bupdate = self._b_update(e, epos, mask)
        self._avalues[mask, :, :, edown] += aupdate - self._a_partial[e, mask]
        self._bvalues[mask, :, edown : edown + 2] += bupdate - self._b_partial[e, mask]
        self._a_partial[e, mask] = aupdate
        self._update_b_partial(e, epos, mask)
        self._configscurrent.move(e, epos, mask)

    def _a_update(self, e, epos, mask):
        r"""
          Calculate a (e-ion) partial sum for electron e
        _a_partial_e is the array :math:`A^p_{iIk} = a_k(r^i_{Ie}` with e fixed
        i is the configuration index
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. a_partial_e might have a smaller configs axis than epos, _configscurrent, and _a_partial because of the mask.
        """
        d = epos.dist.dist_i(self._mol.atom_coords(), epos.configs[mask])
        r = np.linalg.norm(d, axis=-1)
        a_partial_e = np.zeros((*r.shape, self._a_partial.shape[3]))
        for k, a in enumerate(self.a_basis):
            a_partial_e[..., k] = a.value(d, r)
        return a_partial_e

    def _b_update(self, e, epos, mask):
        r"""
          Calculate b (e-e) partial sums for electron e
        _b_partial_e is the array :math:`B^p_{ils} = \sum_s b_l(r^i_{es}`, with e fixed; :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums, not including electron e. 
          :math:`i` is the configuration index.
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. b_partial_e might have a smaller configs axis than epos, _configscurrent, and _b_partial because of the mask.
        """
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        not_e = np.arange(self._nelec) != e
        d = epos.dist.dist_i(
            self._configscurrent.configs[mask][:, not_e], epos.configs[mask]
        )
        r = np.linalg.norm(d, axis=-1)
        b_partial_e = np.zeros((*r.shape[:-1], *self._b_partial.shape[2:]))

        for l, b in enumerate(self.b_basis):
            bval = b.value(d, r)
            b_partial_e[..., l, 0] = bval[..., :sep].sum(axis=-1)
            b_partial_e[..., l, 1] = bval[..., sep:].sum(axis=-1)

        return b_partial_e

    def _b_update_many(self, e, epos, mask, spin):
        r"""
          Calculate b (e-e) partial sums for electron e
        _b_partial_e is the array :math:`B^p_{ils} = \sum_s b_l(r^i_{es}`, with e fixed; :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums, not including electron e. 
          :math:`i` is the configuration index.
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. b_partial_e might have a smaller configs axis than epos, _configscurrent, and _b_partial because of the mask.
        """
        nup = self._mol.nelec[0]
        d = epos.dist.dist_i(self._configscurrent.configs[mask], epos.configs[mask])
        r = np.linalg.norm(d, axis=-1)
        b_partial_e = np.zeros((e.shape[0], *r.shape[:-1], *self._b_partial.shape[2:]))

        for l, b in enumerate(self.b_basis):
            bval = b.value(d, r)
            b_partial_e[..., l, 0] = bval[..., :nup].sum(axis=-1)
            b_partial_e[..., l, 1] = bval[..., nup:].sum(axis=-1)
            b_partial_e[..., l, spin] -= bval[..., e].T

        return b_partial_e

    def _update_b_partial(self, e, epos, mask):
        r"""
          Calculate b (e-e) partial sum contributions from electron e
        _b_partial_e is the array :math:`B^p_{ils} = \sum_s b_l(r^i_{es}`, with e fixed; :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums, not including electron e. 
          Since :math:`B^p_{ils}` is summed over other electrons, moving electron e will affect other partial sums. This function updates all the necessary partial sums instead of just evaluating the one for electron e.
          :math:`i` is the configuration index.
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. b_partial_e might have a smaller configs axis than epos, _configscurrent, and _b_partial because of the mask.
        """
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        not_e = np.arange(self._nelec) != e
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
            self._b_partial[eind, mind, l, edown] += bdiff.transpose((1, 0))
            self._b_partial[e, mask, l, 0] = bval[:, :sep].sum(axis=1)
            self._b_partial[e, mask, l, 1] = bval[:, sep:].sum(axis=1)

    def value(self):
        """Compute the current log value of the wavefunction"""
        u = np.sum(self._bvalues * self.parameters["bcoeff"], axis=(2, 1))

        u += np.einsum("ijkl,jkl->i", self._avalues, self.parameters["acoeff"])
        return (np.ones(len(u)), u)

    def gradient(self, e, epos):
        r"""We compute the gradient for electron e as
        :math:`\nabla_e \ln \Psi_J = \sum_k c_k \left(\sum_{j > e} \nabla_e b_k(r_{ej}) + \sum_{i < e} \nabla_e b_k(r_{ie})\right)` 
        So we need to compute the gradient of the b's for these indices.
        Note that we need to compute distances between electron position given and the current electron distances.
        We will need this for laplacian() as well"""
        nconf, nelec = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        # Get e-e and e-ion distances
        not_e = np.arange(nelec) != e
        dnew = epos.dist.dist_i(self._configscurrent.configs, epos.configs)[:, not_e]
        dinew = epos.dist.dist_i(self._mol.atom_coords(), epos.configs)
        rnew = np.linalg.norm(dnew, axis=-1)
        rinew = np.linalg.norm(dinew, axis=-1)

        grad = np.zeros((3, nconf))

        # Check if selected electron is spin up or down
        eup = int(e < nup)
        edown = int(e >= nup)

        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            bgrad = b.gradient(dnew, rnew)
            grad += c[edown] * np.sum(bgrad[:, : nup - eup], axis=1).T
            grad += c[1 + edown] * np.sum(bgrad[:, nup - eup :], axis=1).T

        for c, a in zip(self.parameters["acoeff"].transpose()[edown], self.a_basis):
            grad += np.einsum("j,ijk->ki", c, a.gradient(dinew, rinew))

        return grad

    def gradient_laplacian(self, e, epos):
        """ """
        nconf, nelec = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        # Get e-e and e-ion distances
        not_e = np.arange(nelec) != e
        dnew = epos.dist.dist_i(self._configscurrent.configs, epos.configs)[:, not_e]
        dinew = epos.dist.dist_i(self._mol.atom_coords(), epos.configs)
        rnew = np.linalg.norm(dnew, axis=-1)
        rinew = np.linalg.norm(dinew, axis=-1)

        eup = int(e < nup)
        edown = int(e >= nup)

        grad = np.zeros((3, nconf))
        lap = np.zeros(nconf)
        # a-value component
        for c, a in zip(self.parameters["acoeff"].transpose()[edown], self.a_basis):
            g, l = a.gradient_laplacian(dinew, rinew)
            grad += np.einsum("j,ijk->ki", c, g)
            lap += np.einsum("j,ijk->i", c, l)

        # b-value component
        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            bgrad, blap = b.gradient_laplacian(dnew, rnew)

            grad += c[edown] * np.sum(bgrad[:, : nup - eup], axis=1).T
            grad += c[1 + edown] * np.sum(bgrad[:, nup - eup :], axis=1).T
            lap += c[edown] * np.sum(blap[:, : nup - eup], axis=(1, 2))
            lap += c[1 + edown] * np.sum(blap[:, nup - eup :], axis=(1, 2))
        return grad, lap + np.sum(grad ** 2, axis=0)

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def testvalue(self, e, epos, mask=None):
        r"""
        Compute the ratio :math:`\Psi_{\rm new}/\Psi_{\rm old}` for moving electron e to epos.
        _avalues is the array for current configurations :math:`A_{Iks} = \sum_s a_{k}(r_{Is})` where :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums.
        _bvalues is the array for current configurations :math:`B_{ls} = \sum_s b_{l}(r_{s})` where :math:`s` indexes over :math:`\uparrow\uparrow` (:math:`\alpha_1 < \alpha_2`), :math:`\uparrow\downarrow` (:math:`\alpha, \beta`), and :math:`\downarrow\downarrow` (:math:`\beta_1 < \beta_2`)  sums.
        The update for _avalues and _b_values from moving one electron only requires computing the new sum for that electron. The sums for the electron in the current configuration are stored in _a_partial and _b_partial.
        deltaa = :math:`a_{k}(r_{Ie})`, indexing (atom, a_basis)
        deltab = :math:`\sum_s b_{l}(r_{se})`, indexing (b_basis, spin s)
        """
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        edown = int(e >= self._mol.nelec[0])
        deltaa = self._a_update(e, epos, mask) - self._a_partial[e, mask]
        a_val = np.einsum(
            "...jk,jk->...", deltaa, self.parameters["acoeff"][..., edown]
        )
        deltab = self._b_update(e, epos, mask) - self._b_partial[e, mask]
        b_val = np.einsum(
            "...jk,jk->...", deltab, self.parameters["bcoeff"][:, edown : edown + 2]
        )
        val = np.exp(b_val + a_val)
        if len(val.shape) == 2:
            val = val.T
        return val

    def testvalue_many(self, e, epos, mask=None):
        r"""
        Compute the ratio :math:`\Psi_{\rm new}/\Psi_{\rm old}` for moving electron e to epos.
        _avalues is the array for current configurations :math:`A_{Iks} = \sum_s a_{k}(r_{Is})` where :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums.
        _bvalues is the array for current configurations :math:`B_{ls} = \sum_s b_{l}(r_{s})` where :math:`s` indexes over :math:`\uparrow\uparrow` (:math:`\alpha_1 < \alpha_2`), :math:`\uparrow\downarrow` (:math:`\alpha, \beta`), and :math:`\downarrow\downarrow` (:math:`\beta_1 < \beta_2`)  sums.
        The update for _avalues and _b_values from moving one electron only requires computing the new sum for that electron. The sums for the electron in the current configuration are stored in _a_partial and _b_partial.
        deltaa = :math:`a_{k}(r_{Ie})`, indexing (atom, a_basis)
        deltab = :math:`\sum_s b_{l}(r_{se})`, indexing (b_basis, spin s)
        """
        s = (e >= self._mol.nelec[0]).astype(int)
        if mask is None:
            mask = [True] * epos.configs.shape[0]

        ratios = np.zeros((epos.configs.shape[0], e.shape[0]))
        for spin in [0, 1]:
            ind = s == spin
            deltaa = (
                self._a_update(e[ind], epos, mask) - self._a_partial[e[ind]][:, mask]
            )
            deltab = (
                self._b_update_many(e[ind], epos, mask, spin)
                - self._b_partial[e[ind]][:, mask]
            )
            a_val = np.einsum(
                "...jk,jk->...", deltaa, self.parameters["acoeff"][..., spin]
            )
            b_val = np.einsum(
                "...jk,jk->...", deltab, self.parameters["bcoeff"][:, spin : spin + 2]
            )
            val = np.exp(b_val + a_val)
            if len(val.shape) == 2:
                val = val.T
            ratios[:, ind] = val
        return ratios

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative
        of all the b's and redo the sums, similar to recompute() """
        return {"bcoeff": self._bvalues, "acoeff": self._avalues}

    def u_components(self, rvec, r):
        """Given positions rvec and their magnitudes r, returns 
        dictionaries of the one-body and two-body Jastrow components.
        Dictionaries are the spin components of U summed across the basis;
        one-body also returns U for different atoms. """
        u_onebody = {"up": [], "dn": []}
        a_value = list(map(lambda x: x.value(rvec, r), self.a_basis))
        u_onebody["up"] = np.einsum(
            "ij,jl->il", self.parameters["acoeff"][:, :, 0], a_value
        )
        u_onebody["dn"] = np.einsum(
            "ij,jl->il", self.parameters["acoeff"][:, :, 1], a_value
        )

        u_twobody = {"upup": [], "updn": [], "dndn": []}
        b_value = list(map(lambda x: x.value(rvec, r), self.b_basis[1:]))
        u_twobody["upup"] = np.dot(self.parameters["bcoeff"][1:, 0], b_value)
        u_twobody["updn"] = np.dot(self.parameters["bcoeff"][1:, 1], b_value)
        u_twobody["dndn"] = np.dot(self.parameters["bcoeff"][1:, 2], b_value)

        return u_onebody, u_twobody
