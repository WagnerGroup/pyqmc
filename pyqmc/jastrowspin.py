import numpy as np
import pyqmc.gpu as gpu


class JastrowSpin:
    r"""
    1 body and 2 body jastrow factor

            The Jastrow form is :math:`e^{U(R)}`, where

        .. math::  U(R) = \sum_{I, i, k} c^{a}_{Ik\sigma(i)} a_{k}(r_{Ii}) + \sum_{i,j,l} c^{b}_{l\sigma(i)\sigma(j)} b_{l}(r_{ij})


    """

    def __init__(self, mol, a_basis, b_basis):
        r"""
        Args:

        mol : a pyscf molecule object

        a_basis : list of func3d objects that comprise the electron-ion basis

        b_basis : list of func3d objects that comprise the electron-electron basis

        """
        self.a_basis = a_basis
        self.b_basis = b_basis
        self.parameters = {}
        self._nelec = np.sum(mol.nelec)
        self._mol = mol
        self.parameters["bcoeff"] = gpu.cp.zeros((len(b_basis), 3))
        self.parameters["acoeff"] = gpu.cp.zeros((self._mol.natm, len(a_basis), 2))
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
        self._bvalues = gpu.cp.zeros((nconf, nexpand, 3))
        self._avalues = gpu.cp.zeros((nconf, self._mol.natm, aexpand, 2))
        self._a_partial = gpu.cp.zeros((nelec, nconf, self._mol.natm, aexpand))
        self._b_partial = gpu.cp.zeros((nelec, nconf, nexpand, 2))
        notmask = np.ones(nconf, dtype=bool)
        for e in range(nelec):
            epos = configs.electron(e)
            self._a_partial[e] = self._a_update(e, epos, notmask)
            self._b_partial[e], _ = self._b_update(e, epos, notmask)

        # electron-electron distances
        nup = self._mol.nelec[0]
        d_upup, ij = configs.dist.dist_matrix(configs.configs[:, :nup])
        d_updown, ij = configs.dist.pairwise(
            configs.configs[:, :nup], configs.configs[:, nup:]
        )
        d_downdown, ij = configs.dist.dist_matrix(configs.configs[:, nup:])

        # Update bvalues according to spin case
        for j, d in enumerate([d_upup, d_updown, d_downdown]):
            d = gpu.cp.asarray(d)
            r = gpu.cp.linalg.norm(d, axis=-1)
            for i, b in enumerate(self.b_basis):
                self._bvalues[:, i, j] = gpu.cp.sum(b.value(d, r), axis=1)

        # electron-ion distances
        di = gpu.cp.zeros((nelec, nconf, self._mol.natm, 3))
        for e in range(nelec):
            di[e] = gpu.cp.asarray(
                configs.dist.dist_i(self._mol.atom_coords(), configs.configs[:, e, :])
            )
        ri = gpu.cp.linalg.norm(di, axis=-1)

        # Update avalues according to spin case
        for i, a in enumerate(self.a_basis):
            avals = a.value(di, ri)
            self._avalues[:, :, i, 0] = gpu.cp.sum(avals[:nup], axis=0)
            self._avalues[:, :, i, 1] = gpu.cp.sum(avals[nup:], axis=0)

        u = gpu.cp.sum(self._bvalues * self.parameters["bcoeff"], axis=(2, 1))
        u += gpu.cp.einsum("ijkl,jkl->i", self._avalues, self.parameters["acoeff"])
        return (np.ones(len(u)), gpu.asnumpy(u))

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        r"""Update a and b sums.
        _avalues is the array for current configurations :math:`A_{Iks} = \sum_s a_{k}(r_{Is})` where :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums.
        _bvalues is the array for current configurations :math:`B_{ls} = \sum_s b_{l}(r_{s})` where :math:`s` indexes over :math:`\uparrow\uparrow` (:math:`\alpha_1 < \alpha_2`), :math:`\uparrow\downarrow` (:math:`\alpha, \beta`), and :math:`\downarrow\downarrow` (:math:`\beta_1 < \beta_2`)  sums.
        The update for _avalues and _b_values from moving one electron only requires computing the new sum for that electron. The sums for the electron in the current configuration are stored in _a_partial and _b_partial"""
        if mask is None:
            mask = [True] * self._configscurrent.configs.shape[0]
        edown = int(e >= self._mol.nelec[0])
        if saved_values is None:
            aupdate = self._a_update(e, epos, mask)
            bupdate, bvals = self._b_update(e, epos, mask)
        else:
            aupdate, bupdate, bvals = [s[mask] for s in saved_values]
        self._avalues[:, :, :, edown][mask] += aupdate - self._a_partial[e][mask]
        self._bvalues[:, :, edown : edown + 2][mask] += (
            bupdate - self._b_partial[e][mask]
        )
        self._a_partial[e][mask] = aupdate
        self._update_b_partial(e, epos, mask, bvals)
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
        d = gpu.cp.asarray(
            epos.dist.dist_i(self._mol.atom_coords(), epos.configs[mask])
        )
        r = gpu.cp.linalg.norm(d, axis=-1)
        a_partial_e = gpu.cp.zeros((*r.shape, self._a_partial.shape[3]))
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
        d = gpu.cp.asarray(
            epos.dist.dist_i(
                self._configscurrent.configs[mask][:, not_e], epos.configs[mask]
            )
        )
        r = gpu.cp.linalg.norm(d, axis=-1)
        b_partial_e = gpu.cp.zeros((*r.shape[:-1], *self._b_partial.shape[2:]))

        bvals = gpu.cp.stack([b.value(d, r) for b in self.b_basis], axis=-2)
        b_partial_e[..., 0] = bvals[..., :sep].sum(axis=-1)
        b_partial_e[..., 1] = bvals[..., sep:].sum(axis=-1)

        # for l, b in enumerate(self.b_basis):
        #    bval = b.value(d, r)
        #    b_partial_e[..., l, 0] = bval[..., :sep].sum(axis=-1)
        #    b_partial_e[..., l, 1] = bval[..., sep:].sum(axis=-1)

        return b_partial_e, bvals

    def _b_update_many(self, e, epos, mask, spin):
        r"""
        Compute the update to b for each electron moving to epos.

          Calculate b (e-e) partial sums for electron e
        _b_partial_e is the array :math:`B^p_{ils} = \sum_s b_l(r^i_{es}`, with e fixed; :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums, not including electron e.
          :math:`i` is the configuration index.
          Args:
              e: fixed electron index
              epos: configs object for electron e
              mask: mask over configs axis, only return values for configs where mask==True. b_partial_e might have a smaller configs axis than epos, _configscurrent, and _b_partial because of the mask.
        """
        nup = self._mol.nelec[0]
        d = gpu.cp.asarray(
            epos.dist.dist_i(self._configscurrent.configs[mask], epos.configs[mask])
        )
        r = gpu.cp.linalg.norm(d, axis=-1)
        b_partial_e = gpu.cp.zeros(
            (e.shape[0], *r.shape[:-1], *self._b_partial.shape[2:])
        )

        for l, b in enumerate(self.b_basis):
            bval = b.value(d, r)
            b_partial_e[..., l, 0] = bval[..., :nup].sum(axis=-1)
            b_partial_e[..., l, 1] = bval[..., nup:].sum(axis=-1)
            # b_partial_e[..., l, spin] -= bval[..., e].T
            b_partial_e[..., l, spin] -= np.moveaxis(bval[..., e], -1, 0)

        return b_partial_e

    def _update_b_partial(self, e, epos, mask, savedbvals):
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
        # d = gpu.cp.asarray(
        #    epos.dist.dist_i(
        #        self._configscurrent.configs[mask][:, not_e], epos.configs[mask]
        #    )
        # )
        # r = gpu.cp.linalg.norm(d, axis=-1)
        dold = gpu.cp.asarray(
            epos.dist.dist_i(
                self._configscurrent.configs[mask][:, not_e],
                self._configscurrent.configs[mask, e],
            )
        )
        rold = gpu.cp.linalg.norm(dold, axis=-1)
        eind, mind = np.ix_(not_e, mask)

        oldbvals = gpu.cp.stack([b.value(dold, rold) for b in self.b_basis], axis=-2)
        bdiff = savedbvals - oldbvals
        self._b_partial[eind, mind, :, edown] += bdiff.transpose((2, 0, 1))
        self._b_partial[e, ..., 0][mask] = savedbvals[..., :sep].sum(axis=-1)
        self._b_partial[e, ..., 1][mask] = savedbvals[..., sep:].sum(axis=-1)

        # for l, b in enumerate(self.b_basis):
        #    bval = b.value(d, r)
        #    bdiff = bval - b.value(dold, rold)
        #    self._b_partial[eind, mind, l, edown] += bdiff.transpose((1, 0))
        #    self._b_partial[e, :, l, 0][mask] = bval[:, :sep].sum(axis=1)
        #    self._b_partial[e, :, l, 1][mask] = bval[:, sep:].sum(axis=1)

    def value(self):
        """Compute the current log value of the wavefunction"""
        u = gpu.cp.sum(self._bvalues * self.parameters["bcoeff"], axis=(2, 1))
        u += gpu.cp.einsum("ijkl,jkl->i", self._avalues, self.parameters["acoeff"])
        return (np.ones(len(u)), gpu.asnumpy(u))

    def gradient(self, e, epos):
        r"""We compute the gradient for electron e as
        :math:`\nabla_e \ln \Psi_J = \sum_l c_l \left(\sum_{j > e} \nabla_e b_l(r_{ej}) + \sum_{i < e} \nabla_e b_l(r_{ie})\right)`
        So we need to compute the gradient of the b's for these indices.
        Note that we need to compute distances between electron position given and the current electron distances.
        We will need this for laplacian() as well"""
        nconf, nelec = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        # Get e-e and e-ion distances
        not_e = np.arange(nelec) != e
        dnew = gpu.cp.asarray(
            epos.dist.dist_i(self._configscurrent.configs, epos.configs)[:, not_e]
        )
        dinew = gpu.cp.asarray(epos.dist.dist_i(self._mol.atom_coords(), epos.configs))
        rnew = gpu.cp.linalg.norm(dnew, axis=-1)
        rinew = gpu.cp.linalg.norm(dinew, axis=-1)

        grad = gpu.cp.zeros((3, nconf))

        # Check if selected electron is spin up or down
        eup = int(e < nup)
        edown = int(e >= nup)

        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            bgrad = b.gradient(dnew, rnew)
            grad += c[edown] * gpu.cp.sum(bgrad[:, : nup - eup], axis=1).T
            grad += c[1 + edown] * gpu.cp.sum(bgrad[:, nup - eup :], axis=1).T

        for c, a in zip(self.parameters["acoeff"].transpose()[edown], self.a_basis):
            grad += gpu.cp.einsum("j,ijk->ki", c, a.gradient(dinew, rinew))

        return gpu.asnumpy(grad)

    def gradient_value(self, e, epos):
        r""""""
        nconf, nelec = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        # Get e-e and e-ion distances
        not_e = np.arange(nelec) != e
        dnew = gpu.cp.asarray(
            epos.dist.dist_i(self._configscurrent.configs[:, not_e], epos.configs)
        )
        dinew = gpu.cp.asarray(epos.dist.dist_i(self._mol.atom_coords(), epos.configs))
        rnew = gpu.cp.linalg.norm(dnew, axis=-1)
        rinew = gpu.cp.linalg.norm(dinew, axis=-1)

        grad = gpu.cp.zeros((3, nconf))

        # Check if selected electron is spin up or down
        eup = int(e < nup)
        edown = int(e >= nup)

        b_partial_e = gpu.cp.zeros((*rnew.shape[:-1], *self._b_partial.shape[2:]))
        bvals = []
        for l, b in enumerate(self.b_basis):
            c = self.parameters["bcoeff"][l]
            bgrad, bval = b.gradient_value(dnew, rnew)
            bvals.append(bval)
            grad += c[edown] * gpu.cp.sum(bgrad[:, : nup - eup], axis=1).T
            grad += c[1 + edown] * gpu.cp.sum(bgrad[:, nup - eup :], axis=1).T
            b_partial_e[..., l, 0] = bval[..., : nup - eup].sum(axis=-1)
            b_partial_e[..., l, 1] = bval[..., nup - eup :].sum(axis=-1)
        bvals = gpu.cp.stack(bvals, axis=-2)

        a_partial_e = gpu.cp.zeros((*rinew.shape, self._a_partial.shape[3]))
        for k, a in enumerate(self.a_basis):
            c = self.parameters["acoeff"][:, k, edown]
            agrad, aval = a.gradient_value(dinew, rinew)
            grad += gpu.cp.einsum("j,ijk->ki", c, agrad)
            a_partial_e[..., k] = aval

        deltaa = a_partial_e - self._a_partial[e]
        a_val = gpu.cp.einsum(
            "...jk,jk->...", deltaa, self.parameters["acoeff"][..., edown]
        )
        deltab = b_partial_e - self._b_partial[e]
        b_val = gpu.cp.einsum(
            "...jk,jk->...", deltab, self.parameters["bcoeff"][:, edown : edown + 2]
        )
        val = gpu.cp.exp(b_val + a_val)
        return gpu.asnumpy(grad), gpu.asnumpy(val), (a_partial_e, b_partial_e, bvals)

    def gradient_laplacian(self, e, epos):
        """ """
        nconf, nelec = self._configscurrent.configs.shape[:2]
        nup = self._mol.nelec[0]

        # Get e-e and e-ion distances
        not_e = np.arange(nelec) != e
        dnew = gpu.cp.asarray(
            epos.dist.dist_i(self._configscurrent.configs, epos.configs)[:, not_e]
        )
        dinew = gpu.cp.asarray(epos.dist.dist_i(self._mol.atom_coords(), epos.configs))
        rnew = gpu.cp.linalg.norm(dnew, axis=-1)
        rinew = gpu.cp.linalg.norm(dinew, axis=-1)

        eup = int(e < nup)
        edown = int(e >= nup)

        grad = gpu.cp.zeros((3, nconf))
        lap = gpu.cp.zeros(nconf)
        # a-value component
        for c, a in zip(self.parameters["acoeff"].transpose()[edown], self.a_basis):
            g, l = a.gradient_laplacian(dinew, rinew)
            grad += gpu.cp.einsum("j,ijk->ki", c, g)
            lap += gpu.cp.einsum("j,ijk->i", c, l)

        # b-value component
        for c, b in zip(self.parameters["bcoeff"], self.b_basis):
            bgrad, blap = b.gradient_laplacian(dnew, rnew)

            grad += c[edown] * gpu.cp.sum(bgrad[:, : nup - eup], axis=1).T
            grad += c[1 + edown] * gpu.cp.sum(bgrad[:, nup - eup :], axis=1).T
            lap += c[edown] * gpu.cp.sum(blap[:, : nup - eup], axis=(1, 2))
            lap += c[1 + edown] * gpu.cp.sum(blap[:, nup - eup :], axis=(1, 2))
        return gpu.asnumpy(grad), gpu.asnumpy(lap + gpu.cp.sum(grad**2, axis=0))

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
        aupdate = self._a_update(e, epos, mask)
        deltaa = aupdate - self._a_partial[e][mask]
        a_val = gpu.cp.einsum(
            "...jk,jk->...", deltaa, self.parameters["acoeff"][..., edown]
        )
        bupdate, savedbvals = self._b_update(e, epos, mask)
        deltab = bupdate - self._b_partial[e][mask]
        b_val = gpu.cp.einsum(
            "...jk,jk->...", deltab, self.parameters["bcoeff"][:, edown : edown + 2]
        )
        val = gpu.cp.exp(b_val + a_val)
        if len(val.shape) == 2:
            val = val.T
        return gpu.asnumpy(val), (aupdate, bupdate, savedbvals)

    def testvalue_many(self, e, epos, mask=None):
        r"""
        Compute the ratio :math:`\Psi_{\rm new}/\Psi_{\rm old}` for moving electrons in e to epos.

        _avalues is the array for current configurations :math:`A_{Iks} = \sum_s a_{k}(r_{Is})` where :math:`s` indexes over :math:`\uparrow` (:math:`\alpha`) and :math:`\downarrow` (:math:`\beta`) sums.
        _bvalues is the array for current configurations :math:`B_{ls} = \sum_s b_{l}(r_{s})` where :math:`s` indexes over :math:`\uparrow\uparrow` (:math:`\alpha_1 < \alpha_2`), :math:`\uparrow\downarrow` (:math:`\alpha, \beta`), and :math:`\downarrow\downarrow` (:math:`\beta_1 < \beta_2`)  sums.
        The update for _avalues and _b_values from moving one electron only requires computing the new sum for that electron. The sums for the electron in the current configuration are stored in _a_partial and _b_partial.
        deltaa = :math:`a_{k}(r_{Ie})`, indexing (atom, a_basis)
        deltab = :math:`\sum_s b_{l}(r_{se})`, indexing (b_basis, spin s)
        """
        s = (e >= self._mol.nelec[0]).astype(int)
        if mask is None:
            mask = [True] * epos.configs.shape[0]

        ratios = gpu.cp.zeros((epos.configs.shape[0], e.shape[0]))
        for spin in [0, 1]:
            ind = s == spin
            deltaa = (
                self._a_update(e[ind], epos, mask) - self._a_partial[e[ind]][:, mask]
            )
            deltab = (
                self._b_update_many(e[ind], epos, mask, spin)
                - self._b_partial[e[ind]][:, mask]
            )
            a_val = gpu.cp.einsum(
                "...jk,jk->...", deltaa, self.parameters["acoeff"][..., spin]
            )
            b_val = gpu.cp.einsum(
                "...jk,jk->...", deltab, self.parameters["bcoeff"][:, spin : spin + 2]
            )
            val = gpu.cp.exp(b_val + a_val)
            if len(val.shape) == 2:
                val = val.T
            ratios[:, ind] = val
        return gpu.asnumpy(ratios)

    def pgradient(self):
        """Given the b sums, this is pretty trivial for the coefficient derivatives.
        For the derivatives of basis functions, we will have to compute the derivative
        of all the b's and redo the sums, similar to recompute()"""
        return {
            "bcoeff": gpu.asnumpy(self._bvalues),
            "acoeff": gpu.asnumpy(self._avalues),
        }

    def u_components(self, rvec, r):
        """Given positions rvec and their magnitudes r, returns
        dictionaries of the one-body and two-body Jastrow components.
        Dictionaries are the spin components of U summed across the basis;
        one-body also returns U for different atoms."""
        u_onebody = {"up": [], "dn": []}
        rvec = gpu.cp.asarray(rvec)
        r = gpu.cp.asarray(r)
        a_value = gpu.cp.asarray(list(map(lambda x: x.value(rvec, r), self.a_basis)))
        u_onebody["up"] = gpu.cp.einsum(
            "ij,jl->il", self.parameters["acoeff"][:, :, 0], a_value
        )
        u_onebody["dn"] = gpu.cp.einsum(
            "ij,jl->il", self.parameters["acoeff"][:, :, 1], a_value
        )

        u_twobody = {"upup": [], "updn": [], "dndn": []}
        b_value = gpu.cp.asarray(
            list(map(lambda x: x.value(rvec, r), self.b_basis[1:]))
        )
        u_twobody["upup"] = gpu.cp.dot(self.parameters["bcoeff"][1:, 0], b_value)
        u_twobody["updn"] = gpu.cp.dot(self.parameters["bcoeff"][1:, 1], b_value)
        u_twobody["dndn"] = gpu.cp.dot(self.parameters["bcoeff"][1:, 2], b_value)

        return u_onebody, u_twobody
