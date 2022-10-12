import numpy as np


class Three_Body_JastrowSpin:
    r"""
    3 body jastrow factor

            The Jastrow form is :math:`e^{U(R)}`, where

        .. math::  U(R) = \sum_{Iklm \sigma_1 \sigma_2}  c_{klmI\sigma_1\sigma_2}    \sum_{i \in \sigma_1, j \in \sigma_2, i \neq j}
        \left(a_k(r_{iI})a_l(r_{jI}) + a_k(r_{jI})a_l(r_{iI})\right)  b_m(r_{ij})


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
        self.parameters["ccoeff"] = np.zeros(
            (self._mol.natm, len(a_basis), len(a_basis), len(b_basis), 3)
        )
        # self.parameters["ccoeef_diagonal"] = np.diagonal(self.C,axis1=1,axis2=2)
        self.iscomplex = False

    def recompute(self, configs):
        """
        Args:
        configs : Openconfigs object of current electron configuration , with configs.configs having shape [nconfig,nelec,3]

        :returns: phase, log value of shape [nconfigs]

        a_basis is evaluated on all electron-ion distances of configs, and is stored in self.a_basis
        """
        self._configscurrent = configs.copy()
        nconf, nelec = configs.configs.shape[:2]
        na, nb = len(self.a_basis), len(self.b_basis)
        # n_elec in first axis to match di format
        # order of spin channel: upup,updown,downdown

        # electron-electron distances
        # d_upup dim is  nconf, nup(nup-1)/2,3
        # d_downdown dim is nconf, ndown(ndown-1)/2,3
        # d_updown dim is nconf, nup*ndown,3
        # nup = int(self._mol.nelec[0])
        # ndown = int(self._mol.nelec[1])
        # d_upup, ij_upup = configs.dist.dist_matrix(configs.configs[:, :nup])
        # d_updown, ij_updown = configs.dist.pairwise(
        #    configs.configs[:, :nup], configs.configs[:, nup:]
        # )
        # d_downdown, ij_downdown = configs.dist.dist_matrix(configs.configs[:, nup:])
        # d_all = [d_upup, d_updown, d_downdown]
        # ij_all = [ij_upup, ij_updown, ij_downdown]
        # r_all = [np.linalg.norm(d, axis=-1) for d in d_all]

        # # bvalues are the evaluations of b bases. bm(rij)
        # b_2d_values = []
        # for s, shape in enumerate([(nup, nup), (nup, ndown), (ndown, ndown)]):
        #    bvalues = np.stack(
        #        [b.value(d_all[s], r_all[s]) for i, b in enumerate(self.b_basis)],
        #        axis=-1,
        #    )
        #    b_2d_values_s = np.zeros((*shape, nconf, nb))
        #    inds = tuple(zip(*ij_all[s]))
        #    b_2d_values_s[inds] = bvalues.swapaxes(0, 1)
        #    b_2d_values.append(b_2d_values_s)

        # electron-ion distances
        di = np.zeros((nelec, nconf, self._mol.natm, 3))
        for e, epos in enumerate(configs.configs.swapaxes(0, 1)):
            di[e] = configs.dist.dist_i(self._mol.atom_coords(), epos)
        ri = np.linalg.norm(di, axis=-1)

        a_values = np.zeros((self._nelec, nconf, self._mol.natm, na))
        for i, a in enumerate(self.a_basis):
            # di dim nconf,I,nelec
            a_values[:, :, :, i] = a.value(di, ri)

        self.C = self.parameters["ccoeff"] + self.parameters["ccoeff"].swapaxes(1, 2)

        self.sum_jC = np.zeros((nelec, nconf))
        arange_e = np.arange(nelec)
        for e, epos in enumerate(configs.configs.swapaxes(0, 1)):
            not_e = arange_e != e
            self.sum_jC[e] = self.single_e_partial(configs, e, epos, a_values[not_e])[0]

        val = 0.5 * self.sum_jC.sum(axis=0)
        self.val = val
        self.a_values = a_values

        return (np.ones(len(val)), val)

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        if mask is None:
            mask = np.ones(len(epos.configs), dtype=bool)
        newconfigs = configs.copy()
        newconfigs.move(e, epos, mask)
        # eventually want to remove recompute from here.
        self.recompute(newconfigs)

    def value(self):
        """Compute the current log value of the wavefunction
        :returns: tuple (phases,values) of shape [nconfig]"""
        val = self.val
        return (np.ones(len(val)), val)

    def single_e_partial(self, configs, e, epos, a_values):
        """Args:
        configs: OpenConfig object with total electron configuration
        e: electron index
        epos: Openconfigs Object with with proposed electron e configuration
        a_values: a_basis evaluated on electron ion distances for configs

        :returns: e_partial: partial sum of U with respect to electron e
        ae: new a_values for distances of electron e with ion

        """
        na, nb = len(self.a_basis), len(self.b_basis)
        nup, ndown = self._mol.nelec
        sep = nup - int(e < nup)
        edown = int(e >= self._mol.nelec[0])
        not_e = np.arange(self._nelec) != e
        nconf = configs.configs.shape[0]

        de = configs.dist.dist_i(configs.configs[:, not_e], epos)
        re = np.linalg.norm(de, axis=-1)

        di_e = configs.dist.dist_i(self._mol.atom_coords(), epos)
        ri_e = np.linalg.norm(di_e, axis=-1)

        ae = np.zeros((nconf, self._mol.natm, na))
        for i, a in enumerate(self.a_basis):
            ae[:, :, i] = a.value(di_e, ri_e)

        b_values = np.zeros((self._nelec - 1, nconf, nb))
        for i, b in enumerate(self.b_basis):
            # swap axes: nconf and nelec. for now doing it here.
            b_values[:, :, i] = b.value(de, re).swapaxes(0, 1)

        e_partial = np.zeros(nconf)
        e_partial += np.einsum(
            "nIk,jnIl,jnm,Iklm->n",
            ae,
            a_values[:sep],
            b_values[:sep],
            self.C[..., edown],
        )
        e_partial += np.einsum(
            "nIk,jnIl,jnm,Iklm->n",
            ae,
            a_values[sep:],
            b_values[sep:],
            self.C[..., edown + 1],
        )
        return e_partial, ae

    def testvalue(self, e, epos, mask=None):
        r"""
        Compute the ratio :math:`\Psi_{\rm new}/\Psi_{\rm old}` for moving electron e to epos.
        """
        configs = self._configscurrent
        nconf, nelec = configs.configs.shape[:2]
        edown = int(e >= self._mol.nelec[0])

        nup = int(self._mol.nelec[0])
        ndown = int(self._mol.nelec[1])
        sep = nup - int(e < nup)
        not_e = np.arange(self._nelec) != e

        e_partial_old, _ = self.single_e_partial(
            configs, e, configs.configs[:, e], self.a_values[not_e]
        )
        e_partial_new, a_e = self.single_e_partial(
            configs, e, epos.configs, self.a_values[not_e]
        )

        val = np.exp(e_partial_new - e_partial_old)
        return val, a_e

    def gradient(self, e, epos):
        r"""We compute the gradient for U with electron e moved to epos, with respect to e as
        :math:`\frac{\partial U}{\partial r_{ed}}
        =
        \sum_{Iklm\sigma_2}
        c_{klmI \sigma(e)\sigma_{2}}
        \sum_{j\in \sigma_2 j \neq e}
        a_l(r_{Ij})
        \left(\frac{\partial a_k(r_{Ie})}{\partial r_{ed}}
        b_m(r_{ej})
        +
        a_k(r_{Ie})
        \frac{\partial b_m(r_{ej})}{\partial r_{ed}}\right)
        Args:
            e: fixed electron index
            epos: configs object for electron e
        :returns: gradient with respect to electron e with shape [3,nconfigs]
        """

        configs = self._configscurrent
        na, nb = len(self.a_basis), len(self.b_basis)
        nconf, nelec = configs.configs.shape[:2]
        nup = int(self._mol.nelec[0])
        not_e = np.arange(self._nelec) != e

        # electron-ion distances for electron e
        di_e = configs.dist.dist_i(self._mol.atom_coords(), epos.configs)
        ri_e = np.linalg.norm(di_e, axis=-1)

        # di is electron -ion distance for all electrons but electron e.
        # electron-electron distances from electron e
        de = configs.dist.dist_i(configs.configs[:, not_e], epos.configs)
        re = np.linalg.norm(de, axis=-1)

        # set values of a basis evaluations needed.
        a_gradients = np.zeros((nconf, self._mol.natm, na, 3))
        a_e = np.zeros((nconf, self._mol.natm, na))
        for k, a in enumerate(self.a_basis):
            # di dim nconf,I,nelec
            a_gradients[:, :, k, :], a_e[..., k] = a.gradient_value(di_e, ri_e)

        # set values of b basis evaluations needed
        b_values = np.zeros((nconf, self._nelec - 1, nb))
        b_gradients = np.zeros((nconf, self._nelec - 1, nb, 3))
        for m, b in enumerate(self.b_basis):
            b_gradients[:, :, m], b_values[:, :, m] = b.gradient_value(de, re)

        edown = int(e >= nup)
        sep = nup - int(e < nup)

        term1 = np.einsum(
            "Iklm,jnIl,nIkd,njm->dn",
            self.C[..., edown],
            self.a_values[not_e][:sep],
            a_gradients,
            b_values[:, :sep],
        )
        term1 += np.einsum(
            "Iklm,jnIl,nIkd,njm->dn",
            self.C[..., edown + 1],
            self.a_values[not_e][sep:],
            a_gradients,
            b_values[:, sep:],
        )

        term2 = np.einsum(
            "Iklm,jnIl,nIk,njmd->dn",
            self.C[..., edown],
            self.a_values[not_e][:sep],
            a_e,
            b_gradients[:, :sep],
        )
        term2 += np.einsum(
            "Iklm,jnIl,nIk,njmd->dn",
            self.C[..., edown + 1],
            self.a_values[not_e][sep:],
            a_e,
            b_gradients[:, sep:],
        )

        grad = term1 + term2
        return grad

    def gradient_laplacian(self, e, epos):
        configs = self._configscurrent
        na, nb = len(self.a_basis), len(self.b_basis)
        nconf, nelec = configs.configs.shape[:2]
        nup = int(self._mol.nelec[0])
        not_e = np.arange(self._nelec) != e

        di_e = configs.dist.dist_i(self._mol.atom_coords(), epos.configs)
        ri_e = np.linalg.norm(di_e, axis=-1)

        # di is electron -ion distance for all electrons but electron e.
        # electron-electron distances from electron e
        de = configs.dist.dist_i(configs.configs[:, not_e], epos.configs)
        re = np.linalg.norm(de, axis=-1)

        # set values of a basis evaluations needed.
        a_gradients = np.zeros((nconf, self._mol.natm, na, 3))
        a_e = np.zeros((nconf, self._mol.natm, na))
        a_double_ders = np.zeros((nconf, self._mol.natm, na, 3))
        for k, a in enumerate(self.a_basis):
            # di dim nconf,I,nelec
            a_gradients[:, :, k, :], a_e[..., k] = a.gradient_value(di_e, ri_e)
            a_double_ders[:, :, k, :] = a.laplacian(di_e, ri_e)

        # set values of b basis evaluations needed
        b_values = np.zeros((nconf, self._nelec - 1, nb))
        b_gradients = np.zeros((nconf, self._nelec - 1, nb, 3))
        b_double_ders = np.zeros((nconf, self._nelec - 1, nb, 3))
        for m, b in enumerate(self.b_basis):
            b_gradients[:, :, m, :], b_values[:, :, m] = b.gradient_value(de, re)
            b_double_ders[:, :, m, :] = b.laplacian(de, re)

        sep = nup - int(e < nup)
        edown = int(e >= nup)

        grad_term1 = np.einsum(
            "Iklm,jnIl,nIkd,njm->dn",
            self.C[..., edown],
            self.a_values[not_e][:sep],
            a_gradients,
            b_values[:, :sep],
        )
        grad_term1 += np.einsum(
            "Iklm,jnIl,nIkd,njm->dn",
            self.C[..., edown + 1],
            self.a_values[not_e][sep:],
            a_gradients,
            b_values[:, sep:],
        )

        grad_term2 = np.einsum(
            "Iklm,jnIl,nIk,njmd->dn",
            self.C[..., edown],
            self.a_values[not_e][:sep],
            a_e,
            b_gradients[:, :sep],
        )
        grad_term2 += np.einsum(
            "Iklm,jnIl,nIk,njmd->dn",
            self.C[..., edown + 1],
            self.a_values[not_e][sep:],
            a_e,
            b_gradients[:, sep:],
        )

        grad = grad_term1 + grad_term2

        # j in upspin term1
        lap = np.einsum(
            "Iklm,nIkd,jnIl,njm->n",
            self.C[..., edown],
            a_double_ders,
            self.a_values[not_e][:sep],
            b_values[:, :sep],
        )
        # downspin term1
        lap += np.einsum(
            "Iklm,nIkd,jnIl,njm->n",
            self.C[..., edown + 1],
            a_double_ders,
            self.a_values[not_e][sep:],
            b_values[:, sep:],
        )
        # upspin term 2
        lap += 2 * np.einsum(
            "Iklm,nIkd,jnIl,njmd->n",
            self.C[..., edown],
            a_gradients,
            self.a_values[not_e][:sep],
            b_gradients[:, :sep],
        )
        # downspin term 2
        lap += 2 * np.einsum(
            "Iklm,nIkd,jnIl,njmd->n",
            self.C[..., edown + 1],
            a_gradients,
            self.a_values[not_e][sep:],
            b_gradients[:, sep:],
        )
        # upspin term 3
        lap += np.einsum(
            "Iklm,nIk,jnIl,njmd->n",
            self.C[..., edown],
            a_e,
            self.a_values[not_e][:sep],
            b_double_ders[:, :sep],
        )
        # downspin term 3
        lap += np.einsum(
            "Iklm,nIk,jnIl,njmd->n",
            self.C[..., edown + 1],
            a_e,
            self.a_values[not_e][sep:],
            b_double_ders[:, sep:],
        )
        return grad, lap + np.sum(grad ** 2, axis=0)

    def laplacian(self, e, epos):
        r"""We compute the laplacian for U with electron e moved to epos, with respect to e as
        :math:\frac{1}{J} \nabla^2_e J = \nabla^2_e U + (\nabla_e.U)^2
        with 
        :math: \frac{\partial^2 U}{\partial r_{ed}^2}
        = 
        \sum_{Iklm\sigma_2} 
        c_{klmI \sigma(e)\sigma_{2}}
        \sum_{j\in \sigma_2 j \neq e} 
        & \frac{\partial^2 a_k(r_{Ie})}{\partial r_{ed}^2} 
        a_l(r_{Ij})
        b_m(r_{ej})
        +
        2\frac{\partial a_k(r_{Ie})}{\partial r_{ed}} 
        a_l(r_{Ij})
        \frac{\partial b_m(r_{ej})}{\partial r_{ed}} \\
        +
        &a_k(r_{Ie})
        a_l(r_{Ij}) 
        \frac{\partial^2 b_m(r_{ej})}{\partial r^2_{ed}}
        Args:
            e: fixed electron index
            epos: configs object for electron e
        :returns: gradient with respect to electron e with shape [3,nconfigs]
        """
        return self.gradient_laplacian(e, epos)[1]

    def pgradient(self):
        configs = self._configscurrent
        nconf, nelec = configs.configs.shape[:2]
        na, nb = len(self.a_basis), len(self.b_basis)
        nup = int(self._mol.nelec[0])
        ndown = int(self._mol.nelec[1])

        c_ders = np.zeros((self._mol.natm, na, na, nb, 3))

        arange_e = np.arange(nelec)

        e_partial = np.zeros(
            (
                nelec,
                nconf,
                self._mol.natm,
                na,
                na,
                nb,
                3,
            )
        )

        for e, epos in enumerate(configs.configs.swapaxes(0, 1)):
            not_e = arange_e != e
            nup, ndown = self._mol.nelec
            sep = nup - int(e < nup)
            edown = int(e >= self._mol.nelec[0])
            not_e = np.arange(self._nelec) != e

            de = configs.dist.dist_i(configs.configs[:, not_e], epos)
            re = np.linalg.norm(de, axis=-1)

            b_values = np.zeros((self._nelec - 1, nconf, nb))
            for i, b in enumerate(self.b_basis):
                # swap axes: nconf and nelec. for now doing it here.
                b_values[:, :, i] = b.value(de, re).swapaxes(0, 1)

            e_partial[e, :, :, :, :, :, edown] = np.einsum(
                "nIk,jnIl,jnm->nIklm",
                self.a_values[e],
                self.a_values[not_e][:sep],
                b_values[:sep],
            )
            e_partial[e, :, :, :, :, :, edown + 1] = np.einsum(
                "nIk,jnIl,jnm->nIklm",
                self.a_values[e],
                self.a_values[not_e][sep:],
                b_values[sep:],
            )
        return {"ccoeff": 0.5 * np.sum(e_partial, axis=0)}

        # order of spin channel: upup,updown,downdown

        # electron-electron distances
        # d_upup dim is  nconf, nup(nup-1)/2,3
        # d_downdown dim is nconf, ndown(ndown-1)/2,3
        # d_updown dim is nconf, nup*ndown,3

        # d_upup, ij_upup = configs.dist.dist_matrix(configs.configs[:, :nup])
        # d_updown, ij_updown = configs.dist.pairwise(
        #    configs.configs[:, :nup], configs.configs[:, nup:]
        # )
        # d_downdown, ij_downdown = configs.dist.dist_matrix(configs.configs[:, nup:])
        # d_all = [d_upup, d_updown, d_downdown]

        # ij_all = [ij_upup, ij_updown, ij_downdown]
        # r_all = [np.linalg.norm(d, axis=-1) for d in d_all]

        # # bvalues are the evaluations of b bases. bm(rij)
        # b_2d_values = []
        # for s, shape in enumerate([(nup, nup), (nup, ndown), (ndown, ndown)]):
        #     bvalues = np.stack(
        #        [b.value(d_all[s], r_all[s]) for i, b in enumerate(self.b_basis)],
        #        axis=-1,
        #     )
        #     b_2d_values_s = np.zeros((*shape, nconf, nb))
        #     inds = tuple(zip(*ij_all[s]))
        #     b_2d_values_s[inds] = bvalues.swapaxes(0, 1)
        #     b_2d_values.append(b_2d_values_s)

        # sum_ij = np.zeros(
        # (
        #     nconf,
        #     self._mol.natm,
        #     na,
        #     na,
        #     nb,
        #     3,
        # ))
        # print(b_2d_values[1][:,:,0,0],'bvals pgrad')

        # sum_ij[:, :, :, :, :, 0] = np.einsum(
        #     "inIk,jnIl,ijnm->nIklm", self.a_values[:nup], self.a_values[:nup], b_2d_values[0]
        # )
        # sum_ij[:, :, :, :, :, 1] = np.einsum(
        #     "inIk,jnIl,ijnm->nIklm", self.a_values[:nup], self.a_values[nup:], b_2d_values[1]
        # )
        # sum_ij[:, :, :, :, :, 2] = np.einsum(
        #     "inIk,jnIl,ijnm->nIklm", self.a_values[nup:], self.a_values[nup:], b_2d_values[2]
        # )

        # return {
        #     "ccoeff": sum_ij
        # }
