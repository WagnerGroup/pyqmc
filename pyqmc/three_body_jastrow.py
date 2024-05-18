# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np


class ThreeBodyJastrow:
    r"""
    3 body jastrow factor

        The Jastrow form is :math:`e^{U(R)}`, where

        .. math::

            U(R)= \sum_{Iklm \sigma_1 \sigma_2} c_{klmI\sigma_1\sigma_2} \sum_{i \in \sigma_1, j \in \sigma_2, i<j} \left(a_k(r_{iI})a_l(r_{jI}) + a_k(r_{jI})a_l(r_{iI})\right) b_m(r_{ij})

        alternatively we could write it as

        .. math::

            U(R) = \sum_{klmI\sigma_1 \sigma_2} c_{klmI\sigma_1\sigma_2} \sum_{i \in \sigma_1, j \in \sigma_2, i<j} (a_k(r_{iI})a_l(r_{jI}))b_m(r_{ij})

    with

    .. math:: c_{klmI\sigma_1\sigma_2} = c_{lkmI\sigma_1\sigma_2}

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
        self.dtype = float

    def recompute(self, configs):
        r"""
        P_i is an array with contribution to jastrow from each electron e.

        .. math::

            P_{i} = \sum_{Iklm \sigma_2} c_{klmI\sigma_i\sigma_2} a_k(r_{iI})\sum_{j \in \sigma_2,j \neq i} a_l(r_{jI}))b_m(r_{ij})

        a_values is the evaluation of each a basis, for each ion-electron distance.

        Args:
        configs : Openconfigs object of current electron configuration , with configs.configs having shape [nconfig,nelec,3]

        :returns: phase, log value of shape [nconfigs]

        """
        self._configscurrent = configs.copy()
        nconf, nelec = configs.configs.shape[:2]
        na = len(self.a_basis)

        # electron-ion distances
        di = np.zeros((nelec, nconf, self._mol.natm, 3))
        for e, epos in enumerate(configs.configs.swapaxes(0, 1)):
            di[e] = configs.dist.dist_i(self._mol.atom_coords(), epos)
        ri = np.linalg.norm(di, axis=-1)

        a_values = np.zeros((self._nelec, nconf, self._mol.natm, na))
        for i, a in enumerate(self.a_basis):
            # di dim nconf,I,nelec
            a_values[:, :, :, i] = a.value(di, ri)

        self.C = (
            self.parameters["ccoeff"] + self.parameters["ccoeff"].swapaxes(1, 2)
        ) / 2

        self.P_i = np.zeros((nelec, nconf))
        arange_e = np.arange(nelec)
        for e, epos in enumerate(configs.configs.swapaxes(0, 1)):
            not_e = arange_e != e
            self.P_i[e] = self.single_e_partial(configs, e, epos, a_values[not_e])[
                0
            ].sum(axis=0)

        self.val = 0.5 * self.P_i.sum(axis=0)
        self.a_values = a_values

        return self.value()

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        r"""

        .. math::

            P_ie = \sum_{Iklm} c_{Iklm\sigma_i\sigma{j}} a_k(r_{iI}) a_l(r_{jI}) b_m(r_{ij})

            P_i = \sum_{j\ne i} P_{ij}

        update electron e

        .. math::

            P_e^{\rm new} = \sum_{j\ne e} \sum_{Iklm} c_{Iklm\sigma_e\sigma{j}} a_k(r_{eI}') a_l(r_{jI}) b_m(r_{ej}')

            P_{i\ne e}^{new} = \sum_{Iklm} c_{Iklm\sigma_e\sigma{j}} a_k(r_{iI}) a_l(r_{eI}') b_m(r_{ie}') - P_{ie}^{\rm old}

        """
        if mask is None:
            mask = np.ones(len(epos.configs), dtype=bool)
        not_e = np.arange(self._nelec) != e

        configs_mask = configs.mask(mask)
        eind, mind = np.ix_(not_e, mask)
        if saved_values is None:
            P_ie_new, ae = self.single_e_partial(
                configs_mask, e, epos.configs[mask], self.a_values[eind, mind]
            )
        else:
            P_ie_new, ae = saved_values
            P_ie_new = P_ie_new[:, mask]
            ae = ae[mask]
        P_ie_old = self.single_e_partial(
            configs_mask, e, configs_mask.configs[:, e], self.a_values[eind, mind]
        )[0]
        newval = P_ie_new.sum(axis=0)
        self.val[mask] += newval - self.P_i[e, mask]
        self.P_i[e, mask] = newval
        self.P_i[eind, mind] += P_ie_new - P_ie_old
        self.a_values[e, mask] = ae
        self._configscurrent.move(e, epos, mask)

    def value(self):
        """Compute the current log value of the wavefunction

        :returns: tuple (phases,values) of shape [nconfig]"""
        return (np.ones(len(self.val)), self.val.copy())

    def single_e_partial(self, configs, e, epos, a_values):
        r"""Args:
        configs: OpenConfig object with total electron configuration

        e: electron index

        epos: (nconfig, 3) array with with proposed electron e configuration

        a_values: a_basis evaluated on electron ion distances for configs

        :returns:

        e_partial: partial sum of U with respect to electron e. array of shape [nelec-1,nelec,...,nconfig].

        .. math::
            \text{e\_partial}_{ji} = \sum_{Iklm \sigma_i \sigma_j} c_{klmI\sigma_j\sigma_i} a_k(r_{iI}) a_l(r_{jI}) b_m(r_{ij})

        ae: new a_values for distances of electron e with ion. array of shape [nconfig,n_Ions,n_abasis]

        .. math::
            \text{ae}_{nIk} = a_k(r_{eI})

        """
        na, nb = len(self.a_basis), len(self.b_basis)
        nup = self._mol.nelec[0]
        sep = nup - int(e < nup)
        edown = int(e >= self._mol.nelec[0])
        not_e = np.arange(self._nelec) != e

        if len(epos.shape) == 2:
            de = configs.dist.dist_i(configs.configs[:, not_e], epos)
            di_e = configs.dist.dist_i(self._mol.atom_coords(), epos)
        else:
            de = configs.dist.pairwise(configs.configs[:, not_e], epos)
            di_e = configs.dist.pairwise(self._mol.atom_coords()[np.newaxis], epos)
            de = np.moveaxis(de, 2, 0)
            di_e = np.moveaxis(di_e, 2, 0)

        re = np.linalg.norm(de, axis=-1)
        ri_e = np.linalg.norm(di_e, axis=-1)

        ae = np.zeros((*epos.shape[-2::-1], self._mol.natm, na))
        for i, a in enumerate(self.a_basis):
            ae[..., :, i] = a.value(di_e, ri_e)

        b_values = np.zeros((*epos.shape[-2::-1], self._nelec - 1, nb))
        for i, b in enumerate(self.b_basis):
            b_values[..., :, i] = b.value(de, re)
        # epos shape nconfig,naux,3
        e_partial = np.zeros((self._nelec - 1, *epos.shape[-2::-1]))

        # shift a_values[:sep] to a_values[not_e][:sep] and dont pass a_values as input but instead use self.a_values?

        e_partial[:sep] = np.einsum(
            "...nIk,j...nIl,...njm,Iklm->j...n",
            ae,
            a_values[:sep],
            b_values[..., :sep, :],
            self.C[..., edown],
        )
        e_partial[sep:] = np.einsum(
            "...nIk,j...nIl,...njm,Iklm->j...n",
            ae,
            a_values[sep:],
            b_values[..., sep:, :],
            self.C[..., edown + 1],
        )
        return e_partial, ae

    def single_e_partial_many(self, configs, e, epos, spin):
        r"""Args:
        configs: OpenConfig object with total electron configuration

        e: array of electron indexes to move


        epos: Openconfigs Object with with proposed configuration for electrons

        spin: spin of e electrons

        :returns:

        e_partial: partial sum of U with respect to electron e. array of shape [nelec-1,nelec,...,nconfig].

        .. math::
            \text{e\_partial}_{ji}  =  \sum_{Iklm \sigma_i \sigma_j} c_{klmI\sigma_j\sigma_i} a_k(r_{iI}) a_l(r_{jI}) b_m(r_{ij})

        ae: new a_values for distances of electron e with ion. array of shape [nconfig,n_Ions,n_abasis]

        .. math::
            \text{ae}_{nIk} = a_k(r_{eI})

        """

        na, nb = len(self.a_basis), len(self.b_basis)
        nup = self._mol.nelec[0]

        de = configs.dist.dist_i(configs.configs, epos)
        re = np.linalg.norm(de, axis=-1)

        di_e = configs.dist.dist_i(self._mol.atom_coords(), epos)
        ri_e = np.linalg.norm(di_e, axis=-1)

        # *epos.shape[-2::-1] is naux,nconf or just nconf
        ae = np.zeros((*epos.shape[-2::-1], self._mol.natm, na))
        for i, a in enumerate(self.a_basis):
            ae[..., :, i] = a.value(di_e, ri_e)

        b_values = np.zeros((*epos.shape[-2::-1], self._nelec, nb))
        for i, b in enumerate(self.b_basis):
            b_values[..., :, i] = b.value(de, re)
        # epos shape nconfig,naux,3
        e_partial = np.zeros((e.shape[0], *epos.shape[-2::-1]))
        e_partial_common = np.zeros((self._nelec, *epos.shape[-2::-1]))
        e_partial_common[:nup, :] = np.einsum(
            "...nIk,j...nIl,...njm,Iklm->j...n",
            ae,
            self.a_values[:nup],
            b_values[..., :nup, :],
            self.C[..., spin],
        )
        e_partial_common[nup:, :] = np.einsum(
            "...nIk,j...nIl,...njm,Iklm->j...n",
            ae,
            self.a_values[nup:],
            b_values[..., nup:, :],
            self.C[..., spin + 1],
        )
        e_partial[:, :] = e_partial_common.sum(axis=0) - e_partial_common[e]
        return e_partial

    def testvalue(self, e, epos, mask=None):
        r"""
        Compute the ratio :math:`\Psi_{\rm new}/\Psi_{\rm old}` for moving electron e to epos.
        """
        configs = self._configscurrent
        nconf = configs.configs.shape[0]
        if mask is None:
            mask = np.ones(nconf, dtype=bool)

        not_e = np.arange(self._nelec) != e

        e_partial_new, a_e = self.single_e_partial(
            configs.mask(mask), e, epos.configs[mask], self.a_values[not_e][:, mask]
        )

        val = np.exp(e_partial_new.sum(axis=0) - self.P_i[e, mask])
        # if val is dim 2 naux,nconf, val.T flips it, else it leaves it be. for a 1d array, A : A = A.T
        return val.T, (e_partial_new, a_e)

    def testvalue_many(self, e, epos, mask=None):
        r"""Args:
        e: array of electron indexes to move

        epos: Openconfigs Object with with proposed configuration for electrons

        compute the ratio :

            .. math:: \Psi_{\rm new}/\Psi_{\rm old}

         for moving electrons in e to epos.

        """

        configs = self._configscurrent
        nconf = configs.configs.shape[0]
        ne = e.shape[0]
        if mask is None:
            mask = np.ones(nconf, dtype=bool)
        val = np.zeros((ne, nconf))
        s = (e >= self._mol.nelec[0]).astype(int)
        for spin in [0, 1]:
            ind = s == spin
            e_partial_new = self.single_e_partial_many(
                configs.mask(mask), e[ind], epos.configs[mask], spin
            )

            val[ind, :] = np.exp(e_partial_new - self.P_i[e[ind]][:, mask])

        return val.T

    def gradient(self, e, epos):
        r"""We compute the gradient for U with electron e moved to epos, with respect to e as

        .. math::     

            \frac{\partial U}{\partial r_{ed}} = \sum_{Iklm\sigma_2}  c_{klmI \sigma(e)\sigma_{2}} & \sum_{j\in \sigma_2 ,e<j} \frac{\partial a_k(r_{Ie})}{\partial r_{ed}} a_l(r_{Ij}) b_m(r_{ej}) + a_k(r_{Ie})a_l(r_{Ij}) \frac{\partial b_m(r_{ej})}{\partial r_{ed}} \\
                                                                                                   & + \sum_{i\in \sigma_2 , e>i} a_k(r_{Ii}) \frac{\partial a_l(r_{Ie})}{\partial r_{ed}} b_m(r_{ei})+ a_k(r_{Ii})a_l(r_{Ie})\frac{\partial b_m(r_{ie})}{\partial r_{ed}}
            
        Args:

            e: fixed electron index
            epos: configs object for electron e

        :returns: gradient with respect to electron e with shape [3,nconfigs]

        """

        configs = self._configscurrent
        na, nb = len(self.a_basis), len(self.b_basis)
        nconf = configs.configs.shape[0]
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

    def gradient_value(self, e, epos):
        r"""
        computes the log value, and gradient. This way we can reuse evaluations of the bases.
        """

        configs = self._configscurrent
        na, nb = len(self.a_basis), len(self.b_basis)
        nconf = configs.configs.shape[0]
        nup = self._mol.nelec[0]
        not_e = np.arange(self._nelec) != e
        edown = int(e >= nup)
        sep = nup - int(e < nup)

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
        b_gradvals = np.zeros((nconf, self._nelec - 1, nb, 4))
        for m, b in enumerate(self.b_basis):
            b_gradvals[:, :, m, 1:], b_gradvals[:, :, m, 0] = b.gradient_value(de, re)

        spin_up = (np.arange(self._nelec - 1) < sep).astype(float)
        spin = np.stack([spin_up, 1 - spin_up], axis=0)
        Cab_j = np.einsum(
            "jnIl,njmd,Iklms,sj->djnIk",
            self.a_values[not_e],
            b_gradvals,
            self.C[..., edown : edown + 2],
            spin,
        )

        e_partial_new = np.einsum("nIk,jnIk->jn", a_e, Cab_j[0])
        Cab = Cab_j.sum(axis=1)

        val = np.exp(np.sum(e_partial_new, axis=0) - self.P_i[e])

        grad_term1 = np.einsum("nIkd,nIk->dn", a_gradients, Cab[0])
        grad_term2 = np.einsum("nIk,dnIk->dn", a_e, Cab[1:])
        return grad_term1 + grad_term2, val, (e_partial_new, a_e)

    def gradient_laplacian(self, e, epos):
        r"""computes gradient and laplacian, so we can reuse evaluations of the basis and its derivative."""
        configs = self._configscurrent
        na, nb = len(self.a_basis), len(self.b_basis)
        nconf = configs.configs.shape[0]
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
        return grad, lap + np.sum(grad**2, axis=0)

    def laplacian(self, e, epos):
        r"""We compute the laplacian for U with electron e moved to epos, with respect to e as

        .. math::  \frac{1}{J} \nabla^2_e J = \nabla^2_e U + (\nabla_e.U)^2

        with 

        .. math:: 
        
            \nabla^2_e U = \sum_d \sum_{Iklm\sigma_2} c_{klmI \sigma(e)\sigma_{2}} \sum_{j\in \sigma_2 j \neq e} \frac{\partial^2 a_k(r_{Ie})}{\partial r_{ed}^2} a_l(r_{Ij}) b_m(r_{ej}) + & 2\frac{\partial a_k(r_{Ie})}{\partial r_{ed}} a_l(r_{Ij}) \frac{\partial b_m(r_{ej})}{\partial r_{ed}} \\
                                                                                                                                                                                          + & a_k(r_{Ie}) a_l(r_{Ij}) \frac{\partial^2 b_m(r_{ej})}{\partial r^2_{ed}}

        Args:
            e: fixed electron index
            epos: configs object for electron e

        :returns: gradient with respect to electron e with shape [3,nconfigs]
        """
        return self.gradient_laplacian(e, epos)[1]

    def pgradient(self):
        r"""
        computes the parameter gradients, given by

        .. math::

            \frac{\partial U}{\partial c_{ijkI\sigma_1\sigma_2}} = \sum_{i \in \sigma_1, j \in \sigma_2, i<j} \left(a_k(r_{iI})a_l(r_{jI}) + a_k(r_{jI})a_l(r_{iI})\right) b_m(r_{ij})

        """
        configs = self._configscurrent
        nconf, nelec = configs.configs.shape[:2]
        na, nb = len(self.a_basis), len(self.b_basis)
        nup, ndown = self._mol.nelec

        # order of spin channel: upup,updown,downdown
        d_all, ij = configs.dist.dist_matrix(configs.configs)
        r_all = np.linalg.norm(d_all, axis=-1)
        bvalues = np.stack([b.value(d_all, r_all) for b in self.b_basis], axis=-1)
        inds = tuple(zip(*ij))
        b_2d_values = np.zeros((nelec, nelec, nconf, nb))
        b_2d_values[inds] = bvalues.swapaxes(0, 1)

        a = self.a_values
        up, down = slice(0, nup), slice(nup, None)
        c_ders = np.zeros((nconf, self._mol.natm, na, na, nb, 3))
        einstr = "inIk,jnIl,ijnm->nIklm"
        c_ders[..., 0] = np.einsum(einstr, a[up], a[up], b_2d_values[up, up])
        c_ders[..., 1] = np.einsum(einstr, a[up], a[down], b_2d_values[up, down])
        c_ders[..., 2] = np.einsum(einstr, a[down], a[down], b_2d_values[down, down])
        c_ders += c_ders.swapaxes(2, 3)

        return {"ccoeff": 0.5 * c_ders}
