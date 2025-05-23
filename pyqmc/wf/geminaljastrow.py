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
import pyqmc.gpu as gpu
import pyqmc.wf.orbitals
import pyqmc.pbc.supercell


class GeminalJastrow:
    r"""
    Jastrow factor defined by Casula, Attaccalite, and Sorella, J. Chem. Phys. 121, 7110 (2004); https://doi.org/10.1063/1.1794632

    .. math:: J_G(\mathbf{R}) = \sum_{i\ne j} \sum_{mn} \tilde g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

    :math:`\chi_m(\mathbf{r})` is a set of basis orbitals.

    We can split the sum over all :math:`i, j` into two sums over all pairs :math:`i<j` (the second term exchanges all pairs of particles from the first, :math:`\mathbf{r}_i \leftrightarrow \mathbf{r}_j`)

    .. math:: J_G(\mathbf{R}) =
        \underbrace{\sum_{i< j} \sum_{mn} \tilde g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)}_{J_G^1(\mathbf{R})}
      + \underbrace{\sum_{i< j} \sum_{mn} \tilde g_{mn} \chi_m(\mathbf{r}_j) \chi_n(\mathbf{r}_i)}_{J_G^2(\mathbf{R})}

    Reordering the product in :math:`J_G^2` and reindexing :math:`m\leftrightarrow n` yields

    .. math:: J_G^2(\mathbf{R}) = \sum_{i< j} \sum_{mn} \tilde g_{nm} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j),

    showing that exchanging two electrons is equivalent to exchanging the indices of :math:`\tilde g_{mn}`.
    The coefficient matrix :math:`\tilde g_{mn}` has extra degrees of freedom: only the symmetric part contributes.

    .. math:: J_G(\mathbf{R}) = \sum_{i< j} \sum_{mn} (\tilde g_{mn}+\tilde g_{nm}) \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

    .. math:: J_G(\mathbf{R}) = \sum_{i< j} \sum_{m\le n} g_{mn} [\chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j) + \chi_n(\mathbf{r}_i) \chi_m(\mathbf{r}_j)]

    The independent parameters are :math:`g_{mn} = \tilde g_{mn} + \tilde g_{nm}, \; m \le n`.
    """

    def __init__(self, mol, orbitals=None, eval_gto_precision=None):
        if orbitals is None:
            if hasattr(mol, "lattice_vectors"):
                if not hasattr(mol, "original_cell"):
                    mol = pyqmc.pbc.supercell.get_supercell(mol, np.eye(3))
                else:
                    mol = make_pbc_supercell_for_gamma_aos(mol)
                kpts = [[0, 0, 0]]
                self.orbitals = pyqmc.wf.orbitals.PBCOrbitalEvaluatorKpoints(
                    mol, kpts=kpts, eval_gto_precision=eval_gto_precision
                )
            else:
                self.orbitals = pyqmc.wf.orbitals.MoleculeOrbitalEvaluator(mol, [0, 0])
        else:
            self.orbitals = orbitals
        randpos = np.random.random((1, 3))
        dim = mol.eval_gto("GTOval_sph", randpos).shape[-1]
        self.parameters = {"gcoeff": gpu.cp.zeros(int(dim * (dim + 1) / 2))}
        self.dtype = float
        self.optimize = "greedy"

    def recompute(self, configs):
        r"""
        Initializes the 2D array :math:`g_{mn}` from the 1D parameter vector
        """
        nconf, self.nelec = configs.configs.shape[:2]
        # shape of arrays:
        # ao_val: (nconf, nelec, nbasis)
        aos = self.orbitals.aos("GTOval_sph", configs)
        self.ao_val = aos.reshape(nconf, self.nelec, aos.shape[-1])
        self.gcoeff = gpu.cp.zeros((aos.shape[-1], aos.shape[-1]))
        triu_inds = gpu.cp.triu_indices(aos.shape[-1])
        if len(triu_inds[0]) != len(self.parameters["gcoeff"]):
            raise ValueError("Wrong number of parameters. Maybe the parameters are from an incompatible version.")
        self.gcoeff[triu_inds] = self.parameters["gcoeff"]
        self.gcoeff = self.gcoeff + self.gcoeff.T

        return self.value()

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        """
        AO values are saved (nconfig, nelec, nao)
        """
        if mask is None:
            mask = [True] * epos.configs.shape[0]
        if saved_values is None:
            aoval = self.orbitals.aos("GTOval_sph", epos, mask=mask)[0]
        else:
            aoval = saved_values[mask]
        self.ao_val[mask, e, :] = aoval

    def value(self):
        r"""
        .. math:: \sum_{m\le n} \sum_{i<j} g_{mn} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

        :math:`g_{mn}` only has nonzero elements where :math:`m\le n`.
        """
        mask = gpu.cp.tril(gpu.cp.ones((self.nelec, self.nelec)), -1)
        vals = gpu.cp.einsum(
            "mn,cim, cjn, ij-> c",
            self.gcoeff,
            self.ao_val,
            self.ao_val,
            mask,
            optimize=self.optimize,
        )
        signs = np.ones(len(vals))
        return (signs, gpu.asnumpy(vals))

    def _compute_value(self, ao_e, ao, e):
        r"""
        .. math:: \sum_{m\le n} \sum_{i<e} g_{mn} \chi_m(\mathbf{r}_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \chi_n(\mathbf{r}_e) \chi_m(\mathbf{r}_j)

        For derivatives,

        .. math:: \sum_{m\le n} \sum_{i<e} g_{mn} \nabla\chi_m(\mathbf{r}_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \nabla\chi_n(\mathbf{r}_e) \chi_m(\mathbf{r}_j)

        :math:`g_{mn}` only has nonzero elements where :math:`m\le n`.

        Parameters:
        :parameter ndarray(nconfig, nao) ao_e: ao values or derivatives of electron e. If derivatives, shape is (nderiv, nconfig, nao)
        :parameter ndarray(nconfig, nelec, nao) ao: ao values of all electrons.
        :parameter int e: electron index

        """
        # `...` is for derivatives
        curr_val = gpu.cp.einsum(
            "mn, ...cm, cjn -> ...c",
            self.gcoeff,
            ao_e,
            ao[:, :e, :],
            optimize=self.optimize,
        )
        curr_val += gpu.cp.einsum(
            "mn, ...cn, cim -> ...c",
            self.gcoeff,
            ao_e,
            ao[:, e + 1:,:],
            optimize=self.optimize,
        )
        return curr_val

    def gradient_value(self, e, epos):
        ao = self.orbitals.aos("GTOval_sph_deriv1", epos)[0]
        deriv = self._compute_value(ao, self.ao_val, e)
        curr_val = self._compute_value(self.ao_val[:, e], self.ao_val, e)
        val_ratio = gpu.cp.exp(deriv[0] - curr_val)
        return gpu.asnumpy(deriv[1:]), gpu.asnumpy(val_ratio), ao[0]

    def gradient(self, e, epos):
        r"""
        .. math:: \sum_{m\le n} \sum_{i<e} g_{mn} \nabla\chi_m(\mathbf{r}'_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \nabla\chi_n(\mathbf{r}'_e) \chi_m(\mathbf{r}_j)
        """

        ao = self.orbitals.aos("GTOval_sph_deriv1", epos)[0]
        grad = self._compute_value(ao[1:], self.ao_val, e)
        return gpu.asnumpy(grad)

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def gradient_laplacian(self, e, epos):
        r"""
        .. math:: \nabla_e J_G = \sum_{m\le n} \sum_{i<e} g_{mn} \nabla\chi_m(\mathbf{r}'_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \nabla\chi_n(\mathbf{r}'_e) \chi_m(\mathbf{r}_j)
        .. math:: \nabla_e^2 J_G = \sum_{m\le n} \sum_{i<e} g_{mn} \nabla^2\chi_m(\mathbf{r}'_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \nabla^2\chi_n(\mathbf{r}'_e) \chi_m(\mathbf{r}_j)
        .. math:: \nabla_e^2 e^{J_G(\mathbf{R})} / e^{J_G(\mathbf{R})} = \nabla_e^2 J_G + |\nabla_e J_G|^2
        """
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)[0][1:]
        #ao = gpu.cp.concatenate(
        #    [ao[1:4, ...], ao[[4, 7, 9], ...].sum(axis=0, keepdims=True)], axis=0
        #)
        deriv = self._compute_value(ao, self.ao_val, e)
        grad = deriv[:3]
        lap3 = gpu.cp.einsum("dc,dc->c", grad, grad)
        return gpu.asnumpy(grad), gpu.asnumpy(deriv[3] + lap3)

    def pgradient(self):
        r"""
        .. math:: \frac{\partial J_G}{\partial g_{mn}} = \sum_{i<j} \chi_m(\mathbf{r}_i) \chi_n(\mathbf{r}_j)

        The upper triangular entries are extracted as a 1D array to match the shape of parameters["gcoeff"].
        """
        mask = gpu.cp.tril(
            gpu.cp.ones((self.nelec, self.nelec)), -1
        )  # to prevent double counting of electron pairs
        coeff_grad = gpu.cp.einsum(
            "cim, cjn, ij-> cmn", self.ao_val, self.ao_val, mask, optimize=self.optimize
        )
        coeff_grad = coeff_grad + coeff_grad.transpose(0, 2, 1)
        tui = gpu.cp.triu_indices(coeff_grad.shape[-1])  # select only m <= n
        return {"gcoeff": coeff_grad[:, tui[0], tui[1]]}

    def testvalue(self, e, epos, mask=None):
        r"""
        .. math:: \sum_{m\le n} \sum_{i<e} g_{mn} \chi_m(\mathbf{r}'_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \chi_n(\mathbf{r}'_e) \chi_m(\mathbf{r}_j)
                 -\left(\sum_{m\le n} \sum_{i<e} g_{mn} \chi_m(\mathbf{r}_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \chi_n(\mathbf{r}_e) \chi_m(\mathbf{r}_j)\right)
        """
        if mask is None:
            mask = [True] * self.ao_val.shape[0]
        masked_ao_val = self.ao_val[mask]
        curr_val = self._compute_value(masked_ao_val[:, e], masked_ao_val, e)
        aos = self.orbitals.aos("GTOval_sph", epos, mask=mask)[0]
        new_ao_val = aos.reshape(
            len(curr_val), *epos.configs.shape[1:-1], aos.shape[-1]
        )
        # `...` is for extra dimension for ECP aux coordinates
        new_val = gpu.cp.einsum(
            "mn, c...m, cjn -> c...",
            self.gcoeff,
            new_ao_val,
            masked_ao_val[:, :e, :],
            optimize=self.optimize,
        )
        new_val += gpu.cp.einsum(
            "mn, c...n, cim -> c...",
            self.gcoeff,
            new_ao_val,
            masked_ao_val[:, e + 1:, :],
            optimize=self.optimize,
        )
        return gpu.asnumpy(gpu.cp.exp((new_val.T - curr_val).T)), new_ao_val

    def testvalue_many(self, e_, epos, mask=None):
        r"""
        .. math:: \sum_{m\le n} \sum_{i<e} g_{mn} \chi_m(\mathbf{r}'_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \chi_n(\mathbf{r}'_e) \chi_m(\mathbf{r}_j)
                 -\left(\sum_{m\le n} \sum_{i<e} g_{mn} \chi_m(\mathbf{r}_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \chi_n(\mathbf{r}_e) \chi_m(\mathbf{r}_j)\right)
        """
        if mask is None:
            mask = [True] * self.ao_val.shape[0]
        aos = self.orbitals.aos("GTOval_sph", epos, mask=mask)[0]
        new_ao_val = aos.reshape(
            len(aos), *epos.configs.shape[1:-1], aos.shape[-1]
        )
        masked_ao_val = self.ao_val[mask]

        out = np.zeros((len(e_), aos.shape[0]))
        for i, e in enumerate(e_):
            curr_val = self._compute_value(masked_ao_val[:, e], masked_ao_val, e)
            new_val = self._compute_value(new_ao_val, masked_ao_val, e)
            out[i] =  gpu.asnumpy(gpu.cp.exp((new_val.T - curr_val).T))
        return out.T


def make_pbc_supercell_for_gamma_aos(scell, S=None, **kwargs):
    import pyscf.pbc.gto as gto

    cell = scell.original_cell
    if S is None:
        S = scell.S
    scale = np.abs(int(np.round(np.linalg.det(S))))
    superlattice = np.dot(S, cell.lattice_vectors())
    Rpts = pyqmc.pbc.supercell.get_supercell_copies(cell.lattice_vectors(), S)
    atom = []
    for name, xyz in cell._atom:
        atom.extend([(name, xyz + R) for R in Rpts])

    newcell = gto.Cell(
        atom=atom,
        a=superlattice,
        unit="Bohr",
        spin=cell.spin * scale,
    )
    for k in ["basis", "ecp", "exp_to_discard"]:
        if k not in kwargs.keys() and k in cell.__dict__.keys():
            kwargs[k] = cell.__dict__[k]
    for k, v in kwargs.items():
        newcell.__dict__[k] = v
    newcell.build()
    newsupercell = pyqmc.pbc.supercell.get_supercell(newcell, np.eye(3))
    return newsupercell
