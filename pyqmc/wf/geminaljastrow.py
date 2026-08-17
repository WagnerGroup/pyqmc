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

r"""
The geminal Jastrow factor, in two implementations of the same wave function.

:class:`GeminalJastrow` is the one to use. :class:`GeminalJastrowReference` is
the straightforward transcription of the formulas, kept because it is easy to
check against the algebra; :class:`GeminalJastrow` derives from it, returns the
same numbers to machine precision, and is tested against it method by method in
``tests/unit/test_geminaljastrow_equivalence.py``.
"""

import numpy as np
import pyqmc.gpu as gpu
import pyqmc.wf.orbitals
import pyqmc.pbc.supercell


class GeminalJastrowReference:
    r"""
    Jastrow factor defined by Casula, Attaccalite, and Sorella, J. Chem. Phys. 121, 7110 (2004); https://doi.org/10.1063/1.1794632

    This is the reference implementation, written to follow the derivation below
    directly. :class:`GeminalJastrow` is the same wave function evaluated more
    efficiently, and is what you normally want.

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
            raise ValueError(
                "Wrong number of parameters. Maybe the parameters are from an incompatible version."
            )
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
            ao[:, e + 1 :, :],
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


    def gradient_laplacian(self, e, epos):
        r"""
        .. math:: \nabla_e J_G = \sum_{m\le n} \sum_{i<e} g_{mn} \nabla\chi_m(\mathbf{r}'_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \nabla\chi_n(\mathbf{r}'_e) \chi_m(\mathbf{r}_j)
        .. math:: \nabla_e^2 J_G = \sum_{m\le n} \sum_{i<e} g_{mn} \nabla^2\chi_m(\mathbf{r}'_e) \chi_n(\mathbf{r}_i)
                             + \sum_{e<j} g_{mn} \nabla^2\chi_n(\mathbf{r}'_e) \chi_m(\mathbf{r}_j)
        .. math:: \nabla_e^2 e^{J_G(\mathbf{R})} / e^{J_G(\mathbf{R})} = \nabla_e^2 J_G + |\nabla_e J_G|^2
        """
        ao = self.orbitals.aos("GTOval_sph_deriv2", epos)[0][1:]
        # ao = gpu.cp.concatenate(
        #    [ao[1:4, ...], ao[[4, 7, 9], ...].sum(axis=0, keepdims=True)], axis=0
        # )
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
            masked_ao_val[:, e + 1 :, :],
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
        new_ao_val = aos.reshape(len(aos), *epos.configs.shape[1:-1], aos.shape[-1])
        masked_ao_val = self.ao_val[mask]

        out = np.zeros((len(e_), aos.shape[0]))
        for i, e in enumerate(e_):
            curr_val = self._compute_value(masked_ao_val[:, e], masked_ao_val, e)
            new_val = self._compute_value(new_ao_val, masked_ao_val, e)
            out[i] = gpu.asnumpy(gpu.cp.exp((new_val.T - curr_val).T))
        return out.T


class GeminalJastrow(GeminalJastrowReference):
    r"""The geminal Jastrow factor of :class:`GeminalJastrowReference`, which
    documents the wave function and the derivation, evaluated more efficiently.

    Same constructor arguments, same numbers to machine precision. Two pieces of
    arithmetic are reorganized. On H2O with a 138-function even-tempered geminal
    basis, ``_compute_value`` was 93% of the run time of a VMC step, so this is
    most of what the wave function costs.

    **The summed-orbital contraction.**
    :meth:`GeminalJastrowReference._compute_value` builds the value from two
    contractions, one over the electrons before ``e`` and one over those after
    it. ``recompute`` symmetrizes :math:`g_{mn}`, and under that symmetry both
    terms have the same form, so the two sums merge into a single sum over every
    electron but ``e``:

    .. math:: \sum_{m\le n} \Big[ \sum_{i<e} g_{mn} \chi_m(\mathbf{r}_e)\chi_n(\mathbf{r}_i)
                                + \sum_{j>e} g_{mn} \chi_n(\mathbf{r}_e)\chi_m(\mathbf{r}_j) \Big]
              = \sum_{mn} g_{mn} \chi_m(\mathbf{r}_e) S^e_n,
              \qquad S^e_n = \sum_{j\ne e} \chi_n(\mathbf{r}_j)

    Evaluating :math:`T^e_m = \sum_n g_{mn} S^e_n` first leaves only one
    (nconfig, nbasis) x (nbasis, nbasis) product -- the sole :math:`O(M^2)` step
    -- followed by an :math:`O(M)` contraction per derivative component. The
    reference form contracts the full :math:`g` against the derivative-carrying
    orbital values twice per call, so it pays :math:`O(M^2)` once for each of
    the (up to ten) derivative components, and its ``c`` index is a batch index
    shared between two operands and the output, which numpy cannot route to
    BLAS. Measured on ``_compute_value`` alone: 24x at M=42, 54x at M=81, 85x at
    M=138.

    **Caching.** :math:`T^e` depends only on the stored orbital values and on
    ``e``, not on the position being tested, so every call for the same electron
    between two moves reuses one product; in a VMC step that is six calls
    sharing one. Keeping :math:`T^e` also makes the value itself cheap to
    maintain: moving electron ``e`` changes :math:`\log\Psi` by
    :math:`\sum_m \Delta\chi_m(\mathbf{r}_e) T^e_m`, so :meth:`updateinternals`
    carries the value forward instead of :meth:`value` rebuilding the full
    double sum on every call. That matters for the ensemble sampler, which asks
    for ``value()`` once per electron per step.

    A full VMC run with Slater + two-body Jastrow + geminal at M=138 goes from
    4.35s to 0.19s, a 23x speedup, with block energies agreeing to 3.6e-15.

    Note that the rewrite relies on :math:`g_{mn}=g_{nm}`. ``recompute`` builds
    ``gcoeff`` from the upper triangle and adds its transpose, so that holds by
    construction; it is not re-checked at run time because it cannot fail
    without someone assigning to ``self.gcoeff`` directly, which is unsupported
    here in a way it is not for the reference class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._invalidate()
        self._value = None

    def _invalidate(self):
        """Drop the cached T vector. Called whenever the stored orbitals change."""
        self._tvec_e = None
        self._tvec_cache = None

    def _tvec_from(self, ao, e):
        r""":math:`T^e_m = \sum_n g_{mn} \sum_{j\ne e} \chi_n(\mathbf{r}_j)`.

        This is the only :math:`O(\rm nbasis^2)` work in the wave function.

        :parameter ndarray(nconfig, nelec, nbasis) ao: stored orbital values
        :parameter int e: electron index
        :returns: (nconfig, nbasis) array
        """
        S_e = ao.sum(axis=1) - ao[:, e, :]
        return S_e @ self.gcoeff

    def _tvec(self, e):
        """:meth:`_tvec_from` on the stored orbitals, cached across calls.

        The cache holds one electron at a time, which is all that is needed:
        within a move every call is for the same ``e``, and the next move
        invalidates it through :meth:`updateinternals`.
        """
        if self._tvec_cache is None or self._tvec_e != e:
            self._tvec_cache = self._tvec_from(self.ao_val, e)
            self._tvec_e = e
        return self._tvec_cache

    def _compute_value(self, ao_e, ao, e):
        """Same quantity and signature as the reference, evaluated through T.

        `ao` is honoured as given, so callers that pass a subset of the
        configurations still get the right answer; the cache is only used when
        they pass the stored orbitals themselves.
        """
        t = self._tvec(e) if ao is self.ao_val else self._tvec_from(ao, e)
        return gpu.cp.einsum("...cm,cm->...c", ao_e, t)

    def recompute(self, configs):
        """Rebuild the stored orbitals, and with them the cached value."""
        self._invalidate()
        self._value = None
        # the parent ends by calling self.value(), which fills the cache
        return super().recompute(configs)

    def _full_value(self):
        """The double sum, from scratch. Only used to seed the cache."""
        mask = gpu.cp.tril(gpu.cp.ones((self.nelec, self.nelec)), -1)
        return gpu.cp.einsum(
            "mn,cim, cjn, ij-> c",
            self.gcoeff,
            self.ao_val,
            self.ao_val,
            mask,
            optimize=self.optimize,
        )

    def value(self):
        """The stored value, rebuilt only if it has not been computed yet.

        The result is copied out. The reference returns a freshly computed array
        every call, so callers hold on to it and expect to own it, while here it
        would otherwise be the live cache that ``updateinternals`` mutates --
        note that ``gpu.asnumpy`` is the identity on CPU.
        """
        if self._value is None:
            self._value = self._full_value()
        return (np.ones(len(self._value)), gpu.asnumpy(self._value).copy())

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        r"""Move electron ``e`` and carry the value along with it.

        Only the pairs involving ``e`` change, and their total is
        :math:`\sum_m \chi_m(\mathbf{r}_e) T^e_m`, so the change in the value is
        :math:`\sum_m [\chi_m(\mathbf{r}'_e) - \chi_m(\mathbf{r}_e)] T^e_m` with
        the same :math:`T^e` the move's ratio was computed from.
        """
        if mask is None:
            mask = np.ones(self.ao_val.shape[0], dtype=bool)
        else:
            mask = np.asarray(mask)
        if saved_values is None:
            aoval = self.orbitals.aos("GTOval_sph", epos, mask=mask)[0]
        else:
            aoval = saved_values[mask]

        # T uses the orbitals from before the move, so this must come first
        t = self._tvec(e)[mask]
        if self._value is None:
            self._value = self._full_value()
        self._value[mask] += gpu.cp.einsum(
            "cm,cm->c", aoval - self.ao_val[mask, e, :], t
        )

        self.ao_val[mask, e, :] = aoval
        self._invalidate()

    def testvalue(self, e, epos, mask=None):
        """Same as the reference, sharing one T between the old and new values."""
        t_full = self._tvec(e)
        if mask is None:
            masked_ao_val, t = self.ao_val, t_full
        else:
            mask = np.asarray(mask)
            masked_ao_val, t = self.ao_val[mask], t_full[mask]

        curr_val = gpu.cp.einsum("cm,cm->c", masked_ao_val[:, e, :], t)
        aos = self.orbitals.aos("GTOval_sph", epos, mask=mask)[0]
        new_ao_val = aos.reshape(
            len(curr_val), *epos.configs.shape[1:-1], aos.shape[-1]
        )
        # `...` is for the extra dimension for ECP aux coordinates
        new_val = gpu.cp.einsum("c...m,cm->c...", new_ao_val, t)
        return gpu.asnumpy(gpu.cp.exp((new_val.T - curr_val).T)), new_ao_val

    def testvalue_many(self, e_, epos, mask=None):
        """Same as the reference; one T per electron in ``e_`` instead of two."""
        if mask is not None:
            mask = np.asarray(mask)
        aos = self.orbitals.aos("GTOval_sph", epos, mask=mask)[0]
        new_ao_val = aos.reshape(len(aos), *epos.configs.shape[1:-1], aos.shape[-1])
        masked_ao_val = self.ao_val if mask is None else self.ao_val[mask]

        out = np.zeros((len(e_), aos.shape[0]))
        for i, e in enumerate(e_):
            t = self._tvec(e) if mask is None else self._tvec(e)[mask]
            curr_val = gpu.cp.einsum("cm,cm->c", masked_ao_val[:, e, :], t)
            new_val = gpu.cp.einsum("c...m,cm->c...", new_ao_val, t)
            out[i] = gpu.asnumpy(gpu.cp.exp((new_val.T - curr_val).T))
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
