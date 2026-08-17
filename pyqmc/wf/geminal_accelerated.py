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
A faster :class:`pyqmc.wf.geminaljastrow.GeminalJastrow`.

The wave function is unchanged -- every method returns the same numbers as the
parent class to machine precision -- but two pieces of arithmetic are
reorganized. On H2O with a 138-function even-tempered geminal basis,
``_compute_value`` was 93% of the run time of a VMC step, so this is most of what
the wave function costs.

**The summed-orbital contraction.** :meth:`GeminalJastrow._compute_value` builds
the value from two contractions, one over the electrons before ``e`` and one
over the electrons after it. ``recompute`` symmetrizes :math:`g_{mn}`, and under
that symmetry both terms have the same form, so the two sums merge into a single
sum over every electron but ``e``:

.. math:: \sum_{m\le n} \Big[ \sum_{i<e} g_{mn} \chi_m(\mathbf{r}_e)\chi_n(\mathbf{r}_i)
                            + \sum_{j>e} g_{mn} \chi_n(\mathbf{r}_e)\chi_m(\mathbf{r}_j) \Big]
          = \sum_{mn} g_{mn} \chi_m(\mathbf{r}_e) S^e_n,
          \qquad S^e_n = \sum_{j\ne e} \chi_n(\mathbf{r}_j)

Evaluating it as :math:`T^e_m = \sum_n g_{mn} S^e_n` first leaves only one
(nconfig, nbasis) x (nbasis, nbasis) product -- the sole :math:`O(M^2)` step --
followed by an :math:`O(M)` contraction per derivative component. The original
form contracts the full :math:`g` against the derivative-carrying orbital values
twice per call, so it pays :math:`O(M^2)` once for each of the (up to ten)
derivative components, and its ``c`` index is a batch index shared between two
operands and the output, which numpy cannot route to BLAS.

**Caching.** :math:`T^e` depends only on the stored orbital values and on ``e``,
not on the position being tested, so every call for the same electron between
two moves reuses one product; in a VMC step that is six calls sharing one.
Keeping :math:`T^e` also makes the value itself cheap to maintain: moving
electron ``e`` changes :math:`\log\Psi` by :math:`\sum_m \Delta\chi_m(\mathbf{r}_e) T^e_m`,
so ``updateinternals`` can carry the value forward instead of ``value()``
rebuilding the full double sum on every call. That matters for the ensemble
sampler, which asks for ``value()`` once per electron per step.
"""

import numpy as np

import pyqmc.gpu as gpu
from pyqmc.wf.geminaljastrow import GeminalJastrow


class GeminalJastrowAccelerated(GeminalJastrow):
    r"""Drop-in replacement for :class:`pyqmc.wf.geminaljastrow.GeminalJastrow`.

    Takes the same constructor arguments and produces the same wave function;
    see the module docstring for what is done differently.

    Note that the rewrite relies on :math:`g_{mn}=g_{nm}`. ``recompute`` builds
    ``gcoeff`` from the upper triangle and adds its transpose, so that holds by
    construction; it is not re-checked at run time because it cannot fail
    without someone assigning to ``self.gcoeff`` directly, which would be
    unsupported here in a way it is not for the parent class.
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
        """Same quantity and signature as the parent, evaluated through T.

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

        The result is copied out. The parent returns a freshly computed array
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
        """Same as the parent, sharing one T between the old and new values."""
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
        """Same as the parent; one T per electron in ``e_`` instead of two."""
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
