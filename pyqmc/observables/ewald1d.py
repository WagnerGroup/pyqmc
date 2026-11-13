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

r"""Ewald summation for a system periodic in one direction, with no images in the other two.

The cell is a 3D orthorhombic cell; only the lattice vector along `axis` (length :math:`A`) is a
period. Separations transverse to it are taken as the minimum image in the cell, which is the
true separation when every pair of particles is closer than half the cell size transversely.

For a neutral set of charges :math:`q_i` with images along the axis,

.. math::

    E = \sum_{i<j} q_i q_j \phi(\vec{r}_{ij}) + \phi_{\rm self} \sum_i q_i^2,

    \phi(\vec{r}) = \sum_{|n|\le n_{\rm lat}} \frac{{\rm erfc}(\alpha |\vec{r} + nA\hat{z}|)}{|\vec{r} + nA\hat{z}|}
        + \frac{2}{A} \sum_{m\ge1} \cos(k_m z) K\!\left(\frac{k_m^2}{4\alpha^2}, \alpha^2\rho^2\right)
        - \frac{1}{A} {\rm Ein}(\alpha^2\rho^2),

    \phi_{\rm self} = \sum_{1\le n\le n_{\rm lat}} \frac{{\rm erfc}(\alpha n A)}{nA}
        + \frac{1}{A} \sum_{m\ge1} E_1\!\left(\frac{k_m^2}{4\alpha^2}\right) - \frac{\alpha}{\sqrt{\pi}},

with :math:`k_m = 2\pi m / A`, :math:`z` the separation along the axis, :math:`\rho` the
transverse separation,

.. math::

    K(x, y) = \int_1^\infty t^{-1} e^{-xt - y/t}\,dt = \int_0^\infty e^{-x e^s - y e^{-s}}\,ds,
    \qquad {\rm Ein}(y) = \int_0^y \frac{1 - e^{-t}}{t}\,dt = \gamma + \ln y + E_1(y).

Derivation: :math:`{\rm erf}(\alpha r)/r = (2/\sqrt{\pi}) \int_0^\alpha e^{-r^2 t^2} dt`. Poisson
summation of the Gaussian over the images gives
:math:`(2/A) \sum_k e^{ikz} \int_0^\alpha t^{-1} e^{-t^2\rho^2 - k^2/4t^2} dt`. For :math:`k \ne 0`,
:math:`t = \alpha/\sqrt{\tau}` gives :math:`K`. For :math:`k = 0` the integral diverges at
:math:`t \to 0` by a constant times :math:`(\sum_i q_i)^2 = 0`; the remainder is
:math:`-{\rm Ein}(\alpha^2\rho^2)/A`. As :math:`\alpha \to \infty`, :math:`K \to 2 K_0(k\rho)`, the
potential of a line of charges.

:math:`K` is tabulated for each :math:`k_m` on a grid in :math:`\rho` and interpolated with a cubic
spline.
"""
import numpy as np
import scipy.special as sp
from scipy.interpolate import CubicSpline
import pyqmc.configurations.distance
import pyqmc.observables.ewald as ewald3d


def ein(y):
    r""":math:`{\rm Ein}(y) = \gamma + \ln y + E_1(y)`, with the series for small y."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y < 1e-2
    ys = y[small]
    out[small] = ys - ys**2 / 4 + ys**3 / 18 - ys**4 / 96
    yl = y[~small]
    out[~small] = np.euler_gamma + np.log(yl) + sp.exp1(yl)
    return out


def incomplete_bessel(x, y, nquad=400):
    r""":math:`K(x, y) = \int_0^\infty e^{-x e^s - y e^{-s}} ds` for scalar x > 0 and an array y >= 0."""
    smax = np.log(800.0 / x) + 2.0  # x e^s > 800 beyond
    s, w = np.polynomial.legendre.leggauss(nquad)
    s = 0.5 * smax * (s + 1)
    w = 0.5 * smax * w
    y = np.asarray(y, dtype=float)
    return np.exp(-x * np.exp(s)[None, :] - y.reshape(-1, 1) * np.exp(-s)[None, :]) @ w


class Ewald:
    """Coulomb energy of a system periodic along one lattice vector.

    Same interface as :class:`pyqmc.observables.ewald.Ewald`: ``energy(configs) -> (ee, ei, ii)``.
    """

    def __init__(self, cell, axis=0, nlatvec=1, alpha_scaling=5.0, tol=1e-12, ngrid=4000):
        """
        :parameter cell: pyscf Cell with an orthorhombic (diagonal) lattice.
        :parameter int axis: the periodic direction. pyscf's convention for a cell with
            ``dimension = 1`` is axis 0.
        :parameter int nlatvec: real-space images n in [-nlatvec, nlatvec] along the axis.
        :parameter float alpha_scaling: alpha = alpha_scaling / A.
        :parameter float tol: drop k_m with (2/A) E1(k_m^2 / 4 alpha^2) < tol.
        :parameter int ngrid: number of transverse grid points in the K tables.
        """
        latvec = np.asarray(cell.lattice_vectors())
        if not np.allclose(latvec, np.diag(np.diag(latvec))):
            raise ValueError("ewald1d requires an orthorhombic cell (diagonal lattice vectors)")
        self.atom_coords = cell.atom_coords()[np.newaxis]
        self.atom_charges = np.asarray(cell.atom_charges(), dtype=float)
        if abs(np.sum(self.atom_charges) - np.sum(cell.nelec)) > 1e-8:
            raise ValueError("ewald1d is only valid for a neutral cell")
        self.axis = axis
        self.transverse = [i for i in range(3) if i != axis]
        self.latvec = latvec
        self.A = float(latvec[axis, axis])
        self.alpha = alpha_scaling / self.A
        self.dist = pyqmc.configurations.distance.MinimalImageDistance(latvec)
        n = np.arange(-nlatvec, nlatvec + 1)
        self.lattice_displacements = np.outer(n, latvec[axis])
        self.nlatvec = nlatvec

        m = np.arange(1, 1000)
        k = 2 * np.pi * m / self.A
        x = k**2 / (4 * self.alpha**2)
        keep = (2 / self.A) * sp.exp1(x) > tol
        self.k, self.x = k[keep], x[keep]
        rho_max = 0.5 * np.hypot(*np.diag(latvec)[self.transverse]) * 1.001
        rho = np.linspace(0, rho_max, ngrid)
        self.splines = [
            CubicSpline(rho, incomplete_bessel(xm, self.alpha**2 * rho**2)) for xm in self.x
        ]

        nn = np.arange(1, nlatvec + 1) * self.A
        self.phi_self = (
            np.sum(sp.erfc(self.alpha * nn) / nn)
            + np.sum(sp.exp1(self.x)) / self.A
            - self.alpha / np.sqrt(np.pi)
        )
        self.ewald_ion_ion = self._ion_ion()

    def phi(self, d):
        """Pair potential for minimum-image separations d (..., 3)."""
        real = ewald3d.real_cij(d, self.lattice_displacements, self.alpha)
        rho = np.hypot(d[..., self.transverse[0]], d[..., self.transverse[1]])
        z = d[..., self.axis]
        recip = np.zeros(rho.shape)
        for km, spl in zip(self.k, self.splines):
            recip += np.cos(km * z) * spl(rho)
        return np.asarray(real) + (2 / self.A) * recip - ein(self.alpha**2 * rho**2) / self.A

    def _ion_ion(self):
        q = self.atom_charges
        e = np.sum(q**2) * self.phi_self
        if len(q) > 1:
            d, ij = self.dist.dist_matrix(self.atom_coords)
            ij = np.asarray(ij)
            e += float(np.sum(q[ij[:, 0]] * q[ij[:, 1]] * self.phi(d)[0]))
        return e

    def ewald_elec_ion(self, configs):
        d = configs.dist.pairwise(self.atom_coords, configs.configs)  # (nconf, natom, nelec, 3)
        return -np.einsum("a,cae->c", self.atom_charges, self.phi(d))

    def ewald_elec_elec(self, configs):
        nconf, nelec, _ = configs.configs.shape
        e = nelec * self.phi_self * np.ones(nconf)
        if nelec > 1:
            d, _ = configs.dist.dist_matrix(configs.configs)  # (nconf, npairs, 3)
            e = e + np.sum(self.phi(d), axis=1)
        return e

    def energy(self, configs):
        """:returns: ee, ei (each (nconf,)), and the constant ii"""
        return self.ewald_elec_elec(configs), self.ewald_elec_ion(configs), self.ewald_ion_ion
