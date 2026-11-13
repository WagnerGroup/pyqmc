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

"""Tests of the Ewald sum periodic in one direction, on point charges."""
import numpy as np
import pytest
import scipy.integrate
import scipy.special as sp

import pyqmc.observables.ewald as ewald3d
import pyqmc.observables.ewald1d as ewald1d
from pyqmc.configurations.coord import PeriodicConfigs


def make_cell(L, A=6.0, axis=2, dimension=3):
    """Two H atoms in an L x L box, periodic with length A along `axis`."""
    import pyscf.pbc.gto

    def place(transverse, along):
        r = np.empty(3)
        r[[i for i in range(3) if i != axis]] = transverse
        r[axis] = along
        return r

    cell = pyscf.pbc.gto.Cell()
    cell.atom = [
        ("H", place([L / 2 + 0.3, L / 2 - 0.2], 0.4)),
        ("H", place([L / 2 - 0.5, L / 2 + 0.1], 3.1)),
    ]
    lengths = np.full(3, L)
    lengths[axis] = A
    cell.a = np.diag(lengths)
    cell.unit = "Bohr"
    cell.basis = "sto-3g"
    cell.dimension = dimension
    if dimension == 1:
        cell.low_dim_ft_type = "inf_vacuum"  # pyscf builds a 1D cell only with this
    cell.verbose = 0
    cell.build()
    return cell


def make_configs(cell, L, A=6.0, axis=2):
    rng = np.random.default_rng(7)
    transverse = L / 2 + rng.normal(0, 1.0, (2, 3, 2))
    along = rng.uniform(0, A, (3, 2))
    x = np.empty((3, 2, 3))  # (nconf, nelec, 3)
    x[..., [i for i in range(3) if i != axis]] = np.moveaxis(transverse, 0, -1)
    x[..., axis] = along
    return PeriodicConfigs(x, cell.lattice_vectors())


def direct_sum(pos, q, A, M, axis=2):
    """Sum over images |n| <= M of a neutral set (pair and self-image terms)."""
    n = np.arange(-M, M + 1)
    e = 0.0
    for i in range(len(q)):
        for j in range(len(q)):
            d = pos[i] - pos[j]
            dz = d[axis] + n * A
            r = np.sqrt(np.sum(np.delete(d, axis) ** 2) + dz**2)
            if i == j:
                r = r[n != 0]
            e += 0.5 * q[i] * q[j] * np.sum(1 / r)
    return e


def total(ew, configs):
    ee, ei, ii = ew.energy(configs)
    return np.asarray(ee) + np.asarray(ei) + ii


@pytest.mark.parametrize("x,y", [(0.05, 0.0), (0.4, 0.3), (2.0, 5.0), (0.01, 20.0)])
def test_incomplete_bessel(x, y):
    ref = scipy.integrate.quad(lambda t: np.exp(-x * t - y / t) / t, 1, np.inf, limit=500)[0]
    assert abs(ewald1d.incomplete_bessel(x, [y])[0] - ref) < 1e-10 * max(1, ref)


def test_bessel_limit():
    """K(k^2/4a^2, a^2 rho^2) -> 2 K0(k rho) for large alpha."""
    k, rho, a = 1.3, 0.7, 50.0
    val = ewald1d.incomplete_bessel(k**2 / (4 * a**2), [a**2 * rho**2], nquad=2000)[0]
    assert abs(val - 2 * sp.k0(k * rho)) < 1e-8


def test_alpha_independent():
    L = 30.0
    cell = make_cell(L)
    configs = make_configs(cell, L)
    e = [total(ewald1d.Ewald(cell, axis=2, alpha_scaling=s, nlatvec=2), configs) for s in (3.0, 5.0, 8.0)]
    assert np.max(np.abs(e[0] - e[1])) < 1e-8
    assert np.max(np.abs(e[2] - e[1])) < 1e-8


def test_direct_sum():
    L, A = 30.0, 6.0
    cell = make_cell(L, A)
    configs = make_configs(cell, L, A)
    e = total(ewald1d.Ewald(cell, axis=2), configs)
    q = np.concatenate([cell.atom_charges(), [-1.0, -1.0]])
    for c in range(3):
        pos = np.concatenate([cell.atom_coords(), configs.configs[c]])
        ed = np.array([direct_sum(pos, q, A, M) for M in (2000, 4000, 8000)])
        # truncation error ~ 1 / M^2 (neutral cells with a dipole): extrapolate from the last two
        einf = ed[2] + (ed[2] - ed[1]) / 3
        assert abs(einf - e[c]) < 1e-7, (c, einf, e[c], ed)


def test_axis_permutation():
    """The same system periodic along x and along z has the same energy."""
    L, A = 30.0, 6.0
    e = []
    for axis in (0, 2):
        cell = make_cell(L, A, axis=axis)
        e.append(total(ewald1d.Ewald(cell, axis=axis), make_configs(cell, L, A, axis=axis)))
    assert np.allclose(e[0], e[1], atol=1e-10)


def test_3d_limit():
    """The 3D Ewald sum approaches the 1D one as the transverse box grows (corrections ~ 1/L^2)."""
    diffs = []
    for L in (20.0, 40.0, 80.0):
        cell = make_cell(L)
        configs = make_configs(cell, L)
        e3 = total(ewald3d.Ewald(cell), configs)
        diffs.append(np.max(np.abs(e3 - total(ewald1d.Ewald(cell, axis=2), configs))))
    assert diffs[2] < diffs[1] < diffs[0]
    assert diffs[2] < 2e-3


def test_energy_accumulator_selects_1d():
    import pyqmc.api as pyq

    cell = make_cell(20.0, axis=0, dimension=1)
    acc = pyq.EnergyAccumulator(cell)
    assert isinstance(acc.coulomb, ewald1d.Ewald) and acc.coulomb.axis == 0


def test_charged_cell_rejected():
    import pyscf.pbc.gto

    cell = pyscf.pbc.gto.M(atom="He 1 1 1", a=np.diag([6.0, 20.0, 20.0]), basis="sto-3g",
                           unit="bohr", charge=1, spin=1, verbose=0)
    with pytest.raises(ValueError, match="neutral"):
        ewald1d.Ewald(cell)
