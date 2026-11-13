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

"""The plane-wave Jastrow is the collective-coordinate form sum_q c_q (|rho_q|^2 - N_e) / 2."""
import numpy as np

import pyqmc.api as pyq


def test_value_is_rho_q(H_pbc_sto3g_krks):
    mol, mf = H_pbc_sto3g_krks
    supercell = pyq.get_supercell(mol, S=2 * np.eye(3))
    gindices = np.array([[1, 0, 0], [0, 1, 1], [2, -1, 0]])
    coeffs = np.array([-0.05, -0.02, 0.01])
    jastrow, to_opt = pyq.generate_plane_wave_jastrow(supercell, gindices)
    jastrow.parameters["bcoeff"][:] = coeffs[:, np.newaxis]  # same in all spin channels
    assert to_opt["bcoeff"].shape == (3, 3)

    configs = pyq.initial_guess(supercell, 7)
    _, logval = jastrow.recompute(configs)

    qvecs = 2 * np.pi * gindices @ np.linalg.inv(supercell.lattice_vectors()).T
    nelec = configs.configs.shape[1]
    ref = np.zeros(configs.configs.shape[0])
    for q, c in zip(qvecs, coeffs):
        rho = np.exp(1j * configs.configs @ q).sum(axis=1)
        ref += c * (np.abs(rho) ** 2 - nelec) / 2
    assert np.allclose(logval, ref, atol=1e-10)
