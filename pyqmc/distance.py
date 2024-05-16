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


class RawDistance:
    """Compute distance vectors using open boundary conditions"""

    def __init__(self):
        """ """
        pass

    def dist_i(self, a, b):
        # a ([m,] n, 3)
        # b (m, 3)
        # returns shape (m, n, 3)
        assert (
            len(b.shape) <= 2
        ), f"dist_i argument b has {len(b.shape)} dimensions -- can have max 2"
        return b[:, np.newaxis, :] - a

    def dist_matrix(self, configs):
        """
        All pairwise distances within the set of positions.

        Returns:

          dist: array of size nconf x n(n-1)/2 x 3

          ij : list of size n(n-1)/2 tuples that document i,j
        """
        nconf, n = configs.shape[:2]
        npairs = int(n * (n - 1) / 2)
        if npairs == 0:
            return np.zeros((nconf, 0, 3)), []

        vs = []
        ij = []
        for i in range(n):
            vs.append(self.dist_i(configs[:, i + 1 :, :], configs[:, i, :]))
            ij.extend([(i, j) for j in range(i + 1, n)])
        vs = np.concatenate(vs, axis=1)

        return vs, ij

    def pairwise(self, config1, config2):
        """
        All pairwise distances from config1 to config2
        Parameters:
            config1 (m1, n1, 3): m1 may be 1, e.g. for all atoms to all electrons
            config2 (m2, n2, 3): m2 must equal m1 if neither equals 1
        Returns:

          dist: array (nconf, n1, n2, 3)
        """
        m1, n1 = config1.shape[:2]
        m2, n2 = config2.shape[:2]
        assert (
            m1 == m2 or m1 == 1 or m2 == 1
        ), f"can't broadcast first axis {m1} and {m2}"
        if n1 == 0 or n2 == 0:
            return np.zeros((config1.shape[0], 0, 3))
        vs = config2[:, np.newaxis, :] - config1[:, :, np.newaxis]
        return vs


def _is_diagonal(M, tol):
    return np.all(np.abs(M - np.diag(np.diagonal(M))) < tol)


class MinimalImageDistance(RawDistance):
    """Compute distance vectors under a minimal image condition
    using periodic boundary conditions."""

    def __init__(self, latvec):
        """latvec should be a 3x3 set of lattice vectors, each row is a vector
        One strategy:
        * Find reduced basis
        * Find Wigner-Seitz cell
        * Find which parallelpiped units the WS cell interacts with
        * Build list of lattice points to consider

        Can also do something smarter by dividing the unit cell up into pieces that need to be determined or not.
        """
        ortho_tol = 1e-10
        diagonal = _is_diagonal(latvec, ortho_tol)
        self.dimension = latvec.shape[-1]
        if diagonal:
            self._minimal_dist = self.diagonal_dist
        else:
            L_ovlp = np.dot(latvec, latvec.T)
            orthogonal = _is_diagonal(L_ovlp, ortho_tol)
            if orthogonal:
                self._minimal_dist = self.orthogonal_dist
                # print("Orthogonal lattics vectors")
            else:
                self._minimal_dist = self.general_dist
                # print("Non-orthogonal lattics vectors")
        self._latvec = latvec
        self._invvec = np.linalg.inv(latvec)
        # list of all 26 neighboring cells
        mesh_grid = np.meshgrid(
            *[np.array(range(self.dimension)) for _ in range(self.dimension)]
        )
        self.point_list = np.stack([m.ravel() for m in mesh_grid], axis=0).T - 1
        self.shifts = np.dot(self.point_list, self._latvec)
        # TODO build a minimal list instead of using all 27

    def dist_i(self, a, b):
        # a ([m,] n, 3)
        # b (m, 3)
        # returns shape (m, n, 3)
        assert (
            len(b.shape) <= 2
        ), f"dist_i argument b has {len(b.shape)} dimensions -- can have max 2"
        return self._minimal_dist(b[:, np.newaxis, :] - a)

    def pairwise(self, a, b):
        return self._minimal_dist(super().pairwise(a, b))

    def general_dist(self, d1):
        """returns a list of electron-electron distances from an electron at position 'vec'
        configs will most likely be [nconfig,electron,dimension], and vec will be [nconfig,dimension]
        """
        shifts = self.shifts.reshape((-1, *[1] * (len(d1.shape) - 1), 3))
        d1all = d1[np.newaxis] + shifts
        dists = np.sum(d1all**2, axis=-1)
        mininds = np.argmin(dists, axis=0)
        inds = np.meshgrid(*[np.arange(n) for n in mininds.shape], indexing="ij")
        return d1all[(mininds, *inds)]

    def orthogonal_dist(self, d1):
        """Like general_dist, but assuming lattice vectors are orthogonal
        It's about 10x faster than the general one checking all 27 lattice points
        """
        frac_disps = np.einsum("...ij,jk->...ik", d1, self._invvec)
        frac_disps = (frac_disps + 0.5) % 1 - 0.5
        return np.einsum("...ij,jk->...ik", frac_disps, self._latvec)

    def diagonal_dist(self, d1):
        """Like general_dist, but assuming lattice vectors are diagonal -- orthogonal and aligned with xyz
        It's about 10x faster than the general one checking all 27 lattice points
        """
        for i in range(3):
            L = self._latvec[i, i]
            d1[..., i] = (d1[..., i] + L / 2) % L - L / 2
        return d1
