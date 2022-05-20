import numpy as np


class RawDistance:
    """Compute distance vectors using open boundary conditions"""

    def __init__(self):
        """ """
        pass

    def dist_i(self, configs, vec):
        """returns a list of electron-electron distances from an electron at position 'vec'
        configs will most likely be [nconfig,electron,dimension], and vec will be [nconfig,dimension]
        """
        if len(vec.shape) == 3:
            v = vec.transpose((1, 0, 2))[:, :, np.newaxis]
        else:
            v = vec[:, np.newaxis, :]
        return v - configs

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

        Returns:

          dist: array of size nconf x n1*n2 x 3

          ij : list of size n1*n2 tuples that document i,j
        """
        n1 = config1.shape[1]
        n2 = config2.shape[1]
        if n1 == 0 or n2 == 0:
            return np.zeros((config1.shape[0], 0, 3)), []
        vs = []
        ij = []
        for i in range(n2):
            vs.append(self.dist_i(config1, config2[:, i, :]))
            ij.extend([(i, j) for j in range(n1)])
        vs = np.concatenate(vs, axis=1)

        return vs, ij


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
        diagonal = np.all(np.abs(latvec - np.diag(np.diagonal(latvec))) < ortho_tol)
        if diagonal:
            self.dist_i = self.diagonal_dist_i
        else:
            orthogonal = (
                np.abs(np.dot(latvec[0], latvec[1])) < ortho_tol
                and np.abs(np.dot(latvec[1], latvec[2])) < ortho_tol
                and np.abs(np.dot(latvec[2], latvec[0])) < ortho_tol
            )
            if orthogonal:
                self.dist_i = self.orthogonal_dist_i
                # print("Orthogonal lattics vectors")
            else:
                self.dist_i = self.general_dist_i
                # print("Non-orthogonal lattics vectors")
        self._latvec = latvec
        self._invvec = np.linalg.inv(latvec)
        # list of all 26 neighboring cells
        self.point_list = (
            np.array([m.ravel() for m in np.meshgrid(*[[0, 1, 2]] * 3)]).T - 1
        )
        self.shifts = np.dot(self.point_list, self._latvec)
        # TODO build a minimal list instead of using all 27

    def general_dist_i(self, configs, vec):
        """returns a list of electron-electron distances from an electron at position 'vec'
        configs will most likely be [nconfig,electron,dimension], and vec will be [nconfig,dimension]
        """
        if len(vec.shape) == 3:
            v = vec.transpose((1, 0, 2))[:, :, np.newaxis]
        else:
            v = vec[:, np.newaxis, :]
        d1 = v - configs
        shifts = self.shifts.reshape((-1, *[1] * (len(d1.shape) - 1), 3))
        d1all = d1[np.newaxis] + shifts
        dists = np.sum(d1all**2, axis=-1)
        mininds = np.argmin(dists, axis=0)
        inds = np.meshgrid(*[np.arange(n) for n in mininds.shape], indexing="ij")
        return d1all[(mininds, *inds)]

    def orthogonal_dist_i(self, configs, vec):
        """Like dist_i, but assuming lattice vectors are orthogonal
        It's about 10x faster than the general one checking all 27 lattice points
        """
        if len(vec.shape) == 3:
            v = vec.transpose((1, 0, 2))[:, :, np.newaxis]
        else:
            v = vec[:, np.newaxis, :]
        d1 = v - configs
        frac_disps = np.einsum("...ij,jk->...ik", d1, self._invvec)
        frac_disps = (frac_disps + 0.5) % 1 - 0.5
        return np.einsum("...ij,jk->...ik", frac_disps, self._latvec)

    def diagonal_dist_i(self, configs, vec):
        """Like dist_i, but assuming lattice vectors are orthogonal
        It's about 10x faster than the general one checking all 27 lattice points
        """
        if len(vec.shape) == 3:
            v = vec.transpose((1, 0, 2))[:, :, np.newaxis]
        else:
            v = vec[:, np.newaxis, :]
        d1 = v - configs
        for i in range(3):
            L = self._latvec[i, i]
            d1[..., i] = (d1[..., i] + L / 2) % L - L / 2
        return d1
