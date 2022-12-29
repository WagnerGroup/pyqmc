import numpy as np
import pyqmc.distance as distance
import pyqmc.pbc as pbc
import copy


class OpenElectron:
    def __init__(self, epos, dist):
        self.configs = epos
        self.dist = dist


class OpenConfigs:
    def __init__(self, configs):
        self.configs = configs
        self.dist = distance.RawDistance()

    def electron(self, e):
        return OpenElectron(self.configs[:, e], self.dist)

    def mask(self, mask):
        return OpenConfigs(self.configs[mask])

    def make_irreducible(self, e, vec, mask=True):
        """
        Input:
          e: unused electron index
          vec: a (nconfig, 3) vector
        Output: OpenConfigs object with just one electron
        """
        return OpenElectron(vec, self.dist)

    def move(self, e, new, accept):
        """
        Change coordinates of one electron
        Args:
          e: int, electron index
          new: OpenConfigs with (nconfig, 3) new coordinates
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept, e, :] = new.configs[accept, :]

    def move_all(self, new, accept):
        """
        Change coordinates of all electrons
        Args:
          new: OpenConfigs with configs.shape new coordinates
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept] = new.configs[accept]

    def resample(self, newinds):
        """
        Resample configs by new indices (e.g. for DMC branching)
        Args:
          newinds: (nconfigs,) array of indices
        """
        self.configs = self.configs[newinds]

    def split(self, npartitions):
        """
        Split configs into npartitions new configs objects for parallelization
        Args:
          npartitions: int, number of partitions to divide configs into
        Returns:
          configslist: list of new configs objects
        """
        return [OpenConfigs(c) for c in np.array_split(self.configs, npartitions)]

    def join(self, configslist, axis=0):
        """
        Merge configs into this object to collect from parallelization
        Args:
          configslist: list of OpenConfigs objects
        """
        self.configs = np.concatenate([c.configs for c in configslist], axis=axis)

    def copy(self):
        return copy.deepcopy(self)

    def reshape(self, shape):
        self.configs = self.configs.reshape(shape)

    def initialize_hdf(self, hdf):
        hdf.create_dataset(
            "configs",
            self.configs.shape,
            chunks=True,
            maxshape=(None, *self.configs.shape[1:]),
        )

    def to_hdf(self, hdf):
        hdf["configs"].resize(self.configs.shape)
        hdf["configs"][...] = self.configs

    def load_hdf(self, hdf):
        """Note that the number of configurations will change to reflect the number in the hdf file."""
        # The ... seems to be necessary to avoid changing the dtype and screwing up
        # pyscf's calls.
        self.configs[...] = np.array(hdf["configs"])


class PeriodicElectron:
    """
    Represents the coordinates of a test electron position, for many walkers and
    potentially several different points.

    configs is a 2D or 3D vector with elements [config, point, dimension]
    wrap is same shape as configs
    lvec and dist will most likely be references to the parent object
    """

    def __init__(self, epos, lattice_vectors, dist, wrap=None):
        self.configs = epos
        self.lvec = lattice_vectors
        self.wrap = wrap if wrap is not None else np.zeros_like(epos)
        self.dist = dist


class PeriodicConfigs:
    def __init__(self, configs, lattice_vectors, wrap=None):
        configs, wrap_ = pbc.enforce_pbc(lattice_vectors, configs)
        self.configs = configs
        self.wrap = wrap_
        if wrap is not None:
            self.wrap += wrap
        self.lvecs = lattice_vectors
        self.dist = distance.MinimalImageDistance(lattice_vectors)

    def electron(self, e):
        return PeriodicElectron(
            self.configs[:, e], self.lvecs, self.dist, wrap=self.wrap[:, e]
        )

    def mask(self, mask):
        return PeriodicConfigs(self.configs[mask], self.lvecs, wrap=self.wrap[mask])

    def make_irreducible(self, e, vec, mask=None):
        """
        Input: a (nconfig, 3) vector or a (nconfig, N, 3) vector
        Output: A Periodic Electron
        """
        if mask is None:
            mask = np.ones(vec.shape[0:-1], dtype=bool)
        epos_, wrap_ = pbc.enforce_pbc(self.lvecs, vec[mask])
        epos = vec.copy()
        epos[mask] = epos_
        wrap = self.wrap[:, e, :].copy()
        if len(vec.shape) == 3:
            wrap = np.repeat(self.wrap[:, e][:, np.newaxis], vec.shape[1], axis=1)
        wrap[mask] += wrap_
        return PeriodicElectron(epos, self.lvecs, wrap=wrap, dist=self.dist)

    def move(self, e, new, accept):
        """
        Change coordinates of one electron
        Args:
          e: int, electron index
          new: PeriodicConfigs with (nconfig, 3) new coordinates
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept, e, :] = new.configs[accept, :]
        self.wrap[accept, e, :] = new.wrap[accept, :]

    def move_all(self, new, accept):
        """
        Change coordinates of all electrons
        Args:
          new: PeriodicConfigs with configs.shape new coordinates
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept] = new.configs[accept]
        self.wrap[accept] = new.wrap[accept]

    def resample(self, newinds):
        """
        Resample configs by new indices (e.g. for DMC branching)
        Args:
          newinds: (nconfigs,) array of indices
        """
        self.configs = self.configs[newinds]
        self.wrap = self.wrap[newinds]

    def split(self, npartitions):
        """
        Split configs into npartitions new configs objects for parallelization
        Args:
          npartitions: int, number of partitions to divide configs into
        Returns:
          configslist: list of new configs objects
        """
        clist = np.array_split(self.configs, npartitions)
        wlist = np.array_split(self.wrap, npartitions)
        return [PeriodicConfigs(c, self.lvecs, w) for c, w in zip(clist, wlist)]

    def join(self, configslist, axis=0):
        """
        Merge configs into this object to collect from parallelization
        Args:
          configslist: list of PeriodicConfigs objects
        """
        self.configs = np.concatenate([c.configs for c in configslist], axis=axis)
        self.wrap = np.concatenate([c.wrap for c in configslist], axis=axis)

    def copy(self):
        return copy.deepcopy(self)

    def reshape(self, shape):
        self.configs = self.configs.reshape(shape)
        self.wrap = self.wrap.reshape(shape)

    def initialize_hdf(self, hdf):
        hdf.create_dataset(
            "configs",
            self.configs.shape,
            chunks=True,
            maxshape=(None, *self.configs.shape[1:]),
        )
        hdf.create_dataset(
            "wrap", self.wrap.shape, chunks=True, maxshape=(None, *self.wrap.shape[1:])
        )

    def to_hdf(self, hdf):
        hdf["configs"].resize(self.configs.shape)
        hdf["configs"][...] = self.configs
        hdf["wrap"].resize(self.wrap.shape)
        hdf["wrap"][...] = self.wrap

    def load_hdf(self, hdf):
        # The ... seems to be necessary to avoid changing the dtype and screwing up
        # pyscf's calls.
        self.configs[...] = hdf["configs"][()]
        self.wrap[...] = hdf["wrap"][()]
