import numpy as np
from pyqmc.distance import MinimalImageDistance, RawDistance
from pyqmc.pbc import enforce_pbc
import copy


class OpenConfigs:
    def __init__(self, configs):
        self.configs = configs
        self.dist = RawDistance()

    def electron(self, e):
        return OpenConfigs(self.configs[:, e])

    def mask(self, mask):
        return OpenConfigs(self.configs[mask])

    def make_irreducible(self, e, vec):
        """ 
          Input: 
            e: unused electron index
            vec: a (nconfig, 3) vector 
          Output: OpenConfigs object with just one electron
        """
        return OpenConfigs(vec)

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

    def join(self, configslist):
        """
        Merge configs into this object to collect from parallelization
        Args:
          configslist: list of OpenConfigs objects; total number of configs must match
        """
        self.configs[:] = np.concatenate([c.configs for c in configslist], axis=0)[:]

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
        self.configs = np.array(hdf["configs"])


class PeriodicConfigs:
    def __init__(self, configs, lattice_vectors, wrap=None):
        configs, wrap_ = enforce_pbc(lattice_vectors, configs)
        self.configs = configs
        self.wrap = wrap_
        if wrap is not None:
            self.wrap += wrap
        self.lvecs = lattice_vectors
        self.dist = MinimalImageDistance(lattice_vectors)

    def electron(self, e):
        return PeriodicConfigs(self.configs[:, e], self.lvecs, wrap=self.wrap[:, e])

    def mask(self, mask):
        return PeriodicConfigs(self.configs[mask], self.lvecs, wrap=self.wrap[mask])

    def make_irreducible(self, e, vec):
        """ 
         Input: a (nconfig, 3) vector 
         Output: a tuple with the wrapped vector and the number of wraps
        """
        epos, wrap = enforce_pbc(self.lvecs, vec)
        currentwrap = self.wrap if len(self.wrap.shape) == 2 else self.wrap[:, e]
        if len(vec.shape) == 3:
            currentwrap = currentwrap[:, np.newaxis]
        return PeriodicConfigs(epos, self.lvecs, wrap=wrap + currentwrap)

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

    def join(self, configslist):
        """
        Merge configs into this object to collect from parallelization
        Args:
          configslist: list of PeriodicConfigs objects; total number of configs must match
        """
        self.configs[:] = np.concatenate([c.configs for c in configslist], axis=0)[:]
        self.wrap[:] = np.concatenate([c.wrap for c in configslist], axis=0)[:]

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
        hdf["configs"].resize(self.wrap.shape)
        hdf["configs"][:, :, :] = self.configs
        hdf["wrap"][:, :, :] = self.wrap

    def load_hdf(self, hdf):
        self.configs = np.array(hdf["configs"])
        self.wrap = np.array(hdf["wrap"])
