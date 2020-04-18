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
        return [OpenConfigs(c) for c in np.split(self.configs, npartitions)]

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
        hdf.create_dataset("configs", self.configs.shape)

    def to_hdf(self, hdf):
        hdf["configs"][:, :, :] = self.configs

    def load_hdf(self, hdf):
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
        clist = np.split(self.configs, npartitions)
        wlist = np.split(self.wrap, npartitions)
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
        hdf.create_dataset("configs", self.configs.shape)
        hdf.create_dataset("wrap", self.wrap.shape)

    def to_hdf(self, hdf):
        hdf["configs"][:, :, :] = self.configs
        hdf["wrap"][:, :, :] = self.wrap

    def load_hdf(self, hdf):
        self.configs = np.array(hdf["configs"])
        self.wrap = np.array(hdf["wrap"])


def test():
    from pyscf.pbc import gto, scf
    import pyqmc
    import pandas as pd

    L = 4
    mol = gto.M(
        atom="""H     {0}      {0}      {0}""".format(0.0),
        basis="sto-3g",
        a=np.eye(3) * L,
        spin=1,
        unit="bohr",
    )
    mf = scf.UKS(mol)
    mf.xc = "pbe"
    mf = mf.density_fit().run()
    wf = pyqmc.PySCFSlaterUHF(mol, mf)

    #####################################
    ## evaluate KE in PySCF
    #####################################
    ke_mat = mol.pbc_intor("int1e_kin", hermi=1, kpts=np.array([0, 0, 0]))
    dm = mf.make_rdm1()
    pyscfke = np.einsum("ij,ji", ke_mat, dm[0])
    print("PySCF kinetic energy: {0}".format(pyscfke))

    #####################################
    ## evaluate KE integral on grid
    #####################################
    X = np.linspace(0, 1, 20, endpoint=False)
    XYZ = np.meshgrid(X, X, X, indexing="ij")
    pts = [np.outer(p.ravel(), mol.a[i]) for i, p in enumerate(XYZ)]
    coords = np.sum(pts, axis=0).reshape((-1, 1, 3))

    phase, logdet = wf.recompute(coords)
    psi = phase * np.exp(logdet)
    lap = wf.laplacian(0, coords.reshape((-1, 3)))
    gridke = np.sum(-0.5 * lap * psi ** 2) / np.sum(psi ** 2)
    print("grid kinetic energy: {0}".format(gridke))

    #####################################
    ## evaluate KE integral with VMC
    #####################################
    coords = pyqmc.initial_guess(mol, 600, 0.7)
    coords = PeriodicConfigs(coords, mol.a)
    warmup = 10
    df, coords = pyqmc.vmc(
        wf,
        coords,
        nsteps=128 + warmup,
        tstep=L * 0.6,
        accumulators={"energy": pyqmc.accumulators.EnergyAccumulator(mol)},
    )
    df = pd.DataFrame(df)
    reblocked = pyqmc.reblock.optimally_reblocked(df["energyke"][warmup:])
    print(
        "VMC kinetic energy: {0} $\pm$ {1}".format(
            reblocked["mean"], reblocked["standard error"]
        )
    )


if __name__ == "__main__":
    test()
