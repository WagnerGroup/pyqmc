import numpy as np
from pyqmc.distance import MinimalImageDistance, RawDistance
from pyqmc.pbc import enforce_pbc

class OpenConfigs:
    def __init__(self, configs):
        self.configs = configs
        self.dist = RawDistance()

    def make_irreducible(self, vec):
        """ 
          Input: a (nconfig, 3) vector 
          Output: a tuple with the same vector and None, with no extra information
        """
        return (vec,None)

    def move(self, e, vec, not_used, accept):
        """
        Change coordinates of one electron
        Args:
          e: int, electron index
          vec: (nconfig, 3) new coordinates
          not_used: not used, same interface as PeriodicConfigs
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept,e,:] = vec[accept,:]

    def resample(self, newinds):
        """
        Resample configs by new indices (e.g. for DMC branching)
        Args:
          newinds: (nconfigs,) array of indices
        """
        self.configs = self.configs[newinds]

class PeriodicConfigs: 
    def __init__(self, configs, lattice_vectors, wrap=None):
        self.configs = configs
        self.wrap = np.zeros(configs.shape) if wrap is None else wrap
        self.lvecs = lattice_vectors
        self.dist = MinimalImageDistance(lattice_vectors)

    def make_irreducible(self, vec):
        """ 
         Input: a (nconfig, 3) vector 
         Output: a tuple with the wrapped vector and the number of wraps
        """
        return enforce_pbc(self.lvecs, vec)

    def move(self, e, vec, wrap, accept):
        """
        Change coordinates of one electron
        Args:
          e: int, electron index
          vec: (nconfig, 3) new coordinates
          wrap: (nconfig, 3) number of times to wrap electron back into the cell to get to new configs 
          accept: (nconfig,) boolean for which configs to update
        """
        self.configs[accept,e,:] = vec[accept,:]
        self.wrap[accept,e,:] += wrap[accept,:]

    def resample(self, newinds):
        """
        Resample configs by new indices (e.g. for DMC branching)
        Args:
          newinds: (nconfigs,) array of indices
        """
        self.configs = self.configs[newinds]
        self.wrap = self.wrap[newinds]


def test():
    from pyscf.pbc import gto, scf
    import pyqmc
    import pandas as pd

    L = 4 
    mol = gto.M(
        atom = '''H     {0}      {0}      {0}'''.format(0.0),
        basis='sto-3g',
        a = np.eye(3)*L,
        spin=1,
        unit='bohr',
    )
    mf = scf.UKS(mol)
    mf.xc = "pbe"
    mf = mf.density_fit().run()
    wf = pyqmc.PySCFSlaterUHF(mol, mf) 

    #####################################
    ## evaluate KE in PySCF
    #####################################
    ke_mat = mol.pbc_intor('int1e_kin', hermi=1, kpts=np.array([0,0,0]))
    dm = mf.make_rdm1() 
    pyscfke = np.einsum('ij,ji',ke_mat, dm[0])
    print('PySCF kinetic energy: {0}'.format(pyscfke))

    #####################################
    ## evaluate KE integral on grid
    #####################################
    X = np.linspace(0, 1, 20, endpoint=False)
    XYZ = np.meshgrid(X, X, X, indexing='ij')
    pts = [np.outer(p.ravel(), mol.a[i]) for i,p in enumerate(XYZ)]
    coords = np.sum(pts, axis=0).reshape((-1,1,3))
    
    phase, logdet = wf.recompute(coords)
    psi = phase*np.exp(logdet)
    lap = wf.laplacian(0, coords.reshape((-1,3)))
    gridke = np.sum(-0.5*lap*psi**2)/np.sum(psi**2)
    print('grid kinetic energy: {0}'.format(gridke))

    #####################################
    ## evaluate KE integral with VMC
    #####################################
    coords = pyqmc.initial_guess(mol, 600, .7)
    coords = PeriodicConfigs(coords, mol.a)
    warmup = 10
    df, coords = pyqmc.vmc(
        wf, coords, nsteps=128+warmup, tstep=L*0.6, accumulators={"energy": pyqmc.accumulators.EnergyAccumulator(mol)}
    )
    df = pd.DataFrame(df)
    reblocked = pyqmc.reblock.optimally_reblocked(df["energyke"][warmup:])
    print('VMC kinetic energy: {0} $\pm$ {1}'.format(reblocked['mean'],reblocked['standard error']))


if __name__=="__main__":
    test()
