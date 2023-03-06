HOWTOs
--------------------

### Periodic systems

```
def run_si_scf(chkfile="si_scf.chk", a=5.43):
    cell = gto.Cell(
        atom="Si 0. 0. 0.; Si {0} {0} {0}".format(a / 4),
        unit="angstrom",
        basis="ccecp-ccpvtz",
        ecp="ccecp",
        a=(np.ones((3, 3)) - np.eye(3)) * a / 2,
    )
    cell.exp_to_discard = 0.1
    cell.build()

    kpts = cell.make_kpts([8, 8, 8])
    mf = scf.KRKS(cell, kpts=kpts)
    mf.chkfile = chkfile
    mf.run()
```

Run QMC on `n_conventional` conventional unit cells of silicon.
```
import pyqmc.api as pyq
import numpy as np

def run_si_qmc(chkfile="si_scf.chk", n_conventional=2):
    # Define periodic supercell in PyQMC
    conventional_S = np.ones((3, 3)) - 2 * np.eye(3)
    S = n_conventional * conventional_S

    pyq.OPTIMIZE(chkfile, "si_opt.chk", S=S)
    pyq.DMC(chkfile, "si_dmc.chk", load_parameters="si_opt.chk", S=S, slater_kws={'twist':0} )
```

To get the available twists: 
```
import pyqmc.api as pyq
cell, mf = pyq.recover_pyqmc("si_scf.chk")
conventional_S = np.ones((3, 3)) - 2 * np.eye(3)
S = n_conventional * conventional_S
cell = pyq.get_supercell(cell, S)
twists=pyq.create_supercell_twists(cell,mf)
print(twists)
```



### Orbital optimization

```
pyq.OPTIMIZE(chkfile, "opt.chk",slater_kws={'optimize_orbitals':True,'optimize_zeros':False})
```

### Selected CI wave function

Most of the effort is in setting up and saving the CI coefficients correctly, which is done in `run_hci()` here. 
You can copy `run_hci()` and use it on any system.

If you are using pyscf 2.0 or above, you will need to run 
```
pip install git+git://github.com/pyscf/naive-hci
```
to get the simple selected CI method.

```
import pyqmc.api as pyq
import pyscf
import pyscf.hci
import numpy as np
from pyscf import gto, scf

def run_mf(chkfile):
    mol = gto.M(
        atom="""
        O 0.0000000, 0.000000, 0.00000000
        H 0.761561 , 0.478993, 0.00000000
        H -0.761561, 0.478993, 0.00000000""",
        basis="ccecp-ccpvdz",
        ecp={"O": "ccecp"},
    )
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.run()


def run_hci(hf_chkfile, chkfile, select_cutoff=0.1, nroots=4):
    mol, mf = pyqmc.recover_pyscf(hf_chkfile, cancel_outputs=False)
    cisolver = pyscf.hci.SCI(mol)
    cisolver.select_cutoff=select_cutoff
    cisolver.nroots=nroots
    nmo = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
    e, civec = cisolver.kernel(h1, h2, nmo, nelec)
    pyscf.lib.chkfile.save(chkfile,'ci',
    {'ci':np.array(civec),
        'nmo':nmo,
        'nelec':nelec,
        '_strs':cisolver._strs,
        'select_cutoff':select_cutoff,
        'energy':e+mol.energy_nuc(),
    })


if __name__=="__main__":
    run_mf("mf.chk")
    run_hci("mf.chk","hci.chk")
    pyq.OPTIMIZE("mf.chk", "opt.chk", ci_checkfile="hci.chk", nconfig=1000)
```

### Create a wavefunction with 3 body Jastrow factor.

```
import pyscf.gto as gto
import pyscf.scf as scf
import pyqmc.api as pyq
import pyqmc.wftools as wftools
import pyqmc.mc as mc

def linemin(mol,mf):
    #create wavefunction object. note jastrow_kws have to be passed even if empty
    wf, to_opt = wftools.generate_wf(mol, mf,jastrow=[wftools.generate_jastrow,wftools.generate_jastrow3],jastrow_kws=[{},{}]
)                   
    nconf = 100
    configs = mc.initial_guess(mol, nconf)
    wf, dfgrad = pyq.line_minimization(
        wf, configs, pyq.gradient_generator(mol, wf, to_opt),max_iterations=50,verbose=True,hdf_file='3_jastrow.chk'
    )

  
def H2_ccecp_uhf():
    r = 2
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. %g" % r,
        ecp="ccecp",
        basis="ccpvdz",
        unit="bohr",
        verbose=1,
    )
    mf = scf.UHF(mol).run()
    return mol, mf


if __name__ == "__main__":
    mol,mf = H2_ccecp_uhf()
    linemin(mol,mf)
```


### MPI parallelization

You must put the execution in an `if __name__=="__main__"` block; otherwise `mpi4py` will not work.

```
from mpi4py.futures import MPIPoolExecutor
import mpi4py.MPI
import pyqmc.recipes

if __name__=="__main__":
    comm = mpi4py.MPI.COMM_WORLD
    npartitions= comm.Get_size()-1
    with MPIPoolExecutor(max_workers=npartitions) as client:
        pyqmc.recipes.OPTIMIZE("h2o.hdf5", "h2o_opt_mpi.hdf5", client=client, npartitions=npartitions)
```

One of the MPI ranks gets used as the main thread. So if you want to run it on 2 processors, you can do:
```
mpiexec -n 3 python -m mpi4py.futures test_mpi.py
```


### Single node parallelization (without MPI)

```
from concurrent.futures import ProcessPoolExecutor
if __name__=="__main__":
    npartitions = 2
    with ProcessPoolExecutor(max_workers=npartitions) as client:
        pyqmc.recipes.OPTIMIZE("h2o.hdf5", "h2o_opt_mpi.hdf5", client=client, npartitions=npartitions)
```

### Define a new accumulator

This simple accumulator computes the average x,y, and z position of electrons. 
This is done in the `__call__` function; `configs.configs` is a numpy array of [walker, electron, xyz].
Note that if you run in parallel, you must define the class at the global scope. 

```
import numpy as np
class DipoleAccumulator:
    def __init__(self):
        pass

    def __call__(self, configs, wf):
        return {'electric_dipole':np.sum(configs.configs,axis=1) } 

    def shapes(self):
        return {"electric_dipole": (3,)}

    def avg(self, configs, wf):
        d = {}
        for k, it in self(configs, wf).items():
            d[k] = np.mean(it, axis=0)
        return d

    def keys(self):
        return self.shapes().keys()

import pyqmc.recipes
pyqmc.recipes.VMC("h2o.hdf5", "dipole.hdf5", 
                  load_parameters="h2o_sj_800.hdf5", 
                  accumulators={'extra_accumulators':{'dipole':DipoleAccumulator()}})
```



### Read in the 1-RDM

You should have computed the 1-RDM by setting `accumulators={'rdm1':True}` in pyqmc.recipes.VMC or DMC.

```
import pyqmc.obdm
def avg(vec):
    nblock = vec.shape[0]
    avg = np.mean(vec,axis=0)
    std = np.std(vec,axis=0)
    return avg, std/np.sqrt(nblock)

with h5py.File("h2o_sj_vmc.hdf5") as f:
    warmup=2
    en, en_err = avg(f['energytotal'][warmup:,...])
    rdm1, rdm1_err=avg(f['rdm1value'][warmup:,...])
    rdm1norm, rdm1norm_err = avg(f['rdm1norm'][warmup:,...])
    rdm1=pyqmc.obdm.normalize_obdm(rdm1,rdm1norm)
    rdm1_err=pyqmc.obdm.normalize_obdm(rdm1_err,rdm1norm)
```

### Compute the density from the 1-RDM

```
mol, mf = pyqmc.recover_pyscf("h2o.hdf5")
import pyscf.tools
ao_rdm1 = np.einsum('pi,ij,qj->pq', mf.mo_coeff, rdm1, mf.mo_coeff.conj())
resolution=0.05
dens=pyscf.tools.cubegen.density(mol, "h2o_sj_density.cube",ao_rdm1,resolution=resolution)
```

### Plot the density from a cube file

```
import ase.io
data = ase.io.cube.read_cube(open("h2o_sj_density.cube"))
print(data.keys())
print(data['data'].shape)
yzplane = data['data'][int(data['data'].shape[0]/2),:,:]
fig = plt.figure(figsize=(8,8))
vmax=np.max(yzplane)
plt.contourf(yzplane,levels=[vmax*x for x in np.logspace(-6,-1,80)], cmap='magma')
plt.xticks([])
plt.yticks([])
```


