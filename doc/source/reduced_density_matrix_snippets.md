Reduced density matrix snippets
------------------------------------


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