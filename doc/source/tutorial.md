Tutorial
---------------
We will step through a simple ground state energy calculation for H<sub>2</sub>O.

### Run Hartree-Fock

Here we run Hartree-Fock using `pyscf` and save the result in `mf.chk`.

```
import pyscf
mol = pyscf.gto.M(
            atom = "O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587", 
            basis='ccecpccpvdz', 
            ecp='ccecp', 
            unit='bohr'
)

mf = pyscf.scf.RHF(mol)
mf.chkfile = "mf.chk"
mf.kernel()
```

### Optimize a Slater-Jastrow wave function

We first do a rough optimization 
```
import pyqmc.recipes
pyqmc.recipes.OPTIMIZE("h2o.hdf5",
                       "h2o_sj_200.hdf5",
                       nconfig=200, 
                       linemin_kws={'max_iterations':10})
```

And then do a more precise optimization starting from the previous iteration by setting `start_from`.
```
pyqmc.recipes.OPTIMIZE("h2o.hdf5",
                       "h2o_sj_800.hdf5", 
                       start_from="h2o_sj_200.hdf5", 
                       nconfig=800, 
                       linemin_kws={'max_iterations':10})
```

### Check on optimization
We want to check whether our energy optimization has converged. 
We do that by reading the information out of the HDF files using `pyqmc.recipes.read_opt`, which returns a `pandas` dataframe that can be plotted easily.

```
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.concat([pyqmc.recipes.read_opt(f"h2o_sj_{n}.hdf5") for n in [200,800]])
g = sns.FacetGrid(hue='fname',data=df)
g.map(plt.errorbar,'iteration','energy','error', marker='o')
g.add_legend()
plt.savefig("optimization.pdf", bbox_inches='tight')
```

When you do this, note that the energy jumps around a little bit. 


### Evaluate the Slater-Jastrow energies

We would like to do a long Monte Carlo evaluation to see how good our wave functions are. 
In this case, we'll do it in one go by looping over `n`. 

```
import pyqmc.recipes
for n in [200,800]:
    pyqmc.recipes.VMC("h2o.hdf5",
                      f"h2o_sj_vmc_{n}.hdf5", 
                      start_from=f"h2o_sj_{n}.hdf5", 
                      vmc_kws=dict(nblocks=100))
```

This will save two evaluations in `h2o_sj_vmc_200.hdf5` and `h2o_sj_vmc_800.hdf5`.

### Plot the quality of wave functions.

Ideally we continue to increase the number of configurations until the wave function is fully optimized. 
We use `read_mc_output` in `pyqmc.recipes` to average the data.
For this example, we will stop at 800 configurations, but often it takes a few thousand to get mHartree accuracy.

```
df = pd.DataFrame([pyqmc.recipes.read_mc_output(f"h2o_sj_vmc_{n}.hdf5") for n in [200,800]])
df['nconfig'] = [int(x.split('_')[3].replace('.hdf5','')) for x in df['fname']]
print(df)
plt.errorbar("nconfig","energytotal","energytotal_err", data=df, marker='o')
plt.xlabel("nconfig")
plt.ylabel("energy (Ha)")
plt.savefig("energy.pdf", bbox_inches='tight')
```

### Run DMC on the best wave function

```
import pyqmc.recipes
pyqmc.recipes.DMC("h2o.hdf5",f"h2o_sj_dmc_800.hdf5", start_from=f"h2o_sj_800.hdf5")
```

### Check DMC warmup

To analyze the DMC resutls we can use `read_mc_output` as well. 

```
import pyqmc.recipes
import matplotlib.pyplot as plt
df = pd.DataFrame([pyqmc.recipes.read_mc_output("h2o_sj_dmc_800.hdf5", warmup=warmup) for warmup in [10,20,30,40,50, 60,70, 100, 150]])
plt.errorbar("warmup",'energytotal', 'energytotal_err',data=df, marker='o')
```