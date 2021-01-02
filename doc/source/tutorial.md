Tutorial
----------------

Each of these snippets can be run in a separate 

### Mean-field

```
from pyscf import gto, scf

mol = gto.M(
    atom="""
    O 0.0000000, 0.000000, 0.00000000
    H 0.761561 , 0.478993, 0.00000000
    H -0.761561, 0.478993, 0.00000000""",
    basis="ccecp-ccpvtz",
    ecp={"O": "ccecp"},
)
mf = scf.RHF(mol)
mf.chkfile = "mf.chk"
mf.run()
```


### Optimize wave function

```
import pyqmc.recipes

pyqmc.recipes.OPTIMIZE("mf.chk", "opt.chk", nconfig=1000,
    linemin_kws={"max_iterations": 30 },
    slater_kws={"optimize_orbitals": False},
)
```

### VMC evaluation

```
import pyqmc.recipes

pyqmc.recipes.VMC("scf.chk", "vmc.chk", 
    start_from="opt.chk", 
    accumulators={"energy": True, "rdm1": True},
    vmc_kws={"nblocks": 100},
)
```

### DMC evaluation

```
import pyqmc.recipes

pyqmc.recipes.DMC("scf.chk", "dmc.chk", 
    start_from="opt.chk", 
    accumulators={"energy": True, "rdm1": True},
    vmc_kws={"nblocks": 100},
)
```
