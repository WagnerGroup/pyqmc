Snippets
--------------------

### Periodic systems

```
def run_si_scf(chkfile="si_scf.chk"):
    a = 5.43
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

```
import pyqmc.recipes
import numpy as np

def run_si_qmc(chkfile="si_scf.chk"):
    # Define periodic supercell in PyQMC
    conventional_S = np.ones((3, 3)) - 2 * np.eye(3)
    S = 2 * conventional_S
    pyqmc.recipes.OPTIMIZE(chkfile, "si_opt.chk", S=S)
    pyqmc.recipes.DMC(chkfile, "si_dmc.chk", start_from="si_opt.chk", S=S)
```