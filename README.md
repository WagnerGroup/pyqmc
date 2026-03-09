![Python package](https://github.com/WagnerGroup/pyqmc/workflows/Python%20package/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyqmc/badge/?version=latest)](https://pyqmc.readthedocs.io/en/latest/?badge=latest)

# PyQMC

PyQMC is a Python package for real-space quantum Monte Carlo (QMC) electronic structure calculations, designed to interoperate closely with [PySCF](https://pyscf.org/).

Full documentation is available at [pyqmc.readthedocs.io](https://pyqmc.readthedocs.io/en/latest/).

## Features

- **Variational Monte Carlo (VMC)** and **Diffusion Monte Carlo (DMC)**
- **Wavefunction optimization** via line minimization and stochastic reconfiguration
- **Trial wavefunctions**: Slater-Jastrow, multi-determinant (CASSCF/selected CI), geminal, and three-body Jastrow
- **Observables**: energy, one- and two-body density matrices, extensible to your problem.
- **Periodic boundary conditions** with twist averaging and supercell support
- **Parallel execution** via MPI (`mpi4py`) or Dask
- **GPU and JAX backends** for high-performance evaluation
- HDF5-based checkpointing for restartable workflows

## Installation

```bash
pip install pyqmc
```
to get the latest development version, 
```bash 
pip install git+https://github.com/WagnerGroup/pyqmc.git
```

**Requirements:** Python >= 3.10, PySCF >= 2.8, SciPy, h5py, pandas.

## Quick Start

The high-level `recipes` interface handles a complete optimize -> VMC -> DMC workflow:

```python
import pyscf
import pyqmc.recipes

# 1. Run a DFT/HF calculation with PySCF and save a checkpoint
mol = pyscf.gto.M(atom="He 0. 0. 0.", basis="ccECP_cc-pVDZ", ecp="ccecp", unit="bohr")
mf = pyscf.scf.RHF(mol)
mf.chkfile = "he_dft.hdf5"
mf.kernel()

# 2. Optimize the Slater-Jastrow wavefunction
pyqmc.recipes.OPTIMIZE("he_dft.hdf5", "he_sj.hdf5", slater_kws={"optimize_orbitals": True})

# 3. Run VMC
pyqmc.recipes.VMC("he_dft.hdf5", "he_sj_vmc.hdf5", load_parameters="he_sj.hdf5", nblocks=40)

# 4. Run DMC
pyqmc.recipes.DMC("he_dft.hdf5", "he_sj_dmc.hdf5", load_parameters="he_sj.hdf5",
                  nblocks=4000, tstep=0.02)
```

Results are saved as HDF5 files and can be read back with `pyqmc.recipes.read_mc_output`.

## Parallel Execution

PyQMC supports parallelism via MPI using `mpi4py.futures` or any other futures object such as `dask` or `concurrent`:

```python
import mpi4py.futures
import pyqmc.recipes

if __name__ == "__main___":
    npartitions = 4
    with mpi4py.futures.MPIPoolExecutor(max_workers=npartitions) as client:
        pyqmc.recipes.OPTIMIZE("he_dft.hdf5", "he_sj.hdf5", client=client, npartitions=npartitions)
```

Run with: `mpiexec -n 5 python -m mpi4py.futures script.py`

## Package Structure

| Module | Description |
|---|---|
| `pyqmc.recipes` | High-level `OPTIMIZE`, `VMC`, `DMC` functions |
| `pyqmc.wf` | Wavefunctions: `Slater`, `JastrowSpin`, `MultiplyWF`, `AddWF`, geminal |
| `pyqmc.method` | Core algorithms: VMC, DMC, line minimization, variance optimization |
| `pyqmc.observables` | Energy, one body density matrix, two body density matrix, ECP |
| `pyqmc.pbc` | Periodic boundary conditions, supercell construction, twist averaging |
| `pyqmc.wf.jax` | JAX-based wavefunction and GTO evaluation |

## Citation

If you use PyQMC in your research, please cite the relevant papers listed in the [documentation](https://pyqmc.readthedocs.io/en/latest/).

## License

MIT License. Copyright (c) 2019-2026 The PyQMC Developers.
