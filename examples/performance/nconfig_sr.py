"""Self-contained example of the planning helpers, on H2.

Builds an H2 Slater-Jastrow wavefunction from scratch with PySCF + pyqmc (no
checkpoint files needed), then runs both planning helpers:

  1. benchmark_nconfig   -- per-configuration cost and memory vs nconfig
  2. recommend_optimizer -- peak memory / solve time of SR vs minSR vs CG-SR

Run:  python example_planning.py
"""
import pyscf.gto
import pyscf.scf
import pyqmc.api as pyq

from planning import benchmark_nconfig, recommend_optimizer
import numpy as np

def build_h2():
    """An H2 Slater-Jastrow wavefunction, built on the fly from a PySCF RHF.

    Returns (mol, wf, to_opt): the pyscf molecule, the pyqmc wavefunction, and
    its optimization mask. The default `generate_wf` gives a single-determinant
    Slater-Jastrow with the Jastrow coefficients marked optimizable.
    """
    mol = pyscf.gto.M(atom="H 0 0 0; H 0 0 1.4", basis="ccpvdz", unit="bohr")
    mf = pyscf.scf.RHF(mol).run()
    wf, to_opt = pyq.generate_wf(mol, mf)
    return mol, wf, to_opt


if __name__ == "__main__":
    mol, wf, to_opt = build_h2()

    print("\n== benchmark_nconfig: per-configuration cost & wf memory vs nconfig ==")
    print(benchmark_nconfig(mol, wf, nconfigs=[2**i for i in range(5,16)]).to_string(index=False))

    # For H2 (which is very small), on my system, ~8192 configurations 
    # is about when the cost per config starts leveling out.
    # This will depend on your system and the computer you are using.
    # This is how many configurations *per core* or *per GPU* that you are using.

    # To choose the SR solver, multiply the number of optimal configurations per core
    # by the number of cores.
    npartitions= 2
    nconfig_optimal = 8192*npartitions

    print("\n== recommend_optimizer: SR / minSR / CG-SR cost at nconfig=1000 ==")
    print(recommend_optimizer(to_opt, nconfig=nconfig_optimal).to_string(index=False))

    # With these settings, we should use the regular SR solver. 
