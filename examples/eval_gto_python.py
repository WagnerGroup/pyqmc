import pyqmc.api as pyq
import pyscf
import time
import scipy.stats
import numpy as np

mol = pyscf.gto.M(atom="H 0. 0. 0.; Cl 0. 0. 2.", basis="ccecp-ccpvtz", ecp="ccecp", unit="bohr")
mf = pyscf.scf.RHF(mol)
mf.run()
enacc = pyq.EnergyAccumulator(mol)
configs = pyq.initial_guess(mol, 10)

# Use pyscf's eval_gto() function
wf, to_opt = pyq.generate_wf(mol, mf, slater_kws=dict(evaluate_orbitals_with="pyscf"))
print(wf.wf_factors[0].orbitals.eval_gto)
t0 = time.perf_counter()
pyq.vmc(wf, configs, accumulators={"energy": enacc}, nblocks=1)
print("time pyscf", time.perf_counter() - t0)

# Use pyqmc's numba implementation. Sometimes faster
wf, _ = pyq.generate_wf(mol, mf, slater_kws=dict(evaluate_orbitals_with="numba"))
print(wf.wf_factors[0].orbitals.eval_gto)
# The first time you call numba functions, they must compile, so this might take a bit longer than eval_gto() for a short run
for i in range(2):
    t0 = time.perf_counter()
    pyq.vmc(wf, configs, accumulators={"energy": enacc}, nblocks=1)
    print("time numba", i, time.perf_counter() - t0)
