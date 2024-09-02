import pyqmc.api as pyq
import pyscf
import time
import scipy.stats
import numpy as np

mol = pyscf.gto.M(atom="H 0. 0. 0.; Cl 0. 0. 2.", basis="ccecp-ccpvtz", ecp="ccecp", unit="bohr")
mf = pyscf.scf.RHF(mol)
mf.run()

wf_pyscf, to_opt = pyq.generate_wf(mol, mf, slater_kws=dict(evaluate_orbitals_with="pyscf"))
wf_numba, to_opt = pyq.generate_wf(mol, mf, slater_kws=dict(evaluate_orbitals_with="numba"))

print(wf_pyscf.wf_factors[0].orbitals.eval_gto)
print(wf_numba.wf_factors[0].orbitals.eval_gto)

enacc = pyq.EnergyAccumulator(mol)
configs = pyq.initial_guess(mol, 10)

# compile
pyq.vmc(wf_pyscf, configs, accumulators={"energy": enacc}, nsteps=1)
pyq.vmc(wf_numba, configs, accumulators={"energy": enacc}, nsteps=1)

def timevmc(wf, enacc, seed, nconfig=100):
    np.random.seed(seed)
    configs = pyq.initial_guess(mol, nconfig)
    t0 = time.perf_counter()
    pyq.vmc(wf, configs, accumulators={"energy": enacc})
    t1 = time.perf_counter()
    return t1 - t0

nreps = 10
t = dict(pyscf=np.zeros(nreps), numba=np.zeros(nreps))
for n in range(nreps):
    t["pyscf"][n] = timevmc(wf_pyscf, enacc, n)
    t["numba"][n] = timevmc(wf_numba, enacc, n)

for k, v in t.items():
    print(f"{k} \t{v.mean():.4f} \t{scipy.stats.sem(v):.4f}")
