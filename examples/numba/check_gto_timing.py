import pyqmc.api as pyq
import pyscf
import time
import scipy.stats
import numpy as np


def timevmc(wf, enacc, nconfig):
    configs = pyq.initial_guess(mol, nconfig)
    t0 = time.perf_counter()
    pyq.vmc(wf, configs, accumulators={"energy": enacc})
    t1 = time.perf_counter()
    return t1 - t0

def check_timing(eval_orbs_with, mol, mf, seed=1, nreps=10, nconfig=10):
    wf, _ = pyq.generate_wf(mol, mf, slater_kws=dict(evaluate_orbitals_with=eval_orbs_with))
    print(wf.wf_factors[0].orbitals.eval_gto)
    np.random.seed(seed)
    configs = pyq.initial_guess(mol, nconfig)
    enacc = pyq.EnergyAccumulator(mol)
    # compile
    pyq.vmc(wf, configs, accumulators={"energy": enacc}, nsteps=1)

    time_array = np.zeros(nreps)
    for n in range(nreps):
        time_array[n] = timevmc(wf, enacc, nconfig)
    return time_array

if __name__ == "__main__":

    mol = pyscf.gto.M(atom="H 0. 0. 0.; Cl 0. 0. 2.", basis="ccecp-ccpvtz", ecp="ccecp", unit="bohr")
    mf = pyscf.scf.RHF(mol)
    mf.run()

    t = dict()
    t["pyscf"] = check_timing("pyscf", mol, mf)
    t["numba"] = check_timing("numba", mol, mf)

    for k, v in t.items():
        print(f"{k} {len(v)} reps: \t{v.mean():.4f} \t{scipy.stats.sem(v):.4f}")
