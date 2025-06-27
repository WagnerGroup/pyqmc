import numpy as np
import pyscf
import pyqmc
import pyqmc.api as pyq


def evaluate_local_energy_variance(mol, wf, configs, use_old_ecp):
    # Evaluates Var(H*Psi(R)/Psi(R)) using <nconfig> different R
    eV_per_Har = 27.2114
    enacc = pyq.EnergyAccumulator(mol, use_old_ecp=use_old_ecp , threshold = -1)
    ens = enacc(configs, wf)
    return ens['ecp']

def test_accelerated_ecp():
    # Test that the accelerated ECP implementation gives the same results as the old one
    # for a simple molecule (H2) with a small basis set.
    mol = pyscf.gto.M(
        atom="H 0. 0. 0.; H 0. 0. 1.4", basis="ccECP_cc-pVTZ", ecp="ccecp", unit="bohr"
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10)
    num_random_walks = 2
    for i in range(num_random_walks):
        data, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)
        energies = {}
        for use_old_ecp in [False, True]:
            energies[use_old_ecp] = evaluate_local_energy_variance(
                mol, wf, configs, use_old_ecp
            )
        assert(np.allclose(energies[True],  energies[False]))

def test_accelerated_C(C_ccecp_rohf):
    mol, mf = C_ccecp_rohf
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10000)
    data, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)
    energies = {}
    for use_old_ecp in [False, True]:
        energies[use_old_ecp] = evaluate_local_energy_variance(
            mol, wf, configs, use_old_ecp
        )
    assert(np.allclose(np.mean(energies[True]),  np.mean(energies[False]), rtol=.01)), f"Accelerated ECP implementation does not match old implementation for C. {np.mean(energies[True])} != {np.mean(energies[False])}"



def test_accelerated_PBC(diamond_primitive):
    mol, mf = diamond_primitive
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10000)
    data, configs = pyqmc.method.mc.vmc(wf, configs, nblocks=1)
    energies = {}
    for use_old_ecp in [False, True]:
        energies[use_old_ecp] = evaluate_local_energy_variance(
            mol, wf, configs, use_old_ecp
        )
    assert(np.allclose(np.mean(energies[True]),  np.mean(energies[False]), rtol=.01)), f"Accelerated ECP implementation does not match old implementation for PBC system. {np.mean(energies[True])} != {np.mean(energies[False])}"
