import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pyscf
import pyqmc.multislater_orbs
import pyscf.hci


def avg(vec):
    nblock = vec.shape[0]
    avg = np.mean(vec, axis=0)
    std = np.std(vec, axis=0)
    return avg, std / np.sqrt(nblock)


def run_test():
    mol = pyscf.gto.M(
        atom="O 0. 0. 0.; H 0. 0. 2.0",
        basis="ccecpccpvtz",
        ecp="ccecp",
        unit="bohr",
        charge=-1,
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    e_hf = mf.energy_tot()
    cisolver = pyscf.hci.SCI(mol)
    cisolver.select_cutoff = 0.1
    nmo = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
    e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)
    cisolver.ci = civec[0]
    ci_energy = mf.energy_nuc() + e

    tol = 0.0
    configs = pyqmc.initial_guess(mol, 1000)
    wf = pyqmc.multislater_orbs.MultiSlater(mol, mf, cisolver, tol=tol)
    data, configs = pyqmc.vmc(
        wf,
        configs,
        nblocks=40,
        verbose=True,
        accumulators={"energy": pyqmc.EnergyAccumulator(mol)},
    )
    en, err = avg(data["energytotal"][1:])
    nsigma = 4
    assert len(wf.parameters["det_coeff"]) == len(cisolver.ci)
    assert en - nsigma * err < e_hf
    assert en + nsigma * err > ci_energy


if __name__ == "__main__":
    run_test()
