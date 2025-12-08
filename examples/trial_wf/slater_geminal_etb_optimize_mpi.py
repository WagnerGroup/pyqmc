from pyscf import gto, scf
import pyqmc.api as pyq
from rich import print
import numpy as np
from pyqmc.wf.geminaljastrow import GeminalJastrow
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import h5py
from mpi4py.futures import MPIPoolExecutor
import os
import itertools
"""
Generate a Slater + 2-body Jastrow + geminal wave function for H2. 

Note that this can be used for any system, including periodic systems.
"""


def run_mf(chkfile):
    mol = gto.M(
        atom='O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587', ecp="ccecp", basis="ccecp-ccpvtz", unit="bohr"
    )
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()
    return mf.chkfile


def run_optimizer(n_s=2, alpha_s=0.2,
                  n_p=1, alpha_p=0.2,
                  n_d = 0, alpha_d = 0.2,
                  pool =None,
                  workers = 4):
    chkfile = f"{__file__}.mf.hdf5"
    if not os.path.exists(chkfile):
        chkfile = run_mf(chkfile)
    mol, mf = pyq.recover_pyscf(chkfile)

    to_opts = [None] * 3
    slater, to_opts[0] = pyq.generate_slater(mol, mf)
    cusp, to_opts[1] = pyq.generate_jastrow(mol, na=1, nb=3)

    mol_geminal = mol.copy()
    # here we use an even tempered Gaussian, which can be more efficient than
    # using the atomic basis.
    mol_geminal.basis = {'H': gto.etbs([
                         (0, n_s, alpha_s, 2), # s orbitals
                         (1, n_p, alpha_p, 2), # p orbitals
                         (2, n_d, alpha_d, 2) # p orbitals
                        ]),
                        'O': gto.etbs([
                            (0, n_s, alpha_s, 2), # s orbitals
                            (1, n_p, alpha_p, 2), # p orbitals
                            (2, n_d, alpha_d, 2) # p orbitals
                        ]),
    }
    mol_geminal.build()

    geminal = GeminalJastrow(mol_geminal)
    to_opts[2] = {"gcoeff": np.ones(geminal.parameters["gcoeff"].shape).astype(bool)}
    to_opt = {}
    for i, t_o in enumerate(to_opts):
        to_opt.update({f"wf{i + 1}" + k: v for k, v in t_o.items()})
    print("to_opt", to_opt["wf3gcoeff"].shape)

    wf = pyq.MultiplyWF(slater, cusp, geminal)

    #  Optimize Jastrow
    pgrad = pyq.gradient_generator(mol, wf, to_opt, eps=1e-3)
    coords = pyq.initial_guess(mol, nconfig=4000)
    hdf_file = f"data/slater_geminal_etb_{n_s}_{alpha_s}_{n_p}_{alpha_p}_{n_d}.hdf5"
    start = time.perf_counter()
    pyq.line_minimization(wf, coords, pgrad,
                          max_iterations=100,
                          verbose=False,
                          hdf_file=hdf_file,
                          client=pool, npartitions=workers)
    end = time.perf_counter()
    with h5py.File(hdf_file, "a") as f:
        f['time'] = end-start
if __name__ == "__main__":
    with MPIPoolExecutor(max_workers=4) as pool:
        with ThreadPoolExecutor(max_workers=10) as threader:
            runs =[]
            for n_s, n_p, n_d in itertools.product( [2], [3], [1,2,3]):
                print(f"submitting {n_s} {n_p} {n_d} ")
                runs.append(threader.submit(run_optimizer, n_s = n_s, n_p = n_p, n_d = n_d,
                                            pool = pool, workers=1))
            for run in runs:
                run.result()