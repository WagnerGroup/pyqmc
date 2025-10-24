from pyscf import gto, scf
import pyqmc.api as pyq
from rich import print
import numpy as np
from pyqmc.wf.geminaljastrow import GeminalJastrow

"""
Generate a Slater + 2-body Jastrow + geminal wave function for H2. 

Note that this can be used for any system, including periodic systems.
"""


def run_mf():
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. 1.4", ecp="ccecp", basis="ccecp-ccpvtz", unit="bohr"
    )
    mf = scf.RHF(mol)
    mf.chkfile = f"{__file__}.mf.hdf5"
    mf.kernel()
    return mf.chkfile


if __name__ == "__main__":
    chkfile = run_mf()
    mol, mf = pyq.recover_pyscf(chkfile)

    to_opts = [None] * 3
    slater, to_opts[0] = pyq.generate_slater(mol, mf)
    cusp, to_opts[1] = pyq.generate_jastrow(mol, na=1, nb=3)

    geminal = GeminalJastrow(mol)
    to_opts[2] = {"gcoeff": np.ones(geminal.parameters["gcoeff"].shape).astype(bool)}
    to_opt = {}
    for i, t_o in enumerate(to_opts):
        to_opt.update({f"wf{i + 1}" + k: v for k, v in t_o.items()})
    print("to_opt", to_opt["wf3gcoeff"].shape)

    wf = pyq.MultiplyWF(slater, cusp, geminal)

    #  Optimize Jastrow
    pgrad = pyq.gradient_generator(mol, wf, to_opt, eps=1e-3)
    coords = pyq.initial_guess(mol, nconfig=1000)
    pyq.line_minimization(wf, coords, pgrad, verbose=True, hdf_file=f"{__file__}.hdf5")
