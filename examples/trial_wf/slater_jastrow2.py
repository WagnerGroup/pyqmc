from pyscf import gto, scf
import pyqmc.api as pyq
from rich import print

"""
Generate a Slater + 2-body Jastrow for H2. 
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
    # This will optimize orbitals, but leave orbital coefficients that are near zero constant.
    # No need to optimize determinants since there is just one.
    wf, to_opt = pyq.generate_wf(
        mol,
        mf,
        slater_kws=dict(
            optimize_orbitals=True, optimize_zeros=False, optimize_determinants=False
        ),  # control which parameters to optimize in to_opt.
        jastrow_kws=dict(na=2),
    )
    print(to_opt)
