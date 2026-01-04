from pyscf import gto, scf
import pyqmc.api as pyq
from rich import print
from pyscf import fci, ao2mo, lib
import numpy as np

"""
Generate a Slater + 2-body Jastrow for H2. 
"""


def run_sci(cutoff = 0.01):
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. 1.4", ecp="ccecp", basis="ccecp-ccpvtz", unit="bohr"
    )
    mf = scf.RHF(mol)
    mf.chkfile = f"{__file__}.mf.hdf5"
    mf.kernel()


    cisolver = fci.SCI(mol)
    
    cisolver.select_cutoff=cutoff 
    #cisolver.nroots=nstates
    nmo = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = ao2mo.full(mol, mf.mo_coeff)
    e, civec = cisolver.kernel(h1, h2, nmo, nelec)
    
    lib.chkfile.save(mf.chkfile,'ci',
    {'ci':np.array(civec),
        'nmo':nmo,
        'nelec':nelec,
        '_strs':cisolver._strs,
        'select_cutoff':cutoff,
        'energy':e+mol.energy_nuc(),
    })

    return mf.chkfile


if __name__ == "__main__":
    chkfile = run_sci()
    mol, mf, mc = pyq.recover_pyscf(chkfile, ci_checkfile = chkfile)
    wf, to_opt = pyq.generate_wf(
        mol,
        mf,
        mc=mc,
        slater_kws=dict(
            optimize_orbitals=True, optimize_zeros=False, optimize_determinants=True
        ),  # control which parameters to optimize in to_opt.
        jastrow_kws=dict(na=2),
    )
    print(to_opt)
