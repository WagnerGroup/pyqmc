import numpy as np
import os
import pyqmc.api as pyq
import pytest


def run_scf(chkfile):
    from pyscf import gto, scf
    from pyscf.scf.addons import remove_linear_dep_

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4", basis="ccecpccpvdz", ecp="ccecp", unit="bohr"
    )
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf = remove_linear_dep_(mf)
    energy = mf.kernel()


@pytest.mark.slow
def test():
    chkfile = "h2.hdf5"
    optfile = "linemin.hdf5"
    run_scf(chkfile)
    mol, mf = pyq.recover_pyscf(chkfile)
    noise = (np.random.random(mf.mo_coeff.shape) - 0.5) * 0.2
    mf.mo_coeff = mf.mo_coeff * 1j + noise

    slater_kws = {"optimize_orbitals": True}
    wf, to_opt = pyq.generate_wf(mol, mf, slater_kws=slater_kws)

    configs = pyq.initial_guess(mol, 100)
    acc = pyq.gradient_generator(mol, wf, to_opt)
    pyq.line_minimization(
        wf, configs, acc, verbose=True, hdf_file=optfile, max_iterations=5
    )

    assert os.path.isfile(optfile)
    os.remove(chkfile)
    os.remove(optfile)


if __name__ == "__main__":
    test()
