import numpy as np
import os
import pyqmc.api as pyq
import copy
import h5py


def test_complex_linemin(H2_ccecp_rhf, optfile="linemin.hdf5"):
    """Test linemin for the case of complex orbital coefficients.
    We check whether it completes successfully and whether the energy has decreased.
    """
    mol, mf = H2_ccecp_rhf
    mf = copy.copy(mf)
    mol.output = None
    mol.stdout = None
    mf.output=None
    mf.stdout=None
    noise = (np.random.random(mf.mo_coeff.shape) - 0.5) * 0.2
    mf.mo_coeff = mf.mo_coeff * 1j + noise

    slater_kws = {"optimize_orbitals": True}
    wf, to_opt = pyq.generate_wf(mol, mf, slater_kws=slater_kws)

    configs = pyq.initial_guess(mol, 1000)
    acc = pyq.gradient_generator(mol, wf, to_opt)
    pyq.line_minimization(
        wf, configs, acc, verbose=True, hdf_file=optfile, max_iterations=5
    )
    assert os.path.isfile(optfile)
    with h5py.File(optfile, "r") as f:
        en = f["energy"][()]
    assert en[0] > en[-1]
    os.remove(optfile)
