import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pyscf.gto as gto
import pyscf.scf as scf
import pandas as pd
import pyqmc.api
import pyqmc.wftools as wftools
import numpy as np
from scipy.stats import sem


def test_linemin(H2_ccecp_uhf):
    """Optimize a Slater-Jastrow wave function and check that it's better than Hartree-Fock"""
    mol,mf =H2_ccecp_uhf
    wf, to_opt = wftools.generate_wf(mol, mf,jastrow=[wftools.generate_gps_jastrow,wftools.generate_jastrow])
    nconf = 1000
    configs = pyqmc.mc.initial_guess(mol, nconf)
    wf, dfgrad = pyqmc.api.line_minimization(
        wf, configs, pyqmc.api.gradient_generator(mol, wf, to_opt),max_iterations=50,verbose=True,hdf_file="opt_h2_bondlength_3_ccecp_gps_default_jastrow_wf.hdf"
    )


    mfen = mf.energy_tot()
    enfinal = dfgrad["energy"].values[-1]
    print("enfinal",enfinal)
    print("mf en",mfen)
    enfinal_err = dfgrad["energy_error"].values[-1]
    assert mfen > enfinal

  


if __name__ == "__main__":
    test_linemin()
