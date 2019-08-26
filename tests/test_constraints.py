# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import pandas as pd
from pyscf import lib, gto, scf, mcscf
from pyqmc import gradient_generator, default_msj
import numpy as np 
from pyqmc.multislater import MultiSlater 
from pyqmc.multiplywf import MultiplyWF
from pyqmc.linemin import line_minimization
from pyqmc.mc import initial_guess

def test():
    #Default multi-Slater wave function
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf,ncas=2,nelecas=(1,1))
    mc.kernel()

    wf, to_opt, freeze = default_msj(mol, mf, mc)
    old_parms = wf.parameters

    nconf = 500
    wf, dfgrad, dfline = line_minimization(
        wf, initial_guess(mol, nconf),
        gradient_generator(mol, wf, to_opt = to_opt, freeze = freeze)
    )
    dfgrad = pd.DataFrame(dfgrad)
    dfgrad.to_pickle('const.pickle')
    mfen = mc.e_tot
    enfinal = dfgrad["en"].values[-1]
    enfinal_err = dfgrad["en_err"].values[-1]
    new_parms = wf.parameters
    
    print(enfinal, enfinal_err, mfen)
    assert mfen > enfinal 
    assert new_parms['wf1det_coeff'][0] == old_parms['wf1det_coeff'][0]
    assert np.sum(new_parms['wf2bcoeff'][0,:] - old_parms['wf2bcoeff'][0,:]) == 0

if __name__ == "__main__":
    test()
