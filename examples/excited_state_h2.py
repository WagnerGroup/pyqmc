from pyscf import lib, gto, scf
import pyscf.pbc
import numpy as np
import pyqmc.api as pyq
import pyqmc.accumulators
from rich import print
from pyqmc.optimize_excited_states import optimize
def H2_casci():
    mol = gto.M(atom="H 0. 0. 0.0; H 0. 0. 1.4",
            basis=f"ccpvtz",  
            unit="bohr", 
            charge=0, 
            spin=0, 
            verbose=1)  
    mf = scf.ROHF(mol).run()
    mc = pyscf.mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 4
    mc.kernel()
    return mol, mf, mc


def run_optimization_best_practice_2states(**kwargs):
    """
    First optimize the ground state and then optimize the excited
    states while fixing the 
    """

    mol, mf, mc = H2_casci()
    import copy
    mf.output=None
    mol.output=None
    mc.output=None
    mc.stdout=None
    mol.stdout=None
    mc.stdout=None
    nstates=2
    mcs = [copy.copy(mc) for i in range(nstates)]
    for i in range(nstates):
        mcs[i].ci = mc.ci[i]

    wfs = []
    to_opts = []
    for i in range(nstates):
        wf, to_opt = pyq.generate_wf(mol, mf,mc=mcs[i], slater_kws=dict(optimize_determinants=True))
        wfs.append(wf)
        to_opts.append(to_opt)
    configs = pyq.initial_guess(mol, 1000)

    pgrad1 = pyq.gradient_generator(mol, wfs[0], to_opt=to_opts[0])
    wfs[0], _= pyq.line_minimization(wfs[0], configs, pgrad1, verbose=True, max_iterations=10)

    for k in to_opts[0]:
        to_opts[0][k] = np.zeros_like(to_opts[0][k])
    to_opts[0]['wf1det_coeff'][0]=True #Bug workaround for linear transform
    for to_opt in to_opts[1:]:
        to_opt['wf1det_coeff'] = np.ones_like(to_opt['wf1det_coeff'])

    transforms = [pyqmc.accumulators.LinearTransform(wf.parameters,to_opt) for wf, to_opt in zip(wfs, to_opts)]
    for wf in wfs[1:]:
        for k in wf.parameters.keys():
            if 'wf2' in k:
                wf.parameters[k] = wfs[0].parameters[k].copy()
    _, configs = pyq.vmc(wfs[0], configs)
    energy =pyq.EnergyAccumulator(mol)
    return optimize(wfs, configs, energy, transforms, **kwargs)


if __name__=="__main__":
    from concurrent.futures import ProcessPoolExecutor
    #with ProcessPoolExecutor(max_workers=2) as client:
    client=None
    for norm_penalty in [0.005]:

            run_optimization_best_practice_2states(hdf_file=f'optimize_norm{norm_penalty}.hdf5', penalty=1.0, max_tstep=0.1, diagonal_approximation=False, norm_relative_penalty=norm_penalty, client=client, npartitions=2, nsteps=200)