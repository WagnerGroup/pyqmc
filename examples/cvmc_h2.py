from pyqmc import dasktools
from dask.distributed import Client, LocalCluster
import pyqmc

ncore = 2
nconfig = 800

def generate_wf():
    from pyscf import gto, scf
    import pyqmc
    
     # ccECP from A. Annaberdiyev et al. Journal of Chemical Physics 149, 134108 (2018)
    basis = {
        "H": gto.basis.parse(
            """ 
    H S
23.843185 0.00411490
10.212443 0.01046440
4.374164 0.02801110
1.873529 0.07588620
0.802465 0.18210620
0.343709 0.34852140
0.147217 0.37823130
0.063055 0.11642410
"""
        )   
    }   
    """ 
H S
0.040680 1.00000000
H S
0.139013 1.00000000
H P
0.166430 1.00000000
H P
0.740212 1.00000000
"""
    ecp = { 
        "H": gto.basis.parse_ecp(
            """ 
    H nelec 0
H ul
1 21.24359508259891 1.00000000000000
3 21.24359508259891 21.24359508259891
2 21.77696655044365 -10.85192405303825
"""
        )   
    }   

    mol = gto.M(
        atom=f"H 0. 0. 0.; H 0. 0. 1.1", unit="bohr", basis=basis, ecp=ecp, verbose=5
    )
    mf = scf.RHF(mol).run()
    wf = pyqmc.slater_jastrow(mol, mf) 
    return mol, mf, wf

def generate_constraints(mol, mf, wf):
    from pyqmc.accumulators import LinearTransform, EnergyAccumulator
    from pyqmc.obdm import OBDMAccumulator
    from pyqmc.cvmc import DescriptorFromOBDM, PGradDescriptor
    from pyscf import lo

    mo_occ = mf.mo_coeff[:, mf.mo_occ > 0]
    a = lo.iao.iao(mol, mo_occ)
    a = lo.vec_lowdin(a, mf.get_ovlp())

    obdm_up = OBDMAccumulator(mol=mol, orb_coeff=a, nstep=5, spin=0)
    obdm_down = OBDMAccumulator(mol=mol, orb_coeff=a, nstep=5, spin=1)

    descriptors = {
        "t": [[(1.0, (0, 1)), (1.0, (1, 0))], [(1.0, (0, 1)), (1.0, (1, 0))]],
        "trace": [[(1.0, (0, 0)), (1.0, (1, 1))], [(1.0, (0, 0)), (1.0, (1, 1))]],
    }
    for i in [0, 1]:
        descriptors[f"nup{i}"] = [[(1.0, (i, i))], []]
        descriptors[f"ndown{i}"] = [[], [(1.0, (i, i))]]

    acc = PGradDescriptor(
        EnergyAccumulator(mol),
        LinearTransform(wf.parameters),
        [obdm_up, obdm_down],
        DescriptorFromOBDM(descriptors, norm=2.0),
    )

    return acc, descriptors

if __name__=="__main__":
    cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
    client = Client(cluster)
    mol,mf,wf=generate_wf()
    from pyqmc.mc import vmc
    from pyqmc.dasktools import distvmc, line_minimization, cvmc
    from pyqmc.dmc import rundmc
    from pyqmc import EnergyAccumulator
    import pandas as pd
    import time 

    df,coords=distvmc(wf,pyqmc.initial_guess(mol,nconfig),client=client,nsteps_per=40,nsteps=40)
    start = time.time()
    wf,datagrad,dataline=line_minimization(wf,coords,pyqmc.gradient_generator(mol,wf),client=client,maxiters=5)
    print(time.time() - start)
    acc, descriptors = generate_constraints(mol, mf, wf)
    
    forcing = {}
    obj = {}
    for k in descriptors:
        forcing[k] = 0.0
        obj[k] = 0.0

    forcing["t"] = 0.5
    forcing["trace"] = 0.5
    obj["t"] = 0.0
    obj["trace"] = 2.0

    datafile = "saveh2.json"

    wf, df = cvmc(
        wf,
        coords,
        acc,
        client=client,
        objective=obj,
        forcing=forcing,
        iters=50,
        tstep=0.1,
        datafile=datafile
    )
