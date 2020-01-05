from dask.distributed import Client, LocalCluster
import pyqmc
from pyscf import gto, scf, lo, mcscf, lib
import h5py 

def setuph2(chkfile, identity, r=2.0 / 0.529177, ncas=2):
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
    H S
    0.040680 1.00000000
    H S
    0.139013 1.00000000
    H P
    0.166430 1.00000000
    H P
    0.740212 1.00000000
"""
        )
    }
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
        atom=f"H 0. 0. 0.; H 0. 0. {r}", unit="bohr", basis=basis, ecp=ecp, verbose=1
    )
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()
    nelecas = (1, 1)
    mc = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
    mc.kernel()
    mf.output = None
    mf.stdout = None
    mf.chkfile = None
    mc.output = None
    mc.stdout = None
    # print(dir(mc))
    with h5py.File(chkfile) as f:
        f.attrs["uuid"] = identity
        f.attrs["r"] = r
        f.create_group("mc")
        f["mc/ncas"] = ncas
        f["mc/nelecas"] = list(nelecas)
        f["mc/ci"] = mc.ci


def pyqmc_from_hdf(chkfile):
    mol = lib.chkfile.load_mol(chkfile)
    mol.output = None
    mol.stdout = None

    mf = scf.RHF(mol)
    mf.__dict__.update(scf.chkfile.load(chkfile, "scf"))
    with h5py.File(chkfile, "r") as f:
        mc = mcscf.CASCI(mf, ncas=int(f["mc/ncas"][...]), nelecas=f["mc/nelecas"][...])
        mc.ci = f["mc/ci"][...]

    wf, to_opt, freeze = pyqmc.default_msj(mol, mf, mc)

    freeze["wf1det_coeff"][...] = False
    pgrad = pyqmc.gradient_generator(mol, wf, to_opt, freeze)

    return {
        "mol": mol,
        "mf": mf,
        "to_opt": to_opt,
        "freeze": freeze,
        "wf": wf,
        "pgrad": pgrad,
    }




ncore = 2
nconfig = ncore*400

if __name__=="__main__":
    cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
    client = Client(cluster)
    from pyqmc.dasktools import distvmc, line_minimization, optimize_orthogonal
    #from pyqmc.optimize_orthogonal import optimize_orthogonal
    from copy import deepcopy
    import os
    savefiles = {'mf':"test.chk", "linemin":"linemin.hdf5", "excited1":"excited1.hdf5",
    'excited2':'excited2.hdf5' } 

    for k, it in savefiles.items():
        if os.path.isfile(it):
            os.remove(it)

    # Run 
    setuph2(savefiles['mf'], "test")
    sys = pyqmc_from_hdf(savefiles['mf'])

    df,coords=distvmc(sys['wf'],pyqmc.initial_guess(sys['mol'],nconfig),client=client,nsteps=10)
    line_minimization(sys['wf'], coords, sys['pgrad'], hdf_file = savefiles['linemin'], client=client)

    # First excited state
    wfs = [sys['wf'], deepcopy(sys['wf'])]
    optimize_orthogonal(wfs, coords, sys['pgrad'], hdf_file = savefiles['excited1'],
        forcing=[5.0], Starget = [0.0], client=client ) 

    # Second excited state
    wfs.append(deepcopy(sys['wf']))
    optimize_orthogonal(wfs, coords, sys['pgrad'], hdf_file = savefiles['excited2'],
        forcing=[5.0, 5.0], Starget = [0.0, 0.0], client=client ) 

    
    client.close()
