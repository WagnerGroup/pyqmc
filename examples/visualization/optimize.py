import pyqmc.api as pyq
from concurrent.futures import ProcessPoolExecutor

if __name__=="__main__":
    mol, mf, mc = pyq.recover_pyscf("benzene.hdf5", "benzene_hci.hdf5", cancel_outputs=False)
    mc.ci = mc.ci[0] # choose first root
    wf, to_opt = pyq.generate_wf(mol, mf,mc= mc)
    updater = pyq.SRUpdate(pyq.EnergyAccumulator(mol), pyq.LinearTransform(wf.parameters, to_opt))
    coords = pyq.initial_guess(mol, 1000)
    nworkers = 4
    with ProcessPoolExecutor(max_workers=4) as pool:
        pyq.optimize(wf, coords, updater, tstep=0.01, verbose=True,
                     client=pool, npartitions=nworkers, hdf_file = 'benzene_vmc.hdf5')
