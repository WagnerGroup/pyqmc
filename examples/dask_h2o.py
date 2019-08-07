from pyqmc import dasktools
from dask.distributed import Client, LocalCluster
import pyqmc

ncore = 2
nconfig = ncore*400

def generate_wfs():
    from pyscf import gto, scf
    import pyqmc
    mol = gto.M(
        atom="O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587", basis="bfd_vtz", ecp="bfd"
        )
    mf = scf.RHF(mol).run()
    wf=pyqmc.slater_jastrow(mol,mf)

    return mol,mf,wf



if __name__=="__main__":
    cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
    client = Client(cluster)
    mol,mf,wf=generate_wfs()
    from pyqmc.mc import vmc
    from pyqmc.dasktools import distvmc,line_minimization
    from pyqmc.dmc import rundmc
    from pyqmc import EnergyAccumulator
    df,coords=distvmc(wf,pyqmc.initial_guess(mol,nconfig),client=client,nsteps_per=10,nsteps=10)
    line_minimization(wf,coords,pyqmc.gradient_generator(mol,wf,["wf2acoeff", "wf2bcoeff"]),client=client)
    dfdmc, configs_, weights_ = rundmc(
        wf,
        coords,
        nsteps=5000,
        branchtime=5,
        accumulators={"energy": EnergyAccumulator(mol)},
        ekey=("energy", "total"),
        tstep=0.02,
        verbose=True,
        propagate=pyqmc.dasktools.distdmc_propagate,
        client=client,
    )

    dfdmc = pd.DataFrame(dfdmc).to_json("dmc.json")