from pyqmc import dasktools
from dask.distributed import Client, LocalCluster
import pyqmc

ncore = 2
nconfig = ncore * 400


def run_scf():
    from pyscf import gto, scf

    mol = gto.M(
        atom="O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587", basis="bfd_vtz", ecp="bfd"
    )
    mf = scf.RHF(mol).run()
    return mol, mf


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
    client = Client(cluster)
    mol, mf = run_scf()
    from pyqmc.dasktools import distvmc, line_minimization
    from pyqmc.dmc import rundmc
    import pandas as pd

    wf, to_opt = pyqmc.default_sj(mol, mf)
    df, coords = distvmc(
        wf, pyqmc.initial_guess(mol, nconfig), client=client, nsteps_per=10, nsteps=10
    )
    line_minimization(
        wf, coords, pyqmc.gradient_generator(mol, wf, to_opt), client=client
    )
    dfdmc, configs, weights = rundmc(
        wf,
        coords,
        nsteps=5000,
        branchtime=5,
        accumulators={"energy": pyqmc.EnergyAccumulator(mol)},
        ekey=("energy", "total"),
        tstep=0.02,
        verbose=True,
        propagate=pyqmc.dasktools.distdmc_propagate,
        client=client,
    )

    dfdmc = pd.DataFrame(dfdmc).to_json("dmc.json")
