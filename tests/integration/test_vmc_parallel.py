import pyqmc.recipes
import concurrent.futures
import os

ncore = 2
nconfig = ncore * 400


def run_scf(chkfile):
    from pyscf import gto, scf

    mol = gto.M(
        atom="H 0 0 0; H 0 0. 1.4", basis="ccecpccpvdz", ecp="ccecp", unit="bohr"
    )
    mf = scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()


def test_parallel():
    run_scf("h2.hdf5")
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncore) as client:
        pyqmc.recipes.OPTIMIZE(
            "h2.hdf5", "linemin.hdf5", nconfig=50, client=client, npartitions=ncore
        )
    assert os.path.isfile("linemin.hdf5")
    os.remove("h2.hdf5")
    os.remove("linemin.hdf5")
