import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from pyqmc import reblock
import pyqmc.api as pyq
import pytest
import uuid


@pytest.mark.slow
def test():
    """Ensure that DMC obtains the exact result for a hydrogen atom"""
    from pyscf import gto, scf
    from pyqmc.dmc import limdrift
    import pandas as pd

    mol = gto.M(atom="H 0. 0. 0.", basis="sto-3g", unit="bohr", spin=1)
    mf = scf.UHF(mol).run()
    nconf = 1000
    configs = pyq.initial_guess(mol, nconf)
    wf, _ = pyq.generate_wf(mol, mf, jastrow_kws=dict(na=0, nb=0))
    enacc = pyq.EnergyAccumulator(mol)

    warmup = 200
    dfdmc, configs_, weights_ = pyq.rundmc(
        wf,
        configs,
        nblocks=800 + warmup,
        nsteps_per_block=5,
        accumulators={"energy": enacc},
        ekey=("energy", "total"),
        tstep=0.01,
        verbose=True,
    )

    dfdmc = pd.DataFrame(dfdmc)
    dfdmc.sort_values("block", inplace=True)

    dfprod = dfdmc[dfdmc.block >= warmup]

    rb_summary = reblock.reblock_summary(
        dfprod[["energytotal", "energyei"]], 20, weights=dfprod["weight"]
    )
    energy, err = [rb_summary[v]["energytotal"] for v in ("mean", "standard error")]
    assert (
        np.abs(energy + 0.5) < 5 * err
    ), "energy not within {0} of -0.5: energy {1}".format(5 * err, np.mean(energy))


def test_dmc_restarts(H_pbc_sto3g_krks, nconf=10):
    """For PBCs, check to make sure there are no
    errors on restart."""
    mol, mf = H_pbc_sto3g_krks
    nconf = 10
    fname = "test_dmc_restart_" + str(uuid.uuid4())

    configs = pyq.initial_guess(mol, nconf)
    wf, _ = pyq.generate_wf(mol, mf, jastrow_kws=dict(na=0, nb=0))
    enacc = pyq.EnergyAccumulator(mol)
    pyq.rundmc(wf, configs, nblocks=2, hdf_file=fname, accumulators={"energy": enacc})
    pyq.rundmc(wf, configs, nblocks=4, hdf_file=fname, accumulators={"energy": enacc})
    pyq.rundmc(wf, configs, nblocks=4, hdf_file=fname, accumulators={"energy": enacc})
    os.remove(fname)


if __name__ == "__main__":
    test()
