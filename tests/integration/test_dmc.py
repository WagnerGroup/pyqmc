# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from pyqmc import reblock
import pyqmc.api as pyq
import pytest


@pytest.mark.slow
def test():
    """ Ensure that DMC obtains the exact result for a hydrogen atom """
    from pyscf import gto, scf
    from pyqmc.dmc import limdrift
    import pandas as pd

    mol = gto.M(atom="H 0. 0. 0.", basis="sto-3g", unit="bohr", spin=1)
    mf = scf.UHF(mol).run()
    nconf = 1000
    configs = pyq.initial_guess(mol, nconf)
    wf, _ = pyq.generate_wf(mol, mf, jastrow_kws=dict(na=0, nb=0))
    enacc = pyq.EnergyAccumulator(mol)

    dfvmc, configs_ = pyq.vmc(
        wf, configs, nsteps=50, accumulators={"energy": enacc}
    )
    dfvmc = pd.DataFrame(dfvmc)
    print(
        "vmc energy",
        np.mean(dfvmc["energytotal"]),
        np.std(dfvmc["energytotal"]) / np.sqrt(len(dfvmc)),
    )

    warmup = 200
    branchtime = 5
    dfdmc, configs_, weights_ = pyq.rundmc(
        wf,
        configs,
        nsteps=4000 + warmup * branchtime,
        branchtime=branchtime,
        accumulators={"energy": enacc},
        ekey=("energy", "total"),
        tstep=0.01,
        verbose=True,
    )

    dfdmc = pd.DataFrame(dfdmc)
    dfdmc.sort_values("step", inplace=True)

    dfprod = dfdmc[dfdmc.step >= warmup]

    rb_summary = reblock.reblock_summary(dfprod[["energytotal", "energyei"]], 20, weights=dfprod["weight"])
    print(rb_summary)
    energy, err = [rb_summary[v]["energytotal"] for v in ("mean", "standard error")]
    assert (
        np.abs(energy + 0.5) < 5 * err
    ), "energy not within {0} of -0.5: energy {1}".format(5 * err, np.mean(energy))


if __name__ == "__main__":
    test()
