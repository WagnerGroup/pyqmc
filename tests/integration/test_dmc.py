# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import pyqmc.testwf as testwf
import pytest
from pyqmc import reblock


def test():
    """ Ensure that DMC obtains the exact result for a hydrogen atom """
    from pyscf import lib, gto, scf
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.dmc import limdrift, rundmc
    from pyqmc.mc import vmc
    from pyqmc.accumulators import EnergyAccumulator
    from pyqmc.func3d import CutoffCuspFunction
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.coord import OpenConfigs
    import pandas as pd

    mol = gto.M(atom="H 0. 0. 0.", basis="sto-3g", unit="bohr", spin=1)
    mf = scf.UHF(mol).run()
    nconf = 1000
    configs = OpenConfigs(np.random.randn(nconf, 1, 3))
    wf1 = PySCFSlaterUHF(mol, mf)
    wf = wf1
    wf2 = JastrowSpin(mol, a_basis=[CutoffCuspFunction(5, 0.2)], b_basis=[])
    wf2.parameters["acoeff"] = np.asarray([[[1.0, 0]]])
    wf = MultiplyWF(wf1, wf2)

    dfvmc, configs_ = vmc(
        wf, configs, nsteps=50, accumulators={"energy": EnergyAccumulator(mol)}
    )
    dfvmc = pd.DataFrame(dfvmc)
    print(
        "vmc energy",
        np.mean(dfvmc["energytotal"]),
        np.std(dfvmc["energytotal"]) / np.sqrt(len(dfvmc)),
    )

    warmup = 200
    dfdmc, configs_, weights_ = rundmc(
        wf,
        configs,
        nsteps=4000 + warmup,
        branchtime=5,
        accumulators={"energy": EnergyAccumulator(mol)},
        ekey=("energy", "total"),
        tstep=0.005,
        drift_limiter=limdrift,
        verbose=False,
    )

    dfdmc = pd.DataFrame(dfdmc)
    dfdmc.sort_values("step", inplace=True)

    dfprod = dfdmc[dfdmc.step >= warmup]

    rb_summary = reblock.reblock_summary(dfprod[["energytotal", "energyei"]], 20)
    print(rb_summary)
    energy, err = [rb_summary[v]["energytotal"] for v in ("mean", "standard error")]
    assert (
        np.abs(energy + 0.5) < 5 * err
    ), "energy not within {0} of -0.5: energy {1}".format(5 * err, np.mean(energy))


if __name__ == "__main__":
    test()
