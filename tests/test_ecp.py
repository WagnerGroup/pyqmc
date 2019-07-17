import pandas as pd
from pyqmc.mc import vmc, initial_guess
from pyscf import gto, scf
from pyqmc.slateruhf import PySCFSlaterUHF
from pyqmc.accumulators import EnergyAccumulator
import numpy as np


def test_ecp():

    mol = gto.M(atom="C 0. 0. 0.", ecp="bfd", basis="bfd_vtz")
    mf = scf.RHF(mol).run()
    nconf = 5000
    wf = PySCFSlaterUHF(mol, mf)
    coords = initial_guess(mol, nconf)
    df, coords = vmc(
        wf, coords, nsteps=100, accumulators={"energy": EnergyAccumulator(mol)}
    )
    df = pd.DataFrame(df)
    warmup = 30
    print(
        "mean field",
        mf.energy_tot(),
        "vmc estimation",
        np.mean(df["energytotal"][warmup:]),
        np.std(df["energytotal"][warmup:]),
    )

    assert abs(mf.energy_tot() - np.mean(df["energytotal"][warmup:])) <= np.std(
        df["energytotal"][warmup:]
    )


if __name__ == "__main__":
    test_ecp()
