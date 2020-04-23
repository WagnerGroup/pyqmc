import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from pyqmc.mc import vmc, initial_guess
from pyscf import lib, gto, scf
from pyqmc.reblock import reblock
from pyqmc.slateruhf import PySCFSlaterUHF
from pyqmc.accumulators import EnergyAccumulator


def test_vmc():
    """
    Test that a VMC calculation of a Slater determinant matches Hartree-Fock within error bars.
    """
    nconf = 1000
    mol = gto.M(
        atom="Li 0. 0. 0.; Li 0. 0. 1.5", basis="cc-pvtz", unit="bohr", verbose=1
    )

    mf_rhf = scf.RHF(mol).run()
    mf_uhf = scf.UHF(mol).run()
    nsteps = 300
    warmup = 30

    for wf, mf in [
        (PySCFSlaterUHF(mol, mf_rhf), mf_rhf),
        (PySCFSlaterUHF(mol, mf_uhf), mf_uhf),
    ]:

        #Without blocks
        coords = initial_guess(mol, nconf)
        df, coords = vmc(
            wf, coords, nsteps=nsteps, accumulators={"energy": EnergyAccumulator(mol)}
        )

        df = pd.DataFrame(df)
        df = reblock(df["energytotal"][warmup:], 20)
        en = df.mean()
        err = df.sem()
        assert en - mf.energy_tot() < 5 * err, "pyscf {0}, vmc {1}, err {2}".format(
            mf.energy_tot(), en, err
        )

        #With blocks
        coords = initial_guess(mol, nconf)
        df, coords = vmc(
            wf, coords, nblocks=int(nsteps/30), nsteps_per_block=30, 
            accumulators={"energy": EnergyAccumulator(mol)}
        )

        df = pd.DataFrame(df)["energytotal"][int(warmup/30):]
        en = df.mean()
        err = df.sem()
        assert en - mf.energy_tot() < 5 * err, "pyscf {0}, vmc {1}, err {2}".format(
            mf.energy_tot(), en, err
        )


def test_accumulator():
    """ Tests that the accumulator gets inserted into the data output correctly.
    """
    import pandas as pd
    from pyqmc.mc import vmc, initial_guess
    from pyscf import gto, scf
    from pyqmc.energy import energy
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.accumulators import EnergyAccumulator

    mol = gto.M(
        atom="Li 0. 0. 0.; Li 0. 0. 1.5", basis="cc-pvtz", unit="bohr", verbose=5
    )
    mf = scf.RHF(mol).run()
    nconf = 5000
    wf = PySCFSlaterUHF(mol, mf)
    coords = initial_guess(mol, nconf)

    df, coords = vmc(
        wf, coords, nsteps=30, accumulators={"energy": EnergyAccumulator(mol)}
    )
    df = pd.DataFrame(df)
    eaccum = EnergyAccumulator(mol)
    eaccum_energy = eaccum(coords, wf)
    df = pd.DataFrame(df)
    print(df["energytotal"][29] == np.average(eaccum_energy["total"]))

    assert df["energytotal"][29] == np.average(eaccum_energy["total"])


if __name__ == "__main__":
    test_vmc()
    test_accumulator()
