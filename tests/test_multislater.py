import numpy as np

def test():
    from pyscf import lib, gto, scf, mcscf
    import pyqmc.testwf as testwf
    from pyqmc.mc import vmc, initial_guess
    from pyqmc.accumulators import EnergyAccumulator
    import pandas as pd
    from pyqmc.multislater import MultiSlater

    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
    for mf in [scf.RHF(mol).run(), scf.ROHF(mol).run(), scf.UHF(mol).run()]:
        print("")
        mc = mcscf.CASCI(mf,ncas=2,nelecas=(1,1))
        mc.kernel()
        wf = pyqmc.multislater.MultiSlater(mol, mf, mc)
        
        nconf = 10
        nelec = np.sum(mol.nelec)
        configs = np.random.randn(nconf, nelec, 3)
        
        print("Testing internals:", testwf.test_updateinternals(wf, configs))
        for delta in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            print(
                "delta",
                delta,
                "Testing gradient",
                testwf.test_wf_gradient(wf, configs, delta=delta),
            )
            print(
                "delta",
                delta,
                "Testing laplacian",
                testwf.test_wf_laplacian(wf, configs, delta=delta),
            )
            print(
                "delta",
                delta,
                "Testing pgradient",
                testwf.test_wf_pgradient(wf, configs, delta=delta),
            )

        print("Testing VMC")
        nconf = 5000
        nsteps = 100
        warmup = 30
        coords = initial_guess(mol, nconf)
        df, coords = vmc(
            wf, coords, nsteps=nsteps, accumulators={"energy": EnergyAccumulator(mol)}
        )

        df = pd.DataFrame(df)
        en = np.mean(df["energytotal"][warmup:])
        err = np.std(df["energytotal"][warmup:]) / np.sqrt(nsteps - warmup)
        print('VMC E = ',en,'+/-',err)

if __name__ == "__main__":
    test()
