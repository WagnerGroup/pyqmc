# MIT License
# 
# Copyright (c) 2019 Lucas K Wagner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

if __name__ == "__main__":
    import pyscf
    import pyqmc.api as pyq

    mol = pyscf.gto.M(atom="He 0. 0. 0.", basis="bfd_vdz", ecp="bfd", unit="bohr")

    mf = pyscf.scf.RHF(mol).run()
    wf, to_opt = pyq.generate_wf(mol, mf)

    nconfig = 1000
    configs = pyq.initial_guess(mol, nconfig)

    acc = pyq.gradient_generator(mol, wf, to_opt)
    pyq.line_minimization(wf, configs, acc,  verbose=True, max_iterations=10)
    quit()
    pyq.rundmc(
        wf,
        configs,
        nblocks=5000,
        accumulators={"energy": pyq.EnergyAccumulator(mol)},
        tstep=0.02,
        hdf_file="he_dmc.hdf5",
        verbose=True,
    )
