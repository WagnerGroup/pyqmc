# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
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
import pyscf
import pyqmc
import pyqmc.recipes

if __name__ == "__main__":

    mol = pyscf.gto.M(
        atom="He 0. 0. 0.", basis="ccECP_cc-pVDZ", ecp="ccecp", unit="bohr"
    )

    mf = pyscf.scf.RHF(mol)
    mf.chkfile = "he_dft.hdf5"
    mf.kernel()
    jastrow_kws = {}
    slater_kws = {"optimize_orbitals": True}

    pyqmc.recipes.OPTIMIZE(
        "he_dft.hdf5", "he_sj.hdf5", jastrow_kws=jastrow_kws, slater_kws=slater_kws, verbose=True
    )

    pyqmc.recipes.VMC(
        "he_dft.hdf5",
        "he_sj_vmc.hdf5",
        load_parameters="he_sj.hdf5",
        accumulators={"rdm1": True},
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        **{"nblocks": 40},
    )

    pyqmc.recipes.DMC(
        "he_dft.hdf5",
        "he_sj_dmc.hdf5",
        load_parameters="he_sj.hdf5",
        accumulators={"rdm1": True},
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        verbose = True,
        **{"nblocks": 4000, "tstep": 0.02},
    )
