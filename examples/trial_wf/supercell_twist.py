#!/usr/bin/env python
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


"""
How to run at various twists in a supercell.
"""

import numpy as np
from pyscf.pbc import gto, scf
import pyqmc.api as pyq
from pyqmc.pbc.supercell import get_supercell
import pyqmc.pbc.twists


def run_scf(nk):
    cell = gto.Cell()
    cell.atom = """
    He 0.000000000000   0.000000000000   0.000000000000
    """
    cell.basis = "ccecp-ccpvdz"
    cell.a = """
    5.61, 0.00, 0.00
    0.00, 5.61, 0.00
    0.00, 0.00, 5.61"""
    cell.unit = "B"
    cell.verbose = 5
    cell.build()

    kmf = scf.KRHF(cell, exxdiv=None).density_fit()
    kmf.kpts = cell.make_kpts([nk, nk, nk])
    ehf = kmf.kernel()
    print("EHF", ehf)
    return cell, kmf


if __name__ == "__main__":
    # Run SCF
    cell, kmf = run_scf(nk=4)

    # Set up wf and configs
    nconfig = 100
    S = np.eye(3) * 2  # 2x2x2 supercell
    supercell = get_supercell(cell, S)

    twist_info = pyqmc.pbc.twists.create_supercell_twists(supercell, kmf)
    print("Here are the twists available in the 2x2x2 supercell:", twist_info["twists"])
    # you access the twist by index
    wf, to_opt = pyq.generate_wf(supercell, kmf, slater_kws=dict(twist=2))

    S = (
        np.eye(3) * 4
    )  # for a 4x4x4 supercell we will only find one twist available because nk=4
    supercell = get_supercell(cell, S)
    twist_info = pyqmc.pbc.twists.create_supercell_twists(supercell, kmf)
    print("Here are the twists available in the 4x4x4 supercell:", twist_info["twists"])
