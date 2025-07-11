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

import numpy as np
import pyqmc.pbc.supercell


def test_supercell_charge(li_cubic_ccecp):
    """We test get_supercell generates a supercell with the correct number of electrons and spin for charged PBC systems
    We generate 1x1x1 and 2x2x2 supercells of a Li cubic cell with -1 charge
    The original cell has 3 valence electrons, and the supercells should have 3 and 17 valence electrons respectively
    The original cell and both supercells should have spin = 1 because of the one unpaired electron
    """
    cell, mf = li_cubic_ccecp
    nelectron_neutral_cell = cell.nelectron
    cell.charge = -1
    cell.spin += 1
    cell.build()
    for multiplier in [1, 2]:
        S = multiplier * np.eye(3)
        scale = abs(int(np.round(np.linalg.det(S))))
        supercell = pyqmc.pbc.supercell.get_supercell(cell, S)
        assert supercell.nelectron == (scale * nelectron_neutral_cell) + 1
        assert supercell.spin == cell.spin
