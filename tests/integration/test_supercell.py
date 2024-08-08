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
import pyqmc.api as pyq

def test_supercell(diamond_primitive):
    mol, mf = diamond_primitive
    S = np.ones((3, 3)) - 2*np.eye(3)
    mol = pyq.get_supercell(mol, S)
    nelec = sum(mol.nelec)
    wf, _ = pyq.generate_slater(mol, mf, eval_gto_precision=1e-4)

    assert True # create the supercell Slater with no errors
