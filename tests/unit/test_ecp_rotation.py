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

import pyqmc.observables.eval_ecp
import pytest
import numpy as np
import scipy.stats


@pytest.mark.parametrize("naip", [6, 12])
def test_rotation_even(naip):
    ncheck = 1000
    rotations = np.zeros((ncheck, naip,3))
    for i in range(ncheck):
        _, rotations[i] = pyqmc.observables.eval_ecp.get_rot(1, naip)
    avg = np.mean(rotations, axis=0)
    sem = scipy.stats.sem(rotations, axis=0)
    # print(avg/sem)
    assert (np.abs(avg / sem) < 5).all()


if __name__ == "__main__":
    test_rotation_even()
