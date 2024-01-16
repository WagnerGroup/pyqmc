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

import pyqmc.eval_ecp
import pytest
import numpy as np
import scipy.stats


@pytest.mark.parametrize("naip", [6, 12])
def test_rotation_even(naip):
    weights, rotations = pyqmc.eval_ecp.get_rot(1000, naip)
    avg = np.mean(rotations, axis=0)
    sem = scipy.stats.sem(rotations, axis=0)
    # print(avg/sem)
    assert (np.abs(avg / sem) < 5).all()


if __name__ == "__main__":
    test_rotation_even()
