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

import h5py
import numpy as np


def setup_hdf(f, data, attr):
    """
    f should be an h5py file object
    data should be a dictionary of numpy arrays.
    attr is a dictionary that should go into attributes

    It's assumed that data consists of representative sizes.
    This function will not insert data into the HDF5 object, only set up the datasets.
    """
    for k, it in data.items():
        itnp = np.array(it)
        f.create_dataset(
            k, (0, *itnp.shape), maxshape=(None, *itnp.shape), dtype=itnp.dtype
        )
    for k, it in attr.items():
        f.attrs[k] = it


def append_hdf(f, data):
    for k, it in data.items():
        if k not in f.keys():
            itnp = np.array(it)
            f.create_dataset(
                k, (0, *itnp.shape), maxshape=(None, *itnp.shape), dtype=itnp.dtype
            )
        currshape = f[k].shape
        f[k].resize((currshape[0] + 1, *currshape[1:]))
        f[k][-1,] = it


if __name__ == "__main__":
    f = h5py.File("testfile.hdf5", "a")
    test = {"a": np.arange(1, 5)}
    attr = {"testval": 3.0}
    setup_hdf(f, test, attr)
    append_hdf(f, test)
    append_hdf(f, test)
    print(np.array(f["a"]))
