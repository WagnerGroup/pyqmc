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
        if k not in data.keys():
            f.create_dataset(
                k, (0, *it.shape), maxshape=(None, *it.shape), dtype=it.dtype
            )
        currshape = f[k].shape
        f[k].resize((currshape[0] + 1, *currshape[1:]))
        f[k][
            -1,
        ] = it


if __name__ == "__main__":
    f = h5py.File("testfile.hdf5", "a")
    test = {"a": np.arange(1, 5)}
    attr = {"testval": 3.0}
    setup_hdf(f, test, attr)
    append_hdf(f, test)
    append_hdf(f, test)
    print(np.array(f["a"]))
