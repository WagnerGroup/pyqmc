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
from pyqmc.wf.slater import sherman_morrison_row
from pyqmc.wf.slater import sherman_morrison_ms


def test_sherman_morrison():
    ratio_err, inv_err = run_sherman_morrison()

    assert ratio_err < 1e-13, f"ratios don't match {ratio_err}"
    assert inv_err < 1e-13, f"inverses don't match {inv_err}"

    ratio_err, inv_err = run_sherman_morrison(ms=True)

    assert ratio_err < 1e-13, f"ratios don't match {ratio_err}"
    assert inv_err < 1e-13, f"inverses don't match {inv_err}"


def construct_mat(nconf, n, ndet=None):
    u, s, v = np.linalg.svd(np.random.randn(n, n))
    if ndet is None:
        shape = (nconf, n)
    else:
        shape = (nconf, ndet, n)
    svals = (np.random.rand(*shape) + 1) * np.random.choice([-1, 1], shape)
    matrix = np.einsum("ij,...hj,jk->...hik", u, svals, v)
    return matrix


def construct_vec(matrix, nconf, n, e, ndet=None):
    if ndet is None:
        coef = np.random.randn(nconf, n - 1)
    else:
        coef = np.random.randn(nconf, ndet, n - 1)
    not_e = np.arange(n) != e
    vec_ = np.einsum("i...j,i...jk->i...k", coef, matrix[..., not_e, :])
    proj = (np.random.random(nconf) - 1) * 2
    proj += np.sign(proj) * 0.5
    vec = vec_ + np.einsum("i...k,i->i...k", matrix[..., e, :], proj)
    return vec


def run_sherman_morrison(ms=False):
    n = 10
    nconf = 4
    e = 2
    ndet = 8 if ms else None

    # construct matrix that isn't near singular
    matrix = construct_mat(nconf, n, ndet=ndet)
    inv = np.linalg.inv(matrix)

    # make sure new matrix isn't near singular
    newmatrix = matrix.copy()
    vec = construct_vec(matrix, nconf, n, e, ndet=ndet)
    newmatrix[..., e, :] = vec

    # compute ratios and inverses directly and by update
    if ndet is None:
        smratio, sminv = sherman_morrison_row(e, inv, vec)
    else:
        smratio, sminv = sherman_morrison_ms(e, inv, vec)
    npratio = np.linalg.det(newmatrix) / np.linalg.det(matrix)
    npinv = np.linalg.inv(newmatrix)

    ratio_err = np.amax(np.abs(npratio - smratio))
    inv_err = np.amax(np.abs(npinv - sminv))

    return ratio_err, inv_err


if __name__ == "__main__":
    r_err, inv_err = list(zip(*[run_sherman_morrison() for i in range(2000)]))
    print(np.amax(r_err))
    print(np.amax(inv_err))

    counts, bins = np.histogram(np.log10(inv_err), bins=np.arange(-16, 0))
    print(np.stack([counts, bins[1:]]))
