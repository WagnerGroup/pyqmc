import numpy as np
from pyqmc.slater import sherman_morrison_row


def test_sherman_morrison():
    ratio_err, inv_err = run_sherman_morrison()

    assert ratio_err < 1e-13, f"ratios don't match {npratio} {smratio}"
    assert inv_err < 1e-13, "inverses don't match"


def run_sherman_morrison():
    n = 10
    nconf = 4
    e = 2
    not_e = np.arange(n) != e

    # construct matrix that isn't near singular
    u, s, v = np.linalg.svd(np.random.randn(n, n))
    svals = (np.random.rand(nconf, n) + 1) * np.random.choice([-1, 1], (nconf, n))
    matrix = np.einsum("ij,hj,jk->hik", u, svals, v)
    inv = np.linalg.inv(matrix)

    # make sure new matrix isn't near singular
    coef = np.random.randn(nconf, n - 1)
    vec_ = np.einsum("ij,ijk->ik", coef, matrix[:, not_e])
    proj = (np.random.random(nconf) - 1) * 2
    proj += np.sign(proj) * 0.5
    vec = vec_ + matrix[:, e] * proj[:, np.newaxis]
    newmatrix = matrix.copy()
    newmatrix[:, e] = vec

    # compute ratios and inverses directly and by update
    smratio, sminv = sherman_morrison_row(e, inv, vec)
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
