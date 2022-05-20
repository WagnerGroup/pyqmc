import numpy as np


class Parameters:
    def __init__(self, dicts):
        self.data = {}
        self.wf_count = len(dicts)
        for (i, d) in enumerate(dicts):
            self.data["wf" + str(i + 1)] = d

    def __setitem__(self, idx, value):
        k1 = idx[0:3]
        k2 = idx[3:]
        self.data[k1][k2] = value

    def __getitem__(self, idx):
        k1 = idx[0:3]
        k2 = idx[3:]
        return self.data[k1][k2]

    def __delitem__(self, idx):
        k1 = idx[0:3]
        k2 = idx[3:]
        del self.data[k1][k2]

    def __iter__(self):
        for i in range(self.wf_count):
            k1 = "wf" + str(i + 1)
            for k2 in self.data[k1].keys():
                yield k1 + k2

    def __len__(self):
        return sum(len(i) for i in self.data)

    def items(self):
        for i in range(self.wf_count):
            k1 = "wf" + str(i + 1)
            for k2 in self.data[k1].keys():
                yield k1 + k2, self.data[k1][k2]

    def __repr__(self):
        return "WFmerger: " + self.data.__repr__()

    def keys(self):
        for i in range(self.wf_count):
            k1 = "wf" + str(i + 1)
            for k2 in self.data[k1].keys():
                yield k1 + k2

    def values(self):
        for i in range(self.wf_count):
            k1 = "wf" + str(i + 1)
            for k2 in self.data[k1].keys():
                yield self.data[k1][k2]


class MultiplyWF:
    """
    A general representation of a wavefunction as a product of multiple wf_factors
    """

    def __init__(self, *wf_factors):
        self.wf_factors = [*wf_factors]
        self.parameters = Parameters([wf.parameters for wf in wf_factors])
        self.iscomplex = bool(sum(wf.iscomplex for wf in wf_factors))
        self.dtype = complex if self.iscomplex else float

    def recompute(self, configs):
        signs = np.ones(len(configs.configs))
        vals = np.zeros(len(configs.configs))
        for wf in self.wf_factors:
            results = wf.recompute(configs)
            signs = signs * results[0]
            vals += results[1]
        return signs, vals

    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        if saved_values is None:
            saved_values = [None] * len(self.wf_factors)
        for wf, saved_val in zip(self.wf_factors, saved_values):
            wf.updateinternals(e, epos, configs, mask=mask, saved_values=saved_val)

    def value(self):
        results = [wf.value() for wf in self.wf_factors]
        results = np.array([*results])
        return np.prod(results[:, 0, :], axis=0), np.sum(results[:, 1, :], axis=0)

    def gradient(self, e, epos):
        grads = [wf.gradient(e, epos) for wf in self.wf_factors]
        return np.sum(grads, axis=0)

    def testvalue(self, e, epos, mask=None):
        testvalues, saved_values = list(
            zip(*[wf.testvalue(e, epos, mask=mask) for wf in self.wf_factors])
        )
        return np.prod(testvalues, axis=0), saved_values

    def testvalue_many(self, e, epos, mask=None):
        testvalues = [wf.testvalue_many(e, epos, mask=mask) for wf in self.wf_factors]
        return np.prod(testvalues, axis=0)

    def gradient_value(self, e, epos):
        grad_vals = [wf.gradient_value(e, epos) for wf in self.wf_factors]
        grads, vals, saved_values = list(zip(*grad_vals))
        return np.sum(grads, axis=0), np.prod(vals, axis=0), saved_values

    def gradient_laplacian(self, e, epos):
        grad_laps = [wf.gradient_laplacian(e, epos) for wf in self.wf_factors]
        grads, laps = list(zip(*grad_laps))
        cross_term = np.zeros(laps[0].shape, dtype=self.dtype)
        nwf = len(self.wf_factors)
        for i in range(nwf):
            for j in range(i + 1, nwf):
                cross_term += np.sum(grads[i] * grads[j], axis=0)
        return np.sum(grads, axis=0), np.sum(laps, axis=0) + cross_term * 2

    def laplacian(self, e, epos):
        return self.gradient_laplacian(e, epos)[1]

    def pgradient(self):
        return Parameters([wf.pgradient() for wf in self.wf_factors])


def test_parameters():
    dicts = [{"coeff" + str(i): np.random.rand(3)} for i in range(10)]
    p = Parameters(dicts)
    # test len
    assert len(p) == 30
    print("len test passed")
    # test getitem
    assert p["wf2coeff2"].all() == dicts[2]["coeff2"].all()
    print("getitem test passed")
    new_coeff = np.random.rand(5)
    # test setitem
    p["wf2coeff2"] = new_coeff
    assert p["wf2coeff2"].all() == new_coeff.all()
    print("setitem test passed")
