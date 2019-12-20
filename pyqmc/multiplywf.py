import numpy as np
import collections
import collections.abc


class WFmerger(collections.abc.MutableMapping):
    def __init__(self, d1, d2):
        self.data = {}
        self.data["wf1"] = d1
        self.data["wf2"] = d2

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
        for k1 in ["wf1", "wf2"]:
            for k2 in self.data[k1].keys():
                yield k1 + k2

    def __len__(self):
        return len(self.d1) + len(self.d2)

    def items(self):
        for k1 in ["wf1", "wf2"]:
            for k2 in self.data[k1].keys():
                yield k1 + k2, self.data[k1][k2]

    def __repr__(self):
        return "WFmerger: " + self.data.__repr__()

    def keys(self):
        for k1 in ["wf1", "wf2"]:
            for k2 in self.data[k1].keys():
                yield k1 + k2


class MultiplyWF:
    """Multiplies two wave functions """

    def __init__(self, wf1, wf2):
        self.wf1 = wf1
        self.wf2 = wf2
        self.parameters = WFmerger(self.wf1.parameters, self.wf2.parameters)

    def recompute(self, configs):
        v1 = self.wf1.recompute(configs)
        v2 = self.wf2.recompute(configs)
        return v1[0] * v2[0], v1[1] + v2[1]

    def updateinternals(self, e, epos, mask=None):
        self.wf1.updateinternals(e, epos, mask=mask)
        self.wf2.updateinternals(e, epos, mask=mask)

    def value(self):
        v1 = self.wf1.value()
        v2 = self.wf2.value()
        return v1[0] * v2[0], v1[1] + v2[1]

    def gradient(self, e, epos):
        return self.wf1.gradient(e, epos) + self.wf2.gradient(e, epos)

    def testvalue(self, e, epos, mask=None):
        return self.wf1.testvalue(e, epos, mask=mask) * self.wf2.testvalue(
            e, epos, mask=mask
        )

    def testvalue_many(self, e, epos, mask=None):
        return self.wf1.testvalue_many(e, epos, mask=mask) * self.wf2.testvalue_many(
            e, epos, mask=mask
        )

    def laplacian(self, e, epos):
        # This is a place where we might want to specialize a vgl function
        # which can save some time if we want both gradient and laplacians
        # Should check to see if that's a limiting factor or not.
        # We typically need the laplacian only for the energy, which is uncommonly
        # evaluated.

        g1, l1 = self.wf1.gradient_laplacian(e, epos)
        g2, l2 = self.wf2.gradient_laplacian(e, epos)
        return l1 + l2 + 2 * np.sum(g1 * g2, axis=0)

    def pgradient(self):
        """Here we need to combine the results"""
        return WFmerger(self.wf1.pgradient(), self.wf2.pgradient())


def test_WFmerger():
    d1 = {"A": 2, "B": 3}
    d2 = {"C": 6}
    d = WFmerger(d1, d2)
    for k in d.keys():
        print(k)

    for k, v in d.items():
        print(k, v)


if __name__ == "__main__":
    test()
    test_WFmerger()
