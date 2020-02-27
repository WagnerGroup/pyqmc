from pyqmc.accumulators import SkAccumulator
from pyqmc.coord import PeriodicConfigs
import numpy as np


def test_fcc():
    # ???
    a = 1
    Lvecs = (np.ones((3, 3)) - np.eye(3)) * a / 2
    run(Lvecs)


def run(Lvecs):
    Gvecs = np.linalg.inv(Lvecs).T * 2 * np.pi
    kvecs = np.stack([x.ravel() for x in np.meshgrid(*[np.arange(5)] * 3)], axis=1)
    kvecs = np.dot(kvecs, Gvecs)
    skacc = SkAccumulator(kvecs)

    configs = np.zeros((1, 2, 3))
    configs = PeriodicConfigs(configs, Lvecs)
    sk = skacc(configs, None)
    skavg = skacc.avg(configs, None)
    print(sk)
    print(skavg)


if __name__ == "__main__":
    test_fcc()
