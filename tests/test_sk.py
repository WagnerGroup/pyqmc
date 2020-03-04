from pyqmc.accumulators import SkAccumulator
from pyqmc.coord import PeriodicConfigs
import numpy as np


def test_fcc():
    # ???
    a = 1
    Lvecs = np.eye(3) * a
    configs = np.zeros((1, 4, 3))
    configs[0, 1:, :] = (np.ones((3, 3)) - np.eye(3)) * a / 2

    run(Lvecs, configs)


def run(Lvecs, configs):
    Gvecs = np.linalg.inv(Lvecs).T * 2 * np.pi
    kvecs = np.stack([x.ravel() for x in np.meshgrid(*[np.arange(4)] * 3)], axis=1)
    kvecs = np.dot(kvecs, Gvecs)
    skacc = SkAccumulator(kvecs)

    configs = PeriodicConfigs(configs, Lvecs)
    sk = skacc(configs, None)["Sk"]
    skavg = skacc.avg(configs, None)
    print(np.round(sk, 4))
    # print(skavg)


if __name__ == "__main__":
    test_fcc()
