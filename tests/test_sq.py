from pyqmc.accumulators import SqAccumulator
from pyqmc.coord import PeriodicConfigs
import numpy as np


def test_config():
    a = 1
    Lvecs = np.eye(3) * a
    configs = np.array(
        [
            [-0.1592848, -0.15798219, 0.04790482],
            [0.03967904, 0.50691437, 0.40398405],
            [0.5295316, -0.11789016, 0.58326953],
            [0.49470142, 0.39850735, 0.02882759],
        ]
    ).reshape(1, 4, 3)

    df = run(Lvecs, configs, 3)

    sqref = np.array(
        [
            4.0,
            0.08956614244510086,
            1.8925934706558083,
            0.1953404868933881,
            0.05121727442047123,
            1.5398266853045084,
            1.4329204824617385,
            0.7457498873351416,
            1.0713898023987862,
            0.2976758438030117,
            0.08202120690018336,
            0.3755969602702992,
            0.933685594722744,
            2.650270169642618,
            0.26674875141672655,
            0.7371957610619541,
            0.777701221323419,
            0.9084042551734659,
            2.170944896653447,
            0.38328335391002477,
            3.5406891971547862,
            1.1884884008703132,
            0.6203428839246292,
            0.7075185940748288,
            0.25780137400339037,
            1.317648046579579,
            0.8699973207672075,
        ]
    )

    diff = np.linalg.norm(df["Sq"] - sqref)
    assert diff < 1e-14, diff


def run(Lvecs, configs, nq):
    Gvecs = np.linalg.inv(Lvecs).T * 2 * np.pi
    qvecs = np.stack([x.ravel() for x in np.meshgrid(*[np.arange(nq)] * 3)], axis=1)
    qvecs = np.dot(qvecs, Gvecs)
    sqacc = SqAccumulator(qvecs)

    configs = PeriodicConfigs(configs, Lvecs)
    sqavg = sqacc.avg(configs, None)
    df = {}
    df.update(sqavg)
    return df


if __name__ == "__main__":
    test_config()
