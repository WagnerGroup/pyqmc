from pyqmc.accumulators import SqAccumulator
from pyqmc.coord import PeriodicConfigs
import numpy as np
import pandas as pd
import pyscf.pbc
import matplotlib.pyplot as plt


def test_config():
    a = 1
    cell = pyscf.pbc.gto.M( # define nelec and lattice vectors
        atom="""Be 0.0 0.0 0.0""",
        a=np.eye(3) * a,
    )
    configs = np.array(
        [
            [-0.1592848, -0.15798219, 0.04790482],
            [0.03967904, 0.50691437, 0.40398405],
            [0.5295316, -0.11789016, 0.58326953],
            [0.49470142, 0.39850735, 0.02882759],
        ]
    ).reshape(1, 4, 3)

    df = run(cell, configs, 1)

    sqref = np.array(
        [0.9692770144196694,
         0.5072793133973733,
         1.3757250294362553,
         1.2743717293794594,
         1.2778376516123164,
         0.2966688755428347,
         0.45430493814236605,
         0.4625463360964845,
         1.5878969982574704,
         0.4656314877050358,
         1.2114862482620417,
         0.7629751960677265,
         2.0217929320128842,
        ]
    )
    spinsqref = np.array(
        [1.1129362308826771,
         1.2553230877280548,
         0.6978705946386531,
         1.436026684490446,
         2.5045121371347414,
         1.0533082766862591,
         0.474001216237328,
         0.5431324262605379,
         0.7727893023374922,
         1.138381838725822,
         0.04571082686242428,
         1.2608863469615232,
         0.08617979133990833,
        ]
    )

    diff = np.linalg.norm(df["Sq"] - sqref)
    assert diff < 1e-14, diff
    diff = np.linalg.norm(df["spinSq"] - spinsqref)
    assert diff < 1e-14, diff


def test_big_cell():
    import time

    a = 1
    ncell = (2, 2, 2)
    cell = pyscf.pbc.gto.M( # define nelec and lattice vectors
        atom="""Ge 0.0 0.0 0.0""",
        a=np.diag(ncell) * a,
    )
    unit_cell = np.zeros((4, 3))
    unit_cell[1:] = (np.ones((3, 3)) - np.eye(3)) * a / 2

    # generate 32 electron positions
    grid = np.meshgrid(*map(np.arange, ncell), indexing="ij")
    shifts = np.stack(list(map(np.ravel, grid)), axis=1)
    supercell = (shifts[:, np.newaxis] + unit_cell[np.newaxis]).reshape(1, -1, 3)
    configs = supercell.repeat(1000, axis=0)
    configs += np.random.randn(*configs.shape) * 0.15

    df = run(cell, configs, 8)
    df = df.groupby("qmag").mean().reset_index()
    df.plot("qmag", "Sq")
    df.plot("qmag", "spinSq")
    plt.show()

    for k in ["Sq", "spinSq"]:
        large_q = df[-35:-10][k]
        mean = np.mean(large_q - 1)
        rms = np.sqrt(np.mean((large_q - 1) ** 2))
        assert np.abs(mean) < 0.01, mean
        assert rms < 0.1, rms


def run(cell, configs, nq):
    sqacc = SqAccumulator(cell, nq=nq)
    configs = PeriodicConfigs(configs, cell.lattice_vectors())
    sqavg = sqacc.avg(configs, None)
    df = {"qmag": np.linalg.norm(sqacc.qlist, axis=1)}
    df.update(sqavg)
    return pd.DataFrame(df)


if __name__ == "__main__":
    test_config()
    test_big_cell()
