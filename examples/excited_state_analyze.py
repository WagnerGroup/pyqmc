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
import pandas as pd

if __name__ == "__main__":
    df = []
    for fname in ["linemin_final.hdf5", "excited1_final.hdf5", "excited2_final.hdf5"]:
        with h5py.File(fname, "r") as f:
            obdm_up = np.mean(np.array(f["obdm_upvalue"]), axis=0)
            obdm_down = np.mean(np.array(f["obdm_downvalue"]), axis=0)
            print(list(f.keys()))
            tbdm = np.mean(np.array(f["tbdmvalue"]), axis=0)
            tbdm_ijkl = np.array(f["tbdmijkl"])[0]
            energy = np.mean(np.array(f["energytotal"]))

        print("energy", energy)

        print(obdm_up * 2, obdm_down * 2)
        print(tbdm * 4)
        print(tbdm_ijkl)
        df.append(
            {
                "energy": energy,
                "t": 2
                * (obdm_up[0, 1] + obdm_up[1, 0] + obdm_down[0, 1] + obdm_down[1, 0]),
                "U": 4 * (tbdm[0] + tbdm[-1]),
                "fname": fname,
                "trace_up": 2 * np.trace(obdm_up),
                "trace_down": 2 * np.trace(obdm_down),
            }
        )

    print(pd.DataFrame(df))
