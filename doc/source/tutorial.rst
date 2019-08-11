Tutorial
**********************************

Import the `pyscf` and `pyqmc` libraries. We also use `pandas` to save data.

.. code-block:: python

    import pyscf
    import pyqmc
    import pandas as pd


Make the molecule using `pyscf`

.. code-block:: python
    
    mol = pyscf.gto.M("He 0. 0. 0.", basis='bfd_vdz', ecp='bfd', unit='bohr')

Run Hartree-Fock using `pyscf`

.. code-block:: python
    
    mf = pyscf.scf.RHF(mol).run()

From the Slater determinant in the mf object, construct a Slater-Jastrow wave function

.. code-block:: python 

    wf = pyqmc.slater_jastrow(mol, mf)

Generate starting sample points

.. code-block:: python 
    nconfig = 1000
    configs = pyqmc.initial_guess(mol, nconfig)

Optimize the Jastrow parameters.

.. code-block:: python 
    acc = pyqmc.gradient_generator(mol, wf, ['wf2acoeff','wf2bcoeff'])
    wf, dfgrad, dfline = pyqmc.line_minimization(wf, coords, acc)
    pd.DataFrame(dfgrad).to_json("optgrad.json")
    pd.DataFrame(dfline).to_json("optline.json")

Diffusion Monte Carlo

.. code-block:: python 
    dfdmc, configs, weights = pyqmc.rundmc(wf, coords, nsteps = 5000,
           accumulators={'energy': pyqmc.EnergyAccumulator(mol) }, tstep = 0.02 )
    pd.DataFrame(dfdmc).to_json("dmc.json")

