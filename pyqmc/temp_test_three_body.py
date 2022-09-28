import pyqmc.api as pyq
import pyqmc.three_body_jastrow
import numpy as np
import pyscf


mol = pyscf.gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="sto-3g", unit="bohr")
a_basis, b_basis = pyqmc.wftools.default_jastrow_basis(mol)
J=pyqmc.three_body_jastrow.Three_Body_JastrowSpin(mol,a_basis,b_basis)
J.parameters["ccoeff"] = np.random.random(J.parameters["ccoeff"].shape) * 0.02 - 0.01
configs = pyq.initial_guess(mol, 10)
print(J.recompute(configs))
