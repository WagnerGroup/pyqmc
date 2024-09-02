import pyqmc.api as pyq
import pyscf

mol = pyscf.gto.M(atom="H 0. 0. 0.; H 0. 0. 2.", basis="ccecp-ccpvtz", ecp="ccecp", unit="bohr")
mf = pyscf.scf.RHF(mol)
mf.run()

wf, to_opt = pyq.generate_wf(mol, mf, slater_kws=dict(evaluate_orbitals_with="numba"))
print(wf.wf_factors[0].orbitals.eval_gto)
