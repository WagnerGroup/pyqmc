import numpy as np 

def buildh2():
    from pyscf import gto
    #H2 molecule test: smallest cas 
    r = 1.1
    basis = {
        "H": gto.basis.parse(
        """
        H S
        23.843185 0.00411490
        10.212443 0.01046440
        4.374164 0.02801110
        1.873529 0.07588620
        0.802465 0.18210620
        0.343709 0.34852140
        0.147217 0.37823130
        0.063055 0.11642410
        """
        )
    }
    ecp = {
        "H": gto.basis.parse_ecp(
        """
        H nelec 0
        H ul
        1 21.24359508259891 1.00000000000000
        3 21.24359508259891 21.24359508259891
        2 21.77696655044365 -10.85192405303825
        """
        )
    }
    mol = gto.M(
        atom=f"H 0. 0. 0.; H 0. 0. {r}", unit="bohr", basis=basis, ecp=ecp
    )
    configs = np.random.randn(100, 2, 3)
    return mol, configs

def buildn2():
    from pyscf import gto
    mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    configs = np.random.randn(100, 14, 3)
    return mol, configs   

def multiSlaterTest(mol, configs):
    print("Multi-Slater testing ----------------------------")
    from pyqmc.testwf import test_updateinternals, test_wf_gradient, test_wf_laplacian, test_wf_pgradient
    from pyscf import gto, scf, mcscf
    from multislater import MultiSlater
    
    for method in ['RHF','ROHF','UHF']:
        if(method == 'RHF'): mf = scf.RHF(mol)
        elif(method  == 'ROHF'): mf = scf.ROHF(mol)
        else: mf = scf.UHF(mol)
   
        print("SCF method: ",method)
        mf.scf()
        print('SCF energy:', mf.e_tot)
        mc = mcscf.CASCI(mf,ncas=2,nelecas=(1,1))
        print('CASCI energy:', mc.kernel()[0])

        wf = MultiSlater(mol, mf, mc)
        ret = test_updateinternals(wf,configs)
        print('Test internals: ',ret)
        ret = test_wf_gradient(wf,configs)
        print('Test gradient: ',ret)
        ret = test_wf_laplacian(wf,configs)
        print('Test laplacian: ',ret)
        ret = test_wf_pgradient(wf,configs)
        print('Test pgrad: ',ret)

def multiSlaterJastrowTest(mol, configs):
    print("Multi-Slater-Jastrow testing ----------------------------")
    from pyqmc.testwf import test_updateinternals, test_wf_gradient, test_wf_laplacian, test_wf_pgradient
    from pyscf import gto, scf, mcscf
    from multislater import MultiSlater
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.func3d import GaussianFunction
    
    abasis = [GaussianFunction(0.2), GaussianFunction(0.4)]
    bbasis = [GaussianFunction(0.2), GaussianFunction(0.4)]
    jastrow = JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    jastrow.parameters["bcoeff"] = np.random.random(jastrow.parameters["bcoeff"].shape)
    jastrow.parameters["acoeff"] = np.random.random(jastrow.parameters["acoeff"].shape)

    for method in ['RHF','ROHF','UHF']:
        if(method == 'RHF'): mf = scf.RHF(mol)
        elif(method  == 'ROHF'): mf = scf.ROHF(mol)
        else: mf = scf.UHF(mol)
   
        print("SCF method: ",method)
        mf.scf()
        print('SCF energy:', mf.e_tot)
        mc = mcscf.CASCI(mf,ncas=2,nelecas=(1,1))
        print('CASCI energy:', mc.kernel()[0])

        ms = MultiSlater(mol, mf, mc)
        wf = MultiplyWF(jastrow,ms)
        ret = test_updateinternals(wf,configs)
        print('Test internals: ',ret)
        ret = test_wf_gradient(wf,configs)
        print('Test gradient: ',ret)
        ret = test_wf_laplacian(wf,configs)
        print('Test laplacian: ',ret)
        ret = test_wf_pgradient(wf,configs)
        print('Test pgrad: ',ret)

if __name__ == '__main__':
  mol, configs = buildh2()
  multiSlaterTest(mol,configs)

  mol, configs = buildn2()
  multiSlaterTest(mol,configs)

  mol, configs = buildh2()
  multiSlaterJastrowTest(mol,configs)

  mol, configs = buildn2()
  multiSlaterJastrowTest(mol,configs)
