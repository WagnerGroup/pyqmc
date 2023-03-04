# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import pyscf
import pyqmc.api as pyq
import pyscf.pbc
import numpy as np
import pyqmc.eval_ecp
np.random.seed(1234)

class SiSuite:
    """
    
    """
    def setup(self):

        self.cell = pyscf.pbc.gto.Cell()
        self.cell.atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751'''
        self.cell.basis = f'ccecpccpvdz'
        self.cell.pseudo = 'ccecp'
        self.cell.a = np.eye(3)*3.5668
        self.cell.exp_to_discard=0.2
        self.cell.build()

        self.configs = pyq.initial_guess(self.cell, 500)
        self.jastrow, self.jastrow_to_opt =  pyq.generate_jastrow(self.cell)
        self.jastrow.recompute(self.configs)

    def time_ecp_jastrow(self):
        pyqmc.eval_ecp.ecp(self.cell,self.configs,self.jastrow, 10)
        
    def time_jastrow_recompute(self):
        self.jastrow.recompute(self.configs)

    def time_jastrow_laplacian(self):
        self.jastrow.gradient_laplacian(0, self.configs.electron(0))

    def time_jastrow_derivatives(self):
        self.jastrow.pgradient()




if __name__=="__main__":
    h2o = SiSuite()
    h2o.setup()
    h2o.time_jastrow_recompute()
