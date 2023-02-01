# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import pyscf
import pyqmc.api as pyq

class H2OSuite:
    """
    
    """
    def setup(self):
        self.mol = pyscf.gto.M(atom="O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587", basis=f'ccecpccpvdz', ecp='ccecp')
        self.mf = pyscf.scf.RHF(self.mol).run()
        self.configs = pyq.initial_guess(self.mol, 500)
        self.slater, self.slater_to_opt =  pyq.generate_slater(self.mol, self.mf, optimize_orbitals=True)
        self.jastrow, self.jastrow_to_opt =  pyq.generate_jastrow(self.mol)
        self.pgrad_acc = pyq.gradient_generator(self.mol, self.slater, self.slater_to_opt)
        self.slater.recompute(self.configs)
        self.jastrow.recompute(self.configs)

    def time_orbital_derivatives(self):
        self.slater.pgradient()

    def time_slater_recompute(self):
        self.slater.recompute(self.configs)

    def time_slater_laplacian(self):
        self.slater.gradient_laplacian(0, self.configs.electron(0))
        
    def time_energy_slater(self):
        self.pgrad_acc.enacc.avg(self.configs, self.slater)

    def time_energy_jastrow(self):
        self.pgrad_acc.enacc.avg(self.configs, self.jastrow)

    def time_pgradient_slater(self):
        self.pgrad_acc.avg(self.configs, self.slater)

    def time_jastrow_recompute(self):
        self.jastrow.recompute(self.configs)

    def time_jastrow_laplacian(self):
        self.jastrow.gradient_laplacian(0, self.configs.electron(0))

    def time_jastrow_derivatives(self):
        self.jastrow.pgradient()




if __name__=="__main__":
    h2o = H2OSuite()
    h2o.setup()
    h2o.time_orbital_derivatives()
