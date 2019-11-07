""" Evaluate the TBDM for a wave function object. """
import numpy as np
from copy import copy
from pyqmc.mc import initial_guess
from pyqmc.obdm import sample_onebody
from sys import stdout

class TBDMAccumulator:
    """ Return the tbdm as an array with indices rho[spin][i][j][k][l] = < c^+_{spin,i} c^+_{spin,k} c_{spin,l} c_{spin,j} > .
    This is in pyscf's notation. ATTENTION: The OBDM is not in pyscf's notation!
        NOTE: We will assume that the localized basis in which the tbdm is written is the same for up and down electrons.

    Args:

      mol (Mole): PySCF Mole object.

      orb_coeff (array): coefficients with size (nbasis,norb) relating mol basis to basis 
        of 2-RDM desired.

      nsweeps (int):

      tstep (float): width of the Gaussian to update two walkers positions for the 
        two extra coordinates.

      warmup (int): number of warmup steps for single-particle orbital sampling.

      naux (int):

      spin: [0,0], [0,1], [1,0] or [1,1] for up-up, up-down, down-up or down-down. Defaults to all electrons.

      electrons: 

      ijkl (array): contains M tbdm matrix elements to calculate with dim (M,4). 
    """

    def __init__(
        self,
        mol,
        orb_coeff,
        nsweeps=10,
        tstep=0.50,
        warmup=100,
        naux=500,
        spin=None,
        electrons=None,
        ijkl=None,
    ):
        assert (
            len(orb_coeff.shape) == 2
        ), "orb_coeff should be a list of orbital coefficients."

        if not (spin is None):
            if np.all(spin == [0,0]):
                self._electrons1 = np.arange(0, mol.nelec[0])
                self._electrons2 = np.arange(0, mol.nelec[0])
            elif np.all(spin == [0,1]):
                self._electrons1 = np.arange(0, mol.nelec[0])
                self._electrons2 = np.arange(mol.nelec[0], np.sum(mol.nelec))
            elif np.all(spin == [1,0]):
                self._electrons1 = np.arange(mol.nelec[0], np.sum(mol.nelec))
                self._electrons2 = np.arange(0, mol.nelec[0])
            elif np.all(spin == [1,1]):
                self._electrons1 = np.arange(mol.nelec[0], np.sum(mol.nelec))
                self._electrons2 = np.arange(mol.nelec[0], np.sum(mol.nelec))
            else:
                raise ValueError("Spin-spin not equal to [0,0], [0,1], [1,0] or [1,1]")
        elif not (electrons is None):
            self._electrons1 = electrons[0]
            self._electrons2 = electrons[1]
        else:
            self._electrons1 = np.arange(0, np.sum(mol.nelec))
            self._electrons2 = np.arange(0, np.sum(mol.nelec))
        self._epairs = np.array(np.meshgrid(self._electrons1,self._electrons2)).T.reshape(-1,2)
        self._epairs = self._epairs[self._epairs[:,0]!=self._epairs[:,1]] # Electron not repeated
            
        self._orb_coeff = orb_coeff
        self._tstep = tstep
        self._mol = mol

        nepairs = len(self._epairs)
        print('nepairs',nepairs)
        self._extra_config = initial_guess(mol, int(naux / nepairs) + 1).configs.reshape(-1, 3)

        self._nsweeps = nsweeps
        self._nstep = nsweeps * nepairs
        
        if not (ijkl is None):
            self._ijkl=ijkl.reshape(-1,4)
        else:
            aux=np.arange(0,self._orb_coeff.shape[1])
            self._ijkl=np.array(np.meshgrid(aux,aux,aux,aux)).T.reshape(-1,4) # All entries of the 2rdm  
        
        for i in range(warmup):
            accept, self._extra_config = sample_onebody(
                mol, orb_coeff, self._extra_config, tstep
            )

            
    def __call__(self, configs, wf, extra_configs=None):
        """ Quantities from equation (10) of DOI:10.1063/1.4793531"""

        nconf = configs.configs.shape[0]
        results = {
            "value": np.zeros(
                (nconf, self._orb_coeff.shape[1], self._orb_coeff.shape[1], self._orb_coeff.shape[1], self._orb_coeff.shape[1])
            ),
            "norm": np.zeros((nconf, self._orb_coeff.shape[1])),
            "acceptance": np.zeros(nconf),
        }
        acceptance = 0
        naux = self._extra_config.shape[0]
        epairs=np.tile(self._epairs, (self._nsweeps,1))

        if extra_configs is None:
            auxassignments0 = np.random.randint(0, int(naux/2), size=(self._nstep, nconf))
            auxassignments1 = np.random.randint(int(naux/2), naux, size=(self._nstep, nconf))
            extra_configs = []
            for step in range(self._nstep):
                extra_configs.append(np.copy(self._extra_config))
                accept, self._extra_config = sample_onebody(
                    self._mol, self._orb_coeff, self._extra_config, tstep=self._tstep
                )
                results["acceptance"] += np.mean(accept)
        else:
            assert auxassignments is not None
        
        for step in range(self._nstep):
            points = np.concatenate([self._extra_config, configs.configs[:, epairs[step,0], :], configs.configs[:, epairs[step,1], :]])
            ao = self._mol.eval_gto("GTOval_sph", points)
            borb = ao.dot(self._orb_coeff)

            # Orbital evaluations at extra coordinates.
            borb_aux = borb[0:naux, :]
            fsum = np.sum(borb_aux * borb_aux, axis=1)
            norm = borb_aux * borb_aux / fsum[:, np.newaxis]
            borb_configs0 = borb[naux:(naux+configs.configs.shape[0]), :]
            borb_configs1 = borb[(naux+configs.configs.shape[0]):, :]
           
            # It would be faster to implement a wf.testvalue_2body()
            epos0 = configs.make_irreducible(
                epairs[step,0], extra_configs[step][auxassignments0[step]]
            )
            epos1 = configs.make_irreducible(
                epairs[step,1], extra_configs[step][auxassignments1[step]]
            )
            wfratio1 = wf.testvalue(epairs[step,0], epos0)
            wf_aux = copy(wf)
            #print('Warning: Shallow copy.')
            wf_aux.updateinternals(epairs[step,0], epos0) # ??? CHECK HERE ???
            wfratio2 = wf_aux.testvalue(epairs[step,1], epos1)
            wfratio = wfratio1 * wfratio2
                        
            orbratio = np.einsum(
                "ij,ik,il,im->ijklm",
                borb_aux[auxassignments0[step], :] / fsum[auxassignments0[step], np.newaxis],
                borb_aux[auxassignments1[step], :] / fsum[auxassignments1[step], np.newaxis],
                borb_configs0, borb_configs1
            )

            #print('Warning: Check if should multiply nelec1 * nelec2.')
            results["value"] += len(self._electrons1) * len(self._electrons2) * np.einsum("i,ijklm->ijklm", wfratio, orbratio) # ??? CHECK THIS NORMALIZATION ???
            results["norm"] += (norm[auxassignments0[step]] + norm[auxassignments1[step]])/2 # ??? CHECK HERE ???

        results["value"] /= self._nstep
        results["norm"] = results["norm"] / self._nstep
        results["acceptance"] /= self._nstep

        #print('value:\n',results["value"][0])
        #print('norm:\n',results["norm"])
        
        return results

    def avg(self, configs, wf):
        d = self(configs, wf)
        davg = {}
        for k, v in d.items():
            # print(k, v.shape)
            davg[k] = np.mean(v, axis=0)
        return davg



def normalize_tbdm(tbdm, norm):
    return tbdm / (norm[np.newaxis, np.newaxis, np.newaxis, :] * norm[np.newaxis, np.newaxis, :, np.newaxis] * norm[np.newaxis, :, np.newaxis, np.newaxis] * norm[:, np.newaxis, np.newaxis, np.newaxis]) ** 0.5 








if __name__ == "__main__":

    import numpy as np
    from pyscf import gto, scf, lo
    from numpy.linalg import solve
    from pyqmc import PySCFSlaterUHF
    from pyqmc.mc import initial_guess, vmc
    from pyqmc.accumulators import EnergyAccumulator
    from pandas import DataFrame

    mol = gto.M(
        atom="Li 0. 0. 0.; H 0. 0. 2.0", basis="cc-pvdz", unit="A", verbose=4
    )
    mf = scf.RHF(mol).run()
    
    # Lowdin orthogonalized AO basis.
    lowdin = lo.orth_ao(mol, "lowdin")

    # MOs in the Lowdin basis.
    mo = solve(lowdin, mf.mo_coeff)

    # make AO to localized orbital coefficients.
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)
    #mftbmd = ...
    
    ### Test TBDM calculation.
    nconf = 5
    nsteps = 100
    tbdm_steps = 4
    warmup = 15
    wf = PySCFSlaterUHF(mol, mf)
    configs = initial_guess(mol, nconf)
    energy = EnergyAccumulator(mol)
    tbdm = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nsweeps=tbdm_steps)

    print('tbdm._mol:\n',tbdm._mol)
    print('tbdm._orb_coeff:\n',tbdm._orb_coeff)
    print('tbdm._nstep:\n',tbdm._nstep)
    print('tbdm._tstep:\n',tbdm._tstep)
    print('tbdm._extra_config:\n',tbdm._extra_config.shape)
    print('tbdm._electrons1:\n',tbdm._electrons1)
    print('tbdm._electrons2:\n',tbdm._electrons2)

    print('Starting VMC...')
    df, coords = vmc(
        wf,
        configs,
        nsteps=nsteps,
        accumulators={
            "energy": energy,
            "tbdm": tbdm,
            #"tbdm_upup": tbdm_upup,
            #"tbdm_updn": tbdm_updn,
            #"tbdm_dndn": tbdm_dndn,
        },
        verbose=True,
    )
    df = DataFrame(df)
    print(df)
    print(df.keys())
    
    tbdm_est = {}
    for k in ["tbdm"]:#, "obdm_up", "obdm_down"]:
        avg_norm = np.array(df.loc[warmup:, k + "norm"].values.tolist()).mean(axis=0)
        avg_obdm = np.array(df.loc[warmup:, k + "value"].values.tolist()).mean(axis=0)
        tbdm_est[k] = normalize_tbdm(avg_tbdm, avg_norm)

    print("Average TBDM(orb,orb)", tbdm_est["tbdm"].diagonal().round(3))
    print("mf tbdm", mftbdm.diagonal().round(3))
    assert np.max(np.abs(tbdm_est["tbdm"] - mftbdm)) < 0.05
    #print(obdm_est["tbdm_upup"].diagonal().round(3))
    #print(obdm_est["tbdm_updn"].diagonal().round(3))
    #print(obdm_est["tbdm_dndn"].diagonal().round(3))
    #assert np.max(np.abs(tbdm_est["tbdm_upup"] + tbdm_est["tbdm_dndn"] - mftbdm)) < 0.05
