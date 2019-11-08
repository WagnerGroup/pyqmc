""" Evaluate the TBDM for a wave function object. """
import numpy as np
from copy import copy, deepcopy
from pyqmc.mc import initial_guess
from pyqmc.obdm import sample_onebody
from sys import stdout

class TBDMAccumulator:
    """ Return the tbdm as an array with indices rho[spin][i][j][k][l] = < c^+_{spin_a,i} c^+_{spin_b,k} c_{spin_b,l} c_{spin_a,j} > .
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
                self._electrons_a = np.arange(0, mol.nelec[0])
                self._electrons_b = np.arange(0, mol.nelec[0])
            elif np.all(spin == [0,1]):
                self._electrons_a = np.arange(0, mol.nelec[0])
                self._electrons_b = np.arange(mol.nelec[0], np.sum(mol.nelec))
            elif np.all(spin == [1,0]):
                self._electrons_a = np.arange(mol.nelec[0], np.sum(mol.nelec))
                self._electrons_b = np.arange(0, mol.nelec[0])
            elif np.all(spin == [1,1]):
                self._electrons_a = np.arange(mol.nelec[0], np.sum(mol.nelec))
                self._electrons_b = np.arange(mol.nelec[0], np.sum(mol.nelec))
            else:
                raise ValueError("Spin-spin not equal to [0,0], [0,1], [1,0] or [1,1]")
        elif not (electrons is None):
            self._electrons_a = electrons[0]
            self._electrons_b = electrons[1]
        else:
            print('Not implemented.')
            exit()
            self._electrons_a = np.arange(0, np.sum(mol.nelec))
            self._electrons_b = np.arange(0, np.sum(mol.nelec))
        if spin[0]==spin[1]:
            self._epairs = np.stack(np.triu_indices(mol.nelec[spin[0]],1)).T + spin[0] * mol.nelec[0]
        else:
            self._epairs = np.array(np.meshgrid(self._electrons_a,self._electrons_b)).T.reshape(-1,2)
        self._epairs = self._epairs[self._epairs[:,0]!=self._epairs[:,1]] # Electron not repeated
        
        if len(self._epairs)==0:
            print('Need to implement single electron sector!')
            exit()
        
        self._orb_coeff = orb_coeff
        self._tstep = tstep
        self._mol = mol

        nepairs = len(self._epairs)
        print('spin',spin,'; nepairs',nepairs,self._epairs)
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
            auxassignments_a = np.random.randint(0, int(naux/2), size=(self._nstep, nconf))
            auxassignments_b = np.random.randint(int(naux/2), naux, size=(self._nstep, nconf))
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
            #print('--> epair:',epairs[step])
            points = np.concatenate([self._extra_config, configs.configs[:, epairs[step,0], :], configs.configs[:, epairs[step,1], :]])
            ao = self._mol.eval_gto("GTOval_sph", points)
            borb = ao.dot(self._orb_coeff)

            # Orbital evaluations at extra coordinates.
            borb_aux = borb[0:naux, :]
            fsum = np.sum(borb_aux * borb_aux, axis=1)
            norm = borb_aux * borb_aux / fsum[:, np.newaxis]
            borb_configs_a = borb[naux:(naux+configs.configs.shape[0]), :]
            borb_configs_b = borb[(naux+configs.configs.shape[0]):, :]
           
            # It would be faster to implement a wf.testvalue_2body()
            epos_a = configs.make_irreducible(
                epairs[step,0], extra_configs[step][auxassignments_a[step]]
            )
            epos_a_orig = configs.electron(epairs[step,0])
            epos_b = configs.make_irreducible(
                epairs[step,1], extra_configs[step][auxassignments_b[step]]
            )
            #print('testvalue2:',wf.testvalue2(epairs[step,0], epairs[step,1], epos_a, epos_b))
            #print('testvalue2:',wf.testvalue2(epairs[step,0], epairs[step,1], configs.make_irreducible(epairs[step,0], configs.configs[:, epairs[step,0], :]), epos_b))
            #print('testvalue:',wf.testvalue(epairs[step,0], epos_a))
            #print('testvalue2:',wf.testvalue2(epairs[step,0], epairs[step,1], epos_a, configs.make_irreducible(epairs[step,1], configs.configs[:, epairs[step,1], :])))
            #print('testvalue:',wf.testvalue(epairs[step,1], epos_b))
            #exit()
            #print('Calculating wfratio_a:\n')
            #print(epairs[step,0], epos_a.configs)
            wfratio_a = wf.testvalue(epairs[step,0], epos_a)
            #wf_aux = copy(wf) ### This line is the issue!!!
            #print('Warning: Shallow copy.')
            wf.updateinternals(epairs[step,0], epos_a) # ??? CHECK HERE ???
            #print('calculating wfratio_b:\n')
            wfratio_b = wf.testvalue(epairs[step,1], epos_b)
            #wfratio_b = np.ones(epos_b.configs.shape[0])
            wfratio = wfratio_a * wfratio_b
            #print('wfratio:',wfratio)
            
            wf.updateinternals(epairs[step,0], epos_a_orig) # ??? CHECK HERE ???
            #print('wfratio_a.nan:',np.isnan(wfratio_a))
            #print('wfratio_b.nan:',np.isnan(wfratio_b))
            #if ( (np.isnan(wfratio_a).any()) | (np.isnan(wfratio_b).any()) ):
            #    print('wfratio_a:\n',wfratio_a)
            #    print('configs_a',configs.configs[:, epairs[step,0], :])
            #    print('epos_a:\n',epos_a.configs)
            #    print('wfratio_b:\n',wfratio_b)
            #    print('epos_b:\n',epos_b.configs)
            #    print('wfratio:\n',wfratio)
            #    exit()
            
            # rho[spin][i][j][k][l] = < c^+_{spin_a,i} c^+_{spin_b,k} c_{spin_b,l} c_{spin_a,j} > .                        
            orbratio = np.einsum(
                "mi,mk,ml,mj->mijkl",
                borb_aux[auxassignments_a[step], :] / fsum[auxassignments_a[step], np.newaxis],
                borb_aux[auxassignments_b[step], :] / fsum[auxassignments_b[step], np.newaxis],
                borb_configs_b, borb_configs_a
            )

            #print('Warning: Check if should multiply nelec1 * nelec2.')
            results["value"] += np.einsum("i,ijklm->ijklm", wfratio, orbratio)
            results["norm"] += (norm[auxassignments_a[step]] + norm[auxassignments_b[step]])/2 # ??? CHECK HERE ???

        results["value"] /= self._nsweeps
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
    #return tbdm / (norm[np.newaxis, np.newaxis, np.newaxis, :] * norm[np.newaxis, np.newaxis, :, np.newaxis] * norm[np.newaxis, :, np.newaxis, np.newaxis] * norm[:, np.newaxis, np.newaxis, np.newaxis]) ** 0.5
    return tbdm / np.einsum('i,j,k,l->ijkl',norm,norm,norm,norm) ** 0.5 








if __name__ == "__main__":

    import numpy as np
    from pyscf import gto, scf, lo
    from numpy.linalg import solve
    from pyqmc import PySCFSlaterUHF
    from pyqmc.mc import initial_guess, vmc
    from pyqmc.accumulators import EnergyAccumulator
    from pandas import DataFrame

    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. 2.0", basis="minao", unit="A", verbose=4
    )
    mf = scf.UHF(mol).run()

    mfobdm = np.array(mf.make_rdm1(mf.mo_coeff,mf.mo_occ))
    
    # Lowdin orthogonalized AO basis.
    lowdin = lo.orth_ao(mol, "lowdin")

    # MOs in the Lowdin basis.
    mo = solve(lowdin, mf.mo_coeff)

    # Construct 1-RMD in MO basis.
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)
    # Computes the TBDM for a single Slater determinant (in AO basis).
    norb=mfobdm.shape[1]
    nelec=mf.nelec
    mftbdm=np.tile(np.nan,[2,2]+[norb,norb,norb,norb])
    for spin in 0,1:
        mftbdm[spin,spin]=np.einsum('ik,jl->ijkl',mfobdm[spin],mfobdm[spin]) - np.einsum('il,jk->ijkl',mfobdm[spin],mfobdm[spin])
    mftbdm[0,1]=np.einsum('ik,jl->ijkl',mfobdm[0],mfobdm[1])
    mftbdm[1,0]=np.einsum('ik,jl->ijkl',mfobdm[1],mfobdm[0])
    # Rotation into pySCF's 2-RDMs notation
    mftbdm=np.transpose(mftbdm,axes=(0,1,2,4,3,5))
    mo_coeff_a=np.dot( (mo[0]).T, mf.get_ovlp() ).T
    mo_coeff_b=np.dot( (mo[1]).T, mf.get_ovlp() ).T
    mo_coeff=[mo_coeff_a,mo_coeff_b]  
    # Transforming the 2-RDM: we change from AO (R_{ijkl}) into MOs or IAOs (R_{alpha,beta,gamma,sigma}) basis: 
    #  R_{alpha,beta,gamma,sigma} = [C^T^*]_{alpha,i} [C^T^*]_{gamma,k} Rab_{ijkl} C_{j,beta} C_{l,sigma}
    tbdm=np.zeros((mftbdm.shape[0],mftbdm.shape[1],mo_coeff_a.T.shape[0],mo_coeff_b.T.shape[0],mo_coeff_a.shape[1],mo_coeff_b.shape[1]))
    for s1 in range(mftbdm.shape[0]):
      for s2 in range(mftbdm.shape[1]):
        rdm2_rot=np.tensordot(mo_coeff[s1].T.conj(), mftbdm[s1,s2], axes=([[1],[0]])) # \sum_i [Ca^T^*]_{alpha}^{i} Rab_{ijkl}
        rdm2_rot=np.transpose( np.tensordot(rdm2_rot, mo_coeff[s1], axes=([[1],[0]])) , axes=(0,3,1,2)) # \sum_j Rab_{ijkl} Ca^{j}_{beta}
        rdm2_rot=np.transpose( np.tensordot(mo_coeff[s2].T.conj(), rdm2_rot, axes=([[1],[2]])), axes=(1,2,0,3)) #\sum_k [Cb^T^*]_{gamma}^{k} Rab_{ijkl}
        mftbdm[s1,s2]=np.tensordot(rdm2_rot, mo_coeff[s2], axes=([[3],[0]])) # \sum_l Rab_{ijkl} Cb^{l}_{sigma}
    print(mftbdm,mftbdm.shape)
    #exit()
    
    ### Test TBDM calculation.
    nconf = 500
    nsteps = 100
    tbdm_steps = 4
    warmup = 15
    wf = PySCFSlaterUHF(mol, mf) # WF without Jastrow
    configs = initial_guess(mol, nconf)
    energy = EnergyAccumulator(mol)
    #tbdm = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nsweeps=tbdm_steps)
    tbdm_upup = TBDMAccumulator(mol=mol, orb_coeff=lowdin, spin=[0,0], nsweeps=tbdm_steps)
    tbdm_updn = TBDMAccumulator(mol=mol, orb_coeff=lowdin, spin=[0,1], nsweeps=tbdm_steps)

    #print('tbdm._mol:\n',tbdm._mol)
    #print('tbdm._orb_coeff:\n',tbdm._orb_coeff)
    #print('tbdm._nstep:\n',tbdm._nstep)
    #print('tbdm._tstep:\n',tbdm._tstep)
    #print('tbdm._extra_config:\n',tbdm._extra_config.shape)
    #print('tbdm._electrons1:\n',tbdm._electrons1)
    #print('tbdm._electrons2:\n',tbdm._electrons2)

    print('tbdm_updn._mol:\n',tbdm_updn._mol)
    print('tbdm_updn._orb_coeff:\n',tbdm_updn._orb_coeff)
    print('tbdm_updn._nstep:\n',tbdm_updn._nstep)
    print('tbdm_updn._tstep:\n',tbdm_updn._tstep)
    print('tbdm_updn._extra_config:\n',tbdm_updn._extra_config.shape)
    print('tbdm_updn._electrons1:\n',tbdm_updn._electrons1)
    print('tbdm_updn._electrons2:\n',tbdm_updn._electrons2)

    print('Starting VMC...')
    df, coords = vmc(
        wf,
        configs,
        nsteps=nsteps,
        accumulators={
            "energy": energy,
            #"tbdm": tbdm,
            "tbdm_upup": tbdm_upup,
            "tbdm_updn": tbdm_updn,
            #"tbdm_dndn": tbdm_dndn,
        },
        verbose=True,
    )
    df = DataFrame(df)
    print(df)
    print(df.keys())
    
    tbdm_est = {}
    for k in ["tbdm_upup","tbdm_updn"]:
        avg_norm = np.array(df.loc[warmup:, k + "norm"].values.tolist()).mean(axis=0)
        std_norm = np.array(df.loc[warmup:, k + "norm"].values.tolist()).std(axis=0)
        avg_tbdm = np.array(df.loc[warmup:, k + "value"].values.tolist()).mean(axis=0)
        std_tbdm = np.array(df.loc[warmup:, k + "value"].values.tolist()).std(axis=0)
        tbdm_est[k] = normalize_tbdm(avg_tbdm, avg_norm)

    #print("Av tbdm(i,i,i,i):\n", tbdm_est["tbdm_updn"].diagonal().round(3))
    #print("Av tbdm(i,i,i,i):\n", tbdm_est["tbdm_updn"].diagonal().round(3))
    #print("Mf tbdm(i,i,i,i):\n", mftbdm[0,1].diagonal().round(3))
    print('mftbdm:\n',mftbdm[0,1])
    print('tbdm_est:\n',tbdm_est["tbdm_updn"])
    print('tbdm_est-mftbdm:\n',tbdm_est["tbdm_updn"] - mftbdm[0,1])
    #assert np.max(np.abs(tbdm_est["tbdm"] - mftbdm[0,1])) < 0.05
    #print(obdm_est["tbdm_upup"].diagonal().round(3))
    #print(obdm_est["tbdm_updn"].diagonal().round(3))
    #print(obdm_est["tbdm_dndn"].diagonal().round(3))
    #assert np.max(np.abs(tbdm_est["tbdm_upup"] + tbdm_est["tbdm_dndn"] - mftbdm)) < 0.05
