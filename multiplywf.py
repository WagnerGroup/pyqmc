import numpy as np
import collections
import collections.abc

class WFmerger(collections.abc.MutableMapping):
    def __init__(self,d1,d2):
        self.data={}
        self.data['wf1']=d1
        self.data['wf2']=d2

    def __setitem__(self,idx,value):
        k1=idx[0:3]
        k2=idx[3:]
        self.data[k1][k2]=value

    def __getitem__(self,idx):
        k1=idx[0:3]
        k2=idx[3:]
        return self.data[k1][k2]

    def __delitem__(self,idx):
        k1=idx[0:3]
        k2=idx[3:]
        del self.data[k1][k2]


    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.d1)+len(self.d2)

    def keys(self):
        return self.data['wf1'].keys()+self.data['wf2'].keys()



class MultiplyWF:
    """Multiplies two wave functions """

    def __init__(self,nconfig,wf1,wf2):
        self.wf1=wf1
        self.wf2=wf2
        #Using a ChainMap here since it makes things easy.
        #But there is a possibility that names collide here. 
        #one option is to use some name-mangling scheme for parameters
        #within each wave function
        self.parameters=WFmerger(self.wf1.parameters,self.wf2.parameters)

        
    def recompute(self,configs):
        v1=self.wf1.recompute(configs)
        v2=self.wf2.recompute(configs)
        return v1[0]*v2[0],v1[1]+v2[1]

    def updateinternals(self,e,epos,mask=None):
        self.wf1.updateinternals(e,epos,mask)
        self.wf2.updateinternals(e,epos,mask)


    def value(self):
        v1=self.wf1.value()
        v2=self.wf2.value()
        return v1[0]*v2[0],v1[1]+v2[1]
    
    def gradient(self,e,epos):
        return self.wf1.gradient(e,epos)+self.wf2.gradient(e,epos)

    def testvalue(self,e,epos):
        return self.wf1.testvalue(e,epos)*self.wf2.testvalue(e,epos)

    def laplacian(self,e,epos):
        # This is a place where we might want to specialize a vgl function 
        # which can save some time if we want both gradient and laplacians
        # Should check to see if that's a limiting factor or not.
        # We typically need the laplacian only for the energy, which is uncommonly 
        # evaluated.
        g1=self.wf1.gradient(e,epos)
        g2=self.wf2.gradient(e,epos)
        l1=self.wf1.laplacian(e,epos)
        l2=self.wf2.laplacian(e,epos)
        return l1+l2+2*np.sum(g1*g2,axis=0)

    def pgradient(self):
        """Here we need to combine the results"""
        return ChainMap(self.wf1.pgradient(),self.wf2.pgradient())





def test():
    from pyscf import lib,gto,scf
    from jastrow import Jastrow2B
    from slater import PySCFSlaterRHF
    nconf=10
    
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    slater=PySCFSlaterRHF(nconf,mol,mf)
    jastrow=Jastrow2B(nconf,mol)
    jastrow.parameters['coeff']=np.random.random(jastrow.parameters['coeff'].shape)
    configs=np.random.randn(nconf,4,3)
    wf=MultiplyWF(nconf,slater,jastrow)
    wf.parameters['coeff']=np.ones(len(wf.parameters['coeff']))
    print(wf.wf2.parameters['coeff'])
    
    import testwf
    for delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        print('delta', delta, "Testing gradient",testwf.test_wf_gradient(wf,configs,delta=delta))
        print('delta', delta, "Testing laplacian", testwf.test_wf_laplacian(wf,configs,delta=delta))
        print('delta', delta, "Testing pgradient", testwf.test_wf_pgradient(wf,configs,delta=delta))
    
    
if __name__=="__main__":
    test()
