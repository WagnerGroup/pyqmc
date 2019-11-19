import numpy as np

class WaveFunction:
    """
    A general representation of a wavefunction as a product of multiple wf_factors 
    """
    def __init__(self, wf_factors):
        self.wf_factors = {}
    
    def __getitem__(self, ind):
        return self.wf_factors[ind]
    
    def __setitem__(self, label, wf):
        self.wf_factors[label] = wf
    
    def pop(self, label):
        self.wf_factors.pop(label)
    
    def  recompute(self, configs):
        results = [wf.recoompute(configs) for wf in self.wf_factors.values()]
        results = np.array([*results])
        return np.prod(results[:,0]), np.sum(results[:,1])

    def updateinternals(self, e, epos, mask=None):
        for wf in self.wf_factors.values():
            wf.updateinternals(e, epos, mask=mask)
    
    def value(self):
        results = [wf.values() for wf in self.wf_factors.values()]
        results = np.array([*results])
        return np.prod(results[:,0]), np.sum(results[:,1])
   
    def gradient(self, e, epos):
        grads = [wf.gradient(e, epos) for wf in self.wf_factors.values()]
        return np.sum(grads, axis=0)
    
    def testvalue(self, e, epos, mask=None):
        testvalues = [wf.testvalue(e, epos, mask=mask) for wf in self.wf_factors.values()]
        return np.prod(testvalues, axis=0)
    
    def laplacian(self, e, epos):
        grad_laps = [wf.gradient_laplacian(e, epos) for wf in self.wf_factors.values()]
        grad_laps = np.array([*grad_laps])
        grads = grad_laps[:,0,:]
        laps = grad_laps[:,1,:]
        corss_term = np.zeros(laps.shape[-1])
        nwf = len(self.wf_factors)
        for i in range(nwf):
            for j in range(i, nwf):
                corss_term += grads[i,:]*grads[j,:]
        return np.sum(laps, axis=0) + corss_term*2
                
    def pgradient(self):
        pass
