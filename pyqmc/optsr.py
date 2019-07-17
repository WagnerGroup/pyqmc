import numpy as np
import pandas as pd
import json


def gradient_descent(
    wf,
    coords,
    pgrad_acc,
    warmup=10,
    step=0.1,
    eps=0.1,
    maxiters=50,
    vmc=None,
    vmcoptions=None,
    datafile=None,
    verbose=2,
    wfsave=None,
):
    """Optimizes energy using gradient descent with stochastic reconfiguration.

    Args:

      wf: initial wave function

      coords: initial configurations

      pgrad_acc: A PGradAccumulator-like object

      vmc: A function that works like mc.vmc()

      vmcoptions: a dictionary of options for the vmc method

      warmup: warmup cutoff for VMC steps

      step: gradient descent step size

      eps: stabilizing constant for the stochastic reconfiguration matrix

      maxiters: maximum number of steps in the gradient descent

      datafile: a file in which the current progress can be dumped in JSON format.

    Returns:

      wf: optimized wave function

      data: dictionary with gradient descent data

    """
    import pandas as pd

    if vmc is None:
        import pyqmc.mc

        vmc = pyqmc.mc.vmc

    if vmcoptions is None:
        vmcoptions = {}

    def gradient_energy_function(x):
        newparms = pgrad_acc.transform.deserialize(x)
        for k in newparms:
            wf.parameters[k] = newparms[k]
        data, newcoords = vmc(
            wf, coords, accumulators={"pgrad": pgrad_acc}, **vmcoptions
        )
        df = pd.DataFrame(data)[warmup:]
        en = np.mean(df["pgradtotal"])

        en_std = np.std(df["pgradtotal"])

        # Sij matrix with stabilizing diagonal
        dpH = np.mean(df["pgraddpH"], axis=0)
        dp = np.mean(df["pgraddppsi"], axis=0)
        dpdp = np.mean(df["pgraddpidpj"], axis=0)
        grad = 2 * (dpH - en * dp)
        Sij = dpdp - np.einsum("i,j->ij", dp, dp) + eps * np.eye(dpdp.shape[0])
        invSij = np.linalg.inv(Sij)
        grad_std = 0
        return grad, grad_std, invSij, en, en_std, len(df)

    x0 = pgrad_acc.transform.serialize_parameters(wf.parameters)
    data = {
        "iter": [],
        "params": [],
        "pgrad": [],
        "pgrad_err": [],
        "totalen": [],
        "totalen_err": [],
    }
    pgrad, pgrad_std, invSij, en, en_std, nsteps = gradient_energy_function(x0)
    data["iter"].append(0)
    data["params"].append(x0)
    data["pgrad"].append(pgrad)
    data["pgrad_err"].append(pgrad_std)
    data["totalen"].append(en)
    data["totalen_err"].append(en_std / np.sqrt(nsteps))
    if verbose > 1:
        print("p =", x0)
        print("grad =", pgrad, flush=True)
    if verbose > 0:
        print(
            "|grad|=%.6f" % np.linalg.norm(pgrad),
            "E=%.5f+-%.5f" % (en, en_std / np.sqrt(nsteps)),
            flush=True,
        )

    # Gradient descent cycles
    for it in range(maxiters):
        x0 -= np.einsum("ij,j->i", invSij, pgrad) * step / (it / 10 + 1)
        pgrad, pgrad_std, invSij, en, en_std, nsteps = gradient_energy_function(x0)
        if verbose > 1:
            print("p =", x0)
            print("grad =", pgrad)
            pgrad_k = pgrad_acc.transform.deserialize(pgrad)
            for k, gradient in pgrad_k.items():
                print("rms gradient for", k, np.linalg.norm(gradient))
        if verbose > 0:
            print(
                "|grad|=%.6f" % np.linalg.norm(pgrad),
                "E=%.5f+-%.5f" % (en, en_std / np.sqrt(nsteps)),
                flush=True,
            )

        data["iter"].append(it + 1)
        data["params"].append(x0.copy())
        data["pgrad"].append(pgrad)
        data["pgrad_err"].append(pgrad_std)
        data["totalen"].append(en)
        data["totalen_err"].append(en_std / np.sqrt(nsteps))
        if not (datafile is None):
            pd.DataFrame(data).to_json(datafile)

        if not (wfsave is None):
            with open(wfsave, "w") as f:
                save = {}
                for k, param in pgrad_acc.transform.deserialize(x0).items():
                    save[k] = param.tolist()
                json.dump(save, f)

    if verbose > 1:
        print("\nGradient descent terminated.")

    return wf, data


def test():
    from pyscf import lib, gto, scf
    from pyqmc.accumulators import EnergyAccumulator, PGradTransform, LinearTransform
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.jastrow import Jastrow2B
    from pyqmc.func3d import GaussianFunction
    from pyqmc.slater import PySCFSlaterRHF
    from pyqmc.mc import initial_guess

    mol = gto.M(atom="H 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", verbose=5)
    mf = scf.RHF(mol).run()
    nconf = 2500
    nsteps = 70
    warmup = 20

    coords = initial_guess(mol, nconf)
    basis = {
        "wf2coeff": [
            GaussianFunction(0.2),
            GaussianFunction(0.4),
            GaussianFunction(0.6),
        ]
    }
    wf = MultiplyWF(PySCFSlaterRHF(mol, mf), Jastrow2B(mol, basis["wf2coeff"]))
    params0 = {"wf2coeff": np.array([-0.8, -0.2, 0.4])}
    for k, p in wf.parameters.items():
        if k in params0:
            wf.parameters[k] = params0[k]

    energy_acc = EnergyAccumulator(mol)
    pgrad_acc = PGradTransform(energy_acc, LinearTransform(wf.parameters))

    # Gradient descent
    wf, data = gradient_descent(
        wf,
        coords,
        pgrad_acc,
        vmcoptions={"nsteps": nsteps},
        warmup=warmup,
        step=0.5,
        eps=0.1,
        maxiters=50,
        datafile="sropt.json",
    )


if __name__ == "__main__":
    test()
