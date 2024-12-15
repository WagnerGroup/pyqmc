# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np
import pyqmc.gpu as gpu

""" 
Collection of 3d function objects. Each has a dictionary parameters, which corresponds
to any variational parameters the funtion has.

"""

@gpu.fuse()
def polypadevalue(r, beta, rcut):
    z = r / rcut
    p = ((3*z - 8) * z + 6) * z**2
    return (1 - p) / (1 + beta * p)


@gpu.fuse()
def polypadegradvalue(r, beta, rcut):
    z1 = r/rcut - 1
    p = (3 * z1 + 4) * z1**2 * z1 + 1
    obp = 1 / (1 + beta * p)
    grad_rvec = (z1 * obp)**2 * (-(1 + beta) * 12 / rcut**2) 
    value = (1 - p) * obp
    return grad_rvec, value


def polypadegradlap(r, beta, rcut):
    z1 = r/rcut - 1
    z12 = z1 * z1
    p = (3 * z12 + 4 * z1) * z12 + 1
    obp = 1 / (1 + beta * p)
    grad_rvec = -(1 + beta) * 12 / rcut**2 * obp**2 * z12
    lap = grad_rvec * (5 + 2 / z1 - 24 * beta * (z1 + 1)**2 * z12 * obp)
    return grad_rvec, lap


class PolyPadeFunction:
    r"""

    .. math:: b(r) = \frac{1-p(z)}{1+\beta p(z)}, \quad z = r/r_{\rm cut}

    where :math:`p(z) = 6z^2 - 8z^3 + 3z^4`

    This function is positive at small r, and is zero for :math:`r \ge r_{\rm cut}`.
    """

    def __init__(self, beta, rcut):
        self.parameters = {
            "beta": gpu.cp.asarray(beta),
            "rcut": gpu.cp.asarray(rcut),
        }

    def value(self, rvec, r):
        return polypadevalue(r, self.parameters["beta"], self.parameters["rcut"])

    def gradient_value(self, rvec, r):
        beta, rcut = self.parameters["beta"], self.parameters["rcut"]
        grad_rvec, value = polypadegradvalue(r, beta, rcut)
        grad = rvec * grad_rvec[..., np.newaxis]
        return grad, value

    def gradient_laplacian(self, rvec, r):
        beta, rcut = self.parameters["beta"], self.parameters["rcut"]
        grad_rvec, lap = polypadegradlap(r, beta, rcut)
        grad = rvec * grad_rvec[..., np.newaxis]
        return grad, lap

    def gradient(self, rvec, r):
        return self.gradient_value(rvec, r)[0]

    def laplacian(self, rvec, r):
        return self.gradient_laplacian(rvec, r)[1]

    def pgradient(self, rvec, r):
        r"""
        :returns: parameter gradient {'rcut': :math:`\frac{\partial b}{\partial r_{\rm cut}}`, 'beta': :math:`\frac{\partial b}{\partial \beta}`}
        :rtype: dictionary
        """
        #mask = r >= self.parameters["rcut"]
        z = r / self.parameters["rcut"]
        z1 = z - 1
        z12 = z1 * z1
        beta = self.parameters["beta"]

        p = (3 * z12 + 4 * z1) * z12 + 1
        obp = 1 / (1 + beta * p)
        dpdz = 12 * z * z12
        dbdp = -(1 + beta) * obp * obp
        derivrcut = dbdp * dpdz * (-z / self.parameters["rcut"])
        derivbeta = -p * (1 - p) * obp * obp
        #derivrcut[mask] = 0.0
        #derivbeta[mask] = 0.0
        pderiv = {"rcut": derivrcut, "beta": derivbeta}
        return pderiv


class CutoffCuspFunction:
    r"""
    .. math:: b(r) = -\frac{p(r/r_{\rm cut})}{1+\gamma*p(r/r_{\rm cut})} + \frac{1}{3+\gamma}

    where
    :math:`p(y) = y - y^2 + y^3/3`

    This function is positive at small r, and is zero for :math:`r \ge r_{\rm cut}`.
    """

    def __init__(self, gamma, rcut):
        self.parameters = {"gamma": gamma, "rcut": rcut}

    def value(self, rvec, r):
        y = r / self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        y1 = y - 1
        p = (y1 * y1 * y1 + 1) / 3
        func = -p / (1 + gamma * p) + 1 / (3 + gamma)
        return func * self.parameters["rcut"]

    def gradient(self, rvec, r):
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        r = r[..., np.newaxis]
        y = r / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        grad = -rvec * a * c * rcut
        return grad

    def gradient_value(self, rvec, r):
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        r = r[..., np.newaxis]
        y = r / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        ogb = 1 / (1 + gamma * b)
        c = ogb * ogb / (rcut * r)

        grad = -rvec * a * c * rcut
        value = -np.squeeze(b * ogb, axis=-1) + 1 / (3 + gamma)
        return grad, value * rcut

    def laplacian(self, rvec, r):
        return self.gradient_laplacian(rvec, r)[1]

    def gradient_laplacian(self, rvec, r):
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        r = r[..., np.newaxis]
        y = r / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        c = 1 / (1 + gamma * b) ** 2 / r

        grad = -a * c * rvec
        temp = y1 - a * a * gamma / (1 + gamma * b)
        temp *= y
        temp += a
        lap = -c * 2 * temp
        return grad, np.squeeze(lap, axis=-1)


    def pgradient(self, rvec, r):
        r"""

        :parameter rvec: (nconf,...,3)
        :parameter r: (nconfig,...)
        :returns: parameter derivatives {'rcut': :math:`\frac{\partial b}{\partial r_{\rm cut}}`, 'gamma': :math:`\frac{\partial b}{\partial \gamma}`}
        :rtype: dict
        """
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        #mask = r > rcut
        y = r / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        ogb = 1 / (1 + gamma * b)
        c = a * ogb * ogb / (rcut * r)
        val = -b * ogb + 1 / (3 + gamma)

        dfdrcut = y * a * ogb * ogb + val
        dfdgamma = ((b * ogb) ** 2 - 1 / (3 + gamma) ** 2) * rcut
        #dfdrcut[mask] = 0.0
        #dfdgamma[mask] = 0.0
        func = {"rcut": dfdrcut, "gamma": dfdgamma}

        return func


def test_func3d_gradient(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 5, 10, 3))  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    numeric = gpu.cp.zeros(grad.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        pos[..., d] -= 2 * delta
        minuval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        numeric[..., d] = (plusval - minuval) / (2 * delta)
    maxerror = np.max(np.abs(grad - numeric))
    return gpu.asnumpy(maxerror)


def test_func3d_laplacian(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 5, 10, 3))  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    lap = bf.laplacian(rvec, r)
    numeric = gpu.cp.zeros(lap.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        r = np.linalg.norm(pos, axis=-1)
        plusval = bf.gradient(pos, r)[..., d]
        pos[..., d] -= 2 * delta
        r = np.linalg.norm(pos, axis=-1)
        minuval = bf.gradient(pos, r)[..., d]
        numeric[...] += (plusval - minuval) / (2 * delta)
    maxerror = np.max(np.abs(lap - numeric))
    return gpu.asnumpy(maxerror)


def test_func3d_gradient_laplacian(bf):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    lap = bf.laplacian(rvec, r)
    andgrad, andlap = bf.gradient_laplacian(rvec, r)
    graderr = np.amax(np.abs(grad - andgrad))
    laperr = np.amax(np.abs(lap - andlap))
    return {"grad": graderr, "lap": laperr}


def test_func3d_gradient_value(bf):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    val = bf.value(rvec, r)
    andgrad, andval = bf.gradient_value(rvec, r)
    graderr = np.linalg.norm((grad - andgrad))
    valerr = np.linalg.norm((val - andval))
    return {"grad": graderr, "val": valerr}


def test_func3d_pgradient(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    pgrad = bf.pgradient(rvec, r)
    numeric = {k: gpu.cp.zeros(v.shape) for k, v in pgrad.items()}
    maxerror = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    for k in pgrad.keys():
        bf.parameters[k] += delta
        plusval = bf.value(rvec, r)
        bf.parameters[k] -= 2 * delta
        minuval = bf.value(rvec, r)
        bf.parameters[k] += delta
        numeric[k] = (plusval - minuval) / (2 * delta)
        maxerror[k] = gpu.asnumpy(np.max(np.abs(pgrad[k] - numeric[k])))
        if maxerror[k] > 1e-5:
            print(k, "\n", pgrad[k] - numeric[k])
    return maxerror



class CutoffFunc3dEvaluator:
    def __init__(self, basis_functions, rcut):
        for b in basis_functions:
            assert b.parameters["rcut"] == rcut
        self.basis_functions = basis_functions
        self.rcut = rcut
        self.nbas = len(basis_functions)

    def __len__(self):
        return self.nbas

    def value(self, d, r):
        #r = np.linalg.norm(d)
        select = r < self.rcut
        out = gpu.cp.zeros((self.nbas, *r.shape))
        rselect = r[select]
        tmp = gpu.cp.zeros(r.shape)
        for l, b in enumerate(self.basis_functions):
            tmp[select] = b.value(None, rselect)
            out[l] = tmp

        return np.moveaxis(out, 0, -1)

    def _grad_x(self, d, r, funcs):
        d = gpu.cp.asarray(d)
        select = r < self.rcut
        dselect = d[select]
        rselect = r[select]
        gradsel = gpu.cp.zeros((*rselect.shape, self.nbas, 3))
        scalsel = gpu.cp.zeros((*rselect.shape, self.nbas)) # lap or val
        for l, f in enumerate(funcs):
            gradsel[..., l, :], scalsel[..., l] = f(dselect, rselect)
        grad = gpu.cp.zeros((*r.shape, self.nbas, 3))
        scal = gpu.cp.zeros((*r.shape, self.nbas))
        grad[select] = gradsel
        scal[select] = scalsel
        return grad, scal

    def gradient_value(self, d, r):
        funcs = [b.gradient_value for b in self.basis_functions]
        return self._grad_x(d, r, funcs)

    def gradient_laplacian(self, d, r):
        funcs = [b.gradient_laplacian for b in self.basis_functions]
        g, l = self._grad_x(d, r, funcs)
        return g, l[..., np.newaxis]
    
    def gradient(self, d, r):
        return self.gradient_value(d, r)[0]

    def laplacian(self, d, r):
        return self.gradient_laplacian(d, r)[1]

    def pgradient(self, d, r):
        raise NotImplementedError()
