import numpy as np


class S2Accumulator:
    """
    Local estimator for the total spin-squared operator <S^2> in VMC.

    For an electronic state with fixed numbers of up and down electrons,
    Sz = (N_up - N_down)/2 is a sharp quantum number, so

        S^2 = Sz(Sz+1) + S_- S_+ .

    Acting on a real-space configuration R = (r_↑, r_↓), the off-diagonal piece
    reduces to a sum over up/down electron pairs of wave-function ratios at
    swapped positions. Working it out in second quantization with electrons
    ordered up-first-then-down,

        S_- S_+ |R⟩ = N_down |R⟩ − Σ_{i∈up, j∈down} |R^{i↔j}⟩,

    where R^{i↔j} has up electron i at the original position of down electron j
    and vice versa. Hence

        S²_loc(R) = Sz(Sz+1) + N_down − Σ_{i∈up, j∈down} ψ(R^{i↔j}) / ψ(R) .

    Each swap ratio ψ(R^{i↔j})/ψ(R) is computed as two consecutive single-electron
    moves with wf.testvalue + wf.updateinternals (factorized through whatever
    wf.testvalue returns); the moves are then unwound so the wf and configs are
    left in their original state.

    Follows the pyqmc accumulator convention: __call__ returns per-config arrays
    in a dict; avg returns averages.

    Parameters
    ----------
    nelec : tuple
        (n_up, n_down). Convention matches pyqmc Slater: electrons 0..n_up-1 are
        spin-up, n_up..n_up+n_down-1 are spin-down.
    """

    def __init__(self, nelec):
        self.nelec = tuple(nelec)
        nu, nd = self.nelec
        self.sz = 0.5 * (nu - nd)

    def _move(self, wf, configs, e, target_pos, accept):
        """Move electron e to target_pos; return ψ_after/ψ_before."""
        new = configs.make_irreducible(e, target_pos)
        ratio, saved = wf.testvalue(e, new)
        configs.move(e, new, accept)
        wf.updateinternals(e, new, configs, mask=accept, saved_values=saved)
        return ratio

    def __call__(self, configs, wf):
        nu, nd = self.nelec
        nconfig = configs.configs.shape[0]
        accept = np.ones(nconfig, dtype=bool)

        orig = configs.configs.copy()  # snapshot of all electron positions

        # Diagonal i=j contribution to <R|S_-S_+|ψ>/ψ(R) is N_down;
        # the swap sum carries a minus sign.
        s_minus_s_plus = np.full(nconfig, float(nd), dtype=wf.dtype)

        for i in range(nu):
            r_i = orig[:, i, :]
            for j in range(nu, nu + nd):
                r_j = orig[:, j, :]

                # Forward swap: up i → r_j, then down j → r_i.
                ratio1 = self._move(wf, configs, i, r_j.copy(), accept)
                ratio2 = self._move(wf, configs, j, r_i.copy(), accept)
                s_minus_s_plus -= ratio1 * ratio2

                # Unwind, in reverse order, so internal caches return to the
                # state that matches the original configs.
                self._move(wf, configs, j, r_j.copy(), accept)
                self._move(wf, configs, i, r_i.copy(), accept)

        return {"S2": self.sz * (self.sz + 1) + s_minus_s_plus}

    def avg(self, configs, wf):
        return {k: np.mean(v, axis=0) for k, v in self(configs, wf).items()}

    def keys(self):
        return self.shapes().keys()

    def shapes(self):
        return {"S2": ()}
