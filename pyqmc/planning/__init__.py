"""Small utilities for planning a QMC calculation.

Each helper answers one sizing question with a short benchmark rather than a
rule of thumb:

  benchmark_nconfig(mol, wf)            -- how many configurations to run
      (applies to any VMC / DMC / optimization run). See planning.nconfig.
  recommend_optimizer(to_opt, nconfig)  -- which SR solver to use for an
      optimization. See planning.sr.

See example_planning.py for a self-contained walkthrough on H2.
"""
from .nconfig import benchmark_nconfig
from .sr import recommend_optimizer

__all__ = ["benchmark_nconfig", "recommend_optimizer"]
