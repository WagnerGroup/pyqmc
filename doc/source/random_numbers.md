Random numbers
-------------------------------

pyqmc draws all of its random numbers from its own numpy `Generator`, in `pyqmc.rng`.
It never reads or changes numpy's global random state, so `np.random.seed()` has no
effect on pyqmc.

### Reproducible runs

Either reseed pyqmc's module generator once at the top of a script:

```python
import pyqmc.rng
pyqmc.rng.seed(1234)
```

or pass `rng` to an entry point (`initial_guess`, `vmc`, `rundmc`, `line_minimization`,
`minsr_optimization`, `cgsr_optimization`, `optimize_ensemble`, and the `OPTIMIZE`,
`VMC`, and `DMC` recipes):

```python
rng = np.random.default_rng(1234)
configs = pyq.initial_guess(mol, 1000, rng=rng)
df, configs = pyq.vmc(wf, configs, rng=rng)
```

`rng` accepts anything `np.random.default_rng` does: `None` (pyqmc's module generator),
an int, a `SeedSequence`, or a `Generator`.

A parallel run is reproducible for a given seed and a given `npartitions`. Changing
`npartitions` changes the results, since each partition gets its own stream.

Without a seed, the module generator is seeded from OS entropy when pyqmc is imported,
so every Python process, including a restarted calculation, gets a fresh stream.

### Pitfall: reusing a seed

Passing the same int to every call in your own loop replays the same random numbers
each time:

```python
for i in range(10):
    df, configs = pyq.vmc(wf, configs, nblocks=1, rng=5)  # wrong
```

Each call proposes moves for walkers that were produced by exactly those random
numbers, so the proposals are correlated with the walkers' history and the sampling
is biased. Seed once with `pyqmc.rng.seed(5)`, or create one `Generator` and pass it to
every call. The same goes for calling `pyqmc.rng.seed` with a fixed value before each
restart of a calculation.

### When a stream continues and when a new one starts

A `Generator` object is a stream. Drawing from it advances it, and anything holding the
same object continues it. A new stream starts only in these cases:

- `import pyqmc`: the module generator is seeded from OS entropy.
- `pyqmc.rng.seed(s)`: the module generator is replaced by `default_rng(s)`.
- `rng=` an int or `SeedSequence`: the entry point creates a new `Generator` once.
- `rng.spawn(n)`: `n` independent child streams; the parent advances.

Pickling a `Generator`, which happens when it is sent to a worker process, makes a copy
with the same state. Two processes given the same `Generator` draw the same numbers.

### How the streams flow through a calculation

1. An entry point resolves `rng` once with `pyqmc.rng.get_rng(rng)` and passes the
   resulting `Generator`, never the int, to everything it calls.
2. Without a client, the worker runs in-process and draws from that `Generator`.
3. With a client, the dispatch function (`vmc_parallel`, `sample_overlap_client`,
   `dmc_propagate_parallel`, and the others that call `client.submit`) spawns one child
   per partition and sends each partition its own child. It does this for every block,
   so blocks never reuse random numbers.
4. In `optimize_ensemble`, the threaded sampler spawns one child per sampling job in the
   main thread before starting any thread, so the streams do not depend on thread
   scheduling.

### Writing code that draws random numbers

- **Accumulators and other code called from a worker** call `pyqmc.rng.get_rng()` with
  no argument at the moment they draw. Do not store a `Generator` on an accumulator:
  it would be pickled into every worker with the same state.
- **Workers**, the functions that do the random work for one partition, are decorated
  with `@pyqmc.rng.worker` and take a keyword-only `rng=None`. The decorator resolves
  `rng`, passes it in, and runs the function inside `pyqmc.rng.using(rng)`, which makes
  it the module generator for the duration of the call. The worker draws its moves from
  `rng`, and accumulators reach the same stream through `get_rng()`.
- **Dispatch functions** that call `client.submit` spawn one child per partition with
  `pyqmc.rng.get_rng(rng).spawn(npartitions)` and pass `rng=child` to each task.
- **Functions that loop** over calls that sample resolve `rng` once before the loop.
- **Correlated sampling** deliberately replays: `correlated_compute_worker` saves
  `rng.bit_generator.state` and restores it before evaluating each parameter set, so
  every set sees the same ECP quadrature rotations.

`using` changes a module-level variable, so it is not safe when several threads run
workers at once in the same process. pyqmc's own threads are safe: with a client they
only spawn and submit, and the workers run in the client's processes; without one,
`optimize_ensemble` runs its sampling jobs one at a time. A client that runs workers as
threads in one process is not supported.

### Background

Before this design, pyqmc drew from numpy's global state. That caused two bugs:

1. A driver script that called `np.random.seed(fixed)` and restarted an optimization
   replayed the same random numbers after every restart. Two restarts in a row
   propagated the saved walkers with the stream that produced them, biasing the energy
   for that iteration.
2. Workers forked by `concurrent.futures.ProcessPoolExecutor` on Linux (the default
   start method before Python 3.14) inherited the parent's global state, so all
   workers drew the same numbers. VMC energies drifted in a sawtooth whose period was
   the number of workers ([issue #503](https://github.com/WagnerGroup/pyqmc/issues/503)).
   macOS starts workers with `spawn`, which reseeds them, so the bug did not appear there.
