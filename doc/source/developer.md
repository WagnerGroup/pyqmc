Developer information 
-------------------------------

### Performance profiling
Benchmarks are present in the /benchmark directory in the repository. These benchmarks can be run directly, or using [asv](https://asv.readthedocs.io).


To run benchmarks on `master`:
```
asv run
```

To compare master to your branch:
```
asv run master..[yourbranch]
```

To compare two versions using SHA tags (you can get them from `git log`).
```
asv compare [version1] [version2]
```
You can also compare your branch to the master branch by doing 
```
asv compare master [yourbranch]
```

To profile a particular benchmark using `snakeviz`

```
asv profile "h2o_benchmark.H2OSuite.time_pgradient_slater()" --gui snakeviz
```