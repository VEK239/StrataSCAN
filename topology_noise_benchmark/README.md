# Replicated topology and density stress benchmark

This standalone PILOT extends the fixed-target diffuse-background experiment
without changing prior suites or outputs. It uses the established project seeds
`23, 42, 73, 151`, one fixed geometry seed, six 300-point targets, and a fixed
background support. Signal mass does not shrink as background increases.

The compact-Gaussian reference is run at all six noise levels (50--99%). Three
shape screens (anisotropic ellipses, curved moon arcs, and rings) run at 75%,
95%, and 99%; two extreme Gaussian density ratios (16x and 64x) run at 95% and
99%. This is an intentionally staged design: topology and density heterogeneity
are not crossed, avoiding an uninterpretable large factorial.

For compact and elliptical targets, recorded local density is the analytical
Gaussian peak. For moon arcs and rings it is the analytical central tube density
(mass divided by curve length and Gaussian normal width). All cells share a
support calibrated so the reference least-dense compact target has 50x local
contrast at 99% background. Metrics use the existing Hungarian target F1,
pairwise F1, background/noise F1, cluster ratio, fragments, and merges.

Run:

```powershell
$env:PYTHONPATH = "src;."
python -m topology_noise_benchmark.run_benchmark --run-id topology_density_4seed_v1
python -m topology_noise_benchmark.report topology_noise_results\topology_density_4seed_v1
```

The runner uses four bounded, single-threaded subprocesses. AMD-DBSCAN is an
explicit resource skip for every cell; other failures and timeouts are retained.
