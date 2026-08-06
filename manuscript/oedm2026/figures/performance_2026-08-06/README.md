# Reproducible OEDM performance figures

These artifacts are generated exclusively from repository data and generators by:

```bash
PYTHONPATH=src:. python manuscript/oedm2026/scripts/make_manuscript_figures.py
```

Outputs:

- `fig1_noise99_dotwhisker`: target-wise, noise, and pairwise F1 on six 99% diffuse-background benchmarks;
- `fig2_noise_fraction_sweep`: response to 50--99% diffuse background;
- `fig3_scalability`: successful-run runtime and categorical completion/failure states;
- `fig4_standard_benchmark_gallery`: first six standard synthetic generators;
- `fig5_noise99_benchmark_gallery`: six exact topology-noise generators;
- `table1_noise99_granularity`: cluster-count granularity summary.

Policy: no image generation, no heatmaps, no missing/failure-to-zero imputation. Pending scalability cells remain pending rather than being scored as failures.
