# Future-shift severity figure

Source: exact summary means and standard deviations supplied in the user request.
GRACE S6 mean and standard deviation are missing from that request. No actual
artifact source was established; the user subsequently instructed us to stop
checking and draw only the given numbers. GRACE S6 is absent from both the plot
and CSV. No value was inferred from the supplied Overall aggregate.

The figure contains 29 points, using external stress multipliers
0.50, 0.75, 1.00, 1.25, 1.50, and 2.00. These are independent of GRACE's internal
Wasserstein radius. Overall is not plotted. Error bars use the reported standard
deviations without smoothing or adjustment; their extent is 79.8–94.4%, within
the explicitly labeled 78–96% axis. No methods were rerun or raw outputs modified.
Run config, command, seed, checkpoint and artifact path for these supplied
aggregates are not available; no unrelated run is assigned as their provenance.

Outputs: `figures/future_shift_severity.pdf`, `figures/future_shift_severity.png`,
and `figures/future_shift_severity_values.csv`. The PDF uses vector graphics and
embedded Liberation Serif, a Times-compatible font matching the manuscript style.
The canvas is 6.75 by 3.65 inches. Reproduce rendering only with:

```sh
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python figures/plot_future_shift_severity.py
```

The requested LaTeX snippet is saved in `figures/future_shift_severity.tex`.
Its image path is relative to the repository root; when compiling from `paper/`,
use `../figures/future_shift_severity.pdf` instead. The manuscript is unchanged.
