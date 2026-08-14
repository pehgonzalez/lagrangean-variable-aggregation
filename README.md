# Variable Aggregation in Lagrangean Decomposition

Code and data accompanying the paper

> P. H. González, R. Schneider, N. Maculan and H. Reinoso, *Variable Aggregation in
> Lagrangean Decomposition: Exactness, Incomparability, and the Support
> Rule* (submitted).

The paper studies the aggregated coupling `sum_k y^k = (p-1) x` in the
Lagrangean decomposition of block-structured 0-1 programs: the complete
partial order of the resulting family of dual bounds (from the merged
decomposition to the continuous-copy variant), exact counterexamples showing
that the aggregated bound is not dominated by the classical Lagrangean bound
(with unbounded failure), and structural conditions (block symmetry, the
support rule, cross-feasibility) under which aggregation is exact or
dominated. Every counterexample carries certificates in exact rational
arithmetic and every theorem is verified computationally.

## Quick start

```bash
pip install -r requirements.txt
python3 run_all.py          # quick mode, a few minutes
python3 run_all.py --full   # full sample sizes of the paper, about 15 minutes
```

Most of the quick-mode time goes to A09, whose point is precisely that the
joint subproblems do not close within their 15-second limits.

The suite runs one experiment per claim (A01-A09), prints the outcome and
writes one report per claim under `results/`, plus `results/00_SUMMARY.txt`.
Exit code 0 means every claim validated. All experiments are deterministic
under the seeds fixed in the scripts.

## Layout

| Path | Contents |
|---|---|
| `run_all.py` | entry point; one experiment per claim of the paper |
| `experiments/verify_certificates.py` | exact rational verification of all counterexample certificates |
| `experiments/battery.py` | randomized battery: thirteen proved statements over five instance families |
| `experiments/corner_battery.py` | hypothesis-variant battery (merged dual, nonempty S, multi-row blocks, paths, rectangular matrices) |
| `experiments/exhaustive_small_universe.py` | exhaustive check of a complete finite universe (16,384 instances, full subset lattice) plus a third independent method |
| `experiments/disaggregation_path.py` | greedy disaggregation path with certified termination |
| `experiments/subgradient_support.py` | support-rule demonstration under a projected subgradient method |
| `experiments/sparse_benchmark.py` | three duals over identical instances: dense vs aggregated vs sparse coupling, bounds and wall times |
| `experiments/four_dual_benchmark.py` | four duals (dense, aggregated, sparse, merged) with instrumented knapsack/MILP time split, 25 seeds, medians |
| `experiments/q3_trajectory.py` | merged dual with truncated (valid-bound) joint evaluations on strongly correlated data: visited multipliers and equal-time comparison |
| `experiments/paired_control.py` | paired control of the two dual-value implementations (hull LP vs epigraph LP of the piecewise linear dual) |
| `experiments/jointly_hard.py` | cost of one exact dual evaluation: blockwise knapsacks vs joint subproblem |
| `data/battery_families.jsonl` | raw battery outputs by family (seeds 100-104); regenerate with `battery.py 2500 SEED out.jsonl` |
| `data/battery_rates.jsonl` | raw battery outputs with rates of the refuted comparisons |

## Requirements

Python 3.10+, `numpy`, `scipy`, `highspy` (the HiGHS solver). Dual values on
small instances are computed exactly through Geoffrion primal
characterizations, as linear programs over enumerated 0-1 points; certificate
verification uses pure fraction arithmetic and no solver.

## License

MIT. See `LICENSE`.

## Citation

Please cite the paper above. Repository: https://github.com/pehgonzalez/lagrangean-variable-aggregation
