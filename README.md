# output_v10 — Normalised-CP weight family (standalone rewrite)

Complete rewrite of the CP pipeline. **Zero imports from output_v8 /
output_v8_ablations / output_v9.** One score family unifies the
parameter-free and learning-based weights the paper discusses:

```
s_w(x_t, a) = s_base(x_t, a) / (1 + w(x_t)),   w >= 0
```

| member | weight `w(x_t)` | calibration set | role |
|---|---|---|---|
| `w0` | 0 | full **and** half 2 | episode-max base (unit ablation) |
| `pf` | `1 − p_max` | full **and** half 2 | parameter-free (paper headline) |
| `mlp` | `w_ψ(φ(x))` learned MLP | half 2 (fit on half 1) | learned member |
| `mlp_noalpha` | same, α feature dropped | half 2 (fit on half 1) | paper `tab:learned` A3 row |
| `hybrid` | `(1 − p_max) + r_ψ(φ(x))` | half 2 (fit on half 1) | **the combination**: parameter-free floor + learned residual |
| `random` | frozen random MLP | half 2 | control: form vs learning |

Calibration unit is always the per-episode maximum over teacher actions;
quantile is the corrected split-conformal one. Prediction sets range over
valid (finite-logit) candidates only. MLP hyperparameters match v8's
`ALPSConfig` (hidden 32, 2 layers, 300 epochs, Adam lr 1e-3, wd 1e-3,
batch 512; SR-adaptive difficulty-ratio target capped at 10).

**Faithfulness check**: on the v8 `duet_full` dump this code reproduces the
paper's numbers exactly — cov_step 0.968, cov_simul 0.822, |C| 6.83,
saturation 0.70, d_TV 0.156 (degree piece 0.070).

## Architecture

One CLI entry point; two packages with one responsibility each.

```
conformal_vln.py             THE entry point (subcommands below)

cp_core/                     pure CP domain (CPU-only, path-agnostic, testable)
├── scores.py                base scores (THR/APS/RAPS) + conformal quantile
├── split.py                 Split: one rollout split as flat per-step arrays
├── weights.py               WeightMLP + WEIGHT_FAMILY registry (add a member
│                            = add one entry) + half-1 fitting
├── evaluation.py            quantiles, coverage/efficiency, conditionals
├── analyses.py              family grid, d_TV (+ sensitivity), object head,
│                            transfer, in-distribution, dense sweep,
│                            qualitative episode
├── reporting.py             verify assertions, figures HTML
└── tests.py                 unit tests (synthetic data, seconds)

vln_backends/                GPU/simulator side (imported lazily by `run` only)
├── config.py                machine paths + dataset config (side-effect free)
├── bootstrap.py             THE one side-effectful import: chdir, MatterSim,
│                            DUET modules (order-sensitive), re-exported
├── isolation.py             sys.modules swap contexts (HAMT / RecBERT)
├── adapters.py              one step interface over HAMT + RecBERT
├── builders.py              DUET / HAMT-R2R / HAMT-REVERIE / RecBERT builders
├── rollouts.py              three record rollouts (deliberately not unified)
└── metrics.py               SR/SPL/nDTW/CLS + the sanity gate
```

## Usage

```bash
PY=/home/vfeliren1/pr65_scratch2/vfvic1/conda/envs/vln_duet_conformal/bin/python

# GPU — one condition end-to-end (build -> rollout -> sanity gate -> dump -> CP)
$PY conformal_vln.py run --backend duet --action_space full      # R2R
$PY conformal_vln.py run --backend duet --action_space local
$PY conformal_vln.py run --backend hamt
$PY conformal_vln.py run --backend recbert --recbert_variant prevalent
$PY conformal_vln.py run --backend recbert --recbert_variant oscar
$PY conformal_vln.py run --backend duet --dataset reverie        # REVERIE (nav+object)
$PY conformal_vln.py run --backend hamt --dataset reverie

# Offline — CPU analyses from dumps/*.pt
$PY conformal_vln.py analyze        # weight-family grid -> results/cp_results.json
$PY conformal_vln.py dense          # 9-alpha sweep      -> results/cp_dense.json
$PY conformal_vln.py qualitative --condition duet_full   # fig_qualitative source
$PY conformal_vln.py transfer       # cross-backbone q-hat matrix
$PY conformal_vln.py indist         # exchangeable val_unseen halves check
$PY conformal_vln.py baselines      # REVIEW items: temperature scaling, weighted
                                    # CP, Mondrian, scan-cluster CIs, ties, latency
$PY conformal_vln.py figures        # aggregate HTML figures
$PY conformal_vln.py paperfigs      # print PNGs -> paper/figures/ (matplotlib)
$PY conformal_vln.py test           # unit tests
$PY conformal_vln.py verify         # assert paper claims vs cp_results.json

# GPU — closed-loop help-seeking (REVIEW item #15, the accept-maker)
$PY conformal_vln.py closedloop     # trigger sweep on DUET-full R2R
                                    # -> results/closedloop.json
```

**RecBERT × REVERIE is an explicit placeholder**: the public
Recurrent-VLN-BERT release (checkpoints on Zenodo) is R2R-only; no REVERIE
code or checkpoint was ever published.

## Paper and review

- **`paper/`** — the modular ICRA draft (`main.tex` → `preamble.tex`,
  `sections/00–08`, `tables/tab_*.tex`). Build:
  `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
  Currently **exactly 8 pages** incl. references — zero margin, so every
  addition must name its donor content.
- **`REVIEW.md`** — three ICRA-2027-style mock reviews of `paper/main.pdf`
  (CP/statistics, VLN/systems, UQ/ML) + AC meta-review, developed into a
  16-item action plan ordered by rebuttal-value ÷ effort. Items tagged
  **[free]** are already computed in `results/` (query-budget curve,
  transfer matrix, tie rates) and only need paper text; the single
  accept-maker is the closed-loop query experiment (**[GPU]**, item #15).

## Guarantee bookkeeping

Parameter-free members may use the full calibration split (the fixed-map
condition of the coverage theorem). Learned members (`mlp`, `mlp_noalpha`,
`hybrid`) are fit on calibration half 1 and take the quantile on held-out
half 2, which preserves exchangeability and the guarantee. `family` blocks in
the results are split-matched (everything on half 2) for a fair comparison;
`family_full` holds the parameter-free members at their full-set entitlement.
