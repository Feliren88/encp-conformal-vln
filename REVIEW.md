# ICRA 2027 Mock Review — "Parameter-Free Conformal Uncertainty for Vision-and-Language Navigation"

**Reviewed artifact:** `output_v10/paper/main.pdf` (modular build, 8 pages
incl. references, IEEEtran). Numbers audited against
`output_v10/results/` (7 conditions, 21/21 `verify` checks).

Three reviews in ICRA style (different expertise per reviewer), an AC
meta-review, and an **action plan** that maps every point to a concrete fix
with an effort tag:
**[free]** = already computed in `output_v10/results/`, paper text only;
**[write]** = text/presentation only; **[offline]** = new CPU analysis from
existing dumps; **[GPU]** = new rollouts.

---

## Reviewer 1 — conformal prediction / statistics (confidence: high)

**Rating: Weak Accept.**

### Summary
The paper identifies a genuine failure of standard CP scores on concentrated
navigation policies (threshold collapse, α-insensitivity), proposes dividing
any base score by `2 − p_max` with an episode-maximum calibration unit, and
proves finite-sample simultaneous trajectory coverage under exchangeable
episodes. Validation spans five backbones and two tasks.

### Strengths
- S1. The failure-mode analysis (Sec. IV-D) is the best part of the paper:
  the collapse is explained from bare order statistics, and Table I
  documents it cleanly (q̂ = 0, 100 % singletons at α = 0.30).
- S2. Theorem 1 is correct. The tie-robust counting argument is standard,
  but its application to the episode maximum of a *fixed* score is exactly
  right, and the paper is unusually clear about where a fitted score loses
  the full calibration set (Step 2 / Remark 2).
- S3. The weight family (Table IV) with parameter-free, learned, no-α,
  combined, and frozen-random members is a well-designed ablation: it
  isolates the divisive form as the operative mechanism.

### Weaknesses
- W1. **The headline guarantee is the quantity that fails in deployment.**
  The abstract promises simultaneous coverage ≥ 1 − α "at every step of the
  walk", yet the deployed (seen→unseen) simultaneous coverage is 0.82–0.89
  at α = 0.10. The paper is honest about this, but the reader must assemble
  the caveat from three places (abstract, Sec. VII-B, Remark 1). State it
  once, sharply: *the theorem's quantity is simultaneous coverage; under
  shift only step-averaged coverage empirically clears target.*
- W2. **d_TV is measured but never connected to coverage by an inequality.**
  A one-line lemma exists: if the joint episode laws satisfy
  d_TV(P, Q) ≤ δ, then coverage under Q ≥ 1 − α − δ. Note the direction
  carefully: your plug-in estimate (0.156) is computed on a *marginal* of
  the episode record, hence it **lower-bounds** the joint distance — it can
  certify that the premise is broken, not that the loss is small. Both the
  bound and this one-sided caveat belong in Sec. VII-B; the observed
  simultaneous coverage 0.82–0.89 ≥ 0.90 − 0.156 = 0.744 is at least
  consistent.
- W3. **Exchangeability across episodes is asserted, not discussed.** R2R
  provides ~3 instructions per trajectory and many trajectories per scan,
  so calibration episodes are *not* independent — they cluster by path and
  building. Exchangeability can survive symmetric sampling, but the
  effective sample size behind n = 1021 is smaller than 1021, which matters
  for the ±1/(n+1) slack the paper quotes. One paragraph (or a
  cluster-bootstrap CI on coverage) would close this.
- W4. **No uncertainty on the reported coverages.** With 2349 test episodes
  a binomial 95 % CI on cov_simul is roughly ±0.016; several claims (e.g.
  "clears target in all 45 cells") are near the boundary at α = 0.30. Report
  CIs at least for the headline cells.
- W5. The frozen-random control (0.982 coverage, |C| = 7.0) undermines the
  framing that anything about `1 − p_max` is special at matched calibration:
  *any* positive weight restores α-response. The paper's own conclusion
  ("the divisive form is doing the work") is right — but then the pitch for
  the specific weight rests on interpretability and the transfer property,
  and W6 below is what carries that.
- W6. **Threshold "transfer" is still observational.** Sec. VI-B reports
  that per-backbone thresholds come out numerically close (0.981–1.000).
  The actual experiment — calibrate q̂ on backbone A, deploy on backbone B,
  report coverage/|C| — is not in the paper. (The authors' repo computes
  exactly this matrix; it should be a 5×5 table or a one-line summary.)

### Questions
- Q1. In Table IV the combined weight matches parameter-free coverage to
  three decimals (0.968). Does the learned residual collapse to ~0? If so,
  say it — it is evidence that `1 − p_max` is (locally) sufficient.
- Q2. How were RAPS hyperparameters (λ = 0.1, k_reg = 2) chosen, and are
  the RAPS conclusions sensitive to them?
- Q3. Step 3's upper bound assumes distinct scores; with fp16 logits, ties
  in episode maxima are not measure-zero. Any empirical tie rate?

---

## Reviewer 2 — VLN / robot systems (confidence: medium-high)

**Rating: Borderline (lean accept).**

### Summary
A practical recipe for calibrated uncertainty on discrete VLN agents,
validated across DUET/HAMT/RecBERT on R2R and DUET/HAMT on REVERIE with
reproduced checkpoints and a sanity gate tying every CP number to published
success rates. No closed-loop use of the sets.

### Strengths
- S1. Reproduction discipline is exemplary: every backbone matches its
  published val-unseen SR within a point before any CP number is reported,
  and the non-intervening rollout guarantees the baseline is untouched.
- S2. The REVERIE object-grounding result is an honest negative: the
  normalisation does not rescue a single-shot, strongly-shifted classifier,
  and in-distribution recalibration isolates the cause. Papers rarely
  include this.
- S3. The forklift framing and the saturation condition give practitioners
  something they can actually evaluate on their own softmax.

### Weaknesses
- W1. **The forklift never moves.** The paper motivates safety–throughput
  trade-offs, but no experiment acts on a prediction set: no query budget,
  no help-seeking policy, no SR-vs-interventions curve. Even an *offline*
  operating curve (query when |C| > τ → what fraction of would-be argmax
  errors is caught at what ask-rate, vs a p_max baseline) would
  substantially strengthen the practical claim. A closed-loop
  `set_restrict`-style experiment is what would move my score to Accept.
- W2. **Saturation 62–94 % at α = 0.10 is close to "ask always".** On
  REVERIE-HAMT (sat 0.94) the set is the whole action space on nearly every
  step; the information content of the set beyond "the policy is unsure" is
  then minimal. The limitation is acknowledged, but Table III would be more
  honest with a "fraction of steps where the set is strictly informative"
  column, or the query-budget view of W1.
- W3. **Discrete panoramic VLN only.** No VLN-CE / continuous control, no
  real robot, and the teacher comes from a shortest-path oracle on the nav
  graph. Fine for scope, but the title's generality ("Vision-and-Language
  Navigation") slightly oversells; one sentence on what breaks in
  continuous action spaces (no finite candidate set → CP over what?) is
  needed.
- W4. RecBERT's absence from REVERIE is properly explained (no public
  checkpoint), but then Table III has two backbones — call it what it is in
  the text ("both agents with public REVERIE checkpoints").
- W5. Runtime/latency is claimed ("no extra computation in the control
  loop") but never measured. One number (µs per step for the membership
  test) makes the claim concrete.

### Questions
- Q1. At matched ask-rate, does |C| > τ outperform a plain p_max threshold
  for triggering help? (This is the decisive practicality question.)
- Q2. Do saturated steps cluster (corridors, staircases) or spread
  uniformly? A per-scan breakdown would tell a roboticist where the method
  is useful.

---

## Reviewer 3 — uncertainty quantification / ML (confidence: medium)

**Rating: Weak Accept.**

### Summary
A parameter-free normalised nonconformity score for sequential
overconfident policies, an episode-max calibration unit, and a broad
empirical study. The method is simple (a strength), the theory is standard
machinery applied cleanly, and the evaluation is wide but shallow on
baselines.

### Strengths
- S1. Simplicity with a guarantee: zero training, full calibration set,
  one subtraction per step. Adoption cost is essentially nil.
- S2. The paper correctly separates what needs a data split (learned
  members) from what does not, and evaluates both under a split-matched
  protocol — this is the right comparison and is often done wrong.
- S3. Dense α-sweep (135 cells) demonstrates the restored α-response is not
  cherry-picked at three levels.

### Weaknesses
- W1. **Missing CP baselines.** Under covariate shift the canonical
  remedies are weighted CP (likelihood-ratio weights), Mondrian /
  group-conditional CP (e.g. by |A_t| or by scan), and CV+/Jackknife+ for
  data efficiency with learned scores. None appears. Weighted CP and
  Mondrian are computable offline from the authors' own dumps; their
  absence is the paper's largest experimental gap.
- W2. **No comparison to cheap recalibration.** Temperature scaling on the
  calibration split followed by standard APS is the obvious "fix the
  overconfidence at the source" baseline. If it also restores α-response,
  the case for the new score narrows to the no-training property; if it
  does not (because a single temperature cannot fix per-step heterogeneity),
  that is a strong argument *for* the paper. Either way the experiment is
  cheap and decisive.
- W3. Related work misses cross-conformal methods (CV+/JK+), conditional
  CP, and the recent sequential/online CP line beyond ACI; the "dependence,
  graphs, shift" paragraph cites the right anchors but the positioning
  against weighted CP is one clause.
- W4. The SR-adaptive difficulty-ratio target for the learned weight is
  unmotivated in the paper (it appears only in code). Since Table IV leans
  on these members, one sentence on what the MLP is trained to predict is
  required for reproducibility.
- W5. Figure 2 (α-sweep) and Table II partially duplicate; with the page
  budget exhausted, consider replacing Figure 2 with the transfer matrix or
  the query-budget curve (higher information density).

### Questions
- Q1. Weighted CP with a simple logistic-regression density ratio on
  φ(x) — does it recover simultaneous coverage under the seen→unseen shift?
- Q2. Mondrian by degree class: does per-class calibration reduce the
  saturation on small action spaces?
- Q3. Why MSE on a capped ratio for the weight MLP rather than pinball /
  quantile losses, which target the calibration quantile more directly?

---

## AC Meta-Review

All three reviewers acknowledge a real, well-diagnosed problem, a clean and
correctly-proven fix, and unusually disciplined reproduction. The shared
reservations are (i) no experiment acts on the sets (R2-W1), (ii) missing
shift-era CP baselines: weighted / Mondrian / temperature-scaled (R3-W1,
R3-W2), and (iii) the simultaneous-vs-step-averaged coverage story needs to
be stated once, sharply, with an inequality connecting d_TV to the loss
(R1-W1, R1-W2). None of these requires new theory, and (ii) is computable
from the authors' released dumps. **Recommendation: accept if the rebuttal
delivers the offline query-budget curve, at least one shift baseline, and
the sharpened coverage statement; otherwise reject as a promising but
incomplete systems-statistics hybrid.**

---

# Action plan (developed from the reviews)

Ordered by (rebuttal value ÷ effort). Page budget is **zero**: every paper
addition names its donor content.

| # | Review pt | Action | Effort | Status / where |
|---|---|---|---|---|
| 1 | R2-W1, R2-Q1 | **Query-budget table**: recall of argmax errors at ask-rates 5/10/20/30 % for set-size vs p_max trigger. | **[free]** | already in `results/*.json → diagnostics.query_budget`; add 4-row table, donor: Fig. 2 (see #10) |
| 2 | R1-W6 | **Transfer matrix**: q̂ from backbone A applied to B, report min/max coverage over the 5×5 grid (one sentence + 2 numbers, or small table). | **[free]** | `results/transfer.json` |
| 3 | R1-W2 | **d_TV inequality remark**: coverage ≥ 1 − α − δ for joint d_TV ≤ δ; plug-in marginal estimate lower-bounds δ (one-sided caveat). | **[write]** | 4 sentences in Sec. VII-B |
| 4 | R1-W1 | One sharp sentence in abstract + Sec. VII-B: theorem = simultaneous; under shift only step-averaged clears target. | **[write]** | rewrite 2 sentences |
| 5 | R1-W4 | Binomial (or cluster-bootstrap, also fixes R1-W3) CIs on headline coverages. | **[offline]** | ~30 lines in `cp_core/analyses.py`, cluster by scan |
| 6 | R3-W2 | **Temperature scaling + base APS** baseline: fit T on val_seen, rerun base CP. Decisive either way. | **[offline]** | ~40 lines; dumps have full logits |
| 7 | R3-W1, R3-Q2 | **Mondrian by degree class**: per-class q̂, conditional coverage + saturation. | **[offline]** | new function in `cp_core`, dumps sufficient |
| 8 | R3-W1, R3-Q1 | **Weighted CP**: logistic-regression density ratio on φ(x), weighted quantile. | **[offline]** | ~60 lines; report in rebuttal even if not in paper |
| 9 | R1-Q1, R3-W4 | State the MLP target (difficulty ratio) and report the combined member's residual magnitude. | **[write]** | 2 sentences at Table IV |
| 10 | R3-W5 | Reclaim page space: drop Fig. 2 (α-sweep; its content = "no collapse", already in Table II + one sentence citing 135/135 cells) → frees ~0.3 page for #1+#2. | **[write]** | layout change |
| 11 | R2-W5 | Measure membership-test latency (µs/step, CPU). | **[offline]** | trivial timing script |
| 12 | R2-W3 | One sentence on continuous VLN-CE: finite candidate set is the boundary of applicability. | **[write]** | limitations §
| 13 | R1-Q3 | Report empirical tie rate in episode-max scores (fp16). | **[free-ish]** | one pass over dumps |
| 14 | R1-W5 | Reframe random-MLP row: form does the work; pf earns its place via transfer (#2) + interpretability. | **[write]** | 1 sentence |
| 15 | R2-W1 | **Closed-loop experiment** (query when \|C\|>τ, human gives teacher action, measure SR vs ask-rate): the accept-maker. | **[GPU]** | needs a guarded intervention rollout in `vln_backends/rollouts.py` (~1 day + ~1 h GPU) |
| 16 | — | Presentation: fix 3 overfull display equations in Sec. V (log lines 524/536/573); Table III caption "both agents with public REVERIE checkpoints" (R2-W4). | **[write]** | cosmetic |

**Suggested order:** #3, #4, #9, #14, #16 (one writing pass) → #10 + #1 + #2
(one layout pass, net page change ≈ 0) → #5, #6, #7, #11, #13 (one offline
batch on existing dumps) → #8 for the rebuttal → #15 if targeting a clear
accept rather than a borderline.

---

## Integration status (post-review revision)

All 16 items are integrated in code and, where they belong there, in the
paper. Implementations: `cp_core/baselines.py` (#5 scan-cluster CIs,
#6 temperature scaling, #7 Mondrian, #8 weighted CP, #11 latency, #13 ties;
`conformal_vln.py baselines` → `results/baselines.json`) and
`vln_backends/rollouts.py::closedloop_duet_split` (#15;
`conformal_vln.py closedloop` → `results/closedloop.json`,
figure `paper/figures/fig_closedloop.png`).

Headline outcomes the revision leans on:
- **#6**: temperature scaling CANNOT restore APS/RAPS alpha-response —
  a temperature preserves ranks and top-rank APS is exactly 0, so the
  q̂=0 collapse survives the best-fit T (2.1–7.1); decisive FOR the paper.
- **#8**: weighted CP partially recovers simultaneous coverage
  (HAMT 0.83→0.91; 0.83–0.93 across conditions) at ESS 259–818/1021.
- **#7**: Mondrian-by-degree shrinks sets (1.1 vs 2.7, small class) but
  undercovers 0.78–0.89 — supports the episode-max design choice.
- **#5**: lowest scan-cluster 95% lower bound on cov_step is 0.943 (HAMT).
- **#2**: cross-applied thresholds hold coverage 0.887–1.000 (20 pairs).
- **#1**: offline, the p_max trigger catches slightly MORE argmax errors
  at matched budgets (e.g. 0.19 vs 0.17 at 10%); the set trigger's case is
  the calibrated, transferable operating point — the paper says so honestly.
- **#13**: ties ≤ 2.3% of episodes; **#11**: membership test 46–95 µs/step.
- **#12**: new Discussion paragraph "Beyond the discrete graph" positions
  continuous VLN-CE / waypoint-head conformalisation as future work.
- **#15 (closed loop, DONE)**: simulated operator answers when the trigger
  fires, 15 policies on DUET-full val-unseen. Set trigger: SR 71.2 -> 76.9
  at 6% ask (tau=15), 82.5 at 14%, 91.2 at 32%, 98.0 at 57%. Tuned p_max
  cutoff slightly stronger at matched mid budgets (95.0 at 34%); the paper
  frames the set trigger's advantage honestly as the calibrated,
  transferable operating point. Figure: paper/figures/fig_closedloop.png;
  data: results/closedloop.json.
