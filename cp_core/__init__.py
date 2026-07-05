"""Pure-CPU conformal-prediction domain for the weight family.

weight family. No simulator, no GPU, no filesystem paths: every function
takes data in and returns data out. See README.md for the method.
"""

from cp_core.scores import (
    SCORES,
    LAM,
    KREG,
    EPS,
    base_scores_all,
    softmax_valid,
    teacher_fallback,
    conformal_quantile,
)
from cp_core.split import Split, T_MAX
from cp_core.weights import (
    WeightMLP,
    WEIGHT_FAMILY,
    WEIGHTS,
    PARAMETER_FREE,
    fit_weight_models,
    difficulty_target,
)
from cp_core.evaluation import (
    epmax_quantile,
    pooled_quantile,
    evaluate,
    conditional_diagnostics,
)
from cp_core.analyses import (
    ALPHAS,
    DENSE_ALPHAS,
    evaluate_condition,
    dtv_plugin,
    dtv_sensitivity,
    evaluate_object_head,
    threshold_transfer,
    evaluate_indist,
    run_condition,
    dense_sweep,
    qualitative_episode,
)
