"""ETBP-TD numerical model (DOI 10.1109/TMC.2024.3489717)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    alpha0: float
    alpha1: float
    sigma: float
    history_window: int
    initial_stage_periods: int
    period_weight_rule: str
    within_period_weight_rule: str
    high_threshold: float = 0.9
    low_threshold: float = 0.4
    beta0: float = 0.3
    beta1: float = 0.4
    history_fraction: float = 0.5

    def validate(self):
        if (
            min(self.alpha0, self.alpha1) < 0
            or not np.isfinite([self.alpha0, self.alpha1]).all()
        ):
            raise ValueError("Finite nonnegative quality coefficients required")
        if (
            not 0 <= self.sigma <= 1
            or self.history_window < 1
            or self.initial_stage_periods < 0
        ):
            raise ValueError(
                "Declare a normalized tolerance, history window, and initial stage length"
            )
        if not 0 <= self.low_threshold < self.high_threshold <= 1:
            raise ValueError("Ordered trust thresholds required")
        if min(self.beta0, self.beta1) < 0 or not 0 <= self.history_fraction <= 1:
            raise ValueError("Valid trust update steps/mixture required")
        for rule in [self.period_weight_rule, self.within_period_weight_rule]:
            if rule not in ["uniform", "linear_recency"]:
                raise ValueError(
                    "Choose and disclose a normalized nondecreasing history-weight rule"
                )


@dataclass
class State:
    trust: np.ndarray
    initial_trusted: np.ndarray
    evaluation_count: np.ndarray
    recruited_count: np.ndarray
    history: list
    period: int = 0


def initialize(n, initial_trusted, cfg):
    cfg.validate()
    initial = np.asarray(initial_trusted, bool)
    if initial.shape != (n,):
        raise ValueError("One registered initial-trusted mask per worker required")
    trust = np.where(initial, 1.0, 0.5)
    return State(trust, initial.copy(), np.ones(n), np.zeros(n, int), [trust.copy()])


def _weights(length, rule):
    weights = (
        np.ones(length) if rule == "uniform" else np.arange(1, length + 1, dtype=float)
    )
    return weights / weights.sum()


def quality(incidence, selected, worker_quality, task_mask=None):
    A = np.asarray(incidence, bool)
    ids = np.asarray(selected, int)
    active = (
        np.ones(A.shape[1], bool) if task_mask is None else np.asarray(task_mask, bool)
    )
    if not len(ids):
        return 0.0
    return float(
        np.max(A[ids] * np.asarray(worker_quality)[ids, None], axis=0)[active].sum()
    )


def _greedy_quality(A, candidates, capacity, worker_quality, task_mask):
    selected = []
    candidates = list(sorted((int(i) for i in candidates)))
    m = A.shape[1]
    maxima = np.zeros(m)
    while candidates and len(selected) < capacity:
        covered = A[selected].any(0) if selected else np.zeros(m, bool)
        if not np.any(task_mask & ~covered):
            break
        gains = np.asarray(
            [
                np.maximum(maxima, A[i] * worker_quality[i])[task_mask].sum()
                - maxima[task_mask].sum()
                for i in candidates
            ]
        )
        pos = int(np.argmax(gains))
        if gains[pos] <= 0:
            break
        worker = candidates.pop(pos)
        selected.append(worker)
        maxima = np.maximum(maxima, A[worker] * worker_quality[worker])
    return selected


def recruit(incidence, K, state, rng, cfg):
    cfg.validate()
    A = np.asarray(incidence, bool)
    n = len(state.trust)
    if A.ndim != 2 or A.shape[0] != n or (not 0 <= K <= n) or (int(K) != K):
        raise ValueError("Worker-by-task incidence and feasible integer K required")
    K = int(K)
    p = state.period + 1
    high = np.flatnonzero(state.trust > cfg.high_threshold)
    unknown = np.flatnonzero(
        (state.trust > cfg.low_threshold) & (state.trust < cfg.high_threshold)
    )
    high_primary = _greedy_quality(A, high, K, state.trust, np.ones(A.shape[1], bool))
    high_coverage = (
        A[high_primary].any(0) if high_primary else np.zeros(A.shape[1], bool)
    )
    remaining = K - len(high_primary)
    verification_reward = (A & high_coverage).sum(1).astype(float)
    exploration = np.sqrt(
        (remaining + 1) * np.log(state.evaluation_count.sum()) / state.evaluation_count
    )
    ucb_quality = (
        cfg.alpha0 * state.trust + cfg.alpha1 * verification_reward + exploration
    )
    unknown_primary = _greedy_quality(
        A, unknown, remaining, ucb_quality, ~high_coverage
    )
    selected = list(high_primary) + list(unknown_primary)
    (second_initial, second_unknown, second_evaluated) = ([], [], [])

    def choose_random(candidates, count):
        values = np.asarray(
            sorted(set((int(i) for i in candidates)) - set(selected)), int
        )
        number = min(count, len(values))
        return (
            list(map(int, rng.choice(values, size=number, replace=False)))
            if number
            else []
        )

    if len(selected) < K and p <= cfg.initial_stage_periods:
        second_initial = choose_random(
            np.flatnonzero(state.initial_trusted), K - len(selected)
        )
        selected += second_initial
    if len(selected) < K:
        candidates = sorted(
            set(map(int, unknown)) - set(selected), key=lambda i: (-ucb_quality[i], i)
        )
        second_unknown = candidates[: K - len(selected)]
        selected += second_unknown
    if len(selected) < K:
        candidates = set(map(int, high)) - set(np.flatnonzero(state.initial_trusted))
        second_evaluated = choose_random(candidates, K - len(selected))
        selected += second_evaluated
    if len(set(selected)) != len(selected):
        raise AssertionError("A cardinality set cannot contain duplicate workers")
    return {
        "period": p,
        "selected": np.asarray(selected, int),
        "K": K,
        "primary_high": np.asarray(high_primary, int),
        "primary_unknown": np.asarray(unknown_primary, int),
        "second_initial": np.asarray(second_initial, int),
        "second_unknown": np.asarray(second_unknown, int),
        "second_evaluated": np.asarray(second_evaluated, int),
        "incidence": A.copy(),
        "pre_trust": state.trust.copy(),
        "verification_reward": verification_reward,
        "ucb_quality": ucb_quality,
        "capacity_shortfall": K - len(selected),
    }


def truth_values(reports, selected, pre_trust, cfg):
    data = np.asarray(reports, float)
    use = np.zeros(len(data), bool)
    use[np.asarray(selected, int)] = True
    valid = np.isfinite(data) & use[:, None]
    high = valid & (pre_trust[:, None] > cfg.high_threshold)
    unknown = (
        valid
        & (pre_trust[:, None] > cfg.low_threshold)
        & (pre_trust[:, None] < cfg.high_threshold)
    )
    priority = np.where(high.any(0)[None, :], high, unknown)
    weights = np.where(priority, pre_trust[:, None], 0.0)
    denominator = weights.sum(0)
    numerator = np.where(valid, data, 0.0) * weights
    truth = np.divide(
        numerator.sum(0),
        denominator,
        out=np.full(data.shape[1], np.nan),
        where=denominator > 0,
    )
    return (truth, high.sum(0), unknown.sum(0))


def feedback(reports, selection, state, Dmax, cfg):
    cfg.validate()
    if not np.isfinite(Dmax) or Dmax <= 0:
        raise ValueError("Strictly positive public historical Dmax required")
    if selection["period"] != state.period + 1 or not np.array_equal(
        selection["pre_trust"], state.trust
    ):
        raise ValueError("Selection and feedback must share the same pre-period state")
    raw = np.asarray(reports, float)
    A = selection["incidence"]
    if raw.shape != A.shape:
        raise ValueError("Actual worker-by-task reports required")
    selected = selection["selected"]
    admitted = np.zeros(len(state.trust), bool)
    admitted[selected] = True
    data = np.where(A & admitted[:, None], raw, np.nan)
    pre = state.trust
    (truth, N_high, N_unknown) = truth_values(data, selected, pre, cfg)
    differences = data - truth[None, :]
    reference = np.isfinite(data) & (pre[:, None] >= cfg.high_threshold)
    wr = np.where(reference, pre[:, None], 0.0)
    denom = wr.sum(0)
    benchmark = np.divide(
        (np.where(reference, differences, 0.0) * wr).sum(0),
        denom,
        out=np.full(A.shape[1], np.nan),
        where=(denom > 0) & (N_high >= 1),
    )
    new_trust = pre.copy()
    actual_evaluations = np.zeros(len(pre), int)
    history = np.asarray(state.history[-cfg.history_window :])
    historical_mean = _weights(len(history), cfg.period_weight_rule) @ history
    updates = []
    for worker in selected:
        within = [float(pre[worker])]
        for task in np.flatnonzero(np.isfinite(data[worker])):
            permitted = N_high[task] > 1 or (
                N_high[task] == 1 and pre[worker] < cfg.high_threshold
            )
            if not permitted or not np.isfinite(benchmark[task]):
                continue
            xi = abs(differences[worker, task] - benchmark[task]) / Dmax
            within_mean = float(
                _weights(len(within), cfg.within_period_weight_rule) @ within
            )
            base = (
                cfg.history_fraction * historical_mean[worker]
                + (1 - cfg.history_fraction) * within_mean
            )
            previous = within[-1]
            if xi <= cfg.sigma:
                increment = cfg.beta0 * (1 - xi) * (1 - previous)
                value = min(base + increment, 1.0)
            else:
                increment = -cfg.beta1 * xi * previous
                value = max(base + increment, 0.0)
            within.append(float(value))
            actual_evaluations[worker] += 1
            updates.append(
                {
                    "worker": int(worker),
                    "task": int(task),
                    "normalized_deviation": float(xi),
                    "previous_within": previous,
                    "history_base": float(base),
                    "next_trust": float(value),
                }
            )
        new_trust[worker] = within[-1]
    if not np.isfinite(new_trust).all() or np.any((new_trust < 0) | (new_trust > 1)):
        raise AssertionError("Native trust update left [0,1]")
    counts = state.evaluation_count.copy()
    (primary_high, primary_unknown) = (
        selection["primary_high"],
        selection["primary_unknown"],
    )
    primary_coverage = (
        A[primary_high].any(0) if len(primary_high) else np.zeros(A.shape[1], bool)
    )
    for worker in primary_high:
        counts[worker] += (
            np.count_nonzero(A[worker] & (primary_coverage & ~A[worker])) + 1
        )
    for worker in primary_unknown:
        counts[worker] += np.count_nonzero(A[worker] & primary_coverage)
    recruited = state.recruited_count.copy()
    recruited[selected] += 1
    next_state = State(
        new_trust,
        state.initial_trusted.copy(),
        counts,
        recruited,
        state.history + [new_trust.copy()],
        state.period + 1,
    )
    return {
        "truth": truth,
        "next_state": next_state,
        "actual_evaluation_count": actual_evaluations,
        "native_evaluation_count": counts,
        "N_high": N_high,
        "N_unknown": N_unknown,
        "benchmark_difference": benchmark,
        "updates": updates,
        "implementation_scope": "Full paper-based numerical three-stage recruitment/weighted TD/history trust update; plaintext-equivalent only. Printed ambiguities and non-default history parameters are disclosed.",
    }


def recruit_offers(incidence, K, state, rng, cfg, offers):
    cfg.validate()
    A = np.asarray(incidence, bool)
    n = len(state.trust)
    if A.ndim != 2 or A.shape[0] != n or (not 0 <= K <= n) or (int(K) != K):
        raise ValueError("Worker-by-task incidence and feasible integer K required")
    offers = np.asarray(offers, bool)
    if offers.shape != (n,):
        raise ValueError("Public registered-worker participation mask required")
    K = int(K)
    if K > int(offers.sum()):
        raise ValueError("Native quota must be capped by public availability")
    p = state.period + 1
    high = np.flatnonzero(offers & (state.trust > cfg.high_threshold))
    unknown = np.flatnonzero(
        offers & (state.trust > cfg.low_threshold) & (state.trust < cfg.high_threshold)
    )
    high_primary = _greedy_quality(A, high, K, state.trust, np.ones(A.shape[1], bool))
    high_coverage = (
        A[high_primary].any(0) if high_primary else np.zeros(A.shape[1], bool)
    )
    remaining = K - len(high_primary)
    verification_reward = (A & high_coverage).sum(1).astype(float)
    exploration = np.sqrt(
        (remaining + 1) * np.log(state.evaluation_count.sum()) / state.evaluation_count
    )
    ucb_quality = (
        cfg.alpha0 * state.trust + cfg.alpha1 * verification_reward + exploration
    )
    unknown_primary = _greedy_quality(
        A, unknown, remaining, ucb_quality, ~high_coverage
    )
    selected = list(high_primary) + list(unknown_primary)
    (second_initial, second_unknown, second_evaluated) = ([], [], [])

    def choose_random(candidates, count):
        values = np.asarray(
            sorted(set((int(i) for i in candidates)) - set(selected)), int
        )
        number = min(count, len(values))
        return (
            list(map(int, rng.choice(values, size=number, replace=False)))
            if number
            else []
        )

    if len(selected) < K and p <= cfg.initial_stage_periods:
        second_initial = choose_random(
            np.flatnonzero(state.initial_trusted & offers), K - len(selected)
        )
        selected += second_initial
    if len(selected) < K:
        candidates = sorted(
            set(map(int, unknown)) - set(selected), key=lambda i: (-ucb_quality[i], i)
        )
        second_unknown = candidates[: K - len(selected)]
        selected += second_unknown
    if len(selected) < K:
        candidates = set(map(int, high)) - set(np.flatnonzero(state.initial_trusted))
        second_evaluated = choose_random(candidates, K - len(selected))
        selected += second_evaluated
    if len(set(selected)) != len(selected):
        raise AssertionError("A cardinality set cannot contain duplicate workers")
    return {
        "period": p,
        "selected": np.asarray(selected, int),
        "K": K,
        "primary_high": np.asarray(high_primary, int),
        "primary_unknown": np.asarray(unknown_primary, int),
        "second_initial": np.asarray(second_initial, int),
        "second_unknown": np.asarray(second_unknown, int),
        "second_evaluated": np.asarray(second_evaluated, int),
        "incidence": A.copy(),
        "pre_trust": state.trust.copy(),
        "verification_reward": verification_reward,
        "ucb_quality": ucb_quality,
        "capacity_shortfall": K - len(selected),
    }
