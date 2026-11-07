"""Budget-feasible admission-aware planning with private historical inputs."""
import copy
from dataclasses import dataclass
import numpy as np
from numba import njit
from . import recruitment,trust
from .progress import make_profile,progress_state,counts_from_history,transition_kernel,step_progress,planning_depth

@dataclass(frozen=True)
class ScoredRound:
    worker: int
    round: int
    pre_r: float
    residuals: tuple
    a: float
    b: float
    next_r: float

@dataclass(frozen=True)
class ReleasedFrame:
    round: int
    packet: dict
    offers: np.ndarray
    requested_tasks: np.ndarray
    virtual_tasks: np.ndarray

@dataclass(frozen=True)
class LegalState:
    round: int
    seed: int
    packet: dict
    offers: np.ndarray
    reputation: np.ndarray
    requested_tasks: np.ndarray
    virtual_tasks: np.ndarray
    privacy_config: object
    trust_config: object
    budget: float
    public_reference: np.ndarray
    support: np.ndarray
    support_receipt: dict
    scored_rounds: tuple
    previous_offer_counts: np.ndarray
    previous_rounds: int
    past_frames: tuple
    support_prior: object

def select_additive(packet, offers, reputation, tasks, cfg, budget, tcfg, support, credible_first=False):
    p = np.asarray(support, float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Support must be a finite probability matrix')
    (n, D) = p.shape
    if np.asarray(packet['bits']).shape != (n, D):
        raise ValueError('Support and released packet dimensions must agree')
    (weights, classes, _) = recruitment.mapweights(reputation, tcfg)
    eligible = np.asarray(offers, bool) & (classes != 0)
    gain = np.rint(1000000 * weights * p[:, np.asarray(tasks, int)].sum(1)).astype(np.int64)
    profile = recruitment.PriceProfile.from_config(cfg)
    if not profile.monotone:
        raise ValueError('A monotone registered quote is required')
    limit = recruitment.budget_limit(profile, budget)
    nz = np.asarray(packet['nonzero'], np.int64)
    hi = np.asarray(packet['hi'], np.int64)
    N = np.zeros(D, np.int64)
    H = N.copy()
    current = 0
    chosen = np.zeros(n, bool)
    trace = []
    while True:
        pool = np.flatnonzero(eligible & ~chosen)
        if not len(pool):
            break
        costs = recruitment.quote_counts(N[None] + nz[pool], H[None] + hi[pool], profile)
        positions = np.flatnonzero((costs <= limit) & (gain[pool] > 0))
        if credible_first and (not np.any(chosen & (classes == 2))):
            credible = positions[classes[pool[positions]] == 2]
            if len(credible):
                positions = credible
        if not len(positions):
            break
        dc = costs - current
        zero = positions[dc[positions] == 0]
        if len(zero):
            best = int(zero[np.argmax(gain[pool[zero]])])
        else:
            best = int(positions[0])
            for j in positions[1:]:
                if int(gain[pool[j]]) * int(dc[best]) > int(gain[pool[best]]) * int(dc[j]):
                    best = int(j)
        i = pool[best]
        chosen[i] = True
        N += nz[i]
        H += hi[i]
        current = int(costs[best])
        trace.append(dict(slot=int(i), gain=int(gain[i]), cost_numerator=current))
    return dict(members=np.flatnonzero(chosen), cost_numerator=current, cost_denominator=profile.cost_denominator, trace=trace)

@njit(cache=True)
def _quote(N, H, P, Q, lo, hi):
    d = 2 * P - Q
    total = 0
    for j in range(len(N)):
        recovered = Q * H[j] - (Q - P) * N[j]
        if recovered < 0:
            recovered = 0
        if recovered > d * N[j]:
            recovered = d * N[j]
        total += lo * d * N[j] + (hi - lo) * recovered
    return total

@njit(cache=True)
def _quote_added(N, H, nz, high, P, Q, lo, hi):
    d = 2 * P - Q
    total = 0
    for j in range(len(N)):
        count = N[j] + nz[j]
        ones = H[j] + high[j]
        recovered = Q * ones - (Q - P) * count
        if recovered < 0:
            recovered = 0
        if recovered > d * count:
            recovered = d * count
        total += lo * d * count + (hi - lo) * recovered
    return total

@njit(cache=True)
def _future_greedy(z, offers, task_support, nz, high, limit, P, Q, lo, hi, amplitude, identified_low, identified_high):
    (n, D) = nz.shape
    chosen = np.zeros(n, np.bool_)
    N = np.zeros(D, np.int64)
    H = np.zeros(D, np.int64)
    identified = 0
    for i in range(n):
        if z[i] == 0 or z[i] == 4:
            identified += 1
    eta = identified / n
    wc = amplitude if eta <= identified_low else amplitude - (amplitude - 1) * (eta - identified_low) / (identified_high - identified_low) if eta <= identified_high else 1.0
    x = (eta - identified_low) / (1 - identified_low)
    wu = 1.0 if eta <= identified_low else 1.0 + (amplitude - 1) * max(0.0, 1.0 - abs(2 * x - 1.0))
    gains = np.zeros(n, np.int64)
    for i in range(n):
        if z[i] != 0:
            gains[i] = int(np.rint(1000000 * (wc if z[i] == 4 else wu) * task_support[i]))
    current = 0
    for _ in range(n):
        best = -1
        best_gain = 0
        best_delta = 0
        best_cost = 0
        for i in range(n):
            if chosen[i] or not offers[i] or z[i] == 0 or (gains[i] <= 0):
                continue
            cost = _quote_added(N, H, nz[i], high[i], P, Q, lo, hi)
            if cost > limit:
                continue
            delta = cost - current
            gain = gains[i]
            if best < 0 or (delta == 0 and best_delta != 0) or (delta == 0 and best_delta == 0 and (gain > best_gain)) or (delta > 0 and best_delta > 0 and (gain * best_delta > best_gain * delta)):
                best = i
                best_gain = gain
                best_delta = delta
                best_cost = cost
        if best < 0:
            break
        chosen[best] = True
        N += nz[best]
        H += high[best]
        current = best_cost
    return (chosen, current)

@njit(cache=True)
def _rollout_fast(initial, z0, offers, task_support, nz, high, task_mask, exposure, routes, coins, kernel, limit, P, Q, lo, hi, amplitude, identified_low, identified_high):
    (scenarios, depth, n, D) = routes.shape
    totals = np.zeros(scenarios)
    picks = np.zeros((scenarios, depth, n), np.bool_)
    costs = np.zeros((scenarios, depth), np.int64)
    states = np.zeros((scenarios, depth + 1, n), np.int8)
    rewards = np.zeros((scenarios, depth))
    for s in range(scenarios):
        z = z0.copy()
        states[s, 0] = z
        for h in range(depth):
            if h == 0:
                chosen = initial.copy()
                N = np.zeros(D, np.int64)
                HH = np.zeros(D, np.int64)
                for i in range(n):
                    if chosen[i]:
                        N += nz[s, h, i]
                        HH += high[s, h, i]
                cost = _quote(N, HH, P, Q, lo, hi)
            else:
                (chosen, cost) = _future_greedy(z, offers[s, h], task_support[s, h], nz[s, h], high[s, h], limit, P, Q, lo, hi, amplitude, identified_low, identified_high)
            picks[s, h] = chosen
            costs[s, h] = cost
            reward = 0.0
            ntasks = 0
            counts = np.zeros(D, np.int64)
            for j in range(D):
                remaining = 1.0
                for i in range(n):
                    if chosen[i] and z[i] == 4:
                        remaining *= 1 - exposure[s, h, i, j]
                        if routes[s, h, i, j]:
                            counts[j] += 1
                if task_mask[s, h, j]:
                    reward += 1 - remaining
                    ntasks += 1
            reward /= ntasks
            rewards[s, h] = reward
            totals[s] += reward
            nxt = z.copy()
            for i in range(n):
                if not chosen[i] or z[i] == 0:
                    continue
                referenced = False
                for j in range(D):
                    if routes[s, h, i, j] and counts[j] - (1 if z[i] == 4 else 0) > 0:
                        referenced = True
                        break
                if referenced:
                    cumulative = 0.0
                    destination = 4
                    for k in range(5):
                        cumulative += kernel[z[i], k]
                        if coins[s, h, i] < cumulative:
                            destination = k
                            break
                    nxt[i] = destination
            z = nxt
            states[s, h + 1] = z
    return (totals, picks, costs, states, rewards)

class AdmissionPlanner:

    def __init__(self, state, remaining_contract_rounds, scenarios=4, allow_admission=True):
        if scenarios != 4:
            raise ValueError('The frozen development profile uses four scenarios')
        if any((frame.round < 0 or frame.round >= state.round for frame in state.past_frames)):
            raise ValueError('Every forecast frame must be strictly past')
        self.state = state
        self.profile = make_profile(state.trust_config)
        self.counts = counts_from_history(state.scored_rounds, state.round, self.profile)
        self.kernel = transition_kernel(self.counts, self.profile)
        self.allow_admission = bool(allow_admission)
        if not self.allow_admission:
            self.kernel = self.kernel.copy()
            for u in (1, 2, 3):
                self.kernel[u, u] += self.kernel[u, 4]
                self.kernel[u, 4] = 0.0
            self.kernel.setflags(write=False)
        self.z0 = np.asarray(progress_state(state.reputation, self.profile), np.int8)
        self.requested_depth = planning_depth(remaining_contract_rounds, self.profile)
        self.cold_start = not bool(state.past_frames)
        self.depth = min(1, self.requested_depth) if self.cold_start else self.requested_depth
        self.scenarios = scenarios
        self.n = len(self.z0)
        self.D = state.support.shape[1]
        self.cache = {}
        self.price = recruitment.PriceProfile.from_config(state.privacy_config)
        if not self.price.monotone:
            raise ValueError('A monotone registered quote is required')
        self.limit = recruitment.budget_limit(self.price, state.budget)
        if 1000000 * state.trust_config.map_amplitude * self.D * self.limit > np.iinfo(np.int64).max:
            raise OverflowError('Density cross-products exceed the registered int64 profile')
        if self.price.twice_maximum * self.price.difference * self.n * self.D > np.iinfo(np.int64).max:
            raise OverflowError('Aggregate quote exceeds int64')
        self._prepare()

    def _prepare(self):
        s = self.state
        S = self.scenarios
        H = self.depth
        n = self.n
        D = self.D
        self.offers = np.zeros((S, H, n), bool)
        self.support = np.zeros((S, H, n, D))
        self.task_support = np.zeros((S, H, n))
        self.nz = np.zeros((S, H, n, D), np.int64)
        self.high = self.nz.copy()
        self.tasks = np.zeros((S, H, D), bool)
        self.exposure = np.zeros((S, H, n, D))
        self.routes = np.zeros((S, H, n, D), bool)
        rng = np.random.default_rng(np.random.SeedSequence([s.seed, 98501, s.round]))
        self.coins = rng.random((S, H, n))
        route_coins = rng.random((S, H, n))
        self.frame_indices = np.full((S, H), -1, int)
        if H > 1 and s.past_frames:
            self.frame_indices[:, 1:] = rng.integers(0, len(s.past_frames), (S, H - 1))
        corr = np.nan_to_num(np.corrcoef(s.public_reference)) ** 2
        corr[corr < 0.5] = 0
        corr /= np.maximum(corr.sum(0, keepdims=True), 1e-10)
        prepared = {}
        for scenario in range(S):
            for h in range(H):
                index = int(self.frame_indices[scenario, h])
                if index not in prepared:
                    frame = ReleasedFrame(s.round, s.packet, s.offers, s.requested_tasks, s.virtual_tasks) if index < 0 else s.past_frames[index]
                    if index >= 0 and frame.round >= s.round:
                        raise ValueError('Forecast frames must be strictly past')
                    model = copy.deepcopy(s.support_prior)
                    (p, receipt) = model.predict(frame.packet['bits'], s.privacy_config.eps1, frame.offers, trust.classify(s.reputation, s.trust_config), s.round)
                    prepared[index] = (frame, p, copy.deepcopy(model._last_models), 1 - np.exp(-3 * p @ corr))
                    if index < 0 and (not np.array_equal(p, s.support)):
                        raise AssertionError('Current forecast differs from frozen current support')
                (frame, p, models, E) = prepared[index]
                self.offers[scenario, h] = frame.offers
                self.support[scenario, h] = p
                self.nz[scenario, h] = frame.packet['nonzero']
                self.high[scenario, h] = frame.packet['hi']
                self.tasks[scenario, h, frame.requested_tasks] = True
                self.task_support[scenario, h] = p[:, frame.requested_tasks].sum(1)
                self.exposure[scenario, h] = E
                for i in np.flatnonzero(frame.offers):
                    (routes, _, weights) = models[int(i)]
                    k = min(len(weights) - 1, int(np.searchsorted(np.cumsum(weights), route_coins[scenario, h, i], side='right')))
                    self.routes[scenario, h, i] = routes[k]

    def _membership(self, members):
        ids = np.asarray(members)
        if ids.size and (not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError('Integer worker slots required')
        ids = ids.astype(int, copy=False)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= self.n)):
            raise ValueError('Distinct valid worker slots required')
        if not np.all(self.state.offers[ids] & (self.z0[ids] != 0)):
            raise ValueError('Current members must be eligible offers')
        chosen = np.zeros(self.n, bool)
        chosen[ids] = True
        charge = int(recruitment.quote_numerators(chosen[None], self.state.packet, self.price)[0])
        if charge > self.limit:
            raise ValueError('Current choice exceeds the exact payment budget')
        return (chosen, charge)

    def value(self, members, backend='numba'):
        (chosen, charge) = self._membership(members)
        key = np.packbits(chosen).tobytes()
        if backend == 'numba' and key in self.cache:
            return self.cache[key]
        if self.depth == 0:
            result = dict(value=0.0, scenario_values=np.zeros(4), current_charge=charge)
        elif backend == 'numba':
            p = self.price
            c = self.state.trust_config
            arrays = _rollout_fast(chosen, self.z0, self.offers, self.task_support, self.nz, self.high, self.tasks, self.exposure, self.routes, self.coins, self.kernel, self.limit, p.probability_numerator, p.probability_denominator, p.twice_minimum, p.twice_maximum, c.map_amplitude, c.identified_low, c.identified_high)
            result = dict(value=float(arrays[0].mean()), scenario_values=arrays[0], selected=arrays[1], charges=arrays[2], states=arrays[3], rewards=arrays[4], current_charge=charge)
        elif backend == 'python':
            result = self._reference(chosen, charge)
        else:
            raise ValueError('Unknown rollout backend')
        if backend == 'numba':
            self.cache[key] = result
        return result

    def _reference(self, chosen, current_charge):
        S = self.scenarios
        H = self.depth
        n = self.n
        picks = np.zeros((S, H, n), bool)
        costs = np.zeros((S, H), np.int64)
        states = np.zeros((S, H + 1, n), np.int8)
        rewards = np.zeros((S, H))
        for scenario in range(S):
            z = self.z0.copy()
            states[scenario, 0] = z
            for h in range(H):
                if h == 0:
                    mask = chosen.copy()
                    cost = current_charge
                else:
                    reputation = np.where(z == 4, 0.9, np.where(z == 0, 0.1, 0.5))
                    packet = dict(bits=np.zeros((n, self.D), bool), nonzero=self.nz[scenario, h], hi=self.high[scenario, h])
                    answer = select_additive(packet, self.offers[scenario, h], reputation, np.flatnonzero(self.tasks[scenario, h]), self.state.privacy_config, self.state.budget, self.state.trust_config, self.support[scenario, h])
                    mask = np.zeros(n, bool)
                    mask[answer['members']] = True
                    cost = answer['cost_numerator']
                picks[scenario, h] = mask
                costs[scenario, h] = cost
                C = mask & (z == 4)
                U = mask & (z != 0)
                rewards[scenario, h] = np.mean((1 - np.prod(1 - self.exposure[scenario, h, C], axis=0))[self.tasks[scenario, h]])
                actual = self.routes[scenario, h] & U[:, None]
                credible = actual & C[:, None]
                reference = (actual & (credible.sum(0)[None, :] - credible > 0)).any(1)
                z = step_progress(z, reference, self.coins[scenario, h], self.kernel)
                states[scenario, h + 1] = z
        return dict(value=float(rewards.sum(1).mean()), scenario_values=rewards.sum(1), selected=picks, charges=costs, states=states, rewards=rewards, current_charge=current_charge)

    def marginal(self, members, candidate):
        base = self.value(members)
        trial = self.value(np.r_[np.asarray(members, int), int(candidate)])
        return dict(delta=trial['value'] - base['value'], delta_charge=trial['current_charge'] - base['current_charge'], trial=trial)

    def select(self):
        if not self.depth:
            return dict(members=np.array([], int), source='ended-contract', trace=[], value=0.0, cost_numerator=0, cost_denominator=self.price.cost_denominator, value_evaluations=0)
        s = self.state
        chosen = np.zeros(self.n, bool)
        trace = []
        current = self.value([])
        if self.cold_start:
            answer = select_additive(s.packet, s.offers, s.reputation, s.requested_tasks, s.privacy_config, s.budget, s.trust_config, s.support)
            val = self.value(answer['members'])
            return dict(**answer, source='template-additive', greedy_trace=[], greedy_members=np.array([], int), value=val['value'], candidate_values=[val['value']], value_evaluations=len(self.cache), cold_start=True)
        for _ in range(self.n):
            candidates = []
            for i in np.flatnonzero(s.offers & (self.z0 != 0) & ~chosen):
                ids = np.r_[np.flatnonzero(chosen), i]
                try:
                    trial = self.value(ids)
                except ValueError:
                    continue
                gain = int(np.rint((trial['value'] - current['value']) * 1000000))
                if gain > 0:
                    candidates.append((int(i), gain, trial['current_charge'] - current['current_charge'], trial))
            if not candidates:
                break
            best = candidates[0]
            for item in candidates[1:]:
                if item[2] == 0 and best[2] != 0 or (item[2] == 0 and best[2] == 0 and (item[1] > best[1])) or (item[2] > 0 and best[2] > 0 and (item[1] * best[2] > best[1] * item[2])):
                    best = item
            chosen[best[0]] = True
            current = best[3]
            trace.append(dict(slot=best[0], gain=best[1], cost_numerator=current['current_charge'], value=current['value']))
        additive = select_additive(s.packet, s.offers, s.reputation, s.requested_tasks, s.privacy_config, s.budget, s.trust_config, s.support)
        first = select_additive(s.packet, s.offers, s.reputation, s.requested_tasks, s.privacy_config, s.budget, s.trust_config, s.support, True)
        members = [additive['members'], first['members'], np.flatnonzero(chosen)]
        values = [self.value(ids) for ids in members]
        best = int(np.argmax([v['value'] for v in values]))
        return dict(members=members[best], source=['template-additive', 'template-first', 'rollout-greedy'][best], greedy_trace=trace, greedy_members=members[2], value=values[best]['value'], cost_numerator=values[best]['current_charge'], cost_denominator=self.price.cost_denominator, candidate_values=[v['value'] for v in values], value_evaluations=len(self.cache))

