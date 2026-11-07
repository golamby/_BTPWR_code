"""Private paid-footprint history and collision-consistent RR support."""
import json,hashlib,time
import numpy as np

VERSION = 'paid-template-support-v2-public-basis'

FIXED = dict(own_templates=8, background_templates=32, own_mass=0.75, global_mass=0.125, singleton_mass=0.125, cold_global_mass=0.5, cold_singleton_mass=0.5, own_decay_rounds=10, vlr_draws=32, initial_lengths=(1, 2, 4, 8, 16, 32, 64), max_D=100, max_epsilon=2.1, singleton_likelihood='exact-public-neighbors', template_order='lexicographic-packed-bytes', history_order='round-then-registered-slot')

_POPCOUNT = np.array([bin(x).count('1') for x in range(256)], np.uint8)

def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def _bits(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or not np.isin(a, (0, 1)).all():
        raise ValueError(name + ' must be a Boolean array of the expected shape')
    return a.astype(bool, copy=False)

class _Original:

    def __init__(self, n, public_reference, seed):
        H = np.asarray(public_reference, float)
        if H.ndim != 2 or H.shape[1] < 2 or (not np.isfinite(H).all()):
            raise ValueError('Finite public historical field is required')
        (self.n, self.D, self.seed) = (int(n), len(H), int(seed))
        corr = np.nan_to_num(np.corrcoef(H))
        np.fill_diagonal(corr, 0)
        self.nb = []
        for j in range(self.D):
            ids = np.argsort(-np.abs(corr[j]), kind='stable')[:2]
            ids = ids[np.abs(corr[j, ids]) >= 0.93]
            self.nb.append(ids if len(ids) else np.array([j]))
        self.history = [[] for _ in range(self.n)]
        (self.paid_pool, self.virtual_cache) = ([], {})
        self.count = np.zeros((self.n, self.D))
        self.seen = np.zeros(self.n, int)
        self.last_update = -1
        self.pending = None
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 98101]))
        self.initial = np.zeros((FIXED['background_templates'], self.D), bool)
        for (k, a) in enumerate(self.initial):
            length = min(self.D, FIXED['initial_lengths'][k % 7])
            a[rng.choice(self.D, length, replace=False)] = True

    def _virtual(self, footprint):
        key = np.packbits(footprint).tobytes()
        if key not in self.virtual_cache:
            words = np.frombuffer(hashlib.sha256(key).digest()[:16], '<u4')
            rng = np.random.default_rng(np.random.SeedSequence([98102, *words.tolist()]))
            masks = np.zeros((FIXED['vlr_draws'], self.D), bool)
            for j in np.flatnonzero(footprint):
                dest = rng.choice(self.nb[j], FIXED['vlr_draws'])
                masks[np.arange(FIXED['vlr_draws']), dest] = True
            self.virtual_cache[key] = np.packbits(masks, axis=1)
        return self.virtual_cache[key]

class TemplateSupport(_Original):

    def __init__(self, n, public_reference, seed):
        if np.asarray(public_reference).ndim == 2 and len(public_reference) < 2:
            raise ValueError('This registered numeric profile requires 2<=D<=100')
        super().__init__(n, public_reference, seed)
        if not 2 <= self.D <= FIXED['max_D']:
            raise ValueError('This registered numeric profile requires 2<=D<=100')
        self.basis = np.eye(self.D, dtype=bool)
        self.public_reference_hash = _hash(np.asarray(public_reference, '<f8'))
        self._last_models = {}

    def history_digest(self):
        header = dict(version=VERSION, config=FIXED, n=self.n, D=self.D, epoch_seed=self.seed, public_reference=self.public_reference_hash, last_update=self.last_update, replacement_neighbors=[list(map(int, v)) for v in self.nb])
        h = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
        for (t, i, a) in self.paid_pool:
            h.update(int(t).to_bytes(8, 'big'))
            h.update(int(i).to_bytes(8, 'big'))
            h.update(np.packbits(a).tobytes())
        return h.hexdigest()

    def _log_likelihood(self, routes, y, epsilon):
        result = np.empty(len(routes))
        sizes = routes.sum(1)
        single = np.flatnonzero(sizes == 1)
        if len(single):
            physical = routes[single].argmax(1)
            neighbors = np.array([[self.nb[j][0], self.nb[j][-1]] for j in physical])
            dist = int(y.sum()) + 1 - 2 * y[neighbors].astype(int)
            exponent = -epsilon * dist
            center = exponent.max(1)
            result[single] = center + np.log(np.exp(exponent - center[:, None]).mean(1))
        mult = np.flatnonzero(sizes != 1)
        if len(mult):
            virtual = np.asarray([self._virtual(routes[k]) for k in mult])
            distance = _POPCOUNT[virtual ^ np.packbits(y)[None, None]].sum(2)
            exponent = -epsilon * distance
            center = exponent.max(1)
            result[mult] = center + np.log(np.exp(exponent - center[:, None]).mean(1))
        return result

    def predict(self, released_bits, epsilon, offers, classes, t):
        if t != self.last_update + 1 or self.pending is not None:
            raise ValueError('Predict once per round after the preceding observe')
        bits = _bits(released_bits, (self.n, self.D), 'Released bits')
        active = _bits(offers, (self.n,), 'Offer mask')
        cls = np.asarray(classes)
        if cls.shape != (self.n,) or not np.isin(cls, (0, 1, 2)).all():
            raise ValueError('Classes must be the pre-selection 0/1/2 state')
        if not np.isfinite(epsilon) or not 0 <= epsilon <= FIXED['max_epsilon']:
            raise ValueError('This registered numeric profile requires 0<=epsilon<=2.1')
        if any((s >= t for (s, _, _) in self.paid_pool)):
            raise ValueError('Current/future footprints cannot enter prior history')
        tick = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 98103, int(t)]))
        if self.paid_pool:
            bg = np.array([self.paid_pool[k][2] for k in rng.integers(0, len(self.paid_pool), FIXED['background_templates'])])
        else:
            bg = self.initial
        p = np.zeros((self.n, self.D))
        diagnostics = {}
        self._last_models = {}
        for i in np.flatnonzero(active):
            records = self.history[i][-FIXED['own_templates']:]
            if any((s >= t for (s, _) in records)):
                raise ValueError('Only strictly previous paid own footprints are allowed')
            if records:
                age_log = -(t - np.array([s for (s, _) in records])) / FIXED['own_decay_rounds']
                ow = np.exp(age_log - age_log.max())
                ow *= FIXED['own_mass'] / ow.sum()
                raw = np.concatenate([np.array([a for (_, a) in records]), bg, self.basis])
                prior = np.r_[ow, np.full(len(bg), FIXED['global_mass'] / len(bg)), np.full(self.D, FIXED['singleton_mass'] / self.D)]
            else:
                raw = np.concatenate([bg, self.basis])
                prior = np.r_[np.full(len(bg), FIXED['cold_global_mass'] / len(bg)), np.full(self.D, FIXED['cold_singleton_mass'] / self.D)]
            merged = {}
            for (a, w) in zip(raw, prior):
                key = np.packbits(a).tobytes()
                if key not in merged:
                    merged[key] = [a, 0.0]
                merged[key][1] += float(w)
            ordered = [merged[k] for k in sorted(merged)]
            routes = np.asarray([v[0] for v in ordered])
            priors = np.asarray([v[1] for v in ordered])
            log_weight = np.log(priors) + self._log_likelihood(routes, bits[i], float(epsilon))
            weight = np.exp(log_weight - log_weight.max())
            weight /= weight.sum()
            p[i] = weight @ routes
            accumulation = len(routes) * np.finfo(float).eps
            tolerance = 8 * accumulation / (1 - accumulation)
            if not np.isfinite(p[i]).all() or not np.all(p[i] > 0) or np.any(p[i] > 1 + tolerance) or (abs(weight.sum() - 1) > tolerance):
                raise ArithmeticError(f'Invalid full-support probabilities: t={t}, slot={i}, finite={np.isfinite(p[i]).all()}, min={p[i].min():.18g}, max={p[i].max():.18g}, sum_weights={weight.sum():.18g}, templates={len(routes)}')
            p[i] = np.minimum(p[i], 1.0)
            self._last_models[int(i)] = (routes, priors, weight)
            diagnostics[int(i)] = dict(paid_history=int(self.seen[i]), template_count=len(routes), ess=float(1 / np.sum(weight ** 2)), cold_start=int(not records), minimum_marginal=float(p[i].min()))
        self.pending = (int(t), (active & (cls != 0)).copy())
        receipt = dict(version=VERSION, round=int(t), last_updated_round=self.last_update, paid_worker_rounds=len(self.paid_pool), history_digest=self.history_digest(), marginal_sha256=_hash(p), seconds=time.perf_counter() - tick, workers=diagnostics)
        return (p, receipt)

    def observe(self, selected, paid_actual_masks, t):
        if self.pending is None or self.pending[0] != t:
            raise ValueError('Observe requires the matching preceding prediction')
        ids = np.asarray(selected)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Selected IDs must be a one-dimensional integer array')
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= self.n) or (not np.all(self.pending[1][ids])):
            raise ValueError('Paid IDs must be distinct eligible offered workers')
        masks = _bits(paid_actual_masks, (len(ids), self.D), 'Paid report masks')
        if len(ids) and (not masks.any(1).all()):
            raise ValueError('This profile requires an actual report for every paid active route')
        for k in np.argsort(ids, kind='stable'):
            i = int(ids[k])
            a = masks[k].copy()
            self.history[i].append((int(t), a.copy()))
            self.paid_pool.append((int(t), i, a.copy()))
            self.count[i] += a
            self.seen[i] += 1
        self.last_update = int(t)
        self.pending = None

