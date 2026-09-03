"""A small genetic-programming symbolic regressor.

Written rather than pulled in because every knob the spec names has to be a
knob here: the operator set is bounded by sr_operator_set, complexity is a
plain node count so the Pareto front is legible, constants are refittable as
a separate second pass (sr_validation), and the front is reduced to one
expression by an explicitly stated rule (sr_complexity_selection) rather than
by eye. The reference work's operator set -- arithmetic, exp, log and real
constants -- is reproduced directly.

Everything is deterministic given a seed.
"""
from __future__ import annotations

import numpy as np

OPERATOR_SETS = {
    "arithmetic": ["+", "-", "*", "/"],
    "arithmetic_exp_log": ["+", "-", "*", "/", "exp", "log"],
    "arithmetic_exp_log_pow": ["+", "-", "*", "/", "exp", "log", "pow", "sqrt"],
}
BINARY = {"+", "-", "*", "/", "pow"}
UNARY = {"exp", "log", "sqrt"}

_MAXV = 1e12


def _clip(a):
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=_MAXV, neginf=-_MAXV), -_MAXV, _MAXV)


def evaluate(node, X):
    """Evaluate an expression tree on X (n_samples, n_features).

    Guarded throughout: division by a near-zero denominator, the log of a
    non-positive number and an overflowing exp are all finite here, so the
    search explores the whole operator set without producing NaNs that would
    silently poison a fitness comparison.
    """
    kind = node[0]
    if kind == "x":
        return X[:, node[1]]
    if kind == "c":
        return np.full(X.shape[0], node[1], dtype=float)
    if kind in UNARY:
        v = evaluate(node[1], X)
        if kind == "exp":
            return _clip(np.exp(np.clip(v, -50, 50)))
        if kind == "log":
            return _clip(np.log(np.abs(v) + 1e-12))
        return _clip(np.sqrt(np.abs(v)))
    a, b = evaluate(node[1], X), evaluate(node[2], X)
    # np.where evaluates both branches, so the guarded division below still
    # computes a/0 before selecting. The result is discarded; silence the
    # warning rather than let it bury real output.
    if kind == "+":
        return _clip(a + b)
    if kind == "-":
        return _clip(a - b)
    if kind == "*":
        return _clip(a * b)
    if kind == "/":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return _clip(a / np.where(np.abs(b) < 1e-9, np.sign(b) * 1e-9 + 1e-9, b))
    if kind == "pow":
        return _clip(np.power(np.abs(a) + 1e-12, np.clip(b, -4, 4)))
    raise ValueError(f"unknown operator: {kind}")


def complexity(node):
    """Node count -- the axis the Pareto front trades accuracy against."""
    if node[0] in ("x", "c"):
        return 1
    if node[0] in UNARY:
        return 1 + complexity(node[1])
    return 1 + complexity(node[1]) + complexity(node[2])


def to_string(node, names=None):
    k = node[0]
    if k == "x":
        return names[node[1]] if names else f"x{node[1]}"
    if k == "c":
        return f"{node[1]:.6g}"
    if k in UNARY:
        return f"{k}({to_string(node[1], names)})"
    if k == "pow":
        return f"({to_string(node[1], names)})**({to_string(node[2], names)})"
    return f"({to_string(node[1], names)} {k} {to_string(node[2], names)})"


def variables(node):
    """Distinct coordinate indices an expression actually reads."""
    if node[0] == "x":
        return {node[1]}
    if node[0] == "c":
        return set()
    if node[0] in UNARY:
        return variables(node[1])
    return variables(node[1]) | variables(node[2])


def constants(node):
    if node[0] == "c":
        return [node[1]]
    if node[0] in ("x",):
        return []
    if node[0] in UNARY:
        return constants(node[1])
    return constants(node[1]) + constants(node[2])


def set_constants(node, values, pos=0):
    if node[0] == "c":
        return ("c", float(values[pos])), pos + 1
    if node[0] == "x":
        return node, pos
    if node[0] in UNARY:
        child, pos = set_constants(node[1], values, pos)
        return (node[0], child), pos
    left, pos = set_constants(node[1], values, pos)
    right, pos = set_constants(node[2], values, pos)
    return (node[0], left, right), pos


def refit_constants(node, X, y):
    """Refit the free constants after the structural search has fixed the form.

    The reference work refits a second time to avoid error accumulated across
    successive stages of fitting; this separates error in the functional form
    from error in the coefficients.
    """
    from scipy.optimize import least_squares

    c0 = constants(node)
    if not c0:
        return node
    def resid(c):
        cand, _ = set_constants(node, c)
        return evaluate(cand, X) - y
    try:
        sol = least_squares(resid, np.asarray(c0, float), method="lm",
                            max_nfev=200 * (len(c0) + 1))
        cand, _ = set_constants(node, sol.x)
        if np.all(np.isfinite(evaluate(cand, X))):
            return cand
    except Exception:
        pass
    return node


class SymbolicRegressor:
    def __init__(self, operator_set="arithmetic_exp_log", n_features=8,
                 population=600, generations=40, max_complexity=30,
                 tournament=5, parsimony=1e-3, seed=0):
        self.ops = OPERATOR_SETS[operator_set]
        self.bin_ops = [o for o in self.ops if o in BINARY]
        self.un_ops = [o for o in self.ops if o in UNARY]
        self.n_features = n_features
        self.population = population
        self.generations = generations
        self.max_complexity = max_complexity
        self.tournament = tournament
        self.parsimony = parsimony
        self.rng = np.random.default_rng(seed)

    # ---- tree construction ------------------------------------------------
    def _leaf(self):
        if self.rng.random() < 0.75:
            return ("x", int(self.rng.integers(self.n_features)))
        return ("c", float(self.rng.normal(0, 2)))

    def _grow(self, depth):
        if depth <= 0 or self.rng.random() < 0.3:
            return self._leaf()
        if self.un_ops and self.rng.random() < 0.25:
            return (self.un_ops[self.rng.integers(len(self.un_ops))], self._grow(depth - 1))
        op = self.bin_ops[self.rng.integers(len(self.bin_ops))]
        return (op, self._grow(depth - 1), self._grow(depth - 1))

    def _nodes(self, node, path=()):
        out = [(node, path)]
        if node[0] in UNARY:
            out += self._nodes(node[1], path + (1,))
        elif node[0] in BINARY:
            out += self._nodes(node[1], path + (1,))
            out += self._nodes(node[2], path + (2,))
        return out

    @staticmethod
    def _replace(node, path, new):
        if not path:
            return new
        i = path[0]
        child = SymbolicRegressor._replace(node[i], path[1:], new)
        return node[:i] + (child,) + node[i + 1:]

    def _mutate(self, node):
        r = self.rng.random()
        cand = self._nodes(node)
        _, path = cand[self.rng.integers(len(cand))]
        if r < 0.4:
            return self._replace(node, path, self._grow(2))
        if r < 0.7:
            return self._replace(node, path, self._leaf())
        # constant jitter: cheap local refinement between structural moves
        cs = [(n, p) for n, p in cand if n[0] == "c"]
        if not cs:
            return self._replace(node, path, self._grow(1))
        n, p = cs[self.rng.integers(len(cs))]
        return self._replace(node, p, ("c", float(n[1] + self.rng.normal(0, 0.5))))

    def _crossover(self, a, b):
        na, nb = self._nodes(a), self._nodes(b)
        _, pa = na[self.rng.integers(len(na))]
        sub, _ = nb[self.rng.integers(len(nb))]
        return self._replace(a, pa, sub)

    # ---- fitness ----------------------------------------------------------
    @staticmethod
    def losses(node, X, y):
        p = evaluate(node, X)
        if not np.all(np.isfinite(p)):
            return np.inf, np.inf
        return float(np.mean((p - y) ** 2)), float(np.mean(np.abs(p - y)))

    def _fitness(self, node, X, y):
        c = complexity(node)
        if c > self.max_complexity:
            return np.inf
        mse, _ = self.losses(node, X, y)
        return mse * (1.0 + self.parsimony * c)

    # ---- search -----------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        pop = [self._grow(int(self.rng.integers(1, 4))) for _ in range(self.population)]
        # front: complexity -> (mse, mae, expression), best seen at that size
        front: dict[int, tuple] = {}

        def record(node):
            c = complexity(node)
            if c > self.max_complexity:
                return
            mse, mae = self.losses(node, X, y)
            if not np.isfinite(mse):
                return
            if c not in front or mse < front[c][0]:
                front[c] = (mse, mae, node)

        for node in pop:
            record(node)

        for _ in range(self.generations):
            fits = np.array([self._fitness(n, X, y) for n in pop])
            order = np.argsort(fits)
            keep = max(2, self.population // 10)
            newpop = [pop[i] for i in order[:keep]]        # elitism
            while len(newpop) < self.population:
                def pick():
                    idx = self.rng.integers(0, self.population, self.tournament)
                    return pop[idx[np.argmin(fits[idx])]]
                r = self.rng.random()
                if r < 0.5:
                    child = self._crossover(pick(), pick())
                elif r < 0.9:
                    child = self._mutate(pick())
                else:
                    child = self._grow(int(self.rng.integers(1, 4)))
                if complexity(child) <= self.max_complexity:
                    newpop.append(child)
                    record(child)
            pop = newpop

        # Refit each surviving structure's constants before the front is built.
        # Without this the search scores a structure using whatever random
        # constants it happened to be born with, so a correct functional form
        # with bad coefficients looks worse than a bare constant and never
        # survives selection -- which is exactly what a plain constant front
        # looks like. Only the per-complexity best is refitted (tens of calls,
        # not tens of thousands), so this is cheap.
        refit = {}
        for c, (mse, mae, node) in front.items():
            cand = refit_constants(node, X, y)
            cmse, cmae = self.losses(cand, X, y)
            refit[c] = (cmse, cmae, cand) if np.isfinite(cmse) and cmse < mse \
                       else (mse, mae, node)

        # keep only points that are not dominated by a simpler expression
        pareto = []
        best = np.inf
        for c in sorted(refit):
            mse, mae, node = refit[c]
            if mse < best:
                best = mse
                pareto.append({"complexity": c, "mse": mse, "mae": mae, "expr": node})
        self.pareto_ = pareto
        return self


def select_from_front(pareto, rule, complexity_ceiling=20):
    """Reduce the accuracy-complexity front to one expression.

    Picking a point by eye is where symbolic regression quietly becomes
    unfalsifiable, so the rule is fixed in advance and recorded with the
    result.
    """
    if not pareto:
        return None, {}
    if rule == "max_complexity":
        cand = [p for p in pareto if p["complexity"] <= complexity_ceiling] or pareto
        best = min(cand, key=lambda p: p["mae"])
        return best, {"rule": rule, "complexity_ceiling": complexity_ceiling}
    if rule == "best_score":
        # The reference rule: maximise the fractional drop in MAE over the
        # increase in complexity from the next best model.
        scores = []
        for i, p in enumerate(pareto):
            if i == 0:
                scores.append(0.0)
                continue
            prev = pareto[i - 1]
            dc = p["complexity"] - prev["complexity"]
            if dc <= 0 or prev["mae"] <= 0 or p["mae"] <= 0:
                scores.append(0.0)
            else:
                scores.append(float(-(np.log(p["mae"]) - np.log(prev["mae"])) / dc))
        i = int(np.argmax(scores)) if len(scores) > 1 else 0
        return pareto[i], {"rule": rule, "scores": scores, "selected_index": i}
    if rule == "elbow":
        if len(pareto) < 3:
            return pareto[-1], {"rule": rule, "note": "front too short for a curvature test"}
        c = np.array([p["complexity"] for p in pareto], float)
        m = np.log(np.array([max(p["mae"], 1e-300) for p in pareto]))
        cn = (c - c.min()) / max(c.ptp(), 1e-12)
        mn = (m - m.min()) / max(m.ptp(), 1e-12)
        curv = np.abs(np.gradient(np.gradient(mn, cn), cn))
        i = int(np.argmax(curv[1:-1]) + 1)
        return pareto[i], {"rule": rule, "selected_index": i}
    raise ValueError(f"unknown sr_complexity_selection: {rule}")
