#!/usr/bin/env python3
"""Verification suite for cyclemean.minimum.

Independent oracles used here (none share code with cyclemean.py):
  * brute_min_mean        -- exhaustive enumeration of all simple cycles
  * check_witness         -- validates the returned original-edge witness
  * exists_cycle_mean_below -- Bellman-Ford negative-cycle detection on
                             shifted integer weights q*c - p (optimality)
  * is_acyclic            -- Kahn topological sort (validates None answers)

Run: python3 verify_cyclemean.py   (exit code 0 = all checks passed)
"""

import itertools
import random
import sys
import time
from collections import deque
from math import gcd

from cyclemean import minimum


# ---------------------------------------------------------------- oracles

def brute_min_mean(n, edges):
    """Exact minimum cycle mean by enumerating every simple cycle.

    Each simple cycle is generated exactly once, from its smallest vertex.
    Returns (p, q) coprime with q > 0, or None if the graph is acyclic.
    """
    adj = [[] for _ in range(n)]
    for u, v, c in edges:
        adj[u].append((v, c))
    best = None  # (num, den), compared by exact cross-multiplication

    for s in range(n):
        def dfs(v, cost, length, visited, s=s):
            nonlocal best
            for w, c in adj[v]:
                if w == s:  # close a cycle back at the start vertex
                    num, den = cost + c, length + 1
                    if best is None or num * best[1] < best[0] * den:
                        best = (num, den)
                elif w > s and w not in visited:
                    visited.add(w)
                    dfs(w, cost + c, length + 1, visited)
                    visited.discard(w)
        dfs(s, 0, 0, {s})

    if best is None:
        return None
    g = gcd(abs(best[0]), best[1])
    return (best[0] // g, best[1] // g)


def check_witness(n, edges, result):
    """Validate (p, q, edge_ids) against the original graph."""
    p, q, ids = result
    assert isinstance(ids, list) and len(ids) >= 1, "witness must be non-empty"
    assert q > 0 and gcd(abs(p), q) == 1, f"({p},{q}) not coprime with q>0"
    assert all(0 <= i < len(edges) for i in ids), "edge id out of range"
    L = len(ids)
    for t in range(L):  # walking order + closure
        a, b = ids[t], ids[(t + 1) % L]
        assert edges[a][1] == edges[b][0], f"edges {a},{b} not connected"
    starts = [edges[i][0] for i in ids]
    assert len(set(starts)) == L, "repeated vertex (not a simple cycle)"
    total = sum(edges[i][2] for i in ids)
    assert total * q == p * L, f"witness mean {total}/{L} != {p}/{q}"


def exists_cycle_mean_below(n, edges, p, q):
    """True iff some directed cycle has mean strictly below p/q.

    Independent of Karp: a cycle with mean < p/q is a negative cycle in
    shifted integer weights w = q*c - p; detected by Bellman-Ford from a
    virtual super-source (all distances start at 0).
    """
    dist = [0] * n
    for _ in range(n):
        updated = False
        for u, v, c in edges:
            w = q * c - p
            du = dist[u] + w
            if du < dist[v]:
                dist[v] = du
                updated = True
        if not updated:
            return False
    return True


def is_acyclic(n, edges):
    """Kahn's topological sort: True iff the graph has no directed cycle."""
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for u, v, _ in edges:
        adj[u].append(v)
        indeg[v] += 1
    dq = deque([v for v in range(n) if indeg[v] == 0])
    seen = 0
    while dq:
        x = dq.popleft()
        seen += 1
        for y in adj[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                dq.append(y)
    return seen == n


# ------------------------------------------------------------------ tests

def test_directed():
    cases = [
        # (n, edges, expected (p, q) or None)
        (2, [(0, 1, 2), (1, 0, 4), (1, 1, 5)], (3, 1)),      # spec example
        (3, [(0, 1, 1), (1, 2, 1), (0, 2, 5)], None),        # plain DAG
        (1, [], None),                                       # no edges
        (1, [(0, 0, -4)], (-4, 1)),                          # negative self-loop
        (2, [(0, 1, 5), (0, 1, 1), (1, 0, 3)], (2, 1)),      # parallel edges
        (5, [(0, 1, 1), (1, 2, 2), (3, 4, 10), (4, 3, -20)], (-5, 1)),  # disconnected
        (3, [(0, 0, -3), (1, 2, -5), (2, 1, 1)], (-3, 1)),   # self-loop beats 2-cycle
        (4, [(0, 1, 0), (1, 0, 6), (2, 3, 1), (3, 2, 1)], (1, 1)),      # two cycles
        (2, [(0, 1, -1), (1, 0, -1)], (-1, 1)),              # negative cycle
        (2, [(0, 1, 1), (1, 0, -1)], (0, 1)),                # zero-mean cycle
        (2, [(0, 1, 0), (1, 0, 1)], (1, 2)),                 # fractional mean
        (4, [(0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 0, 0)], (1, 2)),  # 2/4 reduced
        (3, [(0, 1, 1), (1, 2, 2), (2, 0, 3)], (2, 1)),      # triangle
        (4, [(0, 1, 2), (1, 2, 2), (2, 3, 2), (3, 0, 2),
             (0, 2, 1), (2, 0, 1)], (1, 1)),                 # chord beats 4-cycle
    ]
    for idx, (n, edges, exp) in enumerate(cases):
        before = list(edges)
        res = minimum(n, edges)
        assert edges == before, f"case {idx}: input mutated"
        if exp is None:
            assert res is None, f"case {idx}: expected None, got {res}"
            assert is_acyclic(n, edges), f"case {idx}: Kahn disagrees"
        else:
            assert res is not None, f"case {idx}: expected {exp}, got None"
            assert (res[0], res[1]) == exp, f"case {idx}: expected {exp}, got {res[:2]}"
            check_witness(n, edges, res)
            assert not exists_cycle_mean_below(n, edges, res[0], res[1]), \
                f"case {idx}: Bellman-Ford found a better cycle"
    print(f"[ok] directed cases (spec example, disconnected, negative, parallel, self-loop): {len(cases)}")


def test_exhaustive_n2():
    """Every multigraph on n=2 with <=3 edges and costs in {-1,0,1}."""
    base = [(u, v, c) for u in (0, 1) for v in (0, 1) for c in (-1, 0, 1)]
    count = 0
    for k in range(4):
        for combo in itertools.combinations_with_replacement(range(len(base)), k):
            edges = [base[i] for i in combo]
            res = minimum(2, edges)
            exp = brute_min_mean(2, edges)
            if exp is None:
                assert res is None, (edges, res)
            else:
                assert res is not None and (res[0], res[1]) == exp, (edges, res, exp)
                check_witness(2, edges, res)
            count += 1
    print(f"[ok] exhaustive n=2, |E|<=3, costs in {{-1,0,1}}: {count} graphs vs brute force")


def test_random_small(trials=2000, seed=20261115):
    rng = random.Random(seed)
    for t in range(trials):
        n = rng.randint(1, 6)
        m = rng.randint(0, 12)
        edges = [(rng.randrange(n), rng.randrange(n), rng.randint(-6, 6))
                 for _ in range(m)]
        before = list(edges)
        res = minimum(n, edges)
        assert edges == before, f"trial {t}: input mutated"
        exp = brute_min_mean(n, edges)
        if exp is None:
            assert res is None, f"trial {t}: expected None, got {res}; {n} {edges}"
        else:
            assert res is not None, f"trial {t}: expected {exp}, got None; {n} {edges}"
            assert (res[0], res[1]) == exp, \
                f"trial {t}: {res[:2]} != brute {exp}; {n} {edges}"
            check_witness(n, edges, res)
    print(f"[ok] random small graphs vs independent brute force: {trials} trials")


def test_random_medium(trials=60, seed=99):
    """Larger random graphs, checked by witness + Bellman-Ford + Kahn."""
    rng = random.Random(seed)
    for t in range(trials):
        n = rng.randint(10, 60)
        m = rng.randint(0, 400)
        edges = [(rng.randrange(n), rng.randrange(n), rng.randint(-100, 100))
                 for _ in range(m)]
        res = minimum(n, edges)
        if res is None:
            assert is_acyclic(n, edges), f"trial {t}: None but Kahn finds a cycle"
        else:
            assert not is_acyclic(n, edges), f"trial {t}: cycle but graph is a DAG"
            check_witness(n, edges, res)
            assert not exists_cycle_mean_below(n, edges, res[0], res[1]), \
                f"trial {t}: Bellman-Ford found a better cycle"
    print(f"[ok] random medium graphs (witness + Bellman-Ford + Kahn oracles): {trials} trials")


def test_dense_cyclic():
    rng = random.Random(7)
    n, m = 500, 20000
    edges = [(rng.randrange(n), rng.randrange(n), rng.randint(-10**9, 10**9))
             for _ in range(m)]
    t0 = time.perf_counter()
    res = minimum(n, edges)
    dt = time.perf_counter() - t0
    assert res is not None
    p, q, ids = res
    check_witness(n, edges, res)
    assert not exists_cycle_mean_below(n, edges, p, q)
    print(f"[ok] dense cyclic n={n} m={m}: mean={p}/{q}, witness {len(ids)} edges, "
          f"optimal (Bellman-Ford), minimum() took {dt:.2f}s")


def test_large_acyclic():
    rng = random.Random(11)
    n, m = 500, 20000
    edges = []
    while len(edges) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u < v:
            edges.append((u, v, rng.randint(-10**9, 10**9)))
    t0 = time.perf_counter()
    res = minimum(n, edges)
    dt = time.perf_counter() - t0
    assert res is None, res
    assert is_acyclic(n, edges)
    print(f"[ok] large acyclic n={n} m={m}: None confirmed by Kahn, minimum() took {dt:.2f}s")


def test_cycle_buried_in_acyclic_region():
    rng = random.Random(13)
    n = 500
    edges = []
    while len(edges) < 19997:  # huge forward-only (acyclic) region
        u = rng.randrange(n - 1)
        v = rng.randrange(u + 1, n)
        edges.append((u, v, rng.randint(0, 1000)))
    edges += [(120, 250, -10), (250, 380, -10), (380, 120, -10)]  # planted cycle
    t0 = time.perf_counter()
    res = minimum(n, edges)
    dt = time.perf_counter() - t0
    assert res is not None
    p, q, ids = res
    assert (p, q) == (-10, 1), (p, q)  # only cycles use edge 380->120; -30/3 is minimal
    check_witness(n, edges, res)
    assert not exists_cycle_mean_below(n, edges, p, q)
    print(f"[ok] cycle buried in 19997-edge acyclic region: mean={p}/{q}, "
          f"witness {ids}, minimum() took {dt:.2f}s")


def main():
    test_directed()
    test_exhaustive_n2()
    test_random_small()
    test_random_medium()
    test_dense_cyclic()
    test_large_acyclic()
    test_cycle_buried_in_acyclic_region()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
