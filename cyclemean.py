"""Minimum-mean directed cycle (Karp's algorithm, exact rational arithmetic).

cyclemean.minimum(n, edges) finds a non-empty directed cycle whose average
edge cost (total cost / number of edges) is minimum among all directed
cycles of the graph.

Constraints assumed (input is guaranteed legal and is never modified):
  * vertices are 0 .. n-1, 1 <= n <= 500
  * edges is a collection of up to 20000 tuples (u, v, cost)
  * cost is an integer with |cost| <= 1e9, may be negative
  * self-loops and parallel edges are allowed

Complexity: O(n * (n + len(edges))) time, O(n * n + len(edges)) space.
No cycle enumeration, no floating point anywhere near the decision: all
ratio comparisons are exact integer cross-multiplications.
"""

from math import gcd

__all__ = ["minimum"]


def minimum(n, edges):
    """Return (p, q, edge_ids) for a minimum-mean directed cycle, or None.

    p/q is the minimum cycle mean over all non-empty directed cycles, as
    coprime integers with q > 0.  edge_ids are indices into the original
    `edges`, in walking order, forming a simple cycle (closed walk with no
    repeated vertex except first == last) whose mean is exactly p/q.
    Returns None when the graph has no directed cycle.
    """
    E = list(edges)  # shallow snapshot; the caller's object is never touched

    # ---- Karp DP --------------------------------------------------------
    # D[k][v]   = minimum cost of a walk of exactly k edges ending at v
    #             (walks may start anywhere: D[0][v] = 0, the standard
    #             super-source formulation).  None marks "no such walk".
    # PAR[k][v] = id of the last edge of an optimal k-edge walk to v.
    D = [[0] * n]
    PAR = [[-1] * n]
    E4 = [(u, v, c, i) for i, (u, v, c) in enumerate(E)]
    for k in range(1, n + 1):
        prev = D[k - 1]
        cur = [None] * n
        par = [-1] * n
        for u, v, c, i in E4:
            du = prev[u]
            if du is not None:
                val = du + c
                cv = cur[v]
                if cv is None or val < cv:
                    cur[v] = val
                    par[v] = i
        D.append(cur)
        PAR.append(par)

    # ---- exact minimum cycle mean ---------------------------------------
    # Karp's theorem: lambda* = min_v  max_{0<=k<=n-1} (D[n][v]-D[k][v])/(n-k)
    # (max over k with D[k][v] reachable).  Exact integer comparisons only.
    Dn = D[n]
    best_num = best_den = None
    vstar = -1
    for v in range(n):
        dn = Dn[v]
        if dn is None:
            continue
        mk_num = mk_den = None
        for k in range(n):
            dk = D[k][v]
            if dk is None:
                continue
            num = dn - dk
            den = n - k  # 1 <= den <= n, always positive
            if mk_num is None or num * mk_den > mk_num * den:
                mk_num, mk_den = num, den
        if mk_num is None:
            continue
        if best_num is None or mk_num * best_den < best_num * mk_den:
            best_num, best_den, vstar = mk_num, mk_den, v

    if best_num is None:
        return None  # acyclic: no walk of n edges exists

    g = gcd(abs(best_num), best_den)  # best_den > 0, result coprime with q > 0
    p, q = best_num // g, best_den // g

    # ---- witness cycle --------------------------------------------------
    # Walk back n steps from v* along parent edges: a walk W of n edges and
    # n+1 vertices, so some vertex repeats.  Karp's bound is tight at v*,
    # which forces every simple cycle extracted from W to have mean exactly
    # lambda*; the first repeated vertex yields such a simple cycle, built
    # purely from original edge ids.
    walk_edges = [0] * n
    walk_vert = [0] * (n + 1)
    v = vstar
    for k in range(n, 0, -1):
        e = PAR[k][v]
        walk_edges[k - 1] = e
        walk_vert[k] = v
        v = E[e][0]
    walk_vert[0] = v

    first_pos = {}
    cyc_edges = None
    for i, vtx in enumerate(walk_vert):
        j = first_pos.get(vtx)
        if j is not None:
            cyc_edges = walk_edges[j:i]  # edges of the simple cycle x_j -> .. -> x_i
            break
        first_pos[vtx] = i

    return (p, q, cyc_edges)
