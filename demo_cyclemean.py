#!/usr/bin/env python3
"""Demo of cyclemean.minimum on the specification example and more."""

from cyclemean import minimum


def show(label, n, edges):
    print(f"{label}: n={n}, edges={edges}")
    res = minimum(n, edges)
    if res is None:
        print("  -> no directed cycle, returns None\n")
        return
    p, q, ids = res
    total = sum(edges[i][2] for i in ids)
    path = [edges[ids[0]][0]] + [edges[i][1] for i in ids]
    print(f"  -> minimum mean = {p}/{q}"
          + ("" if q == 1 else f" (={p / q:.6f} approx)")
          + f", witness edge_ids={ids}")
    print(f"     cycle {' -> '.join(map(str, path))}, "
          f"total cost {total}, length {len(ids)}\n")


show("specification example", 2, [(0, 1, 2), (1, 0, 4), (1, 1, 5)])
show("disconnected components (best cycle wins)", 4,
     [(0, 1, 0), (1, 0, 6), (2, 3, 1), (3, 2, 1)])
show("acyclic graph", 3, [(0, 1, 1), (1, 2, 1), (0, 2, 5)])
show("negative self-loop beats negative 2-cycle", 3,
     [(0, 0, -3), (1, 2, -5), (2, 1, 1)])
show("parallel edges (cheapest one is used)", 2,
     [(0, 1, 5), (0, 1, 1), (1, 0, 3)])
show("fractional mean", 2, [(0, 1, 0), (1, 0, 1)])
