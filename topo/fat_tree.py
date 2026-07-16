"""k-ary fat-tree topology generator (Al-Fares et al., 2008).

A k-ary fat-tree has:
  - k pods, each pod contains k/2 edge switches and k/2 aggregation switches.
  - (k/2)^2 core switches.
  - k^3/4 hosts (k/2 hosts per edge switch).
  - 3*k^3/4 links total (host-edge + edge-agg + agg-core).

Bandwidth between any two layers is preserved (full bisection bandwidth)
because the number of switches at each layer balances the fan-in/fan-out.

Numbering convention used here (so it's easy to reason about IDs in plots):
  - Hosts:           [0, H)              where H = k^3 / 4
  - Edge switches:   [H, H + k^2/2)
  - Aggregation:     [H + k^2/2, H + k^2)
  - Core switches:   [H + k^2, H + 5*k^2/4)

For k=4: 16 hosts (0-15), 8 edges (16-23), 8 aggs (24-31), 4 cores (32-35).
For k=8: 128 hosts, 32 edges, 32 aggs, 16 cores → 208 nodes, 384 links.
"""

from typing import List

from .base import Link, Topology


def fat_tree(
    k: int,
    host_bw: str = "100Gbps",
    fabric_bw: str = "100Gbps",
    delay: str = "1000ns",
    error_rate: float = 0.0,
) -> Topology:
    """Build a k-ary fat-tree.

    Args:
        k: must be even and >= 2. k=4 gives 16 hosts (handy for quick sims).
        host_bw: bandwidth on host→edge links (e.g. NIC speed).
        fabric_bw: bandwidth on fabric links (edge↔agg, agg↔core).
        delay: per-link propagation delay.
        error_rate: per-link bit error rate (0.0 = perfect link).
    """
    if k < 2 or k % 2 != 0:
        raise ValueError(f"k must be an even integer >= 2, got {k}")

    half = k // 2
    H = (k ** 3) // 4                   # number of hosts
    E = k * half                         # edge switches = k * k/2 = k^2/2
    A = k * half                         # agg switches
    C = half * half                      # core switches = (k/2)^2

    # ID ranges
    edge_base = H
    agg_base = H + E
    core_base = H + E + A
    n_nodes = H + E + A + C

    switch_ids = list(range(edge_base, core_base + C))
    links: List[Link] = []

    # 1) Host -> edge links.
    #    Each pod has k^2/4 hosts (== k/2 per edge switch). Host h is in pod (h // (k^2/4))
    #    and connects to edge (h // (k/2)) in that pod's edge group.
    hosts_per_pod = (k * k) // 4
    for h in range(H):
        pod = h // hosts_per_pod
        edge_in_pod = (h % hosts_per_pod) // half          # 0 .. k/2-1
        edge_id = edge_base + pod * half + edge_in_pod
        links.append(Link(src=h, dst=edge_id, rate=host_bw, delay=delay, error_rate=error_rate))

    # 2) Edge -> agg links inside each pod (full bipartite k/2 x k/2).
    for pod in range(k):
        for e in range(half):
            edge_id = edge_base + pod * half + e
            for a in range(half):
                agg_id = agg_base + pod * half + a
                links.append(Link(src=edge_id, dst=agg_id, rate=fabric_bw, delay=delay, error_rate=error_rate))

    # 3) Agg -> core links.
    #    Canonical wiring: aggregation switch index `a` within a pod connects to
    #    core switches [a*k/2 .. (a+1)*k/2 - 1]. This guarantees disjoint upward
    #    paths and is the standard fat-tree spec.
    for pod in range(k):
        for a in range(half):
            agg_id = agg_base + pod * half + a
            for j in range(half):
                core_id = core_base + a * half + j
                links.append(Link(src=agg_id, dst=core_id, rate=fabric_bw, delay=delay, error_rate=error_rate))

    topo = Topology(
        n_nodes=n_nodes,
        switch_ids=switch_ids,
        links=links,
        name=f"fat-tree-k{k}",
    )
    topo.validate()
    return topo


if __name__ == "__main__":
    # Quick sanity check — run with `python3 -m topo.fat_tree` inside the container.
    for k in (2, 4, 8):
        t = fat_tree(k)
        expected_hosts = (k ** 3) // 4
        expected_links = 3 * (k ** 3) // 4
        print(t.summary())
        assert len(t.hosts()) == expected_hosts, f"k={k}: host count mismatch"
        assert len(t.links) == expected_links, f"k={k}: link count mismatch"
    print("OK: fat-tree generator self-checks passed.")
