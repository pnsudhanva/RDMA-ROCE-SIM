"""Rail-optimized topology generator (NVIDIA DGX SuperPOD style).

The defining idea: a server holds G GPUs, each with its own NIC, and GPU
*rank* r in EVERY server connects to the same dedicated "rail" switch r.
Rank decides the rail, not physical proximity.

Why this helps AI training:
  NCCL decomposes an all-reduce across a multi-GPU cluster into
    1. intra-server reduce-scatter over NVLink   (never touches the network)
    2. inter-server all-reduce among same-rank GPUs
    3. intra-server all-gather over NVLink       (never touches the network)
  Step 2 is the only network-visible phase, and it only ever involves GPUs
  of the same rank. On a rail-optimized fabric that traffic is confined to
  one rail switch: one hop up, one hop down, and — crucially — it cannot
  collide with any other rank's traffic. Compare with a fat-tree, where
  ECMP may hash two elephant flows onto the same uplink and halve both.

The catch: cross-rail traffic (rank 0 talking to rank 3) has to climb to
the spine tier. A collective whose ring ordering ignores the rail structure
gets the worst of both worlds. That's a real failure mode and one of the
comparisons this project is built to measure.

Topology structure:
  - S servers x G GPUs per server = S*G hosts
  - G rail switches (leaf); rail r has S host-facing ports
  - P spine switches; every rail connects to every spine (full bipartite)
  - Default P = S, which makes each rail non-blocking (S down, S up)

Numbering convention:
  - Hosts:  [0, S*G)              host id = server * G + rank
  - Rails:  [S*G, S*G + G)
  - Spines: [S*G + G, S*G + G + P)

For S=16, G=8: 128 hosts (0-127), 8 rails (128-135), 16 spines (136-151).
That matches a k=8 fat-tree's 128 hosts, so the two are directly comparable.

Note on fairness: a rail switch here needs S host ports + P uplinks (32
ports for S=P=16), while a k=8 fat-tree uses 8-port switches throughout.
Same host count, different switch radix — worth stating explicitly when
reporting results.
"""

from dataclasses import dataclass
from typing import List, Optional

from .base import Link, Topology


@dataclass
class RailTopology(Topology):
    """Topology plus the rail-structure metadata traffic generators need."""
    n_servers: int = 0
    gpus_per_server: int = 0
    n_spine: int = 0

    def rank_of(self, host: int) -> int:
        """Which GPU slot (and therefore which rail) this host occupies."""
        return host % self.gpus_per_server

    def server_of(self, host: int) -> int:
        """Which physical server this host lives in."""
        return host // self.gpus_per_server

    def hosts_on_rail(self, rail: int) -> List[int]:
        """All hosts sharing rail `rail`, one per server."""
        return [s * self.gpus_per_server + rail for s in range(self.n_servers)]


def rail_optimized(
    n_servers: int,
    gpus_per_server: int,
    n_spine: Optional[int] = None,
    host_bw: str = "100Gbps",
    fabric_bw: str = "100Gbps",
    delay: str = "1000ns",
    error_rate: float = 0.0,
) -> RailTopology:
    """Build a rail-optimized fabric.

    Args:
        n_servers: number of servers (S)
        gpus_per_server: GPUs/NICs per server, which is also the rail count (G)
        n_spine: spine switch count (P). Defaults to n_servers, giving each
            rail equal up/down capacity.
        host_bw: bandwidth on GPU->rail links (NIC speed)
        fabric_bw: bandwidth on rail->spine links
        delay: per-link propagation delay
        error_rate: per-link bit error rate
    """
    if n_servers < 1:
        raise ValueError(f"n_servers must be >= 1, got {n_servers}")
    if gpus_per_server < 1:
        raise ValueError(f"gpus_per_server must be >= 1, got {gpus_per_server}")

    S, G = n_servers, gpus_per_server
    P = n_spine if n_spine is not None else S
    if P < 1:
        raise ValueError(f"n_spine must be >= 1, got {P}")

    H = S * G
    rail_base = H
    spine_base = H + G
    n_nodes = H + G + P

    switch_ids = list(range(rail_base, spine_base + P))
    links: List[Link] = []

    # 1) Host -> rail. Rank decides the rail; the server does not matter.
    for h in range(H):
        rail_id = rail_base + (h % G)
        links.append(Link(src=h, dst=rail_id, rate=host_bw, delay=delay, error_rate=error_rate))

    # 2) Rail -> spine, full bipartite.
    for r in range(G):
        rail_id = rail_base + r
        for p in range(P):
            spine_id = spine_base + p
            links.append(Link(src=rail_id, dst=spine_id, rate=fabric_bw, delay=delay, error_rate=error_rate))

    topo = RailTopology(
        n_nodes=n_nodes,
        switch_ids=switch_ids,
        links=links,
        name=f"rail-s{S}-g{G}-p{P}",
        n_servers=S,
        gpus_per_server=G,
        n_spine=P,
    )
    topo.validate()
    return topo


if __name__ == "__main__":
    # Run with `python3 -m topo.rail_optimized` inside the container.
    for S, G in ((4, 4), (16, 8)):
        t = rail_optimized(S, G)
        expected_hosts = S * G
        expected_links = S * G + G * S  # host->rail, plus rail->spine (P=S)
        print(t.summary())
        assert len(t.hosts()) == expected_hosts, f"S={S} G={G}: host count mismatch"
        assert len(t.links) == expected_links, f"S={S} G={G}: link count mismatch"
        # Every rail should hold exactly one GPU from each server.
        for r in range(G):
            on_rail = t.hosts_on_rail(r)
            assert len(on_rail) == S, f"rail {r} has {len(on_rail)} hosts, expected {S}"
            assert len({t.server_of(h) for h in on_rail}) == S, "rail spans duplicate servers"
    print("OK: rail-optimized generator self-checks passed.")
