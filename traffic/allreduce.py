"""Ring all-reduce traffic patterns.

Background — what a ring all-reduce actually puts on the wire:

  N GPUs form a logical ring. A tensor of D bytes is split into N chunks.
  The collective runs in 2(N-1) steps:
    - N-1 reduce-scatter steps: GPU i sends chunk (i-s) to GPU i+1
    - N-1 all-gather steps:     same communication pattern
  Every step, every GPU simultaneously sends D/N bytes to its successor.
  Total bytes each GPU pushes: 2(N-1)/N * D, i.e. ~2D for large N.

Modelling choice: HPCC's flow.txt schedules flows at absolute times and has
no barrier primitive, so we cannot express "step s+1 begins when step s
completes". We therefore simulate ONE step — every GPU sending D/N bytes to
its ring successor at the same instant. That single step is the fundamental
congestion pattern, and the full collective time is well approximated by

    T_allreduce  ~=  2 * (N - 1) * T_step

This is the standard analytical model for ring all-reduce and it keeps the
simulation honest: we measure exactly what we generate, and scale up in the
analysis rather than faking synchronisation.

Ring ordering matters enormously on a rail-optimized fabric:

  - `parallel_rings` builds G independent rings, one per GPU rank. This is
    what NCCL actually does — the inter-server phase of an all-reduce only
    ever involves same-rank GPUs. On rail-optimized wiring every one of
    those rings is confined to its own rail switch.

  - `single_ring` builds one ring over all hosts in ID order. Consecutive
    hosts are different ranks in the same server, so on rail-optimized
    wiring every hop is cross-rail and must climb to the spine. Great for
    demonstrating that a rail-optimized fabric only pays off when the
    collective is rail-aware.
"""

from typing import List, Sequence

from .base import Flow


def parallel_rings(hosts: Sequence[int], gpus_per_server: int) -> List[List[int]]:
    """Build one ring per GPU rank (the NCCL inter-server decomposition).

    Ring r contains every host whose rank is r, ordered by server. On a
    rail-optimized fabric each ring lives entirely on rail r.
    """
    if gpus_per_server < 1:
        raise ValueError(f"gpus_per_server must be >= 1, got {gpus_per_server}")
    rings: List[List[int]] = [[] for _ in range(gpus_per_server)]
    for h in sorted(hosts):
        rings[h % gpus_per_server].append(h)
    rings = [r for r in rings if len(r) > 1]
    if not rings:
        raise ValueError("no ring has more than one member; check host count")
    return rings


def single_ring(hosts: Sequence[int]) -> List[List[int]]:
    """One ring over every host, in ID order."""
    ring = sorted(hosts)
    if len(ring) < 2:
        raise ValueError("need at least 2 hosts to form a ring")
    return [ring]


def ring_chunk_bytes(tensor_bytes: int, ring_size: int) -> int:
    """Bytes moved per GPU per step: the tensor split across the ring."""
    return max(1, tensor_bytes // ring_size)


def ring_step_count(ring_size: int) -> int:
    """Steps in a full ring all-reduce: 2(N-1)."""
    return 2 * (ring_size - 1)


def ring_allreduce_step(
    rings: Sequence[Sequence[int]],
    chunk_bytes: int,
    start_time: float = 2.0,
    pg: int = 3,
) -> List[Flow]:
    """One synchronised step of ring all-reduce across every ring.

    Each GPU sends `chunk_bytes` to its successor in its own ring. All rings
    fire simultaneously, which is what happens in a real collective.
    """
    flows: List[Flow] = []
    for ring in rings:
        n = len(ring)
        if n < 2:
            continue
        for i, src in enumerate(ring):
            dst = ring[(i + 1) % n]
            flows.append(
                Flow(
                    src=src,
                    dst=dst,
                    size_bytes=chunk_bytes,
                    start_time=start_time,
                    pg=pg,
                    # unique per flow so fct.txt rows stay identifiable
                    dport=100 + len(flows),
                )
            )
    return flows
