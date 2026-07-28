"""Incast traffic pattern: N senders -> 1 receiver, synchronized.

This is the simplest pathological pattern for AI fabrics:
  - All N senders begin transmitting at the same instant
  - They all target a single receiver
  - The receiver's last-hop link becomes a hotspot
  - DCQCN/HPCC has to throttle senders fast enough to prevent PFC pauses,
    or PFC kicks in and we see backpressure cascading upstream

A well-tuned RoCE fabric handles 100:1 incast cleanly. A badly tuned one
generates PFC storms. This pattern is *the* go-to test for measuring how
well a fabric absorbs synchronized bursts.
"""

from typing import List

from .base import Flow


def incast(
    senders: List[int],
    receiver: int,
    size_bytes: int,
    start_time: float = 2.0,
    pg: int = 3,
) -> List[Flow]:
    """Build flows for a synchronized incast.

    Args:
        senders: list of host node IDs that will send
        receiver: host node ID that receives all flows
        size_bytes: per-flow size in bytes
        start_time: when the burst begins (seconds, simulated time)
        pg: priority group (RoCE default is 3)
    """
    if receiver in senders:
        raise ValueError("receiver cannot also be a sender")
    if not senders:
        raise ValueError("need at least one sender")

    return [
        Flow(
            src=s,
            dst=receiver,
            size_bytes=size_bytes,
            start_time=start_time,
            pg=pg,
            # dport per-sender so HPCC's port-tracking doesn't collide
            dport=100 + i,
        )
        for i, s in enumerate(senders)
    ]
