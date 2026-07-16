"""Flow data model and HPCC-format writer.

HPCC's flow.txt format (read by third.cc::ReadFlowInput at line 131):

    <flow_count>
    <src> <dst> <pg> <dport> <packet_count> <start_time_seconds>
    ...one line per flow...

Fields:
  - src/dst: host node IDs from the topology
  - pg: priority group (DCQCN/PFC traffic class); 3 is the common RoCE default
  - dport: destination port (just an identifier so multiple flows between the
    same pair don't collide; HPCC assigns the source port itself)
  - packet_count: flow size in *packets* (NOT bytes). Multiply by
    PACKET_PAYLOAD_SIZE from config.txt to get bytes.
  - start_time: when this flow begins, in seconds (float). HPCC requires
    flows ordered by ascending start_time.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Flow:
    src: int
    dst: int
    packet_count: int
    start_time: float = 2.0   # seconds; HPCC convention: warmup until ~2s
    pg: int = 3               # priority group (RoCE default class)
    dport: int = 100          # destination port


def bytes_to_packets(size_bytes: int, packet_payload_size: int = 1000) -> int:
    """Convert flow size in bytes -> packet count for HPCC's flow.txt."""
    return max(1, size_bytes // packet_payload_size)


def write_hpcc_flow(flows: List[Flow], path: str) -> None:
    """Write flows in HPCC's flow.txt format. Auto-sorts by start_time."""
    flows = sorted(flows, key=lambda f: f.start_time)
    with open(path, "w") as f:
        f.write(f"{len(flows)}\n")
        for fl in flows:
            f.write(
                f"{fl.src} {fl.dst} {fl.pg} {fl.dport} "
                f"{fl.packet_count} {fl.start_time}\n"
            )
