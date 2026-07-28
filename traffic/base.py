"""Flow data model and HPCC-format writer.

HPCC's flow.txt format (read by third.cc::ReadFlowInput at line 131):

    <flow_count>
    <src> <dst> <pg> <dport> <size_bytes> <start_time_seconds>
    ...one line per flow...

Fields:
  - src/dst: host node IDs from the topology
  - pg: priority group (DCQCN/PFC traffic class); 3 is the common RoCE default
  - dport: destination port (just an identifier so multiple flows between the
    same pair don't collide; HPCC assigns the source port itself)
  - size_bytes: flow size in BYTES. third.cc reads this into a field named
    `maxPacketCount`, but that name is a misnomer — it is forwarded to
    RdmaClient's "WriteSize" attribute, documented in rdma-client.cc as
    "The number of bytes to write", and RdmaQueuePair::GetBytesLeft() treats
    it as a byte count. Do NOT pre-divide by PACKET_PAYLOAD_SIZE.
  - start_time: when this flow begins, in seconds (float). HPCC requires
    flows ordered by ascending start_time.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Flow:
    src: int
    dst: int
    size_bytes: int
    start_time: float = 2.0   # seconds; HPCC convention: warmup until ~2s
    pg: int = 3               # priority group (RoCE default class)
    dport: int = 100          # destination port


def write_hpcc_flow(flows: List[Flow], path: str) -> None:
    """Write flows in HPCC's flow.txt format. Auto-sorts by start_time."""
    flows = sorted(flows, key=lambda f: f.start_time)
    with open(path, "w") as f:
        f.write(f"{len(flows)}\n")
        for fl in flows:
            f.write(
                f"{fl.src} {fl.dst} {fl.pg} {fl.dport} "
                f"{fl.size_bytes} {fl.start_time}\n"
            )
