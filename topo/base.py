"""Topology data model and HPCC-format writer.

HPCC's topology.txt format (read by third.cc):

    <total_nodes> <switch_count> <link_count>
    <space-separated switch node IDs>
    <src> <dst> <rate> <delay> <error_rate>
    ...one line per link...

Node IDs are integers. Hosts and switches share the integer space — the
switch-ID line tells the simulator which IDs are switches; everything else
is a host (RDMA endpoint).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Link:
    src: int
    dst: int
    rate: str = "100Gbps"   # e.g. "100Gbps", "25Gbps"
    delay: str = "1000ns"   # e.g. "1000ns", "0.001ms"
    error_rate: float = 0.0


@dataclass
class Topology:
    """An HPCC-compatible network topology."""
    n_nodes: int
    switch_ids: List[int]
    links: List[Link] = field(default_factory=list)
    name: str = "topology"

    def hosts(self) -> List[int]:
        """Return host node IDs (everything that isn't a switch)."""
        sw = set(self.switch_ids)
        return [i for i in range(self.n_nodes) if i not in sw]

    def validate(self) -> None:
        """Sanity checks before writing."""
        assert self.n_nodes > 0, "n_nodes must be positive"
        assert len(self.switch_ids) > 0, "must have at least one switch"
        for s in self.switch_ids:
            assert 0 <= s < self.n_nodes, f"switch id {s} out of range"
        for L in self.links:
            assert 0 <= L.src < self.n_nodes, f"link src {L.src} out of range"
            assert 0 <= L.dst < self.n_nodes, f"link dst {L.dst} out of range"
            assert L.src != L.dst, "self-loop not allowed"

    def write_hpcc(self, path: str) -> None:
        """Write the topology in HPCC's topology.txt format."""
        self.validate()
        with open(path, "w") as f:
            f.write(f"{self.n_nodes} {len(self.switch_ids)} {len(self.links)}\n")
            f.write(" ".join(str(s) for s in self.switch_ids) + "\n")
            for L in self.links:
                f.write(f"{L.src} {L.dst} {L.rate} {L.delay} {L.error_rate:.6f}\n")

    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"{self.name}: {self.n_nodes} nodes "
            f"({len(self.hosts())} hosts + {len(self.switch_ids)} switches), "
            f"{len(self.links)} links"
        )
