"""Experiment orchestrator.

Glues a topology generator + traffic pattern + HPCC config together,
writes everything to /work/results/<name>/, then runs the HPCC simulator.

Run from inside the container (the host-side `make shell` shortcut):
    python3 -m experiments.run \\
        --topo fat-tree --k 4 \\
        --traffic incast --n-senders 4 --message-size 1000000 \\
        --cc dcqcn \\
        --name k4-incast-4to1
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Make sibling packages importable when invoked as `python3 -m experiments.run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topo.base import Topology
from topo.fat_tree import fat_tree
from traffic.base import Flow, write_hpcc_flow
from traffic.incast import incast


# CC_MODE values from HPCC's config_doc.txt
CC_MODE = {
    "dcqcn": 1,
    "hpcc": 3,
    "timely": 7,
    "dctcp": 8,
    "hpcc-pint": 10,
}


def write_trace_all_nodes(topo: Topology, path: str) -> None:
    """Trace every node in the topology (simplest choice for now)."""
    ids = list(range(topo.n_nodes))
    with open(path, "w") as f:
        f.write(f"{len(ids)}\n")
        f.write(" ".join(str(i) for i in ids) + "\n")


def write_config(
    out_dir: Path,
    sim_time: float,
    cc_mode: int,
    packet_payload_size: int,
    enable_trace: int,
) -> Path:
    """Emit HPCC's config.txt with absolute paths so it doesn't matter where
    the simulator is invoked from."""
    cfg = f"""ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1

PACKET_PAYLOAD_SIZE {packet_payload_size}

TOPOLOGY_FILE {out_dir}/topology.txt
FLOW_FILE {out_dir}/flow.txt
TRACE_FILE {out_dir}/trace.txt
TRACE_OUTPUT_FILE {out_dir}/mix.tr
FCT_OUTPUT_FILE {out_dir}/fct.txt
PFC_OUTPUT_FILE {out_dir}/pfc.txt

SIMULATOR_STOP_TIME {sim_time}

CC_MODE {cc_mode}
ALPHA_RESUME_INTERVAL 1
RATE_DECREASE_INTERVAL 4
CLAMP_TARGET_RATE 0
RP_TIMER 900
EWMA_GAIN 0.00390625
FAST_RECOVERY_TIMES 1
RATE_AI 50Mb/s
RATE_HAI 100Mb/s
MIN_RATE 100Mb/s
DCTCP_RATE_AI 1000Mb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN 1
GLOBAL_T 1
VAR_WIN 1
FAST_REACT 1
U_TARGET 0.95
MI_THRESH 0
INT_MULTI 1
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE 1.05
PINT_PROB 1.0

RATE_BOUND 1
ACK_HIGH_PRIO 0

LINK_DOWN 0 0 0

ENABLE_TRACE {enable_trace}

KMAX_MAP 3 25000000000 400 50000000000 800 100000000000 1600
KMIN_MAP 3 25000000000 100 50000000000 200 100000000000 400
PMAX_MAP 3 25000000000 0.2 50000000000 0.2 100000000000 0.2
BUFFER_SIZE 32
QLEN_MON_FILE {out_dir}/qlen.txt
QLEN_MON_START 2000000000
QLEN_MON_END 2010000000
"""
    cfg_path = out_dir / "config.txt"
    cfg_path.write_text(cfg)
    return cfg_path


def build_topology(args) -> Topology:
    if args.topo == "fat-tree":
        return fat_tree(
            k=args.k,
            host_bw=args.bw,
            fabric_bw=args.bw,
            delay=args.delay,
        )
    raise ValueError(f"unknown topology: {args.topo}")


def build_flows(args, topo: Topology):
    hosts = topo.hosts()
    if args.traffic == "incast":
        if args.n_senders + 1 > len(hosts):
            raise ValueError(
                f"need at least {args.n_senders + 1} hosts for incast, "
                f"topology has {len(hosts)}"
            )
        receiver = hosts[0]
        senders = hosts[1 : 1 + args.n_senders]
        return incast(
            senders=senders,
            receiver=receiver,
            size_bytes=args.message_size,
            start_time=args.start_time,
        )
    raise ValueError(f"unknown traffic pattern: {args.traffic}")


def run_simulator(config_path: Path, ns3_home: str = "/opt/hpcc/simulation") -> int:
    """Invoke ns-3 with our config. Returns the process exit code."""
    cmd = ["./waf", "--run", f"scratch/third {config_path}"]
    print(f"\n[run] cwd={ns3_home}")
    print(f"[run] {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ns3_home)
    elapsed = time.time() - t0
    print(f"\n[run] simulator exited with code {result.returncode} after {elapsed:.1f}s")
    return result.returncode


def main():
    p = argparse.ArgumentParser(description="Run an HPCC simulation experiment.")
    # Topology
    p.add_argument("--topo", choices=["fat-tree"], required=True)
    p.add_argument("--k", type=int, default=4, help="fat-tree k parameter")
    p.add_argument("--bw", default="100Gbps", help="link bandwidth")
    p.add_argument("--delay", default="1000ns", help="per-link delay")
    # Traffic
    p.add_argument("--traffic", choices=["incast"], required=True)
    p.add_argument("--n-senders", type=int, default=4)
    p.add_argument("--message-size", type=int, default=1_000_000, help="bytes per flow")
    p.add_argument("--start-time", type=float, default=2.0, help="when flows start (s)")
    p.add_argument("--packet-payload-size", type=int, default=1000)
    # Simulator
    p.add_argument("--cc", choices=list(CC_MODE), default="dcqcn")
    p.add_argument("--sim-time", type=float, default=3.0, help="simulated stop time (s)")
    p.add_argument("--no-packet-trace", action="store_true",
                   help="disable packet-level trace (mix.tr); FCT/PFC/qlen still produced")
    # Output
    p.add_argument("--name", required=True, help="experiment name (directory under out-base)")
    p.add_argument("--out-base", default="/work/results")
    p.add_argument("--dry-run", action="store_true",
                   help="generate config files but do not run ns-3")
    args = p.parse_args()

    out_dir = Path(args.out_base) / args.name
    if out_dir.exists():
        print(f"[run] wiping existing output dir: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1. Topology
    topo = build_topology(args)
    print(f"[gen] {topo.summary()}")
    topo.write_hpcc(str(out_dir / "topology.txt"))

    # 2. Traffic
    flows = build_flows(args, topo)
    print(f"[gen] {len(flows)} flows ({args.traffic})")
    write_hpcc_flow(flows, str(out_dir / "flow.txt"))

    # 3. Trace + config
    write_trace_all_nodes(topo, str(out_dir / "trace.txt"))
    cfg_path = write_config(
        out_dir=out_dir,
        sim_time=args.sim_time,
        cc_mode=CC_MODE[args.cc],
        packet_payload_size=args.packet_payload_size,
        enable_trace=0 if args.no_packet_trace else 1,
    )
    print(f"[gen] config written: {cfg_path}")

    if args.dry_run:
        print("[run] --dry-run: skipping simulator")
        return 0

    # 4. Run
    return run_simulator(cfg_path)


if __name__ == "__main__":
    sys.exit(main())
