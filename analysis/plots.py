"""Generate the project's figures from simulation results.

Run from inside the container, from /work:
    python3 -m analysis.plots

Produces, in plots/:
    fct_cdf.png            distribution of per-flow completion times
    collective_time.png    estimated time for one full ring all-reduce
    size_sweep.png         FCT vs message size, both fabrics, vs analytical model
"""

import glob
import math
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fabric colours, kept consistent across every figure.
C_RAIL = "#0D9488"
C_RAIL_ALT = "#5EBFB6"
C_FAT = "#D97706"
C_MODEL = "#64748B"

PLOT_DIR = "plots"

# Headline comparison: (path, label, colour, steps in a full collective).
# A ring of N GPUs needs 2(N-1) steps; parallel rings have N=16, the single
# ring has N=128.
HEADLINE = [
    ("results/allreduce-rail-parallel/fct.txt",
     "Rail-optimized\nrail-aware rings", C_RAIL, 30),
    ("results/allreduce-rail-single/fct.txt",
     "Rail-optimized\nnaive single ring", C_RAIL_ALT, 254),
    ("results/allreduce-fat-parallel/fct.txt",
     "Fat-tree\nrail-aware rings", C_FAT, 30),
]

# The CDF plots raw per-flow FCT, so it may only compare runs whose flows are
# the same size. The naive single ring splits the tensor 128 ways instead of
# 16, giving 125 KB flows against 1 MB — it would look fastest while being
# 6.6x slower overall. It belongs in the collective-time chart, not here.
CDF_CONFIGS = [c for c in HEADLINE if "naive" not in c[1]]

# Analytical model constants, verified against the simulator in Week 2.
HEADER_BYTES = 48          # per-packet RoCEv2 overhead HPCC charges
PAYLOAD_BYTES = 1000       # PACKET_PAYLOAD_SIZE
LINK_BPS = 100e9
RTT_PER_LINK_NS = 2080     # propagation both ways plus store-and-forward


def read_fct(path: str) -> List[Tuple[int, int, int]]:
    """Return (size_bytes, fct_ns, standalone_ns) per flow."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 8:
                rows.append((int(p[4]), int(p[6]), int(p[7])))
    return rows


def cdf(values: List[float]) -> Tuple[List[float], List[float]]:
    s = sorted(values)
    n = len(s)
    return s, [(i + 1) / n for i in range(n)]


def predicted_fct_ns(size_bytes: int, hops: int) -> float:
    """Analytical no-contention FCT: base RTT plus serialization."""
    packets = math.ceil(size_bytes / PAYLOAD_BYTES)
    on_wire = size_bytes + packets * HEADER_BYTES
    return hops * RTT_PER_LINK_NS + on_wire * 8 / LINK_BPS * 1e9


def style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{PLOT_DIR}/{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {PLOT_DIR}/{name}.png")


def plot_cdf() -> None:
    """Per-flow FCT distribution. The tail is the story."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for path, label, colour, _ in CDF_CONFIGS:
        if not os.path.exists(path):
            print(f"[plot] skipping missing {path}")
            continue
        fcts = [r[1] / 1000.0 for r in read_fct(path)]   # ns -> us
        xs, ys = cdf(fcts)
        ax.step(xs, ys, where="post", color=colour, linewidth=2.2,
                label=label.replace("\n", ", "))
    ax.set_xscale("log")
    ax.set_xlabel("Flow completion time (µs, log scale)")
    ax.set_ylabel("Fraction of flows")
    ax.set_title("One step of ring all-reduce: 128 GPUs, 1 MB per flow\n"
                 "Rail-optimized is a vertical line; the fat-tree drags a 10x tail",
                 fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    save(fig, "fct_cdf")


def plot_collective() -> None:
    """Estimated full-collective time = steps x slowest step."""
    labels, totals, colours = [], [], []
    for path, label, colour, steps in HEADLINE:
        if not os.path.exists(path):
            continue
        worst_ns = max(r[1] for r in read_fct(path))
        labels.append(label)
        totals.append(steps * worst_ns / 1e6)            # ns -> ms
        colours.append(colour)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels, totals, color=colours, height=0.55)
    for bar, total in zip(bars, totals):
        ax.text(bar.get_width() + max(totals) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{total:.2f} ms", va="center", fontsize=10)
    ax.set_xlabel("Estimated time for one full all-reduce (ms)")
    ax.set_xlim(0, max(totals) * 1.18)
    ax.invert_yaxis()
    ax.set_title("128 GPUs, 16 MB tensor\n"
                 "Collective time = steps × slowest step (1 step measured)",
                 fontsize=11)
    style(ax)
    save(fig, "collective_time")


def _sweep_series(prefix: str) -> Dict[int, Tuple[float, float]]:
    """chunk_bytes -> (median_fct_ns, max_fct_ns) for one fabric."""
    out: Dict[int, Tuple[float, float]] = {}
    for path in glob.glob(f"results/{prefix}-c*/fct.txt"):
        m = re.search(rf"{prefix}-c(\d+)/", path)
        if not m:
            continue
        rows = read_fct(path)
        if not rows:
            continue
        fcts = sorted(r[1] for r in rows)
        median = fcts[len(fcts) // 2]
        out[int(m.group(1))] = (median, max(fcts))
    return out


def plot_sweep() -> None:
    """FCT vs message size, both fabrics, with the analytical model overlaid."""
    rail = _sweep_series("sweep-rail")
    fat = _sweep_series("sweep-fat")
    if not rail and not fat:
        print("[plot] no sweep results found; run experiments/sweep.sh first")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for series, colour, label in ((rail, C_RAIL, "Rail-optimized"),
                                  (fat, C_FAT, "Fat-tree")):
        if not series:
            continue
        sizes = sorted(series)
        ax.plot(sizes, [series[s][0] / 1000 for s in sizes], "o-",
                color=colour, linewidth=2.2, markersize=5,
                label=f"{label} (median)")
        ax.plot(sizes, [series[s][1] / 1000 for s in sizes], "^--",
                color=colour, linewidth=1.4, markersize=5, alpha=0.65,
                label=f"{label} (slowest flow)")

    sizes = sorted(rail or fat)
    ax.plot(sizes, [predicted_fct_ns(s, hops=2) / 1000 for s in sizes],
            ":", color=C_MODEL, linewidth=2,
            label="Analytical model (2 hops, no contention)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bytes per flow (log scale)")
    ax.set_ylabel("Flow completion time (µs, log scale)")
    ax.set_title("Latency-bound below ~10 KB, bandwidth-bound above\n"
                 "The fat-tree's ECMP penalty only appears with elephant flows",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    save(fig, "size_sweep")


if __name__ == "__main__":
    plot_cdf()
    plot_collective()
    plot_sweep()
