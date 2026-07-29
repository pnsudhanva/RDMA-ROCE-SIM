"""Summarise one or more HPCC fct.txt files side by side.

fct.txt columns (see third.cc::qp_finish):
    sip dip sport dport size_bytes start_time_ns fct_ns standalone_fct_ns

The headline metric is *slowdown* = fct / standalone_fct: how much worse a
flow did than it would have alone on the network. 1.0 is perfect. The p99
and max slowdown matter more than the mean, because a collective finishes
only when its slowest member finishes.

Usage (inside the container):
    python3 -m analysis.fct_stats /work/results/*/fct.txt
"""

import argparse
import statistics
from pathlib import Path
from typing import List, Tuple


def read_fct(path: str) -> List[Tuple[int, int, int]]:
    """Return (size_bytes, fct_ns, standalone_ns) per flow."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append((int(parts[4]), int(parts[6]), int(parts[7])))
    return rows


def pct(values: List[float], q: float) -> float:
    """Simple nearest-rank percentile; q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q / 100.0 * len(s))) - 1))
    return s[idx]


def summarise(path: str) -> dict:
    rows = read_fct(path)
    if not rows:
        return {"label": Path(path).parent.name, "n": 0}
    fcts = [r[1] for r in rows]
    slow = [r[1] / r[2] for r in rows if r[2] > 0]
    return {
        "label": Path(path).parent.name,
        "n": len(rows),
        "size": rows[0][0],
        "fct_min": min(fcts),
        "fct_p50": statistics.median(fcts),
        "fct_p99": pct(fcts, 99),
        "fct_max": max(fcts),
        "slow_mean": statistics.mean(slow) if slow else float("nan"),
        "slow_max": max(slow) if slow else float("nan"),
    }


def fmt_ns(ns: float) -> str:
    """Render nanoseconds in whatever unit reads most naturally."""
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.1f} us"
    return f"{ns:.0f} ns"


def main():
    p = argparse.ArgumentParser(description="Compare HPCC fct.txt results.")
    p.add_argument("paths", nargs="+", help="one or more fct.txt files")
    args = p.parse_args()

    stats = [summarise(path) for path in args.paths]
    stats = [s for s in stats if s.get("n")]
    if not stats:
        print("no flows found in any input")
        return 1

    width = max(len(s["label"]) for s in stats)
    header = (
        f"{'experiment'.ljust(width)}  {'flows':>5}  {'p50':>9}  {'p99':>9}  "
        f"{'max':>9}  {'slowdown avg':>12}  {'slowdown max':>12}"
    )
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['label'].ljust(width)}  {s['n']:>5}  "
            f"{fmt_ns(s['fct_p50']):>9}  {fmt_ns(s['fct_p99']):>9}  "
            f"{fmt_ns(s['fct_max']):>9}  {s['slow_mean']:>11.2f}x  "
            f"{s['slow_max']:>11.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
