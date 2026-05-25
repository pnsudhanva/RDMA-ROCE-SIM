# RDMA/RoCE GPU Cluster Network Fabric Simulator

Packet-level simulation of a multi-node GPU training fabric using **ns-3** with **RoCEv2 / DCQCN / PFC / ECN**, comparing **fat-tree** and **rail-optimized** topologies under realistic AI workloads (ring all-reduce, tree all-reduce, incast). Findings validated against real `nccl-tests` on cloud GPUs.

## Why this exists

AI training is bottlenecked by the network. A 1ms hiccup across thousands of GPUs costs millions in idle compute. Network engineers who can reason about RDMA fabrics, congestion control, and collective communication patterns are in short supply. This project is a portfolio piece demonstrating that skill set.

## Status

`v0` — scaffolded. Container not yet built.

## Prerequisites

- Docker Desktop (running)
- VS Code with the **Dev Containers** extension
- ~10 GB free disk

On Apple Silicon, Docker Desktop's **"Use Rosetta for x86/amd64 emulation"** must be enabled (Settings → General). The simulator runs as `linux/amd64` because the academic ns-3 fork hasn't been validated on ARM64.

## Quick start

```bash
# One-time: build the simulator container (~15-25 min)
docker build -t rdma-sim .

# Open an interactive shell inside the container
docker run -it --rm -v "$PWD":/work rdma-sim bash
```

Or in VS Code: open this folder, then `Cmd+Shift+P` → **Dev Containers: Reopen in Container**.

## Project structure

```
RDMA-ROCE-SIM/
├── Dockerfile                     # ns-3.30 + Alibaba HPCC RDMA fork on Ubuntu 20.04
├── .devcontainer/devcontainer.json # VS Code dev container config
├── topo/                          # Python topology generators (fat-tree, rail-optimized)
├── traffic/                       # Traffic patterns (ring/tree all-reduce, incast)
├── experiments/                   # Scripts that drive ns-3 with topo + traffic configs
├── analysis/                      # Parse ns-3 output → CSV → plots
├── results/                       # Raw simulation output (gitignored)
└── plots/                         # Final figures
```

## 4-weekend plan

**Weekend 1 — Environment & first packet** *(we are here)*
- Build the Docker image, get ns-3 + RoCE fork compiling
- Run a baseline single-switch RoCEv2 flow, plot throughput from the trace
- Wire VS Code Dev Containers

**Weekend 2 — Topology & traffic library**
- Python module to emit ns-3 config for fat-tree (k=4, k=8) and rail-optimized
- Implement ring all-reduce, tree all-reduce, incast, all-to-all traffic patterns
- Expose DCQCN/PFC/ECN knobs as CLI flags

**Weekend 3 — Experiments & plots**
- Sweeps: throughput vs message size, FCT CDF under incast, PFC pause duration vs load
- Headline comparison: fat-tree vs rail-optimized, ring all-reduce at 70% load
- Export 6-8 publication-quality plots

**Weekend 4 — Real-hardware validation + writeup** *(~$0.50 cloud spend)*
- Rent 2× RTX 3090 on Vast.ai for 1 hour, run `nccl-tests`
- Validate one simulator prediction against real NCCL within 10-20%
- Polish README, plots, methodology section

## Methodology

The simulator is built on the [Alibaba HPCC fork of ns-3](https://github.com/alibaba-edu/High-Precision-Congestion-Control), which extends ns-3.30 with full RoCEv2, DCQCN, PFC, ECN, HPCC, and TIMELY support. We build it inside a reproducible Docker container so the entire experiment is one `docker build` away from anyone on any machine.

## License

MIT (project code). The vendored ns-3 fork retains its original license (GPLv2).
