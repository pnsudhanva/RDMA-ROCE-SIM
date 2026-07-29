# RDMA/RoCE GPU Cluster Network Fabric Simulator

Packet-level simulation of AI training fabrics using **ns-3** with **RoCEv2, DCQCN, PFC and ECN**, comparing **fat-tree** against **rail-optimized** topologies under realistic collective-communication workloads.

**Headline result:** on a 128-GPU ring all-reduce, a rail-optimized fabric completed every one of 128 flows in exactly 91.0 µs — zero variance, 1.03× its theoretical floor. The same collective on a fat-tree had a worst-case flow run **10.4× slower than it should have**. Extrapolated to a full collective, that is 2.73 ms versus 28.8 ms.

---

## The question

Training a large model means thousands of GPUs exchanging gradients dozens of times a second. When the network stalls, the GPUs idle, and idle GPUs are the single largest cost line in an AI cluster. So: **how should you wire one, and how much does the wiring actually matter?**

Two designs dominate:

- **Fat-tree** — the general-purpose datacenter topology. Many equal-cost paths, load-balanced by ECMP hashing.
- **Rail-optimized** — purpose-built for AI (NVIDIA DGX SuperPOD). Each GPU has its own NIC, and GPU *rank* r in every server connects to a dedicated "rail" switch r.

This project builds both, runs identical workloads across them, and measures the difference.

---

## Findings

### 1. Rail-optimized wins by removing path choice, not by shortening paths

![Flow completion time CDF](plots/fct_cdf.png)

One step of ring all-reduce, 128 GPUs, 1 MB per flow, DCQCN congestion control:

| Fabric | median FCT | p99 | worst | worst slowdown |
|---|---|---|---|---|
| Rail-optimized | 91.0 µs | 91.0 µs | 91.0 µs | **1.03×** |
| Fat-tree | 178.7 µs | 539.0 µs | 959.9 µs | **10.42×** |

The rail-optimized CDF is a vertical line: all 128 flows finished at the identical time. That is not luck. When NCCL's inter-server all-reduce phase runs on a rail-optimized fabric, every flow's path is host → rail switch → host. There is exactly **one** route, so ECMP has no decision to make and no hash collision is possible.

The fat-tree's tail is the mirror image of that. Half the ring hops cross pods and must climb to the core, where 64 elephant flows are hashed onto core links. Some links draw two or three flows while others sit idle, and the unlucky flows pay for it.

This is the project's central claim: **for elephant-flow workloads, the value of a topology is largely in how little path ambiguity it leaves.**

### 2. The fabric only pays off if the collective is rail-aware

![Estimated collective time](plots/collective_time.png)

Reorder the ring naively — 0 → 1 → 2 → … → 127 — and every hop becomes cross-rail, because consecutive host IDs are different GPU ranks in the *same* server. Traffic must climb to the spine tier, multipath returns, and so does the tail (3.78× worst-case slowdown).

Same hardware. **6.6× slower**, purely from ring ordering. Buying the topology is not enough; the collective library has to understand it.

> Collective time is estimated as `steps × slowest step`, with one step measured. A ring of N GPUs needs 2(N-1) steps: 30 for the 16-GPU parallel rings, 254 for a single 128-GPU ring.

### 3. Two regimes, and AI training lives in the expensive one

![Message size sweep](plots/size_sweep.png)

| Bytes per flow | Rail median | Fat-tree median | Fat-tree worst | Fat-tree avg slowdown |
|---|---|---|---|---|
| 1 KB | 4.2 µs | 10.6 µs | 12.7 µs | 1.00× |
| 10 KB | 4.9 µs | 12.4 µs | 15.0 µs | 1.06× |
| 100 KB | 12.7 µs | 25.4 µs | 42.2 µs | 1.45× |
| 1 MB | 91.0 µs | 178.7 µs | 959.9 µs | 1.96× |
| 4 MB | 352.0 µs | 689.0 µs | 1.59 ms | 1.99× |

Below ~10 KB the fabric is **latency-bound**: completion time is round-trip propagation, hop count decides, and congestion is irrelevant. Above ~100 KB it is **bandwidth-bound**: the bottleneck link's capacity decides, and contention dominates. Real gradient tensors are megabytes, so AI training sits firmly in the second regime — which is exactly where the fat-tree's ECMP penalty appears.

Rail-optimized held a slowdown between **0.98× and 1.04× across four orders of magnitude** of message size, with the slowest flow always equal to the median.

PFC never fired in any configuration. DCQCN's ECN-based rate control absorbed every burst without needing the emergency brake.

---

## Validating the simulator

Simulator output is only worth as much as your ability to check it. Every result here was cross-checked against a closed-form model derived independently of the simulator:

```
FCT = base_rtt + (bytes_on_wire × 8) / bandwidth
bytes_on_wire = size + ceil(size / payload) × header_bytes
```

The dotted line in the sweep plot is that model. It tracks the rail-optimized measurements to within ~4% across the whole range — expected, since a contention-free fabric should behave exactly like the no-contention formula.

The same discipline caught a real bug. HPCC's `third.cc` reads the flow-size field into a variable named `maxPacketCount`, so the generator initially divided byte counts by the packet payload size. The arithmetic gave it away: a 10 MB transfer finishing in 12 µs would require moving data faster than the link rate. Reading `rdma-client.cc` confirmed the field is wired to an attribute documented as *"the number of bytes to write."* Fixed in [`ad2e414`](../../commit/ad2e414).

---

## How it works

```
experiments/run.py                    orchestrator
  ├── topo/fat_tree.py                k-ary fat-tree (Al-Fares wiring)
  ├── topo/rail_optimized.py          DGX SuperPOD-style rails + spines
  ├── traffic/allreduce.py            ring all-reduce, rail-aware or naive
  └── traffic/incast.py               synchronized N-to-1 incast
        ↓  emits topology.txt, flow.txt, trace.txt, config.txt
  ns-3 + Alibaba HPCC fork            packet-level RoCEv2 / DCQCN / PFC / ECN
        ↓  emits fct.txt, pfc.txt, qlen.txt
  analysis/fct_stats.py               slowdown summaries
  analysis/plots.py                   figures
```

The simulator is the [Alibaba HPCC fork of ns-3](https://github.com/alibaba-edu/High-Precision-Congestion-Control), which adds RoCEv2 queue pairs, DCQCN, HPCC, TIMELY, DCTCP, PFC, ECN and a Broadcom shared-buffer switch model to ns-3.18. It is built inside a pinned Docker image so the whole environment reproduces with one command.

**Ring all-reduce modelling.** HPCC's flow file schedules flows at absolute times and has no barrier primitive, so a full 2(N-1)-step collective cannot be expressed directly. Rather than fake synchronisation, this project simulates one step exactly — every GPU sending its chunk to its ring successor simultaneously — and scales analytically. The measured quantity and the generated workload are therefore always the same thing.

---

## Reproducing

Requires Docker. On Apple Silicon, enable **Settings → General → Use Rosetta for x86/amd64 emulation**.

```bash
docker build -t rdma-sim .
```

```bash
docker run -it --rm -v "$PWD":/work rdma-sim bash
```

Then, inside the container:

```bash
python3 -m experiments.run --topo rail --n-servers 16 --gpus-per-server 8 --traffic ring-allreduce --tensor-size 16000000 --ring-mode parallel --sim-time 2.05 --no-packet-trace --name my-run
```

```bash
bash experiments/sweep.sh && python3 -m analysis.plots
```

Build takes ~4 minutes. A 128-GPU all-reduce step simulates in 12–30 s.

---

## Limitations

Stated plainly, because they matter for interpreting the numbers:

- **One step measured, collective extrapolated.** Real collectives have per-step barriers and jitter that this model does not capture.
- **No intra-node modelling.** NVLink and the intra-server reduce-scatter/all-gather phases are outside the network simulation entirely.
- **Switch radix differs between the fabrics.** A k=8 fat-tree uses 8-port switches; the rail fabric needs 32-port rail switches for the same 128 hosts. Host count and link rate are matched, port cost is not.
- **No real-hardware validation yet.** The next step is `nccl-tests` on a rented multi-GPU instance to check the predictions against NCCL.
- **One congestion-control algorithm in the headline results.** HPCC, TIMELY and DCTCP are available in the simulator and not yet swept.
- **ECMP behaviour is HPCC's implementation**, not any specific vendor's ASIC.

---

## Repository layout

```
topo/          topology generators + HPCC topology.txt writer
traffic/       traffic patterns + HPCC flow.txt writer
experiments/   orchestrator and sweep driver
analysis/      statistics and figures
plots/         committed figures used in this README
results/       simulation output (gitignored)
Dockerfile     ns-3.18 + HPCC RDMA fork, pinned
```

## License

MIT for project code. The vendored ns-3 fork retains its original GPLv2 license.
