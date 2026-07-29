# Design notes

Reasoning behind the choices in this project, written for my own reference and
for anyone reading the code.

## Why a simulator instead of real hardware

The question I wanted to answer — how does fabric topology affect a 128-GPU
all-reduce — needs 128 GPUs and a reconfigurable network. Renting that is
expensive, and rewiring it between experiments is impossible. Simulation lets
me change the topology with a command-line flag and get packet-level detail
that real hardware would not expose without specialised switch telemetry.

The tradeoff is obvious: a simulator only tells you about the model, not the
world. That is why the results are checked against hand arithmetic, and why
real-hardware validation is listed as the top item of future work.

## Why ns-3 with the HPCC fork, and not Mininet or Containerlab

Mininet and Containerlab emulate networks using real Linux kernels and virtual
links. They are excellent for testing routing configurations and control-plane
software. They cannot model RDMA: there is no queue-pair state machine, no
DCQCN rate control, no PFC pause frames, no switch buffer model. Running TCP
through them and calling it RDMA would be misleading.

The [Alibaba HPCC fork of ns-3](https://github.com/alibaba-edu/High-Precision-Congestion-Control)
was published alongside the HPCC paper (SIGCOMM 2019) and implements RoCEv2
queue pairs, DCQCN, HPCC, TIMELY, DCTCP, PFC, ECN, and a Broadcom shared-buffer
switch. It is the standard artifact for this kind of study, which also means
results are comparable with published work.

The cost is that it is a 2018-era research codebase: Python 2 build scripts,
ns-3.18, and no ARM support. Hence the Docker image, which pins Ubuntu 20.04,
installs Python 2 alongside Python 3, and runs as linux/amd64 under Rosetta on
Apple Silicon. That container is the reason the whole thing reproduces in one
command.

## What "one step" of all-reduce means, and why I did not simulate the whole thing

A ring all-reduce over N GPUs runs in 2(N-1) steps. In each step every GPU
sends one chunk of the tensor to its neighbour in the ring, and the next step
cannot begin until the current one finishes everywhere.

HPCC's input format schedules flows at fixed absolute times. There is no
barrier primitive, so "start step 2 when step 1 completes" cannot be expressed.
I had two options: guess the inter-step delays and schedule them, or simulate
one step exactly and scale the result.

I chose the second. Guessing the delays would mean the simulation no longer
measures what it generates — if the guess is wrong, later steps either overlap
or leave the network idle, and the resulting number means nothing. Simulating
one step is honest: the workload and the measurement are the same object. The
full collective is then estimated as `steps x slowest step`, which is stated
as an estimate everywhere it appears.

The weakness of this model is that it assumes every step behaves like the one
measured. That is reasonable here because ECMP hashing is deterministic on the
flow 5-tuple, and the same source-destination pairs repeat every step, so the
same flows collide every time. It would be a bad assumption if paths were
re-randomised per step.

## Why rail-optimized has zero variance

This is the result I care most about, so it is worth stating carefully.

In the inter-server phase of an all-reduce, NCCL only ever has same-rank GPUs
talk to each other. On a rail-optimized fabric, all GPUs of a given rank hang
off the same rail switch. So the path from any GPU to its ring neighbour is:
host to rail switch, rail switch to host. Two links.

There is exactly one such path. No routing decision exists, so ECMP has nothing
to hash and no two flows can be assigned to the same link by accident. Each
host's uplink carries exactly one outgoing flow and its downlink exactly one
incoming flow. The fabric is, for this workload, perfectly provisioned.

That is why all 128 flows completed in the same 91.0 microseconds. It is not a
statistical result; it is structural.

## Why the fat-tree developed a tail

A fat-tree gives every pair of hosts many equal-cost paths, and switches pick
between them by hashing the packet's 5-tuple — ECMP. Hashing distributes
*many small flows* well. AI training produces a *small number of very large
flows*, and with few flows a hash distributes badly: some links get two or
three, others get none. A link carrying three flows runs each of them at a
third of line rate.

In the measured run, 64 of the 128 flows crossed pods and had to traverse the
core. The worst-affected flow took 959.9 microseconds against a 92 microsecond
floor. Since a collective finishes only when its slowest member finishes, that
single flow sets the pace for all 128 GPUs.

## Why the comparison is fair, and where it is not

Matched: host count (128), link rate (100 Gbps), per-link delay, congestion
control (DCQCN), and the logical collective — both fabrics run the identical
flow file, 8 rings of 16 GPUs.

Not matched: switch radix. A k=8 fat-tree is built from 8-port switches; the
rail fabric needs 16 host ports plus 16 uplinks on each rail switch. Same host
count, different hardware cost. A purchasing decision would have to account for
that; this project does not.

Also not modelled: NVLink and the intra-server phases of the collective. Those
never touch the network, so excluding them is correct for a network study, but
it means these numbers are not end-to-end training times.

## Why PFC never fired

PFC is the emergency brake: a switch about to overflow its buffer sends a PAUSE
frame upstream. It is hop-by-hop and blunt, and PFC storms are the classic
failure mode of production RoCE deployments.

It never triggered in any run here, because DCQCN did its job. DCQCN reacts to
ECN marks — switches flag packets when queues start building, receivers echo
that back, and senders reduce their rate before the buffer is ever in danger.
A well-tuned fabric should look exactly like this. Deliberately breaking the
tuning to produce PFC storms is on the future-work list.

## What I would do next, in order

1. **Validate against real hardware.** Rent a multi-GPU instance, run
   `nccl-tests`, and compare measured all-reduce bandwidth against the
   simulator's prediction for the same GPU count and message size.
2. **Sweep congestion control.** HPCC, TIMELY and DCTCP are already compiled
   in. The interesting metric is how much each one gives up against the
   no-contention arithmetic.
3. **Break PFC on purpose.** Shrink switch buffers or disable ECN marking and
   show the pause-storm behaviour that makes RoCE deployments difficult.
4. **Add all-to-all traffic** for mixture-of-experts style workloads, which
   stress the fabric differently from a ring.
