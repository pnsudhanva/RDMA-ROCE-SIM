#!/usr/bin/env bash
# Message-size sweep: same ring all-reduce on both fabrics, varying chunk size.
#
# Shows the crossover between the two regimes:
#   - small chunks  -> latency-bound, FCT set by hop count, fabrics look alike
#   - large chunks  -> bandwidth-bound, ECMP collisions punish the fat-tree
#
# Run from inside the container:
#     bash experiments/sweep.sh
set -euo pipefail

# Chunk sizes in bytes. Ring size is 16 (8 parallel rings over 128 GPUs),
# so tensor = chunk * 16.
CHUNKS=(1000 10000 100000 1000000 4000000)

for chunk in "${CHUNKS[@]}"; do
    tensor=$(( chunk * 16 ))

    python3 -m experiments.run \
        --topo rail --n-servers 16 --gpus-per-server 8 \
        --traffic ring-allreduce --tensor-size "$tensor" --ring-mode parallel \
        --sim-time 2.05 --no-packet-trace \
        --name "sweep-rail-c${chunk}" 2>&1 | grep -E '^\[run\] simulator'

    python3 -m experiments.run \
        --topo fat-tree --k 8 --gpus-per-server 8 \
        --traffic ring-allreduce --tensor-size "$tensor" --ring-mode parallel \
        --sim-time 2.05 --no-packet-trace \
        --name "sweep-fat-c${chunk}" 2>&1 | grep -E '^\[run\] simulator'

    echo "done chunk=${chunk}B"
done

echo "sweep complete"
