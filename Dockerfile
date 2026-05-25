# RDMA/RoCE GPU Cluster Network Fabric Simulator
#
# Base: Ubuntu 20.04 (gcc-9, python3.8) — most compatible with academic ns-3.30 forks.
# Platform: linux/amd64 — forces Rosetta on Apple Silicon. Safer than native ARM64
# because the upstream fork was developed/tested on x86 only.

FROM --platform=linux/amd64 ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies for ns-3 build + Python analysis stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc g++ \
        python2 python2-dev \
        python3 python3-dev python3-pip python3-setuptools \
        git mercurial wget curl \
        pkg-config \
        libxml2 libxml2-dev \
        libgsl-dev \
        libsqlite3-dev \
        gnuplot \
        gdb \
        ca-certificates \
        tzdata \
        vim less procps \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point to python2 — the HPCC fork's waf scripts were written
# for Python 2 (e.g., `print foo` syntax). Our own analysis code uses python3 explicitly.
RUN ln -sf /usr/bin/python2 /usr/bin/python

# Python analysis libs available globally so VS Code / Jupyter pick them up.
RUN pip3 install --no-cache-dir \
        numpy pandas matplotlib seaborn networkx pyyaml jupyter ipython

# Clone the Alibaba HPCC fork — modern ns-3.30 with RoCEv2, DCQCN, PFC, HPCC, TIMELY.
# Pinned to main; we'll lock to a specific commit once we confirm the build is green.
WORKDIR /opt
RUN git clone --depth=1 https://github.com/alibaba-edu/High-Precision-Congestion-Control.git hpcc

# Configure and build ns-3.
# --build-profile=optimized turns on -O3 for faster simulations.
# CXXFLAGS=-Wno-error keeps warnings from being treated as errors (older code).
WORKDIR /opt/hpcc/simulation
RUN CXXFLAGS="-Wno-error" CFLAGS="-Wno-error" \
        ./waf configure --build-profile=optimized
RUN ./waf

# Sanity check — list the built RDMA examples so the build log shows them.
RUN ls -la build/scratch/ 2>/dev/null || true

# /work is bind-mounted from the host (this project folder).
WORKDIR /work
ENV NS3_HOME=/opt/hpcc/simulation
ENV PATH="/opt/hpcc/simulation:${PATH}"

CMD ["/bin/bash"]
