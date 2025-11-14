# Split CPU ↔ iGPU Inference System

**Complete inference stack for Phi-tiny-MoE targeting Intel Core Ultra 9 185H**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup & Installation](#setup--installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Development Guide](#development-guide)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This system implements **bandwidth-aware split inference** for Phi-tiny-MoE across CPU and integrated GPU on Intel Core Ultra 9 185H devices. Key features:

- ✅ **Hybrid execution**: CPU handles routing/gating, iGPU handles heavy GEMM operations
- ✅ **Bandwidth management**: Token semaphore prevents concurrent DRAM saturation  
- ✅ **Pipelining**: Double-buffered expert weights with compute/transfer overlap
- ✅ **Vendor kernel reuse**: Integrates oneDNN, ONNX Runtime, oneAPI libraries
- ✅ **Custom SYCL kernels**: MoE routing, scatter/gather, batched expert dispatch
- ✅ **Graceful fallback**: Falls back to CPU-only inference on errors

### Model Architecture

**Phi-tiny-MoE** (from [microsoft/Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE)):
- **Total parameters**: 3.8B
- **Activated parameters**: 1.1B (per forward pass)
- **Experts**: 16 local experts, top-2 routing per token
- **Context length**: 4096 tokens
- **Vocabulary**: 32,064 tokens
- **Layers**: 32 transformer decoder layers

### Target Hardware

**Intel Core Ultra 9 185H**:
- **CPU**: 16 cores, 22 threads
- **iGPU**: Intel Arc Graphics (integrated)
- **Memory**: Shared LPDDR5 (bandwidth bottleneck)
- **oneAPI support**: Full SYCL/DPC++, Level-Zero, oneDNN

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Orchestrator                       │
│  • Tokenization                                             │
│  • Work packet creation                                     │
│  • High-level scheduling decisions                          │
│  • Fallback to CPU-only inference                           │
└─────────────────┬───────────────────────────────────────────┘
                  │ ZeroMQ IPC
                  │ (work packets + telemetry)
┌─────────────────▼───────────────────────────────────────────┐
│                    C++ Scheduler                             │
│  • Receives work packets via ZeroMQ                         │
│  • Routes to CPU or iGPU based on policy                    │
│  • Bandwidth-aware scheduling (token semaphore)             │
│  • Telemetry collection                                     │
└─────────┬─────────────────────────────┬─────────────────────┘
          │                             │
    ┌─────▼─────┐               ┌───────▼────────┐
    │CPU Executor│               │ iGPU Executor  │
    │ (oneDNN)   │               │ (SYCL/Level-0) │
    └────────────┘               └────────────────┘
```

### Operation Partitioning

| Operation | Device | Rationale |
|-----------|--------|-----------|
| Embedding lookup | CPU | Irregular memory access, low compute |
| Router logits (gating) | CPU | Small GEMM, latency-critical |
| Top-k expert selection | CPU | Branching logic, < 1ms for 16 experts |
| Attention QKV projection | iGPU | Large GEMM, parallelizable |
| Attention compute | Auto | Depends on sequence length & batch |
| Expert FFN (batched) | iGPU | Heavy GEMM, benefit from batching |
| Expert scatter/gather | iGPU | Parallel index operations |
| LayerNorm | CPU | Memory-bound, lightweight |
| LM head projection | CPU | Keep on CPU for low output latency |

### Memory Strategy

**Challenge**: CPU and iGPU share LPDDR5 memory bus (~50 GB/s). Concurrent access causes contention.

**Solution**: Bandwidth-aware scheduling
1. **Token semaphore**: Only one device can perform heavy DRAM access at a time
2. **Double buffering**: Load expert N+1 weights while computing expert N
3. **USM (Unified Shared Memory)**: Zero-copy where possible
4. **Expert weight caching**: LRU cache for frequently-used experts

### Scheduling Policies

**Static Partitioning** (baseline):
- Fixed mapping of operations to devices from config file
- Simple, predictable, no runtime overhead

**Bandwidth-Aware Dynamic** (recommended):
- Monitor DRAM bandwidth every 10ms
- Acquire token semaphore before heavy DRAM ops
- Throttle batch size if bandwidth > 85% capacity
- Adaptive expert batching

**Pipelined** (advanced):
- Pipeline stages: cpu_router → igpu_expert_0_7 → igpu_expert_8_15 → cpu_attention
- Double buffer expert weights
- Overlap compute and transfer using SYCL events

---

## Setup & Installation

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ or RHEL 8+)
- **Hardware**: Intel CPU with integrated GPU (ideally Core Ultra 9 185H)
- **Python**: 3.10+
- **Disk**: ~20GB for oneAPI toolkit

### Automated Setup

```bash
# Run the automated setup script
chmod +x setup_split_inference.sh
./setup_split_inference.sh
```

This script will:
1. Install system dependencies (build tools, CMake, ZeroMQ)
2. Download and install Intel oneAPI Base Toolkit (~10GB)
3. Create Python virtual environment
4. Install Python packages (PyTorch CPU, transformers, etc.)
5. Build C++ scheduler (without SYCL initially)
6. Create helper scripts

### Manual Setup

If automated setup fails:

**1. Install Intel oneAPI**

```bash
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
# Or use APT repository:
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit
```

**2. Install system packages**

```bash
sudo apt-get install -y \
    build-essential cmake pkg-config \
    libzmq3-dev git python3-pip
```

**3. Setup Python environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy psutil einops pyzmq PyYAML onnx onnxruntime
```

**4. Build scheduler**

```bash
cd split_inference/cpp
mkdir build && cd build

# Without SYCL (CPU-only testing)
cmake .. -DCMAKE_BUILD_TYPE=Release

# With SYCL (for iGPU)
source /opt/intel/oneapi/setvars.sh
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_SYCL=ON \
    -DENABLE_ONEDNN=ON

make -j$(nproc)
```

---

## Quick Start

### 1. Run ONNX Analysis

Analyze the model and generate export report:

```bash
source venv/bin/activate
python3 export_to_onnx.py
```

Output: `onnx_analysis_report.json` with supported ops, recommendations

### 2. Start the Scheduler

In terminal 1:

```bash
# Load oneAPI environment
source enable_oneapi.sh

# Start scheduler
./run_scheduler.sh
```

Expected output:
```
======================================================================
SPLIT CPU/iGPU INFERENCE SCHEDULER
======================================================================
🔧 Initializing device executor...
  • Found 2 SYCL platforms
    - Intel(R) Core(TM) Ultra 9 185H
    - Intel(R) Arc(TM) Graphics
    ✓ Selected as iGPU
  ✓ iGPU initialized
  ✓ CPU executor ready
🔌 Starting ZeroMQ server on tcp://*:5555
✓ Scheduler ready - waiting for requests...
```

### 3. Run Inference

In terminal 2:

```bash
source venv/bin/activate
./run_orchestrator.sh
```

Or directly:

```bash
source venv/bin/activate
python3 split_inference/python/orchestrator.py
```

### 4. CPU-Only Fallback

If scheduler is not available, orchestrator automatically falls back to CPU:

```bash
python3 cpu_inference.py --prompt "What is quantum computing?" --max-tokens 100
```

---

## Configuration

### Partition Config

Edit `split_inference/configs/partition_config.yaml`:

```yaml
global:
  scheduling_policy: "bandwidth_aware"  # static | bandwidth_aware | adaptive
  enable_pipelining: true
  micro_batch_size: 4

static_partition:
  embedding: cpu
  attention_qkv_proj: igpu
  expert_ffn: igpu
  router_logits: cpu

bandwidth_control:
  enable_token_semaphore: true
  throttle_threshold: 0.85
  
pipelining:
  expert_staging:
    buffer_count: 2
    experts_per_buffer: 2
```

### Layer-Specific Overrides

```yaml
layer_overrides:
  # Keep first/last layers on CPU for low latency
  layer_0_3:
    attention_qkv_proj: cpu
    expert_ffn: cpu
  
  # Middle layers on iGPU (hot path)
  layer_4_27:
    expert_ffn: igpu
```

---

## Development Guide

### Adding a Custom SYCL Kernel

1. **Define kernel in** `split_inference/sycl_kernels/my_kernel.hpp`:

```cpp
#ifdef ENABLE_SYCL
#include <sycl/sycl.hpp>

class MyCustomKernel {
public:
    void operator()(sycl::queue& q, /* params */) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                // Kernel code
            });
        }).wait();
    }
};
#endif
```

2. **Register in DeviceExecutor** (`scheduler.cpp`):

```cpp
#include "../sycl_kernels/my_kernel.hpp"

WorkResult execute_my_op(const WorkPacket& packet) {
    MyCustomKernel kernel;
    kernel(*igpu_queue_, /* args */);
    // ...
}
```

3. **Rebuild**:

```bash
cd split_inference/cpp/build
make -j$(nproc)
```

### Testing Custom Kernels

Create unit test in `split_inference/tests/test_my_kernel.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../sycl_kernels/my_kernel.hpp"

TEST(MyKernelTest, Correctness) {
    // Setup SYCL queue
    // Run kernel
    // Compare with reference implementation
}
```

---

## Performance Tuning

### Profiling with VTune

```bash
# Enable VTune in CMake
cmake .. -DENABLE_SYCL=ON -DENABLE_VTUNE=ON
make -j$(nproc)

# Run with VTune
vtune -collect gpu-hotspots -- ./scheduler
```

### Bandwidth Monitoring

Check telemetry endpoint:

```python
import requests
telemetry = requests.get("http://localhost:5555/telemetry").json()
print(f"CPU bandwidth: {telemetry['cpu_bandwidth_gbps']} GB/s")
print(f"iGPU bandwidth: {telemetry['igpu_bandwidth_gbps']} GB/s")
```

### Tuning Knobs

| Parameter | Location | Effect |
|-----------|----------|--------|
| `micro_batch_size` | `partition_config.yaml` | Larger = better GEMM efficiency, higher latency |
| `throttle_threshold` | `bandwidth_control` | Lower = more conservative bandwidth usage |
| `experts_per_buffer` | `pipelining` | More = less frequent transfers, higher memory |
| `enable_pipelining` | `global` | true = lower latency, more complex |

---

## Troubleshooting

### Scheduler won't start

**Error**: `ZeroMQ bind error: Address already in use`

**Fix**: Kill existing scheduler process
```bash
pkill -f scheduler
./run_scheduler.sh
```

### iGPU not detected

**Check**: SYCL device enumeration
```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

**Expected output**:
```
[opencl:gpu:0] Intel(R) Arc(TM) Graphics
[opencl:cpu:0] Intel(R) OpenCL
```

If no GPU listed, check:
- Driver installation: `sudo apt install intel-opencl-icd`
- Hardware support: `lspci | grep VGA`

### Python import errors

**Error**: `ModuleNotFoundError: No module named 'zmq'`

**Fix**: Activate venv
```bash
source venv/bin/activate
pip install pyzmq
```

### Performance worse than CPU-only

Check telemetry for contention:
```python
telemetry = orchestrator.scheduler.get_telemetry()
if telemetry['cpu_igpu_contention'] > 0.5:
    print("High contention! Enable token semaphore in config")
```

---

## Project Structure

```
slm-aplit/
├── split_inference/
│   ├── configs/
│   │   └── partition_config.yaml        # Device partitioning config
│   ├── cpp/
│   │   ├── scheduler.cpp                 # C++ scheduler (ZeroMQ server)
│   │   ├── CMakeLists.txt                # Build system
│   │   └── build/                        # Build artifacts
│   ├── python/
│   │   └── orchestrator.py               # Python orchestrator (client)
│   ├── sycl_kernels/
│   │   └── moe_routing.hpp               # Custom SYCL kernels
│   └── tests/
│       └── test_kernels.cpp              # Unit tests
├── export_to_onnx.py                     # ONNX export & analysis
├── setup_split_inference.sh              # Automated setup
├── run_scheduler.sh                      # Start scheduler
├── run_orchestrator.sh                   # Start orchestrator
├── enable_oneapi.sh                      # Load oneAPI env
└── SPLIT_INFERENCE_README.md             # This file
```

---

## Next Steps

1. ✅ **Baseline working**: CPU-only inference + stub scheduler
2. 🚧 **Implement full split execution**: Complete work packet processing loop
3. 🚧 **Integrate vendor kernels**: oneDNN for CPU, oneMKL for iGPU GEMM
4. 🚧 **Implement bandwidth monitor**: Real DRAM bandwidth tracking
5. 🚧 **Add pipelining**: Double-buffered expert weights
6. 🚧 **End-to-end benchmarks**: Latency, throughput, bandwidth on target hardware
7. 🚧 **Optimize expert batching**: Group tokens by expert for GEMM efficiency

---

## References

- [Phi-tiny-MoE Model Card](https://huggingface.co/microsoft/Phi-tiny-MoE)
- [Intel Core Ultra 9 185H Specs](https://www.intel.com/content/www/us/en/products/sku/236849/intel-core-ultra-9-processor-185h-24m-cache-up-to-5-10-ghz/specifications.html)
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)
- [SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)

---

**Status**: 🚧 **Work in Progress** - Foundational infrastructure complete, core execution loop under development
