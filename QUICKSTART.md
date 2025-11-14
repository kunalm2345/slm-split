# 🚀 Quick Start Guide - Split CPU/iGPU Inference

**Goal**: Get the split inference system running in under 30 minutes

---

## Prerequisites

- Linux system (Ubuntu 20.04+ or RHEL 8+)
- Intel CPU (ideally Core Ultra 9 185H with Arc iGPU)
- 20GB free disk space (for oneAPI toolkit)
- Python 3.10+
- Internet connection

---

## Step 1: Clone/Navigate to Repository

```bash
cd "/home/subroto/Studies/Sem_4-1/SysML/MoE_Split/next attempt/slm-aplit"
```

---

## Step 2: Run Automated Setup (10-15 minutes)

```bash
# Make script executable (if not already)
chmod +x setup_split_inference.sh

# Run setup
./setup_split_inference.sh
```

This will:
- Install system dependencies (CMake, ZeroMQ, etc.)
- Download Intel oneAPI (~10GB, takes 5-10 min)
- Create Python virtual environment
- Install PyTorch CPU, transformers, etc.
- Build C++ scheduler

**⚠️ Note**: If oneAPI installation fails, you can install manually:
```bash
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
```

---

## Step 3: Verify Installation

```bash
# Activate Python environment
source venv/bin/activate

# Run tests
python3 split_inference/tests/test_system.py
```

**Expected output**:
```
====================================================================
SPLIT INFERENCE SYSTEM - TEST SUITE
====================================================================

====================================================================
TEST 3: Configuration Loading
====================================================================
✓ Configuration loaded from split_inference/configs/partition_config.yaml
✓ Device for attention_qkv_proj (layer 5): igpu
✓ Device for router_logits (layer 5): cpu
✓ Pipelining enabled: True
✓ Micro-batch size: 4

====================================================================
TEST 1: Scheduler Connection
====================================================================
Attempting to connect to scheduler...
✗ Failed to connect (scheduler may not be running)

====================================================================
TEST 4: CPU Fallback
====================================================================
Initializing CPU inference engine...
✓ CPU engine initialized

====================================================================
TEST SUMMARY
====================================================================
✓ Config Loading: PASSED
⊘ Scheduler Connection: SKIPPED (not running)
⊘ Work Packet: SKIPPED
✓ Cpu Fallback: PASSED

2/2 tests passed

✓ All tests passed!
```

---

## Step 4: Analyze Model (Optional but Recommended)

```bash
# Activate environment (if not already)
source venv/bin/activate

# Run ONNX analysis
python3 export_to_onnx.py
```

This will:
- Load Phi-tiny-MoE model
- Analyze architecture (3.8B params, 16 experts, etc.)
- Attempt ONNX export
- Generate recommendations

**Output**: `onnx_analysis_report.json`

---

## Step 5: Start the Split Inference System

### Terminal 1: Start Scheduler

```bash
# Load oneAPI environment
source enable_oneapi.sh

# Start scheduler
./run_scheduler.sh
```

**Expected output**:
```
======================================================================
SPLIT CPU/iGPU INFERENCE SCHEDULER
======================================================================
🔧 Initializing device executor...
  • Found 2 SYCL platforms
    - Intel(R) Core(TM) ...
    - Intel(R) Arc(TM) Graphics
    ✓ Selected as iGPU
  ✓ iGPU initialized
  ✓ CPU executor ready
🔌 Starting ZeroMQ server on tcp://*:5555
✓ Scheduler ready - waiting for requests...
```

**⚠️ If iGPU not detected**:
- Scheduler will fall back to CPU-only mode
- Check with: `source /opt/intel/oneapi/setvars.sh && sycl-ls`
- Install GPU drivers if needed: `sudo apt install intel-opencl-icd`

### Terminal 2: Run Orchestrator

```bash
# Activate environment
source venv/bin/activate

# Run orchestrator
./run_orchestrator.sh
```

**OR** use CPU-only fallback:

```bash
python3 cpu_inference.py --prompt "What is the meaning of life?" --max-tokens 50
```

---

## Step 6: Verify End-to-End (with Scheduler Running)

In Terminal 2:

```bash
source venv/bin/activate
python3 split_inference/tests/test_system.py
```

Now you should see:
```
✓ Config Loading: PASSED
✓ Scheduler Connection: PASSED
✓ Work Packet: PASSED
✓ Cpu Fallback: PASSED

4/4 tests passed

✓ All tests passed!
```

---

## Common Issues & Fixes

### Issue 1: `oneAPI not found`

**Solution**:
```bash
# Manually source oneAPI
source /opt/intel/oneapi/setvars.sh

# Verify
which icpx  # Should show: /opt/intel/oneapi/compiler/.../icpx
```

### Issue 2: `ZMQ bind error: Address already in use`

**Solution**:
```bash
# Kill existing scheduler
pkill -f scheduler

# Restart
./run_scheduler.sh
```

### Issue 3: `ModuleNotFoundError: No module named 'zmq'`

**Solution**:
```bash
source venv/bin/activate
pip install pyzmq
```

### Issue 4: `iGPU not detected`

**Check drivers**:
```bash
# List SYCL devices
source /opt/intel/oneapi/setvars.sh
sycl-ls

# Expected output should include:
# [opencl:gpu:0] Intel(R) Arc(TM) Graphics
```

**Install drivers if missing**:
```bash
sudo apt install intel-opencl-icd intel-level-zero-gpu
```

### Issue 5: Build errors in C++ scheduler

**Rebuild with verbose output**:
```bash
cd split_inference/cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SYCL=OFF
make VERBOSE=1
```

---

## What's Next?

### For Development:

1. **Enable SYCL** (if you have Intel iGPU):
   ```bash
   cd split_inference/cpp/build
   source /opt/intel/oneapi/setvars.sh
   cmake .. -DENABLE_SYCL=ON -DENABLE_ONEDNN=ON
   make -j$(nproc)
   ```

2. **Implement end-to-end execution loop**:
   - Edit `split_inference/python/orchestrator.py`
   - Implement `_generate_split()` method
   - Break layers into work packets

3. **Integrate vendor kernels**:
   - Add oneDNN for CPU operations
   - Add oneMKL for iGPU GEMM
   - Test ONNX Runtime with OpenVINO EP

### For Testing:

```bash
# Run CPU-only inference (baseline)
python3 cpu_inference.py --benchmark

# Compare with split inference (when ready)
python3 split_inference/python/orchestrator.py --benchmark
```

---

## File Structure Overview

```
slm-aplit/
├── export_to_onnx.py              ← Run first: analyze model
├── setup_split_inference.sh        ← Run second: install everything
├── run_scheduler.sh                ← Terminal 1: start scheduler
├── run_orchestrator.sh             ← Terminal 2: run inference
├── cpu_inference.py                ← Fallback: CPU-only inference
├── split_inference/
│   ├── configs/
│   │   └── partition_config.yaml   ← Configure device mapping
│   ├── cpp/
│   │   ├── scheduler.cpp           ← C++ scheduler
│   │   └── CMakeLists.txt          ← Build system
│   ├── python/
│   │   └── orchestrator.py         ← Python orchestrator
│   ├── sycl_kernels/
│   │   └── moe_routing.hpp         ← Custom SYCL kernels
│   └── tests/
│       └── test_system.py          ← Run tests
└── SPLIT_INFERENCE_README.md       ← Full documentation
```

---

## Performance Expectations

**Current State** (foundation complete):
- ✅ IPC communication working
- ✅ Config-driven device partitioning
- ✅ SYCL kernels implemented (not tested on hardware)
- ⏳ End-to-end execution loop in progress

**Target Performance** (once complete):
- Latency: 10-20 tokens/s (batch=1, int8/fp16)
- Throughput: 50+ tokens/s (batch=4-8)
- Memory: < 8GB (model + activations)
- Bandwidth: < 40 GB/s DRAM (with pipelining)

---

## Getting Help

1. **Check documentation**:
   - `SPLIT_INFERENCE_README.md` - Complete guide
   - `IMPLEMENTATION_STATUS.md` - Current status
   
2. **Run diagnostics**:
   ```bash
   python3 split_inference/tests/test_system.py
   ```

3. **Check logs**:
   - Scheduler: stdout of `./run_scheduler.sh`
   - Orchestrator: stdout of `./run_orchestrator.sh`
   - Telemetry: `./telemetry/*.json` (if enabled)

---

## Success Criteria ✅

You have successfully set up the system if:

- [x] All scripts are executable
- [x] Python venv is activated
- [x] oneAPI is installed (optional for CPU-only)
- [x] Scheduler builds and runs
- [x] Tests pass (at least config + fallback)
- [x] ONNX analysis generates report

**Next**: Start implementing the full execution loop! See `IMPLEMENTATION_STATUS.md` for roadmap.

---

**Estimated Total Time**: 15-30 minutes (depending on download speeds)

**Have fun! 🚀**
