# Telemetry Logging Guide 📊

## Overview

Telemetry logging is now **fully implemented** and records detailed metrics for every operation in the split CPU/iGPU inference pipeline.

---

## Output Files

Every inference session generates **2 files** in `./telemetry/`:

1. **CSV Log** (`session_YYYY-MM-DD_HH-MM-SS.csv`)
   - Detailed per-operation logs
   - One row per work packet
   - Real-time streaming (appends as operations complete)

2. **JSON Summary** (`session_YYYY-MM-DD_HH-MM-SS.json`)
   - Aggregate statistics
   - Per-operation breakdown
   - Expert selection distribution
   - Written at session end

---

## CSV Format

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Unix timestamp | 1763317513.759285 |
| `device` | Execution device | cpu, igpu |
| `worker_instance` | Pipeline stage | cpu_router, igpu_expert_0_7 |
| `duration_ms` | Execution time (ms) | 5.20 |
| `packet_id` | Work packet ID | 42 |
| `layer_idx` | Transformer layer | 0-31, -1 (embedding) |
| `operation` | Operation type | attention_qkv_proj, expert_ffn |
| `success` | Success flag | True, False |
| `memory_used_mb` | Memory consumed (MB) | 128.50 |
| `error_message` | Error details | "iGPU out of memory" |

### Worker Instance Names

- `cpu_embedding` - Token embedding lookup
- `cpu_router` - MoE routing logic
- `igpu_attention_layer_N` - Attention at layer N
- `igpu_expert_ffn_layer_N` - Expert FFN at layer N
- `cpu_layernorm_layer_N` - LayerNorm at layer N
- `cpu_lm_head` - Final vocabulary projection

---

## JSON Summary

```json
{
  "session_name": "session_2024-11-16_14-30-00",
  "total_time_s": 12.345,
  "tokens_generated": 50,
  "tokens_per_second": 4.05,
  
  "total_packets_processed": 1650,
  "cpu_packets": 825,
  "igpu_packets": 800,
  "failed_packets": 25,
  "success_rate": 0.985,
  
  "avg_cpu_time_ms": 0.8,
  "avg_igpu_time_ms": 3.2,
  "cpu_utilization": 0.50,
  "igpu_utilization": 0.48,
  
  "operation_breakdown": {
    "attention_qkv_proj": {
      "count": 160,
      "avg_time_ms": 2.1,
      "cpu_count": 0,
      "igpu_count": 160
    }
  },
  
  "expert_selection_distribution": {
    "expert_3": 245,
    "expert_7": 189
  }
}
```

---

## Usage Examples

### 1. Run Inference with Telemetry

```bash
# Terminal 1: Start scheduler
./run_scheduler.sh

# Terminal 2: Run orchestrator (telemetry auto-enabled)
./run_orchestrator.sh

# Check output
ls -la telemetry/
```

### 2. Analyze Latest Session

```bash
python3 analyze_telemetry.py

# Or analyze specific file
python3 analyze_telemetry.py telemetry/session_2024-11-16_14-30-00.csv
```

### 3. Query CSV with Standard Tools

```bash
# Count operations per device
awk -F',' 'NR>1 {print $2}' telemetry/session_*.csv | sort | uniq -c

# Find slowest operations
awk -F',' 'NR>1 {print $4, $7}' telemetry/session_*.csv | sort -rn | head -n 10

# Get average time per layer
awk -F',' 'NR>1 {sum[$6]+=$4; count[$6]++} END {for(l in sum) print l, sum[l]/count[l]}' telemetry/session_*.csv | sort -n
```

### 4. Import to Pandas

```python
import pandas as pd

df = pd.read_csv('telemetry/session_2024-11-16_14-30-00.csv')

# Average time per device
df.groupby('device')['duration_ms'].mean()

# Operations per layer
df.groupby('layer_idx')['operation'].count()

# Plot timeline
import matplotlib.pyplot as plt
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.plot(x='timestamp', y='duration_ms', kind='scatter', c='device', colormap='viridis')
plt.show()
```

---

## Customization

### Change Output Directory

```python
orchestrator = SplitInferenceOrchestrator(
    model_path=".",
    telemetry_dir="/tmp/my_telemetry"  # Custom path
)
```

### Custom Session Name

```python
from telemetry_logger import TelemetryLogger

logger = TelemetryLogger(
    output_dir="./telemetry",
    session_name="benchmark_run_1"  # Custom name
)
```

---

## Troubleshooting

### No telemetry files created

**Check:** Directory permissions
```bash
mkdir -p telemetry
chmod 755 telemetry
```

### Empty CSV file

**Cause:** No work packets sent to scheduler
**Fix:** Check scheduler connection with `orchestrator.initialize()`

### Missing operation entries

**Cause:** Operation failed before reaching scheduler
**Fix:** Check orchestrator logs for exceptions

---

## Performance Impact

- **CSV writes:** ~0.1-0.5 ms per entry (negligible)
- **Memory:** ~50 KB per 1000 entries
- **Disk:** ~1 MB per 10,000 operations

**Recommendation:** Keep telemetry enabled for production - overhead is minimal and diagnostics are invaluable!

---

## Quick Commands

```bash
# View latest CSV
cat telemetry/session_*.csv | tail -n 20

# View JSON summary
cat telemetry/session_*.json | jq '.operation_breakdown'

# Count total operations
wc -l telemetry/session_*.csv

# Check success rate
awk -F',' 'NR>1 {if($8=="True") s++; t++} END {print s/t*100"%"}' telemetry/session_*.csv

# Find errors
grep "False" telemetry/session_*.csv
```

---

## Files

- `split_inference/python/telemetry_logger.py` - Logger implementation
- `test_telemetry.py` - Test script
- `analyze_telemetry.py` - Analysis tool
- `telemetry/` - Output directory (auto-created)

---

**Happy debugging+x ~/Studies/Sem_4-1/SysML/MoE_Split/next\ attempt/slm-aplit/analyze_telemetry.py ~/Studies/Sem_4-1/SysML/MoE_Split/next\ attempt/slm-aplit/test_telemetry.py* 🎯
