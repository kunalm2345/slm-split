#!/usr/bin/env python3
"""
Test telemetry logger functionality
"""

import sys
sys.path.insert(0, 'split_inference/python')

from telemetry_logger import TelemetryLogger, TelemetryEntry
import time

print("="*60)
print("TESTING TELEMETRY LOGGER")
print("="*60)

# Initialize logger
logger = TelemetryLogger(output_dir="./telemetry", session_name="test_session")

# Simulate some work packets
print("\n📝 Logging test entries...")

# Entry 1: Embedding on CPU
logger.log_work_result(
    packet_id=1,
    layer_idx=-1,
    operation="embedding",
    device_target="cpu",
    result_device="cpu",
    duration_ms=0.5,
    success=True,
    memory_used_mb=10.5
)

# Entry 2: Attention on iGPU
logger.log_work_result(
    packet_id=2,
    layer_idx=0,
    operation="attention_qkv_proj",
    device_target="igpu",
    result_device="igpu",
    duration_ms=2.3,
    success=True,
    memory_used_mb=45.2
)

# Entry 3: Router on CPU
logger.log_work_result(
    packet_id=3,
    layer_idx=0,
    operation="router_logits",
    device_target="cpu",
    result_device="cpu",
    duration_ms=0.8,
    success=True,
    memory_used_mb=5.1
)

# Entry 4: Expert FFN on iGPU
logger.log_work_result(
    packet_id=4,
    layer_idx=0,
    operation="expert_ffn",
    device_target="igpu",
    result_device="igpu",
    duration_ms=5.2,
    success=True,
    memory_used_mb=128.5
)

# Entry 5: Failed operation
logger.log_work_result(
    packet_id=5,
    layer_idx=1,
    operation="attention_qkv_proj",
    device_target="igpu",
    result_device="cpu",  # Fell back to CPU
    duration_ms=3.1,
    success=False,
    memory_used_mb=0.0,
    error_message="iGPU out of memory"
)

print("✓ Logged 5 test entries")

# Check stats
stats = logger.get_summary_stats()
print(f"\n📊 Current stats:")
print(f"   Total packets: {stats['total_packets']}")
print(f"   CPU packets: {stats['cpu_packets']}")
print(f"   iGPU packets: {stats['igpu_packets']}")
print(f"   Failed packets: {stats['failed_packets']}")

# Finalize
print("\n💾 Finalizing telemetry...")
logger.finalize(tokens_generated=5, prompt_tokens=3)

print(f"\n✅ Test complete! Check:")
print(f"   {logger.csv_path}")
print(f"   {logger.json_path}")
