#!/usr/bin/env python3
"""
Simple test script for split inference system
Tests IPC communication and basic functionality
"""

import sys
import time
import subprocess
import signal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from split_inference.python.orchestrator import (
    SchedulerClient, PartitionConfig, WorkPacket
)

def test_scheduler_connection():
    """Test connection to scheduler"""
    print("=" * 60)
    print("TEST 1: Scheduler Connection")
    print("=" * 60)
    
    client = SchedulerClient("tcp://localhost:5555")
    
    # Try to connect (with short timeout)
    print("Attempting to connect to scheduler...")
    connected = client.connect(timeout_ms=2000)
    
    if connected:
        print("✓ Successfully connected to scheduler")
        
        # Get telemetry
        telemetry = client.get_telemetry()
        if telemetry:
            print(f"✓ Telemetry received:")
            for key, value in telemetry.items():
                print(f"  • {key}: {value}")
        
        client.shutdown()
        return True
    else:
        print("✗ Failed to connect (scheduler may not be running)")
        print("\nTo start scheduler, run in another terminal:")
        print("  ./run_scheduler.sh")
        return False


def test_work_packet():
    """Test sending work packet to scheduler"""
    print("\n" + "=" * 60)
    print("TEST 2: Work Packet Execution")
    print("=" * 60)
    
    client = SchedulerClient("tcp://localhost:5555")
    
    if not client.connect(timeout_ms=2000):
        print("✗ Scheduler not available")
        return False
    
    # Create test work packet
    packet = WorkPacket(
        packet_id=1,
        layer_idx=0,
        operation="test_operation",
        device_target="cpu",
        input_shape=[1, 10],
        input_dtype="float32",
        params={},
        priority=0,
        can_pipeline=True,
        memory_requirement_mb=1.0,
        estimated_duration_ms=1.0
    )
    
    print(f"Sending work packet: {packet.operation} on {packet.device_target}")
    
    result = client.send_work_packet(packet)
    
    if result and result.success:
        print(f"✓ Work packet executed successfully")
        print(f"  • Device used: {result.device_used}")
        print(f"  • Duration: {result.actual_duration_ms:.2f} ms")
        print(f"  • Memory used: {result.memory_used_mb:.2f} MB")
        client.shutdown()
        return True
    else:
        print(f"✗ Work packet failed")
        if result:
            print(f"  Error: {result.error_message}")
        client.shutdown()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("TEST 3: Configuration Loading")
    print("=" * 60)
    
    config_path = "split_inference/configs/partition_config.yaml"
    
    try:
        config = PartitionConfig(config_path)
        print(f"✓ Configuration loaded from {config_path}")
        
        # Test device selection
        device = config.get_device_for_operation(5, "attention_qkv_proj")
        print(f"✓ Device for attention_qkv_proj (layer 5): {device}")
        
        device = config.get_device_for_operation(5, "router_logits")
        print(f"✓ Device for router_logits (layer 5): {device}")
        
        # Test settings
        pipelining = config.should_enable_pipelining()
        print(f"✓ Pipelining enabled: {pipelining}")
        
        batch_size = config.get_micro_batch_size()
        print(f"✓ Micro-batch size: {batch_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def test_cpu_fallback():
    """Test CPU fallback functionality"""
    print("\n" + "=" * 60)
    print("TEST 4: CPU Fallback")
    print("=" * 60)
    
    try:
        from cpu_inference import CPUInferenceEngine
        import torch
        
        print("Initializing CPU inference engine...")
        engine = CPUInferenceEngine(
            model_path=".",
            device="cpu",
            dtype=torch.float32
        )
        
        print("✓ CPU engine initialized")
        
        # Test generation (short prompt)
        prompt = "Hello, world!"
        print(f"\nGenerating from prompt: '{prompt}'")
        
        text, metrics = engine.generate(
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.7,
            benchmark=True
        )
        
        print(f"✓ Generation completed")
        print(f"  • Output: {text[:100]}...")
        print(f"  • Tokens/s: {metrics.tokens_per_second:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ CPU fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SPLIT INFERENCE SYSTEM - TEST SUITE")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test 1: Config loading (doesn't need scheduler)
    results['config'] = test_config_loading()
    
    # Test 2: Scheduler connection
    results['connection'] = test_scheduler_connection()
    
    # Test 3: Work packet (only if connected)
    if results['connection']:
        results['work_packet'] = test_work_packet()
    else:
        results['work_packet'] = None
    
    # Test 4: CPU fallback
    results['cpu_fallback'] = test_cpu_fallback()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
            symbol = "⊘"
        elif result:
            status = "PASSED"
            symbol = "✓"
        else:
            status = "FAILED"
            symbol = "✗"
        
        print(f"{symbol} {test_name.replace('_', ' ').title()}: {status}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
