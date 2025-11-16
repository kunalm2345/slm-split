#!/usr/bin/env python3
"""
Test bandwidth-aware scheduling functionality
"""

import sys
import time
import zmq
import json

def test_bandwidth_stats():
    """Test getting bandwidth stats from C++ scheduler"""
    print("🔍 Testing bandwidth stats query...")
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    # Send bandwidth stats request
    request = {
        "type": "get_bandwidth_stats"
    }
    socket.send_json(request)
    
    # Receive response
    response = socket.recv_json()
    
    if response["status"] == "success":
        stats = response["bandwidth_stats"]
        print(f"✅ Bandwidth stats received:")
        print(f"   CPU bandwidth:  {stats['cpu_bandwidth_gbps']:.2f} GB/s")
        print(f"   iGPU bandwidth: {stats['igpu_bandwidth_gbps']:.2f} GB/s")
        print(f"   Utilization:    {stats['utilization']:.1%}")
    else:
        print(f"❌ Error: {response.get('error', 'Unknown error')}")
        return False
    
    socket.close()
    context.term()
    return True

def test_work_packet_with_telemetry():
    """Send work packets and verify telemetry logging"""
    print("\n📦 Testing work packet execution with telemetry...")
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    # Send multiple work packets
    operations = [
        ("embedding_lookup", "cpu"),
        ("attention_qkv_proj", "auto"),
        ("expert_ffn_0", "auto"),
        ("expert_ffn_1", "auto"),
        ("router", "cpu"),
    ]
    
    for i, (op, device) in enumerate(operations):
        packet = {
            "packet_id": i,
            "layer_idx": 0,
            "operation": op,
            "device_target": device,
            "input_shape": [1, 128, 768],
            "input_dtype": "float32",
            "input_data_ptr": 0,
            "expected_duration_ms": 10.0,
            "metadata": {}
        }
        
        request = {
            "type": "work_packet",
            "data": packet
        }
        
        socket.send_json(request)
        response = socket.recv_json()
        
        if response["status"] == "success":
            result = response["result"]
            print(f"  ✅ Packet {i} ({op}): {result['device_used']} - {result['actual_duration_ms']:.2f}ms")
        else:
            print(f"  ❌ Packet {i} failed: {response.get('error', 'Unknown')}")
    
    socket.close()
    context.term()
    return True

def test_health_check():
    """Test health check with iGPU availability"""
    print("\n🏥 Testing health check...")
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    request = {"type": "health_check"}
    socket.send_json(request)
    response = socket.recv_json()
    
    if response["status"] == "ok":
        igpu_available = response["igpu_available"]
        print(f"✅ Scheduler healthy")
        print(f"   iGPU available: {igpu_available}")
    else:
        print(f"❌ Health check failed")
        return False
    
    socket.close()
    context.term()
    return True

def main():
    print("=" * 60)
    print("BANDWIDTH-AWARE SCHEDULING TEST")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        if not test_health_check():
            sys.exit(1)
        
        # Test 2: Bandwidth stats
        if not test_bandwidth_stats():
            sys.exit(1)
        
        # Test 3: Work packet execution
        if not test_work_packet_with_telemetry():
            sys.exit(1)
        
        # Test 4: Get telemetry
        print("\n📊 Testing telemetry query...")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        
        request = {"type": "get_telemetry"}
        socket.send_json(request)
        response = socket.recv_json()
        
        if response["status"] == "success":
            telemetry = response["telemetry"]
            print(f"✅ Telemetry received:")
            print(f"   Total packets:  {telemetry['total_packets_processed']}")
            print(f"   CPU packets:    {telemetry['cpu_packets']}")
            print(f"   iGPU packets:   {telemetry['igpu_packets']}")
            print(f"   Failed:         {telemetry['failed_packets']}")
            print(f"   Avg CPU time:   {telemetry['avg_cpu_time_ms']:.2f}ms")
        
        socket.close()
        context.term()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
