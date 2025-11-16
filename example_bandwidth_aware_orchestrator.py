#!/usr/bin/env python3
"""
Example: Using bandwidth-aware scheduling with Python orchestrator

This demonstrates how to integrate bandwidth monitoring into the
generation loop for adaptive performance optimization.
"""

import time
import zmq
import json
from typing import Dict, Optional

class BandwidthAwareOrchestrator:
    """
    Extension of the base orchestrator with bandwidth awareness
    """
    
    def __init__(self, scheduler_address: str = "tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(scheduler_address)
        
        # Bandwidth tracking
        self.last_bandwidth_check = 0
        self.bandwidth_check_interval = 0.1  # Check every 100ms
        self.current_bandwidth_stats = None
        
        print("✅ Bandwidth-aware orchestrator initialized")
    
    def get_bandwidth_stats(self) -> Optional[Dict]:
        """Query current bandwidth stats from C++ scheduler"""
        try:
            self.socket.send_json({"type": "get_bandwidth_stats"})
            response = self.socket.recv_json()
            
            if response["status"] == "success":
                return response["bandwidth_stats"]
            else:
                print(f"⚠️ Bandwidth query failed: {response.get('error', 'Unknown')}")
                return None
        except Exception as e:
            print(f"⚠️ Bandwidth query exception: {e}")
            return None
    
    def should_check_bandwidth(self) -> bool:
        """Determine if it's time to check bandwidth"""
        now = time.time()
        if now - self.last_bandwidth_check >= self.bandwidth_check_interval:
            self.last_bandwidth_check = now
            return True
        return False
    
    def send_work_packet(self, packet: Dict) -> Dict:
        """
        Send work packet with optional bandwidth-aware device override
        
        If bandwidth utilization is high, this will:
        1. Override device_target to 'cpu' for non-critical ops
        2. Add small delays to reduce burst pressure
        3. Prefer lighter operations when possible
        """
        
        # Periodically check bandwidth
        if self.should_check_bandwidth():
            self.current_bandwidth_stats = self.get_bandwidth_stats()
            
            if self.current_bandwidth_stats:
                util = self.current_bandwidth_stats['utilization']
                
                if util > 0.90:
                    print(f"  ⚠️ High bandwidth utilization: {util:.1%}")
                    
                    # Force CPU for this packet to reduce load
                    if packet.get('device_target') == 'auto' or packet.get('device_target') == 'igpu':
                        print(f"  → Overriding device to CPU")
                        packet['device_target'] = 'cpu'
                    
                    # Add small delay to reduce burst
                    time.sleep(0.005)  # 5ms delay
        
        # Send packet to scheduler
        request = {
            "type": "work_packet",
            "data": packet
        }
        
        self.socket.send_json(request)
        response = self.socket.recv_json()
        
        if response["status"] == "success":
            return response["result"]
        else:
            raise RuntimeError(f"Work packet failed: {response.get('error', 'Unknown')}")
    
    def generate_with_bandwidth_awareness(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ):
        """
        Example generation loop with bandwidth monitoring
        
        This demonstrates how to:
        1. Monitor bandwidth during generation
        2. Adapt strategy based on bandwidth pressure
        3. Log bandwidth stats for analysis
        """
        
        print(f"\n🚀 Generating {max_new_tokens} tokens (bandwidth-aware mode)")
        print(f"   Prompt: '{prompt}'")
        
        # Bandwidth stats log
        bandwidth_log = []
        
        for token_idx in range(max_new_tokens):
            
            # Check bandwidth before each token
            if token_idx % 5 == 0:  # Every 5 tokens
                stats = self.get_bandwidth_stats()
                if stats:
                    bandwidth_log.append({
                        'token': token_idx,
                        'cpu_bw': stats['cpu_bandwidth_gbps'],
                        'igpu_bw': stats['igpu_bandwidth_gbps'],
                        'utilization': stats['utilization']
                    })
                    
                    print(f"\n  📊 Token {token_idx}: "
                          f"CPU {stats['cpu_bandwidth_gbps']:.1f} GB/s, "
                          f"iGPU {stats['igpu_bandwidth_gbps']:.1f} GB/s, "
                          f"util {stats['utilization']:.1%}")
            
            # Create work packets for this token
            # (In real implementation, this would be split across operations)
            packets = self._create_token_packets(token_idx)
            
            # Execute packets with bandwidth awareness
            for packet in packets:
                result = self.send_work_packet(packet)
                print(f"  ✓ {packet['operation']} on {result['device_used']} "
                      f"({result['actual_duration_ms']:.2f}ms)")
        
        print(f"\n✅ Generation complete!")
        print(f"   Bandwidth samples: {len(bandwidth_log)}")
        
        if bandwidth_log:
            avg_util = sum(s['utilization'] for s in bandwidth_log) / len(bandwidth_log)
            print(f"   Avg bandwidth utilization: {avg_util:.1%}")
        
        return bandwidth_log
    
    def _create_token_packets(self, token_idx: int):
        """Create work packets for one token generation (simplified)"""
        layer_idx = 0  # Simplified: just one layer
        
        return [
            {
                "packet_id": token_idx * 10 + 0,
                "layer_idx": layer_idx,
                "operation": "embedding_lookup",
                "device_target": "cpu",
                "input_shape": [1, 128, 768],
                "input_dtype": "float32",
                "input_data_ptr": 0,
                "expected_duration_ms": 5.0,
                "metadata": {}
            },
            {
                "packet_id": token_idx * 10 + 1,
                "layer_idx": layer_idx,
                "operation": "attention_qkv_proj",
                "device_target": "auto",  # Let scheduler decide
                "input_shape": [1, 128, 768],
                "input_dtype": "float32",
                "input_data_ptr": 0,
                "expected_duration_ms": 15.0,
                "metadata": {}
            },
            {
                "packet_id": token_idx * 10 + 2,
                "layer_idx": layer_idx,
                "operation": "expert_ffn_0",
                "device_target": "auto",
                "input_shape": [1, 128, 768],
                "input_dtype": "float32",
                "input_data_ptr": 0,
                "expected_duration_ms": 20.0,
                "metadata": {}
            },
        ]
    
    def close(self):
        """Clean up resources"""
        self.socket.close()
        self.context.term()


def example_usage():
    """Example of using bandwidth-aware orchestrator"""
    
    print("=" * 60)
    print("BANDWIDTH-AWARE ORCHESTRATOR EXAMPLE")
    print("=" * 60)
    
    orchestrator = BandwidthAwareOrchestrator()
    
    try:
        # Example 1: Query bandwidth stats
        print("\n📊 Example 1: Query bandwidth stats")
        stats = orchestrator.get_bandwidth_stats()
        if stats:
            print(f"   CPU bandwidth:  {stats['cpu_bandwidth_gbps']:.2f} GB/s")
            print(f"   iGPU bandwidth: {stats['igpu_bandwidth_gbps']:.2f} GB/s")
            print(f"   Utilization:    {stats['utilization']:.1%}")
        
        # Example 2: Send single work packet
        print("\n📦 Example 2: Send work packet with bandwidth awareness")
        packet = {
            "packet_id": 999,
            "layer_idx": 0,
            "operation": "attention_qkv_proj",
            "device_target": "auto",
            "input_shape": [1, 128, 768],
            "input_dtype": "float32",
            "input_data_ptr": 0,
            "expected_duration_ms": 10.0,
            "metadata": {}
        }
        
        result = orchestrator.send_work_packet(packet)
        print(f"   ✓ Executed on {result['device_used']} in {result['actual_duration_ms']:.2f}ms")
        
        # Example 3: Generate with bandwidth monitoring
        print("\n🔄 Example 3: Generation with bandwidth monitoring")
        bandwidth_log = orchestrator.generate_with_bandwidth_awareness(
            prompt="Hello, I am",
            max_new_tokens=10
        )
        
        print("\n✅ Examples complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        orchestrator.close()


if __name__ == "__main__":
    example_usage()
