#!/usr/bin/env python3
"""
Telemetry Logger for Split Inference
Writes both summary JSON and detailed CSV logs
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class TelemetryEntry:
    """Single telemetry entry for CSV logging"""
    timestamp: float
    device: str  # cpu, igpu
    worker_instance: str  # cpu_router, igpu_expert_0_7, etc.
    duration_ms: float
    packet_id: int
    layer_idx: int
    operation: str  # embedding, attention_qkv_proj, etc.
    success: bool
    memory_used_mb: float = 0.0
    error_message: str = ""


class TelemetryLogger:
    """
    Thread-safe telemetry logger that writes:
    1. CSV file with detailed per-operation logs
    2. JSON file with summary metrics
    """
    
    def __init__(self, output_dir: str = "./telemetry", session_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session name with timestamp
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
        
        self.session_name = session_name
        self.csv_path = self.output_dir / f"{session_name}.csv"
        self.json_path = self.output_dir / f"{session_name}.json"
        
        # In-memory storage for entries
        self.entries: List[TelemetryEntry] = []
        self.lock = threading.Lock()
        
        # Summary metrics
        self.start_time = time.time()
        self.total_packets = 0
        self.failed_packets = 0
        self.cpu_packets = 0
        self.igpu_packets = 0
        self.total_cpu_time_ms = 0.0
        self.total_igpu_time_ms = 0.0
        
        # Expert distribution
        self.expert_selection_count: Dict[str, int] = {}
        
        # Device contention events
        self.contention_events = 0
        
        # Initialize CSV file with header
        self._init_csv()
        
        print(f"📊 Telemetry logging initialized:")
        print(f"   CSV: {self.csv_path}")
        print(f"   JSON: {self.json_path}")
    
    def _init_csv(self):
        """Initialize CSV file with header"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'device',
                'worker_instance',
                'duration_ms',
                'packet_id',
                'layer_idx',
                'operation',
                'success',
                'memory_used_mb',
                'error_message'
            ])
    
    def log_entry(self, entry: TelemetryEntry):
        """Log a single telemetry entry"""
        with self.lock:
            self.entries.append(entry)
            self.total_packets += 1
            
            # Update summary metrics
            if entry.success:
                if entry.device == "cpu":
                    self.cpu_packets += 1
                    self.total_cpu_time_ms += entry.duration_ms
                elif entry.device == "igpu":
                    self.igpu_packets += 1
                    self.total_igpu_time_ms += entry.duration_ms
            else:
                self.failed_packets += 1
            
            # Track expert usage
            if "expert" in entry.operation:
                key = f"{entry.operation}_layer_{entry.layer_idx}"
                self.expert_selection_count[key] = self.expert_selection_count.get(key, 0) + 1
            
            # Write to CSV immediately (append mode)
            self._write_csv_entry(entry)
    
    def _write_csv_entry(self, entry: TelemetryEntry):
        """Append single entry to CSV file"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                entry.timestamp,
                entry.device,
                entry.worker_instance,
                f"{entry.duration_ms:.3f}",
                entry.packet_id,
                entry.layer_idx,
                entry.operation,
                entry.success,
                f"{entry.memory_used_mb:.2f}",
                entry.error_message
            ])
    
    def log_work_result(self, 
                       packet_id: int,
                       layer_idx: int,
                       operation: str,
                       device_target: str,
                       result_device: str,
                       duration_ms: float,
                       success: bool,
                       memory_used_mb: float = 0.0,
                       error_message: str = ""):
        """
        Convenience method to log from work result
        
        Args:
            packet_id: Work packet ID
            layer_idx: Transformer layer index (-1 for embedding, num_layers for final)
            operation: Operation name (embedding, attention_qkv_proj, etc.)
            device_target: Originally requested device (cpu/igpu)
            result_device: Actually used device (cpu/igpu)
            duration_ms: Execution time in milliseconds
            success: Whether operation succeeded
            memory_used_mb: Memory used in MB
            error_message: Error message if failed
        """
        # Determine worker instance from operation and device
        worker_instance = self._get_worker_instance(operation, layer_idx, result_device)
        
        entry = TelemetryEntry(
            timestamp=time.time(),
            device=result_device,
            worker_instance=worker_instance,
            duration_ms=duration_ms,
            packet_id=packet_id,
            layer_idx=layer_idx,
            operation=operation,
            success=success,
            memory_used_mb=memory_used_mb,
            error_message=error_message
        )
        
        self.log_entry(entry)
    
    def _get_worker_instance(self, operation: str, layer_idx: int, device: str) -> str:
        """Map operation to worker instance name"""
        if operation == "embedding":
            return f"{device}_embedding"
        elif "router" in operation or "expert_selection" in operation:
            return "cpu_router"
        elif "expert_ffn" in operation:
            # Determine expert range based on config (0-7 or 8-15)
            # For now, just use device
            if device == "igpu":
                # Could be more sophisticated based on expert ID
                return f"igpu_expert_ffn_layer_{layer_idx}"
            else:
                return f"cpu_expert_ffn_layer_{layer_idx}"
        elif "attention" in operation:
            return f"{device}_attention_layer_{layer_idx}"
        elif "layernorm" in operation or "norm" in operation:
            return f"{device}_layernorm_layer_{layer_idx}"
        elif "lm_head" in operation:
            return f"{device}_lm_head"
        else:
            return f"{device}_{operation}"
    
    def finalize(self, tokens_generated: int = 0, prompt_tokens: int = 0):
        """
        Write final summary JSON file
        
        Args:
            tokens_generated: Number of tokens generated
            prompt_tokens: Number of input prompt tokens
        """
        end_time = time.time()
        total_time_s = end_time - self.start_time
        
        with self.lock:
            summary = {
                "session_name": self.session_name,
                "start_time": self.start_time,
                "end_time": end_time,
                "total_time_s": total_time_s,
                
                # Token metrics
                "prompt_tokens": prompt_tokens,
                "tokens_generated": tokens_generated,
                "total_tokens": prompt_tokens + tokens_generated,
                "tokens_per_second": tokens_generated / total_time_s if total_time_s > 0 else 0,
                
                # Packet metrics
                "total_packets_processed": self.total_packets,
                "cpu_packets": self.cpu_packets,
                "igpu_packets": self.igpu_packets,
                "failed_packets": self.failed_packets,
                "success_rate": (self.total_packets - self.failed_packets) / self.total_packets if self.total_packets > 0 else 0,
                
                # Time metrics
                "total_cpu_time_ms": self.total_cpu_time_ms,
                "total_igpu_time_ms": self.total_igpu_time_ms,
                "avg_cpu_time_ms": self.total_cpu_time_ms / self.cpu_packets if self.cpu_packets > 0 else 0,
                "avg_igpu_time_ms": self.total_igpu_time_ms / self.igpu_packets if self.igpu_packets > 0 else 0,
                
                # Device utilization
                "cpu_utilization": self.cpu_packets / self.total_packets if self.total_packets > 0 else 0,
                "igpu_utilization": self.igpu_packets / self.total_packets if self.total_packets > 0 else 0,
                
                # Expert distribution
                "expert_selection_distribution": self.expert_selection_count,
                
                # Contention
                "cpu_igpu_contention_events": self.contention_events,
                
                # Per-operation breakdown
                "operation_breakdown": self._compute_operation_breakdown(),
                
                # Files
                "csv_log": str(self.csv_path),
                "json_summary": str(self.json_path)
            }
        
        # Write JSON summary
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Telemetry finalized:")
        print(f"   Total packets: {self.total_packets}")
        
        if self.total_packets > 0:
            print(f"   CPU packets: {self.cpu_packets} ({self.cpu_packets/self.total_packets*100:.1f}%)")
            print(f"   iGPU packets: {self.igpu_packets} ({self.igpu_packets/self.total_packets*100:.1f}%)")
            print(f"   Failed packets: {self.failed_packets}")
        else:
            print(f"   ⚠️  No packets were processed (likely used CPU fallback)")
        
        if total_time_s > 0 and tokens_generated > 0:
            print(f"   Tokens/s: {tokens_generated / total_time_s:.2f}")
        
        print(f"   CSV log: {self.csv_path}")
        print(f"   JSON summary: {self.json_path}")
    
    def _compute_operation_breakdown(self) -> Dict[str, Dict]:
        """Compute per-operation statistics"""
        breakdown = {}
        
        for entry in self.entries:
            if entry.operation not in breakdown:
                breakdown[entry.operation] = {
                    "count": 0,
                    "total_time_ms": 0.0,
                    "avg_time_ms": 0.0,
                    "min_time_ms": float('inf'),
                    "max_time_ms": 0.0,
                    "cpu_count": 0,
                    "igpu_count": 0,
                    "failed_count": 0
                }
            
            op = breakdown[entry.operation]
            op["count"] += 1
            
            if entry.success:
                op["total_time_ms"] += entry.duration_ms
                op["min_time_ms"] = min(op["min_time_ms"], entry.duration_ms)
                op["max_time_ms"] = max(op["max_time_ms"], entry.duration_ms)
                
                if entry.device == "cpu":
                    op["cpu_count"] += 1
                elif entry.device == "igpu":
                    op["igpu_count"] += 1
            else:
                op["failed_count"] += 1
        
        # Compute averages
        for op_name, op_data in breakdown.items():
            if op_data["count"] > 0:
                op_data["avg_time_ms"] = op_data["total_time_ms"] / op_data["count"]
        
        return breakdown
    
    def get_summary_stats(self) -> Dict:
        """Get current summary statistics (without finalizing)"""
        with self.lock:
            return {
                "total_packets": self.total_packets,
                "cpu_packets": self.cpu_packets,
                "igpu_packets": self.igpu_packets,
                "failed_packets": self.failed_packets,
                "avg_cpu_time_ms": self.total_cpu_time_ms / self.cpu_packets if self.cpu_packets > 0 else 0,
                "avg_igpu_time_ms": self.total_igpu_time_ms / self.igpu_packets if self.igpu_packets > 0 else 0
            }
