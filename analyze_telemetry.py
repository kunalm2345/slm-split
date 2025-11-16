#!/usr/bin/env python3
"""
Analyze telemetry CSV logs
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

def analyze_telemetry(csv_path):
    """Analyze telemetry CSV file"""
    
    print(f"\n📊 Analyzing: {csv_path}")
    print("="*60)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("⚠️  No data in CSV file")
        return
    
    # Statistics
    total_ops = len(rows)
    successful_ops = sum(1 for r in rows if r['success'] == 'True')
    failed_ops = total_ops - successful_ops
    
    cpu_ops = [r for r in rows if r['device'] == 'cpu']
    igpu_ops = [r for r in rows if r['device'] == 'igpu']
    
    cpu_time = sum(float(r['duration_ms']) for r in cpu_ops)
    igpu_time = sum(float(r['duration_ms']) for r in igpu_ops)
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total operations: {total_ops}")
    print(f"   Successful: {successful_ops} ({successful_ops/total_ops*100:.1f}%)")
    print(f"   Failed: {failed_ops}")
    print(f"   CPU operations: {len(cpu_ops)} ({cpu_time:.2f} ms)")
    print(f"   iGPU operations: {len(igpu_ops)} ({igpu_time:.2f} ms)")
    
    # Per-worker breakdown
    workers = defaultdict(lambda: {'count': 0, 'time': 0.0})
    for row in rows:
        if row['success'] == 'True':
            workers[row['worker_instance']]['count'] += 1
            workers[row['worker_instance']]['time'] += float(row['duration_ms'])
    
    print(f"\n🔧 Per-Worker Breakdown:")
    for worker, stats in sorted(workers.items(), key=lambda x: x[1]['time'], reverse=True):
        avg_time = stats['time'] / stats['count']
        print(f"   {worker:40s} {stats['count']:3d} ops  {stats['time']:8.2f} ms  (avg: {avg_time:.2f} ms)")
    
    # Per-layer breakdown
    layers = defaultdict(lambda: {'count': 0, 'time': 0.0})
    for row in rows:
        if row['success'] == 'True' and row['layer_idx'] != '-1':
            layer_id = int(row['layer_idx'])
            layers[layer_id]['count'] += 1
            layers[layer_id]['time'] += float(row['duration_ms'])
    
    if layers:
        print(f"\n📊 Per-Layer Breakdown:")
        for layer_id in sorted(layers.keys()):
            stats = layers[layer_id]
            avg_time = stats['time'] / stats['count']
            print(f"   Layer {layer_id:2d}: {stats['count']:3d} ops  {stats['time']:8.2f} ms  (avg: {avg_time:.2f} ms)")
    
    # Slowest operations
    sorted_ops = sorted(
        [r for r in rows if r['success'] == 'True'],
        key=lambda x: float(x['duration_ms']),
        reverse=True
    )
    
    print(f"\n🐌 Top 5 Slowest Operations:")
    for i, op in enumerate(sorted_ops[:5]):
        print(f"   {i+1}. {op['operation']:25s} Layer {op['layer_idx']:>3s}  {op['device']:5s}  {float(op['duration_ms']):6.2f} ms")
    
    # Error analysis
    if failed_ops > 0:
        print(f"\n❌ Failed Operations:")
        for row in rows:
            if row['success'] == 'False':
                print(f"   Packet {row['packet_id']}: {row['operation']} at layer {row['layer_idx']}")
                print(f"      Error: {row['error_message']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Find most recent CSV in telemetry/
        telemetry_dir = Path("telemetry")
        if not telemetry_dir.exists():
            print("❌ No telemetry directory found")
            sys.exit(1)
        
        csv_files = list(telemetry_dir.glob("*.csv"))
        if not csv_files:
            print("❌ No CSV files found in telemetry/")
            sys.exit(1)
        
        csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    analyze_telemetry(csv_path)
    
    # Also show JSON summary if available
    json_path = Path(str(csv_path).replace('.csv', '.json'))
    if json_path.exists():
        with open(json_path) as f:
            summary = json.load(f)
        
        print(f"\n📋 Session Summary:")
        print(f"   Session: {summary['session_name']}")
        print(f"   Duration: {summary['total_time_s']:.3f} seconds")
        print(f"   Tokens: {summary['tokens_generated']} generated, {summary['prompt_tokens']} prompt")
        print(f"   Throughput: {summary['tokens_per_second']:.2f} tokens/s")
        print(f"   Success rate: {summary['success_rate']*100:.1f}%")
        print(f"\n   Files:")
        print(f"   📄 CSV: {summary['csv_log']}")
        print(f"   📄 JSON: {summary['json_summary']}")
