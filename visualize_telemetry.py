#!/usr/bin/env python3
"""
Generate visualizations from telemetry CSV and JSON logs
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_data(csv_path, json_path):
    """Load CSV and JSON telemetry data"""
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        summary = json.load(f)
    return df, summary


def plot_device_utilization(df, summary, output_dir):
    """Plot CPU vs iGPU utilization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    sizes = [summary['cpu_packets'], summary['igpu_packets']]
    labels = [f'CPU\n{summary["cpu_packets"]} ops', f'iGPU\n{summary["igpu_packets"]} ops']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Device Utilization (Packet Count)', fontsize=14, fontweight='bold')
    
    # Time breakdown
    cpu_time = summary['total_cpu_time_ms']
    igpu_time = summary['total_igpu_time_ms']
    time_data = [cpu_time, igpu_time]
    time_labels = [f'CPU\n{cpu_time:.1f} ms', f'iGPU\n{igpu_time:.1f} ms']
    
    ax2.pie(time_data, labels=time_labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('Device Time Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'device_utilization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: device_utilization.png")
    plt.close()


def plot_operation_breakdown(summary, output_dir):
    """Plot per-operation statistics"""
    op_breakdown = summary['operation_breakdown']
    
    # Prepare data
    operations = []
    avg_times = []
    counts = []
    cpu_counts = []
    igpu_counts = []
    
    for op_name, op_data in sorted(op_breakdown.items(), key=lambda x: x[1]['avg_time_ms'], reverse=True):
        operations.append(op_name)
        avg_times.append(op_data['avg_time_ms'])
        counts.append(op_data['count'])
        cpu_counts.append(op_data['cpu_count'])
        igpu_counts.append(op_data['igpu_count'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Average time per operation
    bars = ax1.barh(operations, avg_times, color='steelblue')
    ax1.set_xlabel('Average Time (ms)', fontsize=12)
    ax1.set_title('Average Execution Time per Operation', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_times)):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left', fontsize=9)
    
    # Device distribution per operation
    x = np.arange(len(operations))
    width = 0.35
    
    ax2.barh(x - width/2, cpu_counts, width, label='CPU', color='#ff9999')
    ax2.barh(x + width/2, igpu_counts, width, label='iGPU', color='#66b3ff')
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(operations)
    ax2.set_xlabel('Operation Count', fontsize=12)
    ax2.set_title('Device Usage per Operation Type', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'operation_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: operation_breakdown.png")
    plt.close()


def plot_layer_performance(df, output_dir):
    """Plot per-layer execution time"""
    # Filter out non-layer operations (embedding, lm_head)
    layer_df = df[(df['layer_idx'] >= 0) & (df['layer_idx'] < 32)].copy()
    
    if len(layer_df) == 0:
        print("⚠️  No layer data found, skipping layer performance plot")
        return
    
    # Group by layer
    layer_stats = layer_df.groupby('layer_idx').agg({
        'duration_ms': ['sum', 'mean', 'count']
    }).reset_index()
    layer_stats.columns = ['layer_idx', 'total_time', 'avg_time', 'op_count']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Total time per layer
    ax1.bar(layer_stats['layer_idx'], layer_stats['total_time'], color='coral')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Total Time (ms)', fontsize=12)
    ax1.set_title('Total Execution Time per Transformer Layer', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Operation count per layer
    ax2.bar(layer_stats['layer_idx'], layer_stats['op_count'], color='mediumpurple')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Operation Count', fontsize=12)
    ax2.set_title('Number of Operations per Layer', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: layer_performance.png")
    plt.close()


def plot_timeline(df, output_dir, max_packets=500):
    """Plot execution timeline"""
    # Limit to first N packets for readability
    plot_df = df.head(max_packets).copy()
    
    # Normalize timestamp to start from 0
    plot_df['rel_timestamp'] = (plot_df['timestamp'] - plot_df['timestamp'].min()) * 1000  # Convert to ms
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color map
    device_colors = {'cpu': '#ff9999', 'igpu': '#66b3ff'}
    
    # Plot each packet as a bar
    for idx, row in plot_df.iterrows():
        color = device_colors.get(row['device'], 'gray')
        ax.barh(idx, row['duration_ms'], left=row['rel_timestamp'], 
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Packet ID', fontsize=12)
    ax.set_title(f'Execution Timeline (First {max_packets} packets)', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff9999', label='CPU'),
                      Patch(facecolor='#66b3ff', label='iGPU')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_timeline.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: execution_timeline.png")
    plt.close()


def plot_worker_performance(df, output_dir):
    """Plot worker instance performance"""
    worker_stats = df.groupby('worker_instance').agg({
        'duration_ms': ['sum', 'mean', 'count']
    }).reset_index()
    worker_stats.columns = ['worker', 'total_time', 'avg_time', 'count']
    worker_stats = worker_stats.sort_values('total_time', ascending=True).tail(15)  # Top 15
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Total time by worker
    bars1 = ax1.barh(worker_stats['worker'], worker_stats['total_time'], color='teal')
    ax1.set_xlabel('Total Time (ms)', fontsize=12)
    ax1.set_title('Top 15 Workers by Total Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, worker_stats['total_time']):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                va='center', ha='left', fontsize=8)
    
    # Average time by worker
    worker_avg = df.groupby('worker_instance')['duration_ms'].mean().sort_values(ascending=True).tail(15)
    bars2 = ax2.barh(worker_avg.index, worker_avg.values, color='indianred')
    ax2.set_xlabel('Average Time (ms)', fontsize=12)
    ax2.set_title('Top 15 Workers by Average Execution Time', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, worker_avg.values):
        ax2.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'worker_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: worker_performance.png")
    plt.close()


def plot_duration_distribution(df, output_dir):
    """Plot distribution of execution times"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall distribution
    ax1.hist(df['duration_ms'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Duration (ms)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Duration Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # CPU vs iGPU distribution
    cpu_durations = df[df['device'] == 'cpu']['duration_ms']
    igpu_durations = df[df['device'] == 'igpu']['duration_ms']
    
    ax2.hist([cpu_durations, igpu_durations], bins=30, label=['CPU', 'iGPU'],
            color=['#ff9999', '#66b3ff'], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Duration (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('CPU vs iGPU Duration Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Box plot by device
    df.boxplot(column='duration_ms', by='device', ax=ax3, 
              patch_artist=True, grid=False)
    ax3.set_xlabel('Device', fontsize=12)
    ax3.set_ylabel('Duration (ms)', fontsize=12)
    ax3.set_title('Duration by Device (Box Plot)', fontsize=12, fontweight='bold')
    plt.sca(ax3)
    plt.xticks([1, 2], ['CPU', 'iGPU'])
    
    # Violin plot by operation type (top 5)
    top_ops = df['operation'].value_counts().head(5).index
    op_df = df[df['operation'].isin(top_ops)]
    
    parts = ax4.violinplot([op_df[op_df['operation'] == op]['duration_ms'].values 
                           for op in top_ops],
                          positions=range(len(top_ops)),
                          showmeans=True, showmedians=True)
    
    ax4.set_xticks(range(len(top_ops)))
    ax4.set_xticklabels(top_ops, rotation=45, ha='right')
    ax4.set_ylabel('Duration (ms)', fontsize=12)
    ax4.set_title('Duration Distribution by Top 5 Operations', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: duration_distribution.png")
    plt.close()


def plot_cumulative_time(df, output_dir):
    """Plot cumulative execution time over packets"""
    df_sorted = df.sort_values('timestamp').copy()
    df_sorted['cumulative_time'] = df_sorted['duration_ms'].cumsum()
    df_sorted['packet_number'] = range(len(df_sorted))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df_sorted['packet_number'], df_sorted['cumulative_time'], 
           linewidth=2, color='darkblue')
    ax.fill_between(df_sorted['packet_number'], df_sorted['cumulative_time'], 
                    alpha=0.3, color='lightblue')
    
    ax.set_xlabel('Packet Number', fontsize=12)
    ax.set_ylabel('Cumulative Time (ms)', fontsize=12)
    ax.set_title('Cumulative Execution Time', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_time.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: cumulative_time.png")
    plt.close()


def generate_summary_report(df, summary, output_dir):
    """Generate text summary report"""
    report = []
    report.append("=" * 70)
    report.append("TELEMETRY ANALYSIS SUMMARY")
    report.append("=" * 70)
    report.append(f"\nSession: {summary['session_name']}")
    report.append(f"Total Time: {summary['total_time_s']:.3f} seconds")
    report.append(f"Tokens Generated: {summary['tokens_generated']}")
    report.append(f"Throughput: {summary['tokens_per_second']:.2f} tokens/s")
    
    report.append(f"\n{'-' * 70}")
    report.append("DEVICE UTILIZATION")
    report.append(f"{'-' * 70}")
    report.append(f"Total Packets: {summary['total_packets_processed']}")
    report.append(f"CPU Packets: {summary['cpu_packets']} ({summary['cpu_utilization']*100:.1f}%)")
    report.append(f"iGPU Packets: {summary['igpu_packets']} ({summary['igpu_utilization']*100:.1f}%)")
    report.append(f"Failed Packets: {summary['failed_packets']}")
    report.append(f"Success Rate: {summary['success_rate']*100:.1f}%")
    
    report.append(f"\n{'-' * 70}")
    report.append("EXECUTION TIME")
    report.append(f"{'-' * 70}")
    report.append(f"Total CPU Time: {summary['total_cpu_time_ms']:.2f} ms")
    report.append(f"Total iGPU Time: {summary['total_igpu_time_ms']:.2f} ms")
    report.append(f"Average CPU Time: {summary['avg_cpu_time_ms']:.3f} ms/op")
    report.append(f"Average iGPU Time: {summary['avg_igpu_time_ms']:.3f} ms/op")
    
    report.append(f"\n{'-' * 70}")
    report.append("TOP 5 SLOWEST OPERATIONS")
    report.append(f"{'-' * 70}")
    slowest = df.nlargest(5, 'duration_ms')[['operation', 'layer_idx', 'device', 'duration_ms', 'worker_instance']]
    for idx, row in slowest.iterrows():
        report.append(f"  {row['operation']:25s} Layer {row['layer_idx']:>3d}  {row['device']:5s}  {row['duration_ms']:8.3f} ms")
    
    report.append(f"\n{'-' * 70}")
    report.append("TOP 5 MOST FREQUENT OPERATIONS")
    report.append(f"{'-' * 70}")
    top_ops = df['operation'].value_counts().head(5)
    for op, count in top_ops.items():
        report.append(f"  {op:30s} {count:5d} times")
    
    report_text = "\n".join(report)
    
    # Print to console
    print(f"\n{report_text}\n")
    
    # Save to file
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"✓ Saved: summary_report.txt")


def main():
    # Find latest CSV and JSON files
    telemetry_dir = Path("telemetry")
    
    if not telemetry_dir.exists():
        print("❌ Error: telemetry directory not found")
        sys.exit(1)
    
    csv_files = list(telemetry_dir.glob("session_*.csv"))
    if not csv_files:
        print("❌ Error: No CSV files found in telemetry/")
        sys.exit(1)
    
    # Get latest CSV
    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    json_path = csv_path.with_suffix('.json')
    
    if not json_path.exists():
        print(f"❌ Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    print(f"\n📊 Analyzing telemetry data:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
    
    # Load data
    df, summary = load_data(csv_path, json_path)
    
    print(f"\n📈 Generating visualizations...")
    
    # Create output directory for plots
    output_dir = telemetry_dir / f"plots_{csv_path.stem}"
    output_dir.mkdir(exist_ok=True)
    print(f"   Output directory: {output_dir}")
    
    # Generate all plots
    plot_device_utilization(df, summary, output_dir)
    plot_operation_breakdown(summary, output_dir)
    plot_layer_performance(df, output_dir)
    plot_timeline(df, output_dir, max_packets=500)
    plot_worker_performance(df, output_dir)
    plot_duration_distribution(df, output_dir)
    plot_cumulative_time(df, output_dir)
    
    # Generate summary report
    generate_summary_report(df, summary, output_dir)
    
    print(f"\n✅ Analysis complete! All visualizations saved to:")
    print(f"   {output_dir}")
    print(f"\n📁 Generated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"   • {file.name}")


if __name__ == "__main__":
    main()
