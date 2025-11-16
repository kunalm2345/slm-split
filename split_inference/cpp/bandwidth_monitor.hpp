/**
 * Bandwidth Monitor for Split CPU/iGPU Inference
 * Tracks memory bandwidth usage and implements token semaphore
 */

#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <string>

class BandwidthMonitor {
public:
    struct BandwidthStats {
        double cpu_bandwidth_gbps = 0.0;
        double igpu_bandwidth_gbps = 0.0;
        double total_bandwidth_gbps = 0.0;
        double max_bandwidth_gbps = 60.0;  // System max (shared DRAM)
        double utilization = 0.0;           // 0.0 to 1.0
        bool is_saturated = false;
    };

    BandwidthMonitor(double max_bw_gbps = 60.0, double throttle_threshold = 0.85)
        : max_bandwidth_gbps_(max_bw_gbps)
        , throttle_threshold_(throttle_threshold)
        , running_(false)
        , dram_token_held_(false)
    {
        // Define heavy operations that need DRAM token
        heavy_operations_ = {
            "expert_ffn",
            "attention_compute",
            "attention_qkv_proj",
            "large_matmul"
        };
    }

    ~BandwidthMonitor() {
        stop();
    }

    void start() {
        if (running_) return;
        
        running_ = true;
        monitor_thread_ = std::thread(&BandwidthMonitor::monitor_loop, this);
    }

    void stop() {
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }

    BandwidthStats get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return current_stats_;
    }

    bool is_saturated() const {
        auto stats = get_stats();
        return stats.is_saturated;
    }

    // Token semaphore for heavy operations
    bool try_acquire_dram_token(const std::string& operation) {
        if (!is_heavy_operation(operation)) {
            return true;  // Lightweight ops don't need token
        }

        std::lock_guard<std::mutex> lock(token_mutex_);
        if (!dram_token_held_) {
            dram_token_held_ = true;
            current_token_holder_ = operation;
            return true;
        }
        return false;  // Token already held
    }

    void acquire_dram_token_blocking(const std::string& operation) {
        if (!is_heavy_operation(operation)) {
            return;  // Lightweight ops don't need token
        }

        while (!try_acquire_dram_token(operation)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void release_dram_token(const std::string& operation) {
        if (!is_heavy_operation(operation)) {
            return;
        }

        std::lock_guard<std::mutex> lock(token_mutex_);
        if (dram_token_held_ && current_token_holder_ == operation) {
            dram_token_held_ = false;
            current_token_holder_.clear();
        }
    }

    bool is_heavy_operation(const std::string& operation) const {
        return heavy_operations_.count(operation) > 0;
    }

    // Throttling recommendations
    enum class ThrottleAction {
        NONE,
        REDUCE_BATCH,
        DELAY_LAUNCH,
        FALLBACK_CPU
    };

    ThrottleAction get_throttle_action() const {
        auto stats = get_stats();
        
        if (!stats.is_saturated) {
            return ThrottleAction::NONE;
        }

        // Decide based on current load
        if (stats.utilization > 0.95) {
            return ThrottleAction::DELAY_LAUNCH;  // Critical
        } else if (stats.utilization > 0.90) {
            return ThrottleAction::REDUCE_BATCH;  // High
        } else {
            return ThrottleAction::FALLBACK_CPU;  // Moderate
        }
    }

private:
    void monitor_loop() {
        using namespace std::chrono;
        
        while (running_) {
            auto start = steady_clock::now();
            
            // Read memory bandwidth from /proc/meminfo (Linux)
            double cpu_bw = read_cpu_bandwidth();
            double igpu_bw = read_igpu_bandwidth();
            
            // Update stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                current_stats_.cpu_bandwidth_gbps = cpu_bw;
                current_stats_.igpu_bandwidth_gbps = igpu_bw;
                current_stats_.total_bandwidth_gbps = cpu_bw + igpu_bw;
                current_stats_.max_bandwidth_gbps = max_bandwidth_gbps_;
                current_stats_.utilization = current_stats_.total_bandwidth_gbps / max_bandwidth_gbps_;
                current_stats_.is_saturated = current_stats_.utilization > throttle_threshold_;
            }
            
            // Sleep for monitoring interval (10ms)
            auto end = steady_clock::now();
            auto elapsed = duration_cast<milliseconds>(end - start);
            auto sleep_time = milliseconds(10) - elapsed;
            
            if (sleep_time.count() > 0) {
                std::this_thread::sleep_for(sleep_time);
            }
        }
    }

    double read_cpu_bandwidth() {
        // Simplified: Estimate from memory access patterns
        // In production, use Intel PCM or perf counters
        
        // For now, return a simulated value based on token state
        std::lock_guard<std::mutex> lock(token_mutex_);
        if (dram_token_held_ && current_token_holder_.find("cpu") != std::string::npos) {
            return 30.0;  // CPU using ~30 GB/s
        }
        return 5.0;  // Baseline CPU usage
    }

    double read_igpu_bandwidth() {
        // Simplified: Estimate from iGPU activity
        // In production, use Intel Level-Zero metrics
        
        std::lock_guard<std::mutex> lock(token_mutex_);
        if (dram_token_held_ && current_token_holder_.find("igpu") != std::string::npos) {
            return 40.0;  // iGPU using ~40 GB/s
        }
        return 10.0;  // Baseline iGPU usage
    }

    // Configuration
    double max_bandwidth_gbps_;
    double throttle_threshold_;
    
    // Monitoring thread
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    
    // Stats
    mutable std::mutex stats_mutex_;
    BandwidthStats current_stats_;
    
    // Token semaphore
    mutable std::mutex token_mutex_;
    bool dram_token_held_;
    std::string current_token_holder_;
    
    // Heavy operations set
    std::set<std::string> heavy_operations_;
};
