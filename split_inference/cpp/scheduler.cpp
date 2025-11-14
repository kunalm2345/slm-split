/**
 * C++ Scheduler for Split CPU/iGPU Inference
 * Receives work packets from Python orchestrator via ZeroMQ
 * Routes operations to CPU (oneDNN) or iGPU (SYCL/Level-Zero)
 */

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

// Optional: SYCL headers (if installed)
#ifdef ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

// Forward declarations
class TelemetryCollector;
class DeviceExecutor;
class BandwidthMonitor;

/**
 * Work packet structure matching Python WorkPacket
 */
struct WorkPacket {
    int packet_id;
    int layer_idx;
    std::string operation;
    std::string device_target;
    
    std::vector<int> input_shape;
    std::string input_dtype;
    uintptr_t input_data_ptr = 0;
    
    json params;
    
    int priority = 0;
    bool can_pipeline = true;
    float memory_requirement_mb = 0.0f;
    float estimated_duration_ms = 0.0f;
    
    // Parse from JSON
    static WorkPacket from_json(const json& j) {
        WorkPacket packet;
        packet.packet_id = j.at("packet_id").get<int>();
        packet.layer_idx = j.at("layer_idx").get<int>();
        packet.operation = j.at("operation").get<std::string>();
        packet.device_target = j.at("device_target").get<std::string>();
        packet.input_shape = j.at("input_shape").get<std::vector<int>>();
        packet.input_dtype = j.at("input_dtype").get<std::string>();
        
        if (j.contains("input_data_ptr") && !j["input_data_ptr"].is_null()) {
            packet.input_data_ptr = j["input_data_ptr"].get<uintptr_t>();
        }
        if (j.contains("params")) {
            packet.params = j["params"];
        }
        if (j.contains("priority")) {
            packet.priority = j["priority"].get<int>();
        }
        if (j.contains("can_pipeline")) {
            packet.can_pipeline = j["can_pipeline"].get<bool>();
        }
        if (j.contains("memory_requirement_mb")) {
            packet.memory_requirement_mb = j["memory_requirement_mb"].get<float>();
        }
        if (j.contains("estimated_duration_ms")) {
            packet.estimated_duration_ms = j["estimated_duration_ms"].get<float>();
        }
        
        return packet;
    }
};

/**
 * Work result structure matching Python WorkResult
 */
struct WorkResult {
    int packet_id;
    bool success;
    std::vector<int> output_shape;
    std::string output_dtype;
    uintptr_t output_data_ptr = 0;
    
    float actual_duration_ms = 0.0f;
    std::string device_used;
    float memory_used_mb = 0.0f;
    std::string error_message;
    
    // Convert to JSON
    json to_json() const {
        json j;
        j["packet_id"] = packet_id;
        j["success"] = success;
        j["output_shape"] = output_shape;
        j["output_dtype"] = output_dtype;
        j["output_data_ptr"] = output_data_ptr;
        j["actual_duration_ms"] = actual_duration_ms;
        j["device_used"] = device_used;
        j["memory_used_mb"] = memory_used_mb;
        j["error_message"] = error_message;
        return j;
    }
};

/**
 * Telemetry collector for performance metrics
 */
class TelemetryCollector {
public:
    struct Metrics {
        size_t total_packets_processed = 0;
        size_t cpu_packets = 0;
        size_t igpu_packets = 0;
        size_t failed_packets = 0;
        
        double total_cpu_time_ms = 0.0;
        double total_igpu_time_ms = 0.0;
        
        double avg_cpu_time_ms = 0.0;
        double avg_igpu_time_ms = 0.0;
        
        size_t current_queue_depth = 0;
        double peak_memory_usage_mb = 0.0;
        
        double cpu_bandwidth_gbps = 0.0;
        double igpu_bandwidth_gbps = 0.0;
    };
    
    void record_packet(const WorkPacket& packet, const WorkResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        metrics_.total_packets_processed++;
        
        if (result.success) {
            if (result.device_used == "cpu") {
                metrics_.cpu_packets++;
                metrics_.total_cpu_time_ms += result.actual_duration_ms;
                metrics_.avg_cpu_time_ms = metrics_.total_cpu_time_ms / metrics_.cpu_packets;
            } else if (result.device_used == "igpu") {
                metrics_.igpu_packets++;
                metrics_.total_igpu_time_ms += result.actual_duration_ms;
                metrics_.avg_igpu_time_ms = metrics_.total_igpu_time_ms / metrics_.igpu_packets;
            }
        } else {
            metrics_.failed_packets++;
        }
        
        if (result.memory_used_mb > metrics_.peak_memory_usage_mb) {
            metrics_.peak_memory_usage_mb = result.memory_used_mb;
        }
    }
    
    void update_queue_depth(size_t depth) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.current_queue_depth = depth;
    }
    
    void update_bandwidth(double cpu_gbps, double igpu_gbps) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.cpu_bandwidth_gbps = cpu_gbps;
        metrics_.igpu_bandwidth_gbps = igpu_gbps;
    }
    
    Metrics get_metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return metrics_;
    }
    
    json to_json() const {
        auto m = get_metrics();
        json j;
        j["total_packets_processed"] = m.total_packets_processed;
        j["cpu_packets"] = m.cpu_packets;
        j["igpu_packets"] = m.igpu_packets;
        j["failed_packets"] = m.failed_packets;
        j["total_cpu_time_ms"] = m.total_cpu_time_ms;
        j["total_igpu_time_ms"] = m.total_igpu_time_ms;
        j["avg_cpu_time_ms"] = m.avg_cpu_time_ms;
        j["avg_igpu_time_ms"] = m.avg_igpu_time_ms;
        j["current_queue_depth"] = m.current_queue_depth;
        j["peak_memory_usage_mb"] = m.peak_memory_usage_mb;
        j["cpu_bandwidth_gbps"] = m.cpu_bandwidth_gbps;
        j["igpu_bandwidth_gbps"] = m.igpu_bandwidth_gbps;
        return j;
    }
    
private:
    mutable std::mutex mutex_;
    Metrics metrics_;
};

/**
 * Device executor (stub for now - will implement actual kernel dispatch)
 */
class DeviceExecutor {
public:
    DeviceExecutor() {
        std::cout << "🔧 Initializing device executor..." << std::endl;
        
#ifdef ENABLE_SYCL
        try {
            // Enumerate SYCL devices
            auto platforms = sycl::platform::get_platforms();
            std::cout << "  • Found " << platforms.size() << " SYCL platforms" << std::endl;
            
            for (const auto& platform : platforms) {
                auto devices = platform.get_devices();
                for (const auto& device : devices) {
                    std::cout << "    - " << device.get_info<sycl::info::device::name>() << std::endl;
                    if (device.is_gpu()) {
                        igpu_device_ = std::make_unique<sycl::device>(device);
                        igpu_queue_ = std::make_unique<sycl::queue>(*igpu_device_);
                        std::cout << "    ✓ Selected as iGPU" << std::endl;
                    }
                }
            }
            
            if (igpu_queue_) {
                igpu_available_ = true;
                std::cout << "  ✓ iGPU initialized" << std::endl;
            } else {
                std::cout << "  ⚠️  No iGPU found" << std::endl;
            }
        } catch (const sycl::exception& e) {
            std::cerr << "  ✗ SYCL error: " << e.what() << std::endl;
        }
#else
        std::cout << "  ⚠️  SYCL not enabled (compile with -DENABLE_SYCL)" << std::endl;
#endif
        
        std::cout << "  ✓ CPU executor ready" << std::endl;
    }
    
    WorkResult execute(const WorkPacket& packet) {
        WorkResult result;
        result.packet_id = packet.packet_id;
        result.success = false;
        
        auto start = Clock::now();
        
        try {
            // Determine device
            std::string device = packet.device_target;
            if (device == "auto") {
                device = choose_device(packet);
            }
            
            // Execute on chosen device
            if (device == "cpu") {
                result = execute_on_cpu(packet);
            } else if (device == "igpu") {
                if (igpu_available_) {
                    result = execute_on_igpu(packet);
                } else {
                    std::cout << "  ⚠️  iGPU not available, falling back to CPU" << std::endl;
                    result = execute_on_cpu(packet);
                }
            } else {
                result.error_message = "Unknown device: " + device;
            }
            
            auto end = Clock::now();
            result.actual_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
        } catch (const std::exception& e) {
            result.error_message = e.what();
            result.success = false;
        }
        
        return result;
    }
    
    bool is_igpu_available() const { return igpu_available_; }
    
private:
    std::string choose_device(const WorkPacket& packet) {
        // Simple heuristic: heavy compute → iGPU, light → CPU
        if (packet.operation.find("ffn") != std::string::npos ||
            packet.operation.find("expert") != std::string::npos) {
            return igpu_available_ ? "igpu" : "cpu";
        }
        return "cpu";
    }
    
    WorkResult execute_on_cpu(const WorkPacket& packet) {
        // STUB: Actual implementation will call oneDNN/PyTorch ops
        WorkResult result;
        result.packet_id = packet.packet_id;
        result.success = true;
        result.device_used = "cpu";
        result.output_shape = packet.input_shape;
        result.output_dtype = packet.input_dtype;
        result.memory_used_mb = 10.0f;  // Placeholder
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        return result;
    }
    
    WorkResult execute_on_igpu(const WorkPacket& packet) {
        // STUB: Actual implementation will call SYCL kernels
        WorkResult result;
        result.packet_id = packet.packet_id;
        result.success = true;
        result.device_used = "igpu";
        result.output_shape = packet.input_shape;
        result.output_dtype = packet.input_dtype;
        result.memory_used_mb = 20.0f;  // Placeholder
        
#ifdef ENABLE_SYCL
        // Example: Simple kernel invocation
        // Actual kernels will be much more complex
        try {
            igpu_queue_->submit([&](sycl::handler& h) {
                // Placeholder kernel
            }).wait();
        } catch (const sycl::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }
#else
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
#endif
        
        return result;
    }
    
    bool igpu_available_ = false;
    
#ifdef ENABLE_SYCL
    std::unique_ptr<sycl::device> igpu_device_;
    std::unique_ptr<sycl::queue> igpu_queue_;
#endif
};

/**
 * Main scheduler class
 */
class Scheduler {
public:
    Scheduler(const std::string& bind_address = "tcp://*:5555")
        : bind_address_(bind_address), running_(false) {
        
        std::cout << "🚀 Initializing scheduler..." << std::endl;
        executor_ = std::make_unique<DeviceExecutor>();
        telemetry_ = std::make_unique<TelemetryCollector>();
    }
    
    void start() {
        std::cout << "🔌 Starting ZeroMQ server on " << bind_address_ << std::endl;
        
        context_ = std::make_unique<zmq::context_t>(1);
        socket_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::rep);
        socket_->bind(bind_address_);
        
        running_ = true;
        
        std::cout << "✓ Scheduler ready - waiting for requests..." << std::endl;
        
        while (running_) {
            try {
                // Receive request
                zmq::message_t request;
                auto result = socket_->recv(request, zmq::recv_flags::none);
                
                if (!result.has_value()) {
                    continue;
                }
                
                // Parse JSON request
                std::string request_str(static_cast<char*>(request.data()), request.size());
                json request_json = json::parse(request_str);
                
                std::string request_type = request_json["type"].get<std::string>();
                
                // Handle request
                json response;
                if (request_type == "health_check") {
                    response = handle_health_check();
                } else if (request_type == "work_packet") {
                    response = handle_work_packet(request_json["data"]);
                } else if (request_type == "get_telemetry") {
                    response = handle_get_telemetry();
                } else if (request_type == "shutdown") {
                    response = handle_shutdown();
                } else {
                    response["status"] = "error";
                    response["error"] = "Unknown request type: " + request_type;
                }
                
                // Send response
                std::string response_str = response.dump();
                zmq::message_t reply(response_str.size());
                memcpy(reply.data(), response_str.c_str(), response_str.size());
                socket_->send(reply, zmq::send_flags::none);
                
            } catch (const std::exception& e) {
                std::cerr << "✗ Error handling request: " << e.what() << std::endl;
                
                // Send error response
                json error_response;
                error_response["status"] = "error";
                error_response["error"] = e.what();
                std::string error_str = error_response.dump();
                zmq::message_t error_reply(error_str.size());
                memcpy(error_reply.data(), error_str.c_str(), error_str.size());
                socket_->send(error_reply, zmq::send_flags::none);
            }
        }
        
        std::cout << "✓ Scheduler stopped" << std::endl;
    }
    
private:
    json handle_health_check() {
        json response;
        response["status"] = "ok";
        response["igpu_available"] = executor_->is_igpu_available();
        return response;
    }
    
    json handle_work_packet(const json& data) {
        WorkPacket packet = WorkPacket::from_json(data);
        
        std::cout << "📦 Work packet " << packet.packet_id 
                  << " (layer " << packet.layer_idx 
                  << ", op: " << packet.operation << ")" << std::endl;
        
        // Execute
        WorkResult result = executor_->execute(packet);
        
        // Record telemetry
        telemetry_->record_packet(packet, result);
        
        // Build response
        json response;
        response["status"] = result.success ? "success" : "error";
        if (result.success) {
            response["result"] = result.to_json();
        } else {
            response["error"] = result.error_message;
        }
        
        std::cout << "  ✓ Completed on " << result.device_used 
                  << " (" << result.actual_duration_ms << " ms)" << std::endl;
        
        return response;
    }
    
    json handle_get_telemetry() {
        json response;
        response["status"] = "success";
        response["telemetry"] = telemetry_->to_json();
        return response;
    }
    
    json handle_shutdown() {
        std::cout << "🛑 Shutdown requested" << std::endl;
        running_ = false;
        
        json response;
        response["status"] = "ok";
        return response;
    }
    
    std::string bind_address_;
    std::atomic<bool> running_;
    
    std::unique_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
    
    std::unique_ptr<DeviceExecutor> executor_;
    std::unique_ptr<TelemetryCollector> telemetry_;
};

int main(int argc, char** argv) {
    std::cout << "="  << std::string(59, '=') << std::endl;
    std::cout << "SPLIT CPU/iGPU INFERENCE SCHEDULER" << std::endl;
    std::cout << "="  << std::string(59, '=') << std::endl;
    
    std::string bind_address = "tcp://*:5555";
    if (argc > 1) {
        bind_address = argv[1];
    }
    
    try {
        Scheduler scheduler(bind_address);
        scheduler.start();
    } catch (const std::exception& e) {
        std::cerr << "✗ Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
