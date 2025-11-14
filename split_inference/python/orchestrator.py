#!/usr/bin/env python3
"""
Python Orchestrator for Split CPU/iGPU Inference
Handles tokenization, work packet creation, and IPC with C++ scheduler
"""

import zmq
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import logging
from transformers import AutoTokenizer, AutoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WorkPacket:
    """Work packet sent to C++ scheduler"""
    packet_id: int
    layer_idx: int
    operation: str  # e.g., "attention_qkv", "expert_ffn", "router_logits"
    device_target: str  # "cpu", "igpu", "auto"
    
    # Input tensor metadata
    input_shape: List[int]
    input_dtype: str
    input_data_ptr: Optional[int] = None  # Pointer to shared memory (if using)
    
    # Operation-specific parameters
    params: Dict[str, Any] = None
    
    # Scheduling hints
    priority: int = 0  # Higher = more urgent
    can_pipeline: bool = True
    memory_requirement_mb: float = 0.0
    estimated_duration_ms: float = 0.0


@dataclass
class WorkResult:
    """Result received from C++ scheduler"""
    packet_id: int
    success: bool
    output_shape: List[int]
    output_dtype: str
    output_data_ptr: Optional[int] = None
    
    # Telemetry
    actual_duration_ms: float = 0.0
    device_used: str = ""
    memory_used_mb: float = 0.0
    error_message: str = ""


class PartitionConfig:
    """Parser for partition_config.yaml"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load YAML configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_device_for_operation(self, layer_idx: int, operation: str) -> str:
        """Determine target device for an operation"""
        # Check layer-specific overrides first
        if 'layer_overrides' in self.config:
            for layer_range, overrides in self.config['layer_overrides'].items():
                if self._in_range(layer_idx, layer_range) and operation in overrides:
                    return overrides[operation]
        
        # Fall back to static partition
        static_part = self.config.get('static_partition', {})
        layer_ops = static_part.get('layer_operations', {})
        
        return layer_ops.get(operation, 'auto')
    
    def _in_range(self, idx: int, range_str: str) -> bool:
        """Check if index is in range like 'layer_0_3'"""
        if not range_str.startswith('layer_'):
            return False
        parts = range_str.replace('layer_', '').split('_')
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return start <= idx <= end
        return False
    
    def should_enable_pipelining(self) -> bool:
        """Check if pipelining is enabled"""
        return self.config.get('global', {}).get('enable_pipelining', True)
    
    def get_micro_batch_size(self) -> int:
        """Get configured micro-batch size"""
        return self.config.get('global', {}).get('micro_batch_size', 4)
    
    def get_bandwidth_config(self) -> Dict:
        """Get bandwidth control configuration"""
        return self.config.get('bandwidth_control', {})


class SchedulerClient:
    """ZeroMQ client for communicating with C++ scheduler"""
    
    def __init__(self, scheduler_address: str = "tcp://localhost:5555"):
        self.scheduler_address = scheduler_address
        self.context = zmq.Context()
        self.socket = None
        self.packet_counter = 0
        self.connected = False
        
    def connect(self, timeout_ms: int = 5000) -> bool:
        """Connect to scheduler with timeout"""
        try:
            logger.info(f"Connecting to scheduler at {self.scheduler_address}...")
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self.socket.connect(self.scheduler_address)
            
            # Health check
            if self.health_check():
                self.connected = True
                logger.info("✓ Connected to scheduler")
                return True
            else:
                logger.error("✗ Scheduler health check failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Failed to connect to scheduler: {e}")
            return False
    
    def health_check(self) -> bool:
        """Ping scheduler to verify it's alive"""
        try:
            request = {"type": "health_check"}
            self.socket.send_json(request)
            response = self.socket.recv_json()
            return response.get("status") == "ok"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def send_work_packet(self, packet: WorkPacket) -> Optional[WorkResult]:
        """Send work packet and wait for result"""
        if not self.connected:
            logger.error("Not connected to scheduler")
            return None
        
        try:
            # Serialize packet
            request = {
                "type": "work_packet",
                "data": asdict(packet)
            }
            
            # Send
            self.socket.send_json(request)
            
            # Receive result
            response = self.socket.recv_json()
            
            if response.get("status") == "success":
                result_data = response.get("result", {})
                return WorkResult(**result_data)
            else:
                logger.error(f"Work packet failed: {response.get('error')}")
                return None
                
        except zmq.Again:
            logger.error("Timeout waiting for scheduler response")
            return None
        except Exception as e:
            logger.error(f"Error sending work packet: {e}")
            return None
    
    def get_telemetry(self) -> Optional[Dict]:
        """Request current telemetry from scheduler"""
        if not self.connected:
            return None
        
        try:
            request = {"type": "get_telemetry"}
            self.socket.send_json(request)
            response = self.socket.recv_json()
            return response.get("telemetry", {})
        except Exception as e:
            logger.error(f"Error getting telemetry: {e}")
            return None
    
    def shutdown(self):
        """Shutdown scheduler and close connection"""
        if self.connected:
            try:
                request = {"type": "shutdown"}
                self.socket.send_json(request)
                response = self.socket.recv_json()
                logger.info("Scheduler shutdown requested")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        
        if self.socket:
            self.socket.close()
        self.context.term()
        self.connected = False


class SplitInferenceOrchestrator:
    """Main orchestrator for split CPU/iGPU inference"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "split_inference/configs/partition_config.yaml",
                 scheduler_address: str = "tcp://localhost:5555",
                 use_cpu_fallback: bool = True):
        
        self.model_path = Path(model_path)
        self.config = PartitionConfig(config_path)
        self.scheduler = SchedulerClient(scheduler_address)
        self.use_cpu_fallback = use_cpu_fallback
        
        # Load tokenizer and model config
        logger.info("Loading tokenizer and config...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        self.model_config = AutoConfig.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        # CPU fallback model (lazy load)
        self.cpu_model = None
        
        logger.info(f"✓ Model config loaded: {self.model_config.model_type}")
        logger.info(f"  • Layers: {self.model_config.num_hidden_layers}")
        logger.info(f"  • Experts: {self.model_config.num_local_experts}")
        logger.info(f"  • Top-k: {self.model_config.num_experts_per_tok}")
    
    def initialize(self) -> bool:
        """Initialize connection to scheduler"""
        return self.scheduler.connect()
    
    def _load_cpu_fallback(self):
        """Lazy load CPU-only model for fallback"""
        if self.cpu_model is None:
            logger.info("Loading CPU fallback model...")
            from transformers import AutoModelForCausalLM
            self.cpu_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map="cpu"
            )
            logger.info("✓ CPU fallback model loaded")
    
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> Tuple[str, Dict]:
        """
        Generate text using split CPU/iGPU inference
        
        Returns:
            (generated_text, telemetry_metrics)
        """
        start_time = time.time()
        
        # Tokenize input
        logger.info(f"Tokenizing prompt ({len(prompt)} chars)...")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        logger.info(f"Input tokens: {input_ids.shape[1]}")
        
        # Try split inference first
        try:
            generated_ids = self._generate_split(
                input_ids, max_new_tokens, temperature, top_p
            )
        except Exception as e:
            logger.error(f"Split inference failed: {e}")
            
            if self.use_cpu_fallback:
                logger.warning("Falling back to CPU-only inference...")
                self._load_cpu_fallback()
                generated_ids = self._generate_cpu_fallback(
                    input_ids, max_new_tokens, temperature, top_p
                )
            else:
                raise
        
        # Decode output
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        
        end_time = time.time()
        
        # Collect telemetry
        telemetry = self.scheduler.get_telemetry() or {}
        telemetry['total_time_s'] = end_time - start_time
        telemetry['tokens_generated'] = len(generated_ids[0]) - len(input_ids[0])
        telemetry['tokens_per_second'] = telemetry['tokens_generated'] / telemetry['total_time_s']
        
        return generated_text, telemetry
    
    def _generate_split(self, 
                       input_ids: torch.Tensor,
                       max_new_tokens: int,
                       temperature: float,
                       top_p: float) -> torch.Tensor:
        """
        Generate using split CPU/iGPU inference
        
        NOTE: This is a simplified stub. Full implementation requires:
        1. Breaking down each transformer layer into work packets
        2. Orchestrating execution across CPU and iGPU
        3. Managing KV cache and hidden states
        4. Implementing sampling logic
        """
        logger.info("Starting split inference (STUB - needs full implementation)")
        
        # For now, this is a placeholder that shows the structure
        # Real implementation will:
        # - Loop over max_new_tokens
        # - For each token, execute all layers by sending work packets
        # - Accumulate results and sample next token
        
        raise NotImplementedError(
            "Split inference not yet fully implemented. "
            "Use CPU fallback for now."
        )
    
    def _generate_cpu_fallback(self,
                              input_ids: torch.Tensor,
                              max_new_tokens: int,
                              temperature: float,
                              top_p: float) -> torch.Tensor:
        """Generate using CPU-only model"""
        logger.info("Running CPU-only inference...")
        
        with torch.no_grad():
            outputs = self.cpu_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        return outputs
    
    def benchmark(self, 
                 prompts: List[str],
                 max_new_tokens: int = 50) -> Dict:
        """Run benchmark on multiple prompts"""
        logger.info(f"Running benchmark on {len(prompts)} prompts...")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Prompt {i+1}/{len(prompts)}")
            text, metrics = self.generate(prompt, max_new_tokens=max_new_tokens)
            results.append({
                'prompt': prompt[:50] + '...',
                'output': text[:100] + '...',
                'metrics': metrics
            })
        
        # Aggregate metrics
        avg_tokens_per_second = np.mean([r['metrics']['tokens_per_second'] for r in results])
        avg_time = np.mean([r['metrics']['total_time_s'] for r in results])
        
        benchmark_summary = {
            'num_prompts': len(prompts),
            'avg_tokens_per_second': avg_tokens_per_second,
            'avg_time_s': avg_time,
            'detailed_results': results
        }
        
        logger.info(f"Benchmark complete: {avg_tokens_per_second:.2f} tokens/s")
        
        return benchmark_summary
    
    def shutdown(self):
        """Shutdown orchestrator and connections"""
        logger.info("Shutting down orchestrator...")
        self.scheduler.shutdown()


def main():
    """Example usage"""
    print("="*60)
    print("SPLIT CPU/iGPU INFERENCE ORCHESTRATOR")
    print("="*60)
    
    orchestrator = SplitInferenceOrchestrator(
        model_path=".",
        config_path="split_inference/configs/partition_config.yaml"
    )
    
    # Initialize (will attempt to connect to C++ scheduler)
    print("\n🔌 Connecting to scheduler...")
    if not orchestrator.initialize():
        print("⚠️  Scheduler not available - will use CPU fallback")
    
    # Run generation
    print("\n🚀 Generating text...")
    prompt = "What is the meaning of life?"
    
    try:
        text, metrics = orchestrator.generate(prompt, max_new_tokens=50)
        
        print("\n📝 Generated text:")
        print(text)
        
        print("\n📊 Metrics:")
        for key, value in metrics.items():
            print(f"  • {key}: {value}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
