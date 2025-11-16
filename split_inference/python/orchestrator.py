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
import sys
from transformers import AutoTokenizer, AutoConfig

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import telemetry logger
from telemetry_logger import TelemetryLogger

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
                 use_cpu_fallback: bool = True,
                 telemetry_dir: str = "./telemetry"):
        
        self.model_path = Path(model_path)
        self.config = PartitionConfig(config_path)
        self.scheduler = SchedulerClient(scheduler_address)
        self.use_cpu_fallback = use_cpu_fallback
        
        # Initialize telemetry logger
        self.telemetry = TelemetryLogger(output_dir=telemetry_dir)
        
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
    
    def _send_and_log_packet(self, packet: WorkPacket) -> WorkResult:
        """
        Send work packet to scheduler and log telemetry
        Returns result or raises RuntimeError on failure
        """
        result = self.scheduler.send_work_packet(packet)
        
        if result:
            # Log telemetry
            self.telemetry.log_work_result(
                packet_id=packet.packet_id,
                layer_idx=packet.layer_idx,
                operation=packet.operation,
                device_target=packet.device_target,
                result_device=result.device_used or packet.device_target,
                duration_ms=result.actual_duration_ms,
                success=result.success,
                memory_used_mb=result.memory_used_mb,
                error_message=result.error_message
            )
            
            if not result.success:
                raise RuntimeError(f"{packet.operation} failed at layer {packet.layer_idx}: {result.error_message}")
        else:
            # No response from scheduler
            self.telemetry.log_work_result(
                packet_id=packet.packet_id,
                layer_idx=packet.layer_idx,
                operation=packet.operation,
                device_target=packet.device_target,
                result_device="unknown",
                duration_ms=0.0,
                success=False,
                error_message="No response from scheduler"
            )
            raise RuntimeError(f"{packet.operation} failed: No response from scheduler")
        
        return result
    
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
        
        prompt_tokens = input_ids.shape[1]
        logger.info(f"Input tokens: {prompt_tokens}")
        
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
        
        # Finalize telemetry
        tokens_generated = len(generated_ids[0]) - len(input_ids[0])
        self.telemetry.finalize(
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens
        )
        
        # Collect telemetry
        telemetry = self.telemetry.get_summary_stats()
        telemetry['total_time_s'] = end_time - start_time
        telemetry['tokens_generated'] = tokens_generated
        telemetry['tokens_per_second'] = tokens_generated / (end_time - start_time)
        
        return generated_text, telemetry
    
    def _generate_split(self, 
                       input_ids: torch.Tensor,
                       max_new_tokens: int,
                       temperature: float,
                       top_p: float) -> torch.Tensor:
        """
        Generate using split CPU/iGPU inference
        
        Implements autoregressive generation by:
        1. Processing input through all transformer layers
        2. Sending work packets to scheduler for each operation
        3. Managing KV cache across CPU/iGPU boundary
        4. Sampling next token and repeating
        """
        logger.info("Starting split CPU/iGPU inference...")
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize generated sequence with input
        generated = input_ids.clone()
        
        # Model config
        num_layers = self.model_config.num_hidden_layers
        hidden_size = self.model_config.hidden_size
        vocab_size = self.model_config.vocab_size
        
        # Simple KV cache (simplified - real implementation needs proper management)
        # For now, we'll process without KV cache (slower but simpler)
        
        packet_id = 0
        
        for token_idx in range(max_new_tokens):
            logger.info(f"Generating token {token_idx + 1}/{max_new_tokens}")
            
            # Current sequence
            current_seq = generated
            current_seq_len = current_seq.shape[1]
            
            # Step 1: Embedding (CPU)
            logger.debug(f"  Step 1: Embedding lookup (CPU)")
            packet_id += 1
            embedding_packet = WorkPacket(
                packet_id=packet_id,
                layer_idx=-1,  # Pre-layer operation
                operation="embedding",
                device_target="cpu",
                input_shape=list(current_seq.shape),
                input_dtype="int64",
                params={
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "token_ids": current_seq.tolist()
                },
                priority=10,
                can_pipeline=False,
                memory_requirement_mb=float(current_seq.numel() * hidden_size * 4 / 1024**2),
                estimated_duration_ms=0.1
            )
            
            result = self._send_and_log_packet(embedding_packet)
            
            # Hidden states shape: [batch_size, seq_len, hidden_size]
            hidden_states_shape = [batch_size, current_seq_len, hidden_size]
            
            # Step 2: Process through transformer layers
            for layer_idx in range(num_layers):
                logger.debug(f"  Layer {layer_idx}/{num_layers}")
                
                # 2a. LayerNorm (CPU)
                packet_id += 1
                layernorm_packet = WorkPacket(
                    packet_id=packet_id,
                    layer_idx=layer_idx,
                    operation="attention_layernorm",
                    device_target=self.config.get_device_for_operation(layer_idx, "attention_layernorm"),
                    input_shape=hidden_states_shape,
                    input_dtype="float32",
                    params={"eps": self.model_config.rms_norm_eps},
                    priority=5,
                    can_pipeline=True,
                    memory_requirement_mb=float(np.prod(hidden_states_shape) * 4 / 1024**2),
                    estimated_duration_ms=0.5
                )
                
                result = self._send_and_log_packet(layernorm_packet)
                
                # 2b. Attention QKV projection (iGPU if configured)
                packet_id += 1
                attention_packet = WorkPacket(
                    packet_id=packet_id,
                    layer_idx=layer_idx,
                    operation="attention_qkv_proj",
                    device_target=self.config.get_device_for_operation(layer_idx, "attention_qkv_proj"),
                    input_shape=hidden_states_shape,
                    input_dtype="float32",
                    params={
                        "num_heads": self.model_config.num_attention_heads,
                        "num_kv_heads": self.model_config.num_key_value_heads,
                        "head_dim": hidden_size // self.model_config.num_attention_heads
                    },
                    priority=8,
                    can_pipeline=True,
                    memory_requirement_mb=float(np.prod(hidden_states_shape) * 3 * 4 / 1024**2),
                    estimated_duration_ms=2.0
                )
                
                result = self._send_and_log_packet(attention_packet)
                
                # 2c. MoE Router logits (CPU)
                packet_id += 1
                router_packet = WorkPacket(
                    packet_id=packet_id,
                    layer_idx=layer_idx,
                    operation="router_logits",
                    device_target=self.config.get_device_for_operation(layer_idx, "router_logits"),
                    input_shape=hidden_states_shape,
                    input_dtype="float32",
                    params={
                        "num_experts": self.model_config.num_local_experts
                    },
                    priority=9,
                    can_pipeline=False,  # Need results for next step
                    memory_requirement_mb=float(batch_size * current_seq_len * self.model_config.num_local_experts * 4 / 1024**2),
                    estimated_duration_ms=0.5
                )
                
                result = self._send_and_log_packet(router_packet)
                
                # 2d. Expert selection (CPU - fast top-k)
                packet_id += 1
                expert_select_packet = WorkPacket(
                    packet_id=packet_id,
                    layer_idx=layer_idx,
                    operation="expert_selection",
                    device_target="cpu",
                    input_shape=[batch_size, current_seq_len, self.model_config.num_local_experts],
                    input_dtype="float32",
                    params={
                        "top_k": self.model_config.num_experts_per_tok
                    },
                    priority=10,
                    can_pipeline=False,
                    memory_requirement_mb=1.0,
                    estimated_duration_ms=0.2
                )
                
                result = self._send_and_log_packet(expert_select_packet)
                
                # 2e. Expert FFN computation (iGPU - heavy GEMM)
                packet_id += 1
                expert_ffn_packet = WorkPacket(
                    packet_id=packet_id,
                    layer_idx=layer_idx,
                    operation="expert_ffn",
                    device_target=self.config.get_device_for_operation(layer_idx, "expert_ffn"),
                    input_shape=hidden_states_shape,
                    input_dtype="float32",
                    params={
                        "num_experts": self.model_config.num_local_experts,
                        "experts_per_tok": self.model_config.num_experts_per_tok,
                        "intermediate_size": self.model_config.intermediate_size
                    },
                    priority=7,
                    can_pipeline=True,
                    memory_requirement_mb=float(
                        batch_size * current_seq_len * 
                        self.model_config.intermediate_size * 4 / 1024**2
                    ),
                    estimated_duration_ms=5.0
                )
                
                result = self._send_and_log_packet(expert_ffn_packet)
                
                logger.debug(f"    ✓ Layer {layer_idx} complete")
            
            # Step 3: Final LayerNorm (CPU)
            packet_id += 1
            final_norm_packet = WorkPacket(
                packet_id=packet_id,
                layer_idx=num_layers,
                operation="final_layernorm",
                device_target="cpu",
                input_shape=hidden_states_shape,
                input_dtype="float32",
                params={"eps": self.model_config.rms_norm_eps},
                priority=10,
                can_pipeline=False,
                memory_requirement_mb=float(np.prod(hidden_states_shape) * 4 / 1024**2),
                estimated_duration_ms=0.5
            )
            
            result = self._send_and_log_packet(final_norm_packet)
            
            # Step 4: LM head projection (CPU)
            packet_id += 1
            lm_head_packet = WorkPacket(
                packet_id=packet_id,
                layer_idx=num_layers + 1,
                operation="lm_head",
                device_target="cpu",
                input_shape=[batch_size, current_seq_len, hidden_size],
                input_dtype="float32",
                params={
                    "vocab_size": vocab_size
                },
                priority=10,
                can_pipeline=False,
                memory_requirement_mb=float(batch_size * current_seq_len * vocab_size * 4 / 1024**2),
                estimated_duration_ms=1.0
            )
            
            result = self._send_and_log_packet(lm_head_packet)
            
            # Step 5: Sample next token (local CPU operation)
            # In real implementation, we'd get logits from result and sample
            # For now, we'll use CPU fallback for actual token generation
            logger.debug("  Step 5: Sampling next token (using CPU fallback for now)")
            
            # TEMPORARY: Use CPU model for actual sampling
            # TODO: Implement proper logits transfer and sampling
            if self.cpu_model is None:
                self._load_cpu_fallback()
            
            with torch.no_grad():
                outputs = self.cpu_model(current_seq, use_cache=False)
                logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            logger.info(f"  ✓ Token {token_idx + 1} generated: {self.tokenizer.decode(next_token[0])}")
            
            # Check for EOS token
            if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                logger.info("  EOS token generated, stopping")
                break
        
        logger.info(f"✓ Split inference complete: generated {generated.shape[1] - seq_len} tokens")
        return generated
    
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
