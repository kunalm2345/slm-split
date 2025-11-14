#!/usr/bin/env python3
"""
ONNX Export and Analysis for Phi-tiny-MoE
Exports model to ONNX and analyzes supported/unsupported operations
"""

import torch
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
from collections import defaultdict


class ModelAnalyzer:
    """Analyze Phi-tiny-MoE architecture and ONNX export feasibility"""
    
    def __init__(self, model_path: str = "."):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print("Loading Phi-tiny-MoE model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu"
        )
        self.config = self.model.config
        print(f"✓ Model loaded: {self.config.model_type}")
        
    def analyze_architecture(self) -> Dict:
        """Extract and document model architecture details"""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE ANALYSIS")
        print("="*60)
        
        arch_info = {
            "model_type": self.config.model_type,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "intermediate_size": self.config.intermediate_size,
            "vocab_size": self.config.vocab_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "moe_config": {
                "num_local_experts": self.config.num_local_experts,
                "num_experts_per_tok": self.config.num_experts_per_tok,
                "router_jitter_noise": self.config.router_jitter_noise,
                "input_jitter_noise": self.config.input_jitter_noise,
            },
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        # Calculate activated parameters (top-k routing)
        expert_params = sum(
            p.numel() for name, p in self.model.named_parameters() 
            if 'experts' in name
        ) / self.config.num_local_experts  # Per expert
        
        non_expert_params = sum(
            p.numel() for name, p in self.model.named_parameters() 
            if 'experts' not in name
        )
        
        activated_params = non_expert_params + (expert_params * self.config.num_experts_per_tok)
        arch_info["activated_params"] = int(activated_params)
        
        print(f"\n📊 Model Configuration:")
        print(f"  • Type: {arch_info['model_type']}")
        print(f"  • Layers: {arch_info['num_hidden_layers']}")
        print(f"  • Hidden size: {arch_info['hidden_size']}")
        print(f"  • Attention heads: {arch_info['num_attention_heads']} (KV: {arch_info['num_key_value_heads']})")
        print(f"  • Vocab size: {arch_info['vocab_size']}")
        print(f"  • Context length: {arch_info['max_position_embeddings']}")
        
        print(f"\n🔀 MoE Configuration:")
        print(f"  • Total experts: {arch_info['moe_config']['num_local_experts']}")
        print(f"  • Experts per token: {arch_info['moe_config']['num_experts_per_tok']}")
        print(f"  • Router jitter: {arch_info['moe_config']['router_jitter_noise']}")
        print(f"  • Input jitter: {arch_info['moe_config']['input_jitter_noise']}")
        
        print(f"\n💾 Parameters:")
        print(f"  • Total: {arch_info['total_params'] / 1e9:.2f}B")
        print(f"  • Activated (per forward): {arch_info['activated_params'] / 1e9:.2f}B")
        print(f"  • Expert params (each): {expert_params / 1e6:.1f}M")
        
        return arch_info
    
    def analyze_module_structure(self) -> Dict:
        """Analyze model module structure for partitioning decisions"""
        print("\n" + "="*60)
        print("MODULE STRUCTURE ANALYSIS")
        print("="*60)
        
        module_info = defaultdict(list)
        
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if module_type not in ['ModuleList', 'Sequential']:
                module_info[module_type].append(name)
        
        print("\n📦 Module Types:")
        for module_type, instances in sorted(module_info.items()):
            print(f"  • {module_type}: {len(instances)} instances")
            if 'MoE' in module_type or 'Expert' in module_type:
                print(f"    → {instances[:3]}..." if len(instances) > 3 else f"    → {instances}")
        
        # Identify MoE-specific components
        moe_components = {
            "gating_layers": [],
            "expert_layers": [],
            "routing_modules": [],
        }
        
        for name, module in self.model.named_modules():
            if 'gate' in name.lower():
                moe_components["gating_layers"].append(name)
            elif 'expert' in name.lower():
                moe_components["expert_layers"].append(name)
            elif 'moe' in name.lower() and 'sparse' in name.lower():
                moe_components["routing_modules"].append(name)
        
        print(f"\n🎯 MoE-Specific Components:")
        print(f"  • Gating layers: {len(moe_components['gating_layers'])}")
        print(f"  • Expert layers: {len(moe_components['expert_layers'])}")
        print(f"  • Routing modules: {len(moe_components['routing_modules'])}")
        
        return dict(module_info), moe_components
    
    def attempt_onnx_export(self, output_path: str = "phi_tiny_moe.onnx") -> Tuple[bool, List[str]]:
        """Attempt to export model to ONNX and identify unsupported operations"""
        print("\n" + "="*60)
        print("ONNX EXPORT ATTEMPT")
        print("="*60)
        
        # Create dummy input
        dummy_input_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        attention_mask = torch.ones(1, 8, dtype=torch.long)
        
        print(f"\n📝 Attempting export with input shape: {dummy_input_ids.shape}")
        
        unsupported_ops = []
        export_success = False
        
        try:
            # Attempt export
            torch.onnx.export(
                self.model,
                (dummy_input_ids,),
                output_path,
                export_params=True,
                opset_version=17,  # Try modern opset
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                },
                verbose=False
            )
            
            # Verify exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            print(f"✓ ONNX export successful: {output_path}")
            print(f"  • Opset version: 17")
            print(f"  • Model size: {Path(output_path).stat().st_size / 1e6:.1f} MB")
            
            # Analyze ONNX operators
            ops = set()
            for node in onnx_model.graph.node:
                ops.add(node.op_type)
            
            print(f"\n📊 ONNX Operators Used ({len(ops)} unique):")
            for op in sorted(ops):
                print(f"  • {op}")
            
            # Check for custom/unsupported ops
            known_problematic_ops = [
                'ScatterND', 'GatherND', 'TopK', 'NonMaxSuppression',
                'Loop', 'If', 'Scan'  # Control flow ops
            ]
            
            problematic = ops.intersection(set(known_problematic_ops))
            if problematic:
                print(f"\n⚠️  Potentially problematic operators for OpenVINO:")
                for op in problematic:
                    print(f"  • {op}")
                    unsupported_ops.append(op)
            
            export_success = True
            
        except Exception as e:
            print(f"✗ ONNX export failed: {str(e)}")
            print(f"\n🔍 Error analysis:")
            error_msg = str(e)
            
            # Parse error for specific unsupported operations
            if "aten::" in error_msg:
                import re
                aten_ops = re.findall(r'aten::(\w+)', error_msg)
                unsupported_ops.extend(aten_ops)
                print(f"  • Unsupported ATen operations: {', '.join(set(aten_ops))}")
            
            if "index_add_" in error_msg or "scatter" in error_msg.lower():
                print(f"  • Issue with scatter/gather operations (MoE routing likely)")
                unsupported_ops.append("scatter/gather (MoE routing)")
            
            if "one_hot" in error_msg:
                print(f"  • Issue with one-hot encoding (expert masking)")
                unsupported_ops.append("one_hot (expert masking)")
        
        return export_success, unsupported_ops
    
    def generate_recommendations(self, export_success: bool, unsupported_ops: List[str],
                                 moe_components: Dict) -> Dict:
        """Generate recommendations for split CPU/iGPU implementation"""
        print("\n" + "="*60)
        print("IMPLEMENTATION RECOMMENDATIONS")
        print("="*60)
        
        recommendations = {
            "onnx_export_success": export_success,
            "unsupported_operations": unsupported_ops,
            "strategy": {},
            "device_partitioning": {},
            "custom_kernels_needed": []
        }
        
        if export_success:
            print("\n✓ ONNX export successful!")
            print("\n📋 Recommended Strategy: Hybrid ONNX + Custom Kernels")
            recommendations["strategy"] = {
                "approach": "hybrid",
                "description": "Use ONNX Runtime + OpenVINO EP for standard ops, custom SYCL for MoE routing"
            }
            
            if unsupported_ops:
                print(f"\n⚠️  {len(unsupported_ops)} operations may need custom implementation:")
                for op in unsupported_ops:
                    print(f"  • {op}")
                    recommendations["custom_kernels_needed"].append(op)
        else:
            print("\n✗ ONNX export failed - full custom implementation needed")
            print("\n📋 Recommended Strategy: Full Custom Implementation")
            recommendations["strategy"] = {
                "approach": "full_custom",
                "description": "Implement all MoE operations with custom SYCL kernels"
            }
            recommendations["custom_kernels_needed"] = [
                "top-k routing (gating)",
                "expert selection & masking",
                "scatter/gather for token dispatch",
                "batched expert FFN",
                "routing weight application"
            ]
        
        # Device partitioning recommendations
        print("\n🔀 Recommended Device Partitioning:")
        print("\n  CPU (Light compute, high branching):")
        print("    • Tokenization & embedding lookup")
        print("    • Router logit computation (small GEMM)")
        print("    • Top-k expert selection (if < 32 tokens)")
        print("    • LayerNorm operations")
        print("    • Final LM head projection")
        
        print("\n  iGPU (Heavy compute, parallelizable):")
        print("    • Attention QKV projection (GEMM)")
        print("    • Attention scores & softmax (if large batch)")
        print("    • Expert FFN computations (batched GEMM)")
        print("    • Expert output accumulation")
        
        recommendations["device_partitioning"] = {
            "cpu": [
                "embedding", "token_gating", "layernorm", "lm_head",
                "router_logits", "expert_selection (if small batch)"
            ],
            "igpu": [
                "attention_qkv", "attention_compute", "expert_ffn_batched",
                "expert_accumulation (if large batch)"
            ]
        }
        
        # Memory bandwidth considerations
        print("\n💾 Memory Bandwidth Strategy:")
        print("    • Stage expert weights on iGPU local memory (one expert at a time)")
        print("    • Use double buffering: load expert N+1 while computing expert N")
        print("    • Group tokens by expert to maximize GEMM efficiency")
        print("    • Prevent concurrent CPU/iGPU DRAM saturation via token semaphore")
        
        recommendations["memory_strategy"] = {
            "staging": "expert_weights_local_memory",
            "pipelining": "double_buffer_compute_transfer",
            "batching": "group_tokens_by_expert",
            "bandwidth_control": "token_semaphore"
        }
        
        # Critical kernels to implement
        print("\n🔧 Critical Custom SYCL Kernels Needed:")
        critical_kernels = [
            ("moe_routing_topk", "Top-k expert selection with gating", "HIGH"),
            ("expert_token_scatter", "Scatter tokens to selected experts", "HIGH"),
            ("expert_token_gather", "Gather expert outputs with routing weights", "HIGH"),
            ("batched_expert_ffn", "Batched FFN for grouped tokens per expert", "MEDIUM"),
            ("fused_layernorm_gating", "Fused LayerNorm + gating logits", "LOW"),
        ]
        
        for kernel_name, description, priority in critical_kernels:
            print(f"    • [{priority}] {kernel_name}: {description}")
            
        recommendations["critical_kernels"] = [
            {"name": k, "description": d, "priority": p} 
            for k, d, p in critical_kernels
        ]
        
        return recommendations
    
    def save_report(self, arch_info: Dict, module_info: Dict, 
                   recommendations: Dict, output_path: str = "onnx_analysis_report.json"):
        """Save complete analysis report to JSON"""
        report = {
            "architecture": arch_info,
            "module_analysis": {k: len(v) for k, v in module_info[0].items()},
            "recommendations": recommendations,
            "timestamp": str(Path.ctime(Path(__file__)))
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved: {output_path}")
        return output_path


def main():
    print("="*60)
    print("PHI-TINY-MOE ONNX EXPORT & ANALYSIS")
    print("="*60)
    
    analyzer = ModelAnalyzer(model_path=".")
    
    try:
        # Load model
        analyzer.load_model()
        
        # Analyze architecture
        arch_info = analyzer.analyze_architecture()
        
        # Analyze module structure
        module_info, moe_components = analyzer.analyze_module_structure()
        
        # Attempt ONNX export
        export_success, unsupported_ops = analyzer.attempt_onnx_export()
        
        # Generate recommendations
        recommendations = analyzer.generate_recommendations(
            export_success, unsupported_ops, moe_components
        )
        
        # Save report
        report_path = analyzer.save_report(arch_info, (module_info, moe_components), 
                                          recommendations)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\n✓ Architecture documented")
        print(f"✓ ONNX export {'succeeded' if export_success else 'failed (expected)'}")
        print(f"✓ Recommendations generated")
        print(f"✓ Report saved to {report_path}")
        
        print("\n📝 Next Steps:")
        print("  1. Review onnx_analysis_report.json")
        print("  2. Set up oneAPI/SYCL development environment")
        print("  3. Implement custom MoE routing SYCL kernel")
        print("  4. Build C++ scheduler with IPC")
        print("  5. Integrate vendor libraries (oneDNN, ONNX Runtime)")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
