#!/usr/bin/env python3
"""
Quick test of orchestrator with fixes
"""

import sys
sys.path.insert(0, 'split_inference/python')

from orchestrator import SplitInferenceOrchestrator

print("="*60)
print("QUICK ORCHESTRATOR TEST")
print("="*60)

orchestrator = SplitInferenceOrchestrator(
    model_path=".",
    config_path="split_inference/configs/partition_config.yaml",
    telemetry_dir="./telemetry"
)

# Initialize
print("\n🔌 Connecting to scheduler...")
if not orchestrator.initialize():
    print("⚠️  Scheduler not available - will use CPU fallback")
else:
    print("✓ Connected to scheduler")

# Test with very short generation
print("\n🚀 Generating 5 tokens...")
prompt = "Hello"

try:
    text, metrics = orchestrator.generate(prompt, max_new_tokens=5)
    
    print("\n✅ Success!")
    print(f"📝 Generated text: {text[:100]}")
    print(f"📊 Tokens/s: {metrics.get('tokens_per_second', 0):.2f}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    orchestrator.shutdown()
