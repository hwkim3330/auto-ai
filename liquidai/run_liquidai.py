#!/usr/bin/env python3
"""
LiquidAI LFM2 Model Runner
--------------------------
LiquidAI의 LFM2-1.2B 모델을 테스트합니다.
Liquid Foundation Models - 효율적인 온디바이스 AI
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    print("=" * 60)
    print("LiquidAI LFM2-1.2B Model Test")
    print("=" * 60)

    # Model selection - LFM2-1.2B is good balance of speed/quality
    model_id = "LiquidAI/LFM2-1.2B"

    print(f"\n📥 Loading model: {model_id}")
    print("   (First run will download ~2.4GB)")

    start_time = time.time()

    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load model - force CPU (GTX 1050 Ti CUDA 6.1 not supported by PyTorch)
    print("🧠 Loading model...")
    device = "cpu"  # Force CPU - older GPU not supported
    print(f"   Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = model.to(device)

    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f}s")

    # Test prompts
    prompts = [
        "What is Time-Sensitive Networking (TSN)?",
        "Write a Python function to calculate factorial:",
        "오늘 서울의 날씨를 알려주세요."  # Korean test
    ]

    print("\n" + "=" * 60)
    print("🧪 Running inference tests")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"📝 Prompt: {prompt}")

        start_time = time.time()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - start_time

        print(f"🤖 Response:\n{response}")
        print(f"⏱️  Generation time: {gen_time:.2f}s")

    print("\n" + "=" * 60)
    print("✅ LiquidAI LFM2-1.2B Test Complete!")
    print("=" * 60)

    # Model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Parameters: {param_count / 1e9:.2f}B")
    print(f"   Device: {device}")
    print(f"   Dtype: {next(model.parameters()).dtype}")

if __name__ == "__main__":
    main()
