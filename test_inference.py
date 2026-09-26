"""
FinClaw Local Inference Script
==============================
Once you have downloaded your .gguf file to this folder, run this script to test responses!
Requirements:
    pip install llama-cpp-python
"""

import sys
import glob

try:
    from llama_cpp import Llama
except ImportError:
    print("Please install llama-cpp-python first:")
    print("    pip install llama-cpp-python")
    sys.exit(1)

# Find GGUF file in current folder
gguf_files = glob.glob("*.gguf")
if not gguf_files:
    print("Error: No .gguf model found in this folder.")
    print("Please download your trained GGUF file from Google Drive into this directory:")
    print("    d:\\Documents\\finclaw\\")
    sys.exit(1)

model_path = gguf_files[0]
print(f"Loading model: {model_path} ...")

# n_gpu_layers: Offloads ~24 layers to your 4GB RTX 3050 GPU, rest to 16GB system RAM
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=24,
    verbose=False
)

def chat_with_finclaw(prompt: str, system_prompt: str = "You are FinClaw, a specialized financial analysis AI."):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print("\n--- Testing FinClaw Model ---")
    sample_query = "What is the key difference between operating margin and net profit margin?"
    print(f"User: {sample_query}\n")
    reply = chat_with_finclaw(sample_query)
    print(f"FinClaw: {reply}")
