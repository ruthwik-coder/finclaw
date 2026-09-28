"""
FinClaw Local Training Script
=============================
Optimized for Windows PCs with NVIDIA RTX 3050 (4 GB VRAM).
Uses Hugging Face Transformers, PEFT (QLoRA 4-bit), and TRL.

Usage:
    d:\\Documents\\finclaw\\.venv\\Scripts\\python.exe train_local.py
"""

import os
import sys

# Ensure Hugging Face downloads to D: drive instead of filling C: drive
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "huggingface"))
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    print("=" * 60)
    print(" FinClaw Local Fine-Tuning ")
    print("=" * 60)

    # 1. Verify CUDA
    print(f"PyTorch Version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("\n[!] Error: CUDA is not available in this environment.")
        sys.exit(1)
    else:
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)")

    # 2. Select Model (Llama-3.2-1B fits perfectly on 4GB VRAM)
    model_id = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    print(f"\nLoading model: {model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Prepare model for 4-bit LoRA training
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 3. Load Local Dataset
    dataset_path = "finclaw_colab_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found in current folder!")
        sys.exit(1)

    print(f"\nLoading dataset from {dataset_path} ...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def formatting_prompts_func(examples):
        convs = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convs]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Total dataset examples: {len(dataset)}")

    # 4. Training Arguments with SFTConfig
    output_dir = "./finclaw_local_checkpoints"

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=1,       # Safe for 4GB VRAM
        gradient_accumulation_steps=4,      # Effective batch size = 4
        warmup_steps=15,
        max_steps=350,                      # 350 steps (~2-3 epochs over 758 examples)
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=25,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=100,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    print("\nStarting local training on RTX 3050...")
    trainer.train()

    # 5. Save Model Locally
    save_path = "./finclaw_finetuned_adapter"
    print(f"\nSaving model adapter to: {save_path} ...")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("=" * 60)
    print(" Done! Model successfully fine-tuned and saved locally.")
    print("=" * 60)

if __name__ == "__main__":
    main()
