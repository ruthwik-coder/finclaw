"""
FinClaw Model Engine — Fast Local GPU Inference with Multi-turn Memory & Token Streaming
========================================================================================
- Loads the fine-tuned FinClaw LoRA adapter on top of Llama-3.2-1B-Instruct in 4-bit BF16.
- Uses TextIteratorStreamer for sub-second token streaming.
- Supports multi-turn conversation memory.
"""

import os
import sys
import json
from threading import Thread
from typing import Generator, List, Dict, Any, Optional

os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache", "huggingface"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "finclaw_finetuned_adapter"))
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

class FinClawEngine:
    _instance = None

    def __init__(self):
        print(f"[Engine] Loading base model: {BASE_MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        )

        print(f"[Engine] Attaching FinClaw LoRA adapter from {ADAPTER_DIR} ...")
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        self.model.eval()
        print("[Engine] FinClaw model successfully loaded and ready on GPU!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def format_system_prompt(self, ledger: Dict[str, Any]) -> str:
        """
        Creates the exact system prompt schema used during fine-tuning.
        """
        return f"""You are FinClaw, an emotionally-aware financial coaching bot.
Each sample links a psychological/emotional scenario to an exact financial ledger state and a multi-turn coaching conversation.

CURRENT USER FINANCIAL LEDGER:
{json.dumps(ledger, indent=2)}

Provide empathetic emotional validation, analyze their financial ledger state, and guide their financial decision."""

    def stream_chat(
        self,
        messages_history: List[Dict[str, str]],
        ledger: Dict[str, Any],
        max_new_tokens: int = 300,
        temperature: float = 0.3
    ) -> Generator[str, None, None]:
        """
        Streams response tokens in real-time, preserving multi-turn conversation memory.
        messages_history: list of dicts with 'role' ('user' | 'assistant') and 'content'.
        """
        system_content = self.format_system_prompt(ledger)

        # Build formatted prompt: System message + prior conversation memory
        full_conversation = [{"role": "system", "content": system_content}]
        
        # Include past conversation turns for memory (last 6 messages max to stay tight and fast)
        recent_history = messages_history[-6:] if len(messages_history) > 6 else messages_history
        for msg in recent_history:
            full_conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            full_conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        # Yield each token as it is generated in real-time
        for token_text in streamer:
            yield token_text

        thread.join()
