"""
Model serving abstraction.
Local: vLLM with structured outputs (JSON schema constrained).
Designed so swapping to OpenAI later = change config, not code.
"""

from __future__ import annotations
import os
import json
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ── Config ────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "local"
    default_model: str = "Qwen/Qwen3-8B-Instruct"        # laptop default
    adjudicator_model: str = "Qwen/Qwen3-30B-A3B-Instruct"  # harder cases
    vision_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"    # scanned pages
    temperature: float = 0.0                              # deterministic extraction
    max_tokens: int = 2048


_config: Optional[ModelConfig] = None

def get_config() -> ModelConfig:
    global _config
    if _config is None:
        _config = ModelConfig(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            default_model=os.environ.get("DEFAULT_MODEL", "Qwen/Qwen3-8B-Instruct"),
            adjudicator_model=os.environ.get("ADJUDICATOR_MODEL", "Qwen/Qwen3-30B-A3B-Instruct"),
        )
    return _config


# ── Client ────────────────────────────────────────────────────────────────────

class LocalModelClient:
    """
    OpenAI-compatible client pointed at local vLLM server.
    Supports structured outputs via JSON schema.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or get_config()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                )
            except ImportError:
                raise ImportError("pip install openai")
        return self._client

    def extract(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        use_adjudicator: bool = False,
        prompt_version: str = "0.1.0",
    ) -> tuple[T, str]:
        """
        Extract structured output from LLM, schema-constrained.
        Returns (parsed_object, model_used).
        """
        client = self._get_client()
        model = self.config.adjudicator_model if use_adjudicator else self.config.default_model

        # vLLM structured outputs: pass JSON schema as response_format
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        )

        raw = response.choices[0].message.content
        parsed = schema.model_validate_json(raw)
        return parsed, model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        use_adjudicator: bool = False,
    ) -> str:
        """Freeform chat — used only for protocol Q&A UI, not extraction."""
        client = self._get_client()
        model = self.config.adjudicator_model if use_adjudicator else self.config.default_model

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ── vLLM startup helper ───────────────────────────────────────────────────────

VLLM_START_COMMANDS = {
    "qwen3-8b": """
# Start vLLM server (run in terminal before using the pipeline):
vllm serve Qwen/Qwen3-8B-Instruct \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --max-model-len 32768 \\
    --enable-prefix-caching \\
    --dtype auto

# For MoE 30B model (needs more VRAM):
vllm serve Qwen/Qwen3-30B-A3B-Instruct \\
    --host 0.0.0.0 \\
    --port 8001 \\
    --max-model-len 32768 \\
    --enable-prefix-caching \\
    --dtype auto
""",
    "ollama_fallback": """
# Ollama fallback (if vLLM setup is complex):
ollama serve
ollama pull qwen2.5:14b
# Then set: VLLM_BASE_URL=http://localhost:11434/v1
# Note: Ollama does not support strict JSON schema — less reliable structured outputs
""",
}

def print_startup_instructions():
    print(VLLM_START_COMMANDS["qwen3-8b"])
