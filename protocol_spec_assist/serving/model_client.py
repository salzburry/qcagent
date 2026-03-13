"""
Model serving abstraction.
Local: vLLM with structured outputs (JSON schema constrained).
Designed so swapping to OpenAI later = change config, not code.
"""

from __future__ import annotations
import os
import json
import time
import logging
from typing import Optional, Type, TypeVar
from dataclasses import dataclass, field
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    default_base_url: str = "http://localhost:8000/v1"
    adjudicator_base_url: str = "http://localhost:8001/v1"
    api_key: str = "local"
    default_model: str = "Qwen/Qwen3-235B-A22B"            # MoE — fits on H100 80GB
    adjudicator_model: str = "Qwen/Qwen3-235B-A22B"      # same model (single-server)
    vision_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"    # scanned pages
    temperature: float = 0.0                              # deterministic extraction
    max_tokens: int = 16384
    max_retries: int = 2                                  # retry on transient failures
    timeout: float = 120.0                                # seconds


_config: Optional[ModelConfig] = None

def get_config() -> ModelConfig:
    global _config
    if _config is None:
        _config = ModelConfig(
            default_base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            adjudicator_base_url=os.environ.get(
                "ADJUDICATOR_BASE_URL",
                os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1"),
            ),
            api_key=os.environ.get("VLLM_API_KEY", "local"),
            default_model=os.environ.get("DEFAULT_MODEL", "Qwen/Qwen3-235B-A22B"),
            adjudicator_model=os.environ.get("ADJUDICATOR_MODEL", "Qwen/Qwen3-235B-A22B"),
        )
    return _config


# ── Extraction result with raw response ──────────────────────────────────────

@dataclass
class ExtractionResult:
    """Carries both parsed output and raw response for debugging."""
    parsed: object
    model_used: str
    raw_response: str
    prompt_version: str = ""


# ── Client ────────────────────────────────────────────────────────────────────

class LocalModelClient:
    """
    OpenAI-compatible client pointed at local vLLM server.
    Supports structured outputs via JSON schema.
    Maintains separate clients for default and adjudicator endpoints.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or get_config()
        self._default_client = None
        self._adjudicator_client = None

    def _get_client(self, use_adjudicator: bool = False):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        if use_adjudicator:
            if self._adjudicator_client is None:
                self._adjudicator_client = OpenAI(
                    base_url=self.config.adjudicator_base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )
            return self._adjudicator_client
        else:
            if self._default_client is None:
                self._default_client = OpenAI(
                    base_url=self.config.default_base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )
            return self._default_client

    def check_model_available(self, use_adjudicator: bool = False) -> bool:
        """Verify the configured model is served before starting a run."""
        client = self._get_client(use_adjudicator=use_adjudicator)
        model = self.config.adjudicator_model if use_adjudicator else self.config.default_model
        try:
            models = client.models.list()
            available = [m.id for m in models.data]
            if model in available:
                return True
            logger.warning(
                f"Model '{model}' not found. Available: {available}"
            )
            return False
        except Exception as e:
            logger.error(f"Cannot reach model server: {e}")
            return False

    def extract(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        use_adjudicator: bool = False,
        prompt_version: str = "0.3.0",
    ) -> ExtractionResult:
        """
        Extract structured output from LLM, schema-constrained.
        Returns ExtractionResult with parsed object, model_used, and raw response.
        Retries on transient failures. Raises on persistent failure.
        """
        client = self._get_client(use_adjudicator=use_adjudicator)
        model = self.config.adjudicator_model if use_adjudicator else self.config.default_model

        last_error = None
        for attempt in range(1 + self.config.max_retries):
            try:
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

                choice = response.choices[0]
                raw = choice.message.content
                if not raw or not raw.strip():
                    raise ValueError("Empty response content from model")

                # Detect truncated output (length finish instead of stop)
                if getattr(choice, "finish_reason", None) == "length":
                    raise ValueError(
                        f"Response truncated (finish_reason=length, "
                        f"{len(raw)} chars). Increase max_tokens."
                    )

                parsed = schema.model_validate_json(raw)
                return ExtractionResult(
                    parsed=parsed,
                    model_used=model,
                    raw_response=raw,
                    prompt_version=prompt_version,
                )

            except (ValueError, json.JSONDecodeError) as e:
                # Malformed JSON or empty content — retry
                last_error = e
                logger.warning(
                    f"[extract] Attempt {attempt + 1}: parse error: {e}"
                )
                if attempt < self.config.max_retries:
                    time.sleep(1 * (attempt + 1))
                continue

            except Exception as e:
                # Connection/timeout errors — retry
                last_error = e
                logger.warning(
                    f"[extract] Attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                if attempt < self.config.max_retries:
                    time.sleep(2 * (attempt + 1))
                continue

        raise RuntimeError(
            f"Extraction failed after {1 + self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        use_adjudicator: bool = False,
    ) -> str:
        """Freeform chat — used only for protocol Q&A UI, not extraction."""
        client = self._get_client(use_adjudicator=use_adjudicator)
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
    "qwen3-235b-a22b": """
# Start vLLM server (run in terminal before using the pipeline):
# Qwen3-235B-A22B is a Mixture-of-Experts model (~22B active params, fits on H100 80GB)
vllm serve Qwen/Qwen3-235B-A22B \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --max-model-len 32768 \\
    --enable-prefix-caching \\
    --dtype auto \\
    --gpu-memory-utilization 0.90 \\
    --tensor-parallel-size 1
""",
}

def print_startup_instructions():
    print(VLLM_START_COMMANDS["qwen3-235b-a22b"])
