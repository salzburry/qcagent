"""
Model serving abstraction.
Local: vLLM with structured outputs (JSON schema constrained).
Designed so swapping to OpenAI later = change config, not code.

Default model: Qwen3-14B on A100 40GB via vLLM.
Override with DEFAULT_MODEL / ADJUDICATOR_MODEL env vars.
"""

from __future__ import annotations
import copy
import os
import json
import re
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
    default_model: str = "Qwen/Qwen3-14B"
    adjudicator_model: str = "Qwen/Qwen3-14B"
    temperature: float = 0.0                              # deterministic extraction
    max_tokens: int = 4096                                 # headroom for multi-row extractions
                                                          # (eligibility, data prep, censoring)
    max_retries: int = 2                                  # retry on transient failures
    timeout: float = 300.0                                # seconds (5 min — 14B model needs headroom)


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
            default_model=os.environ.get("DEFAULT_MODEL", "Qwen/Qwen3-14B"),
            adjudicator_model=os.environ.get("ADJUDICATOR_MODEL", "Qwen/Qwen3-14B"),
        )
        logger.info(
            f"Model config — base={_config.default_model}, "
            f"adj={_config.adjudicator_model}"
        )
    return _config


# ── Schema flattening for vLLM compatibility ─────────────────────────────────

def _flatten_schema(schema: dict) -> dict:
    """Flatten a Pydantic JSON schema for vLLM/xgrammar compatibility.

    vLLM's xgrammar backend silently mishandles schemas that use:
      - $ref / $defs (nested type references)
      - anyOf (from Optional fields)

    This inlines all $ref references and converts anyOf[type, null]
    to just the type (xgrammar doesn't support nullable unions).

    The result is a self-contained schema with no $ref, $defs, or anyOf,
    which xgrammar can constrain correctly.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def _resolve(node):
        """Recursively resolve $ref and simplify anyOf."""
        if not isinstance(node, dict):
            return node

        # Resolve $ref by inlining the definition
        if "$ref" in node:
            ref_path = node["$ref"]  # e.g. "#/$defs/CandidateExtraction"
            ref_name = ref_path.split("/")[-1]
            if ref_name in defs:
                resolved = copy.deepcopy(defs[ref_name])
                # Merge any extra keys (like "description") from the ref site
                for k, v in node.items():
                    if k != "$ref":
                        resolved.setdefault(k, v)
                return _resolve(resolved)
            return node

        # Simplify anyOf[{type: X}, {type: null}] → {type: X}
        # This is Pydantic's encoding of Optional[X] — xgrammar can't handle it
        if "anyOf" in node:
            variants = node["anyOf"]
            non_null = [v for v in variants if v != {"type": "null"}]
            if len(non_null) == 1:
                # Optional[X] → just X (drop the null variant)
                simplified = copy.deepcopy(non_null[0])
                for k, v in node.items():
                    if k != "anyOf":
                        simplified.setdefault(k, v)
                return _resolve(simplified)
            # Multi-type union — flatten each variant
            node["anyOf"] = [_resolve(v) for v in variants]
            return node

        # Recurse into properties
        if "properties" in node:
            node["properties"] = {
                k: _resolve(v) for k, v in node["properties"].items()
            }

        # Recurse into array items
        if "items" in node:
            node["items"] = _resolve(node["items"])

        # Recurse into additionalProperties
        if "additionalProperties" in node and isinstance(node["additionalProperties"], dict):
            node["additionalProperties"] = _resolve(node["additionalProperties"])

        return node

    result = _resolve(schema)

    # Clean up: remove leftover $defs if any
    result.pop("$defs", None)

    return result


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
        max_tokens: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Extract structured output from LLM, schema-constrained.
        Returns ExtractionResult with parsed object, model_used, and raw response.
        Retries on transient failures. Raises on persistent failure.

        Schema handling:
            Pydantic schemas are flattened to remove $ref/$defs/anyOf before
            sending to vLLM. This is required because vLLM's xgrammar backend
            silently mishandles these features, producing empty/garbage output
            instead of raising an error.

        Args:
            max_tokens: Per-call override. Falls back to config default if None.
        """
        client = self._get_client(use_adjudicator=use_adjudicator)
        model = self.config.adjudicator_model if use_adjudicator else self.config.default_model
        tokens = max_tokens or self.config.max_tokens

        # Flatten schema for vLLM/xgrammar compatibility
        raw_schema = schema.model_json_schema()
        flat_schema = _flatten_schema(raw_schema)

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
                    max_tokens=tokens,
                    extra_body={
                        "guided_json": flat_schema,
                    },
                )

                choice = response.choices[0]
                raw = choice.message.content

                # Strip <think>...</think> blocks (Qwen3 reasoning traces)
                if raw and "<think>" in raw:
                    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

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
