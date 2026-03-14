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
from pydantic import BaseModel, ValidationError

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


def _generate_example(schema: Type[BaseModel]) -> str:
    """Generate a minimal JSON example from a Pydantic schema.

    This gives the model a concrete reference for exact field names and types,
    preventing the #1 failure mode: wrong/hallucinated field names.
    """
    def _example_value(field_name: str, field_info) -> object:
        annotation = field_info.annotation
        if annotation is None:
            return None

        # Handle Optional (typing.Optional[X] -> X | None)
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())

        # Optional[X] shows as Union[X, None]
        import typing
        if origin is typing.Union and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                annotation = non_none[0]
                origin = getattr(annotation, "__origin__", None)
                args = getattr(annotation, "__args__", ())

        # list[X] -> one example item
        if origin is list:
            item_type = args[0] if args else str
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                return [_example_model(item_type)]
            elif item_type is str:
                return ["..."]
            else:
                return ["..."]

        # Nested BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return _example_model(annotation)

        # Literal
        if origin is typing.Literal:
            return args[0] if args else "..."

        # Primitives
        if annotation is str:
            return "..."
        elif annotation is bool:
            return False
        elif annotation is float:
            return 0.0
        elif annotation is int:
            return 0

        return "..."

    def _example_model(model: Type[BaseModel]) -> dict:
        result = {}
        for fname, finfo in model.model_fields.items():
            result[fname] = _example_value(fname, finfo)
        return result

    example = _example_model(schema)
    return json.dumps(example, indent=2)


def _fill_missing_defaults(data: dict, schema: Type[BaseModel]) -> dict:
    """Fill missing required fields with type-appropriate defaults.

    vLLM's xgrammar backend doesn't enforce 'required' fields in JSON schemas,
    so the model may omit them. Rather than adding explicit defaults to every
    field in every schema, we infer safe defaults from the field types:
        str -> "", bool -> False, float/int -> 0, list -> [], Optional -> None
    """
    for field_name, field_info in schema.model_fields.items():
        if field_name in data:
            continue
        # Has an explicit default — Pydantic will handle it
        if field_info.default is not None:
            continue
        # Infer default from annotation
        annotation = field_info.annotation
        if annotation is str:
            data[field_name] = ""
        elif annotation is bool:
            data[field_name] = False
        elif annotation is float:
            data[field_name] = 0.0
        elif annotation is int:
            data[field_name] = 0
        elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            data[field_name] = []
        # Otherwise leave it for Pydantic to error on (truly unknown type)
    return data


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

        # Drop 'required' from flattened schema — all fields have defaults now,
        # and removing it prevents xgrammar/outlines from fighting the model
        # over field presence, which is the #1 cause of schema failures.
        flat_schema.pop("required", None)

        # Generate a minimal JSON example so the model sees exact field names
        example_json = _generate_example(schema)

        # Log flattened schema on first call for diagnostics
        if not hasattr(self, "_logged_schemas"):
            self._logged_schemas = set()
        schema_name = schema.__name__
        if schema_name not in self._logged_schemas:
            self._logged_schemas.add(schema_name)
            logger.info(
                f"[extract] Schema '{schema_name}' flattened for guided_json "
                f"({len(json.dumps(flat_schema))} chars)"
            )

        # ── Two-pass extraction ──────────────────────────────────────────
        # Pass 1: Unconstrained reasoning — let the model think freely
        # Pass 2: Schema-constrained normalization — format into JSON
        # This separates reasoning from formatting (arXiv: "think before
        # constraining") and dramatically improves accuracy for complex schemas.

        # Pass 1: Free reasoning draft
        draft_prompt = (
            f"{system_prompt}\n\n"
            f"First, analyze the protocol text and list your findings as bullet points. "
            f"For each finding, include:\n"
            f"- The exact quoted text from the protocol\n"
            f"- Which chunk_id it came from\n"
            f"- Whether it is explicit or inferred\n"
            f"- Your confidence (0-1)\n"
            f"- Any contradictions between sections\n\n"
            f"Be thorough. List ALL relevant findings."
        )

        try:
            draft_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": draft_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            draft_text = draft_response.choices[0].message.content or ""
            # Strip any <think> blocks from draft too
            draft_text = re.sub(r"<think>.*?</think>\s*", "", draft_text, flags=re.DOTALL).strip()
            logger.info(
                f"[extract] Pass 1 draft for {schema_name}: {len(draft_text)} chars"
            )
        except Exception as e:
            logger.warning(f"[extract] Pass 1 draft failed ({e}), proceeding with single-pass")
            draft_text = ""

        # Pass 2: Schema-constrained normalization
        normalize_system = (
            f"Convert the analysis below into a JSON object matching the required schema.\n\n"
            f"## REQUIRED OUTPUT FORMAT\n"
            f"Your response must be a JSON object with EXACTLY these field names:\n"
            f"```json\n{example_json}\n```\n\n"
            f"Rules:\n"
            f"- Use ONLY the field names shown above — do not invent new ones.\n"
            f"- Fill in real values from the analysis and protocol text.\n"
            f"- If no findings, return empty lists.\n"
            f"- Respond with valid JSON only, no markdown fences."
        )

        if draft_text:
            normalize_user = (
                f"## ANALYSIS\n{draft_text}\n\n"
                f"## ORIGINAL PROTOCOL TEXT\n{user_prompt}\n\n"
                f"Now convert the analysis into the required JSON format."
            )
        else:
            # Fallback to single-pass if draft failed
            normalize_user = user_prompt
            normalize_system = (
                f"{system_prompt}\n\n"
                f"## REQUIRED OUTPUT FORMAT\n"
                f"Your response must be a JSON object with EXACTLY these field names.\n"
                f"Example structure (fill in real values from the protocol):\n"
                f"```json\n{example_json}\n```"
            )

        last_error = None
        last_raw = None
        for attempt in range(1 + self.config.max_retries):
            try:
                # On retry after validation error, add self-healing context
                messages = [
                    {"role": "system", "content": normalize_system},
                    {"role": "user", "content": normalize_user},
                ]
                if attempt > 0 and last_raw and last_error:
                    messages.append({
                        "role": "assistant",
                        "content": last_raw,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your JSON had validation errors:\n{last_error}\n\n"
                            f"Fix the JSON to match the required schema exactly. "
                            f"Use EXACTLY these top-level fields: "
                            f"{list(schema.model_fields.keys())}"
                        ),
                    })

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=tokens,
                    extra_body={
                        "guided_json": flat_schema,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

                choice = response.choices[0]
                raw = choice.message.content

                # Strip <think>...</think> blocks (Qwen3 reasoning traces)
                # Even with enable_thinking=False, some vLLM versions still emit them
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

                # Fill missing required fields with type-appropriate defaults
                # before validation — belt-and-suspenders with schema defaults
                data = json.loads(raw)
                data = _fill_missing_defaults(data, schema)
                parsed = schema.model_validate(data)

                # Back-fill chain_of_thought from draft if empty
                if draft_text and hasattr(parsed, "chain_of_thought") and not parsed.chain_of_thought:
                    parsed.chain_of_thought = draft_text[:2000]

                if attempt > 0:
                    logger.info(
                        f"[extract] Self-healed on attempt {attempt + 1} for {schema_name}"
                    )

                return ExtractionResult(
                    parsed=parsed,
                    model_used=model,
                    raw_response=raw,
                    prompt_version=prompt_version,
                )

            except (ValueError, json.JSONDecodeError, ValidationError) as e:
                # Malformed JSON or validation error — retry with error feedback
                last_error = e
                last_raw = locals().get("raw")
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
