"""LLM-based span extractor with OpenAI-compatible backends (DeepSeek, OpenAI, etc.)."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── .env loader (no external dependency) ────────────────────────────────────

_ENV_LOADED = False


def _load_dotenv(root_marker: str = "pyproject.toml") -> None:
    """Load ``.env`` from the project root into ``os.environ`` (idempotent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    # walk up from this file until we find *root_marker*
    candidate = Path(__file__).resolve().parent
    for _ in range(8):
        if (candidate / root_marker).exists():
            break
        candidate = candidate.parent
    env_path = candidate / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    logger.debug("Loaded .env from %s", env_path)

# ── reusable JSON-ish block parser ──────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json_block(text: str) -> str:
    """Pull the outermost {…} block from *text*, falling back to the whole string."""
    match = _JSON_BLOCK_RE.search(text)
    return match.group(0) if match else text


# ── prompt helpers ──────────────────────────────────────────────────────────

def _load_prompt_strategies(path: str | None = None) -> dict[str, Any]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "prompting", "clause_prompts.yaml")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    # If the YAML wraps everything under a top-level "strategies" key, unwrap it.
    if isinstance(data, dict) and "strategies" in data and len(data) == 1:
        return data["strategies"]
    return data


def _build_messages(
    strategy: dict[str, Any],
    context: str,
    query: str,
    examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build a chat message list from a strategy definition."""
    messages: list[dict[str, str]] = []

    system = strategy.get("system", "")
    if system:
        messages.append({"role": "system", "content": system.strip()})

    user_template = strategy.get("user_template", "{context}\n\n{query}")
    examples_block = ""
    if examples:
        example_template = strategy.get("example_template", "")
        parts = []
        for i, ex in enumerate(examples, 1):
            parts.append(
                example_template.format(
                    index=i,
                    context=ex.get("context", ""),
                    query=ex.get("query", ""),
                    answer=ex.get("answer", ""),
                )
            )
        examples_separator = strategy.get("examples_separator", "\n\n")
        examples_block = examples_separator.join(parts)

    user = user_template.format(
        context=context,
        query=query,
        examples=examples_block,
        num_examples=len(examples) if examples else 0,
    )
    messages.append({"role": "user", "content": user.strip()})
    return messages


# ── span resolution ─────────────────────────────────────────────────────────

def _resolve_span_offsets(source_text: str, span_text: str) -> tuple[int, int] | None:
    """Find *span_text* in *source_text*; return (start, end) or None."""
    idx = source_text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)
    # loose match: collapse whitespace
    collapsed_source = re.sub(r"\s+", " ", source_text)
    collapsed_span = re.sub(r"\s+", " ", span_text).strip()
    idx = collapsed_source.find(collapsed_span)
    if idx != -1:
        # map back to original offsets (approximate)
        return idx, idx + len(collapsed_span)
    return None


# ── client abstraction ──────────────────────────────────────────────────────

class LLMClient:
    """OpenAI-compatible chat client, defaults to DeepSeek endpoint."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        import openai

        _load_dotenv()

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("api_key")
        )
        if not api_key:
            raise ValueError(
                "No API key provided. Set DEEPSEEK_API_KEY or api_key in .env, "
                "or pass --api-key."
            )

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat(self, messages: list[dict[str, str]]) -> str:
        logger.debug("Sending %d messages to %s", len(messages), self._model)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("Received %d chars from LLM", len(content))
        return content

    def extract_spans(
        self,
        context: str,
        query: str,
        strategy: dict[str, Any],
        examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run a single extraction and return parsed result.

        Returns
        -------
        dict with keys:
          - answerable: bool
          - spans: list[dict] with text, start, end, score
          - raw_response: str
          - parse_error: str | None
        """
        messages = _build_messages(strategy, context=context, query=query, examples=examples)
        raw = self.chat(messages)

        result: dict[str, Any] = {
            "answerable": False,
            "spans": [],
            "raw_response": raw,
            "parse_error": None,
        }

        try:
            parsed = json.loads(_extract_json_block(raw))
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"JSON parse failed: {exc}"
            logger.warning("JSON parse failed for query %r: %s", query[:80], exc)
            return result

        if isinstance(parsed, list):
            # LLM returned bare list — treat as spans directly
            parsed = {"answerable": len(parsed) > 0, "spans": parsed}

        result["answerable"] = bool(parsed.get("answerable", len(parsed.get("spans", [])) > 0))

        for span in parsed.get("spans", []):
            text = str(span.get("text", "")).strip()
            if not text:
                continue
            offsets = _resolve_span_offsets(context, text)
            result["spans"].append(
                {
                    "text": text,
                    "start": offsets[0] if offsets else -1,
                    "end": offsets[1] if offsets else -1,
                    "score": float(span.get("score", 0.9)),
                }
            )

        return result


# ── batch extraction helper ─────────────────────────────────────────────────

def _pick_few_shot_examples(
    train_records: list[dict[str, Any]],
    rule_type: str,
    n: int,
    max_context_chars: int = 3000,
) -> list[dict[str, Any]]:
    """Pick up to *n* examples for *rule_type* from *train_records*."""
    candidates = [r for r in train_records if r["rule_type"] == rule_type and r["gold_spans"]]
    examples = []
    for rec in candidates[:n]:
        gold_text = rec["gold_spans"][0]["text"]
        ctx = rec["source_text"]
        # find a window around the answer
        ans_start = rec["gold_spans"][0]["start"]
        ans_end = rec["gold_spans"][0]["end"]
        half = max_context_chars // 2
        window_start = max(0, ans_start - half)
        window_end = min(len(ctx), ans_end + half)
        ctx_excerpt = ctx[window_start:window_end]
        examples.append(
            {
                "context": ctx_excerpt,
                "query": rec["query"],
                "answer": gold_text,
            }
        )
    return examples
