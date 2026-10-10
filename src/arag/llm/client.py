"""LLM client (Ollama or OpenAI) with JSON-schema output, backoff, and caching.

Every call is logged (agent, tokens, latency, cache hit, error). A running cost
estimate is maintained; for the local Ollama provider the cost is $0.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from pydantic import BaseModel, ValidationError

from arag import util
from arag.config import Config
from arag.llm.cache import DiskCache


class LLMError(Exception):
    pass


class InvalidOutputError(LLMError):
    def __init__(self, raw: str):
        self.raw = raw
        super().__init__(f"model output is not valid JSON: {raw[:200]!r}")


@dataclass
class CallRecord:
    agent: str
    model: str
    cache_hit: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    error: str | None = None
    retried: bool = False
    fallback: bool = False


@dataclass
class ValidatedResult:
    obj: Any
    fallback: bool
    retried: bool


class LLMClient:
    def __init__(self, cfg: Config, cache: DiskCache | None = None, logger=None):
        self.cfg = cfg
        self.model = cfg.llm_model
        self.provider = cfg.llm_provider
        self.logger = logger or util.setup_logging("arag.llm")
        self.cache = cache or DiskCache(cfg.path("llm_cache"))
        self.records: list[CallRecord] = []
        self._client = openai.OpenAI(
            base_url=cfg.llm_base_url(),
            api_key=cfg.llm_api_key(),
            timeout=cfg.get("llm.timeout_s", 120),
            max_retries=0,
        )
        self._in_price = cfg.get("budget.prices.input_per_1m", 0.0) / 1e6
        self._out_price = cfg.get("budget.prices.output_per_1m", 0.0) / 1e6
        self.replay_only = os.environ.get("REPLAY_ONLY") == "1"

    # -- cost -------------------------------------------------------------
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def total_tokens(self) -> dict:
        return {
            "prompt": sum(r.prompt_tokens for r in self.records),
            "completion": sum(r.completion_tokens for r in self.records),
        }

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        if self.provider != "openai":
            return 0.0
        return prompt_tokens * self._in_price + completion_tokens * self._out_price

    # -- low-level --------------------------------------------------------
    def _extra_body(self) -> dict | None:
        if self.provider == "ollama" and self.cfg.get("llm.ollama.enable_thinking") is False:
            return {"think": False}
        return None

    def complete_json(self, agent: str, prompt: str, schema: dict, model: str | None = None) -> dict:
        model = model or self.model
        key = DiskCache.key(model, prompt, schema)
        cached = self.cache.get(key)
        if cached is not None:
            self._record(agent, model, True, cached.get("usage", {}), 0.0)
            return cached["content"]

        if self.replay_only:
            raise LLMError(f"REPLAY_ONLY: refusing uncached LLM call for agent={agent}")

        content, usage, latency = self._call_api(agent, model, prompt, schema)
        self._record(agent, model, False, usage, latency)
        self.cache.put(key, {"content": content, "usage": usage})
        return content

    def _call_api(self, agent, model, prompt, schema):
        max_retries = self.cfg.get("llm.max_retries", 3)
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.get("llm.temperature", 0),
                    response_format={"type": "json_schema", "json_schema": schema},
                    extra_body=self._extra_body(),
                )
                msg = resp.choices[0].message
                raw = msg.content or ""
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                }
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise InvalidOutputError(raw) from e
                return parsed, usage, time.time() - t0
            except InvalidOutputError:
                raise
            except openai.APIStatusError as e:
                last_err = e
                if e.status_code not in (429, 500, 502, 503, 504):
                    raise
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last_err = e
            if attempt < max_retries:
                wait = 2.0 ** attempt
                self.logger.warning("%s: retrying (%s) in %.1fs", agent, type(last_err).__name__, wait)
                time.sleep(wait)
        raise LLMError(f"{agent}: failed after {max_retries + 1} attempts: {last_err}")

    def _record(self, agent, model, cache_hit, usage, latency, error=None, retried=False, fallback=False):
        rec = CallRecord(
            agent=agent,
            model=model,
            cache_hit=cache_hit,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
            cost_usd=0.0 if cache_hit else self._cost(
                int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
            ),
            error=error,
            retried=retried,
            fallback=fallback,
        )
        self.records.append(rec)

    # -- validated --------------------------------------------------------
    def complete_validated(
        self,
        agent: str,
        prompt: str,
        schema: dict,
        model_cls: type[BaseModel],
        fallback: Any = None,
    ) -> ValidatedResult:
        error_note: str | None = None
        retried = False
        for attempt in range(2):
            p = prompt if error_note is None else f"{prompt}\n\n{error_note}"
            try:
                content = self.complete_json(agent, p, schema)
            except InvalidOutputError as e:
                error_note = f"Your previous output was not valid JSON. Error: {e}. Return ONLY a valid JSON object matching the requested schema."
                retried = attempt > 0
                continue
            except LLMError as e:
                error_note = f"Error: {e}. Try again."
                retried = attempt > 0
                continue
            try:
                obj = model_cls.model_validate(content)
                return ValidatedResult(obj=obj, fallback=False, retried=attempt > 0)
            except ValidationError as e:
                error_note = f"Your previous output failed schema validation: {e}. Return a JSON object exactly matching the requested schema."
                retried = attempt > 0
        if fallback is None:
            raise LLMError(f"{agent}: validation failed after retry; no fallback configured")
        return ValidatedResult(obj=fallback, fallback=True, retried=retried)
