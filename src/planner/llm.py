"""Provider-agnostic LLM abstraction. REAL clients use only urllib (no SDK needed); keys come from env vars."""
from __future__ import annotations
import json
import logging
import os
import random
import re
import socket
import time
import urllib.request
import urllib.error
from src.planner.model_registry import adapt_request, get_spec
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    is_mock: bool
    tokens_estimated: bool = False
    raw: dict = field(default_factory=dict)


log = logging.getLogger("llm")


class LLMError(RuntimeError):
    """Provider/transport failure (NOT a model-output problem). Never recorded as an experimental outcome."""


class LLMRateLimitError(LLMError):
    def __init__(self, msg, daily: bool = False, retry_after_s: float | None = None):
        super().__init__(msg)
        self.daily = daily
        self.retry_after_s = retry_after_s     # provider's own estimate of when the limit frees up (None = unknown)


_MINUTE = ("per minute", "perminute", "per_minute", "(rpm)", "(tpm)", "tokens per min", "requests per min")
_DAY = ("per day", "perday", "per_day", "daily", "(rpd)", "(tpd)")


def classify_429(text: str) -> str:
    """'daily' only when the provider names a per-day quota and no per-minute one; everything else is a transient limit that waiting fixes.
    (v11 treated any Gemini 'Quota exceeded for metric' as daily, but Gemini words its PER-MINUTE quota the same way, with quotaId '...PerMinute...',
    so free-tier runs stopped early and needed --resume.)"""
    t = text.lower()
    if any(w in t for w in _DAY) and not any(w in t for w in _MINUTE):
        return "daily"
    return "transient"


def parse_wait_s(headers: dict | None, data) -> float | None:
    """Seconds to wait, from a Retry-After header or the provider's message: 'retry in 12.3s', 'try again in 7m12.5s', 'retryDelay': '16s', '1h2m3s'."""
    v = (headers or {}).get("retry-after")
    try:
        if v is not None:
            return float(v)
    except ValueError:
        pass
    try:
        txt = json.dumps(data)
    except Exception:
        txt = str(data)
    m = re.search(r"(?:retry|try again) in ((?:[0-9.]+h)?(?:[0-9.]+m(?!s))?(?:[0-9.]+s)?)", txt, re.I) or re.search(r'"retryDelay"\s*:\s*"([0-9.]+s)"', txt)
    if not m or not m.group(1):
        return None
    total = 0.0
    for num, unit in re.findall(r"([0-9.]+)([hms])", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total or None


def stop_info(exc: BaseException | None, reason: str | None = None) -> dict:
    """Structured reason why a run stopped, written to manifests so `run_free.py` knows whether to wait (quota) or to stop (configuration error)."""
    if exc is None:
        return {"kind": "budget" if reason and "budget" in reason else ("interrupted" if reason and "interrupt" in reason else ("none" if not reason else "other")), "message": reason}
    if isinstance(exc, LLMRateLimitError):
        return {"kind": "daily_quota" if exc.daily else "rate_limit", "retry_after_s": exc.retry_after_s, "message": str(exc)[:400]}
    msg = str(exc)
    if re.search(r"HTTP (401|403)", msg) or "is not set" in msg:
        return {"kind": "auth", "message": msg[:400]}
    if re.search(r"HTTP (400|404)", msg):
        return {"kind": "bad_request", "message": msg[:400]}
    if re.search(r"HTTP 599", msg) or "network error" in msg:
        return {"kind": "network", "message": msg[:400]}
    if re.search(r"HTTP (500|502|503|504|529)", msg):
        return {"kind": "server", "message": msg[:400]}
    return {"kind": "provider_error", "message": msg[:400]}


class LLMClient:
    name = "base"
    is_mock = False
    model = ""

    def generate(self, system: str, user: str, temperature: float, max_tokens: int, seed: int) -> LLMResponse:
        raise NotImplementedError

    def config(self) -> dict:
        return {"provider": self.name, "model": self.model, "is_mock": self.is_mock}


def default_transport(url: str, headers: dict, body: dict, timeout: float = 90.0) -> tuple[int, dict, dict]:
    h = dict(headers or {})
    h.setdefault("User-Agent", "uav-llm-closed-loop/1.0")
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        timeout = float(os.environ.get("CA_CLP_HTTP_TIMEOUT") or timeout)    # large models (Gemma 31B) can need more than 90 s per reply
        with urllib.request.urlopen(req, timeout=timeout) as r:
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            try:
                return r.status, json.loads(r.read().decode()), hdrs
            except ValueError:   # 200 with a non-JSON body (proxy / overload page): treat as a transient server error
                return 502, {"error": "provider returned a non-JSON body"}, hdrs
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            data = json.loads(raw)
        except Exception:
            data = {"error": raw[:500]}
        return e.code, data, {k.lower(): v for k, v in (e.headers or {}).items()}
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
        return 599, {"error": f"network error: {e}"}, {}


class _HTTPClient(LLMClient):
    def __init__(self, model: str, transport=None, max_retries: int = 4, backoff: float = 3.0, max_wait_s: float = 90.0, on_wait=None,
                 max_daily_wait_s: float = 1200.0):
        self.model = model
        self.max_daily_wait_s = max_daily_wait_s   # per-day limits with a short provider-announced reset (rolling windows) are waited for
        self.transport = transport or default_transport
        self.max_retries, self.backoff, self.max_wait_s = max_retries, backoff, max_wait_s
        self.on_wait = on_wait          # optional callback(seconds, attempt, reason) so UIs/CLIs can show progress
        self.n_rate_limit_waits = 0

    @staticmethod
    def _retry_after(headers: dict, data) -> float | None:
        return parse_wait_s(headers, data)

    def _post(self, url, headers, body):
        last, k, long_waits = None, 0, 0
        while True:
            res = self.transport(url, headers, body)
            status, data = res[0], res[1]
            hdrs = res[2] if len(res) > 2 else {}
            if status == 200:
                return data
            last = (status, data)
            text = json.dumps(data)
            wait_hint = parse_wait_s(hdrs, data)
            if status == 429 and classify_429(text) == "daily":
                # rolling windows (e.g. Groq tokens-per-day) free up within minutes: wait when the provider says it is short, otherwise stop
                if wait_hint is not None and wait_hint <= self.max_daily_wait_s and long_waits < 6:
                    long_waits += 1
                    log.warning("HTTP 429 (per-day limit) - provider asks to wait %.0fs; waiting", wait_hint)
                    if self.on_wait:
                        self.on_wait(wait_hint + 2, k + 1, "daily-limit window")
                    time.sleep(wait_hint + 2)
                    continue
                raise LLMRateLimitError(f"HTTP 429: daily quota appears exhausted ({text[:300]}). Wait for the reset, then continue with --resume.",
                                        daily=True, retry_after_s=wait_hint)
            if status in (429, 500, 502, 503, 504, 529, 599) and k < self.max_retries:
                wait = min(self.max_wait_s, max(wait_hint or 0.0, self.backoff * (2 ** k))) + random.uniform(0, 1.0)
                self.n_rate_limit_waits += 1
                log.warning("HTTP %s from provider; waiting %.0fs before retry %d/%d", status, wait, k + 1, self.max_retries)
                if self.on_wait:
                    self.on_wait(wait, k + 1, f"HTTP {status}")
                time.sleep(wait)
                k += 1
                continue
            break
        if last[0] == 429:
            raise LLMRateLimitError(f"HTTP 429 persisted after {self.max_retries} retries: {json.dumps(last[1])[:300]}. "
                                    f"Lower requests_per_minute in config/default.yaml or resume later with --resume.", retry_after_s=parse_wait_s({}, last[1]))
        raise LLMError(f"HTTP {last[0]}: {json.dumps(last[1])[:300]}")

    def _get(self, url, headers):
        """GET helper (model listing). Uses urllib directly; returns (status, data)."""
        req = urllib.request.Request(url, headers={**headers, "User-Agent": "uav-llm-closed-loop/1.0"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode(errors="replace"))
            except Exception:
                return e.code, {}
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ValueError) as e:
            return 599, {"error": str(e)}


class AnthropicClient(_HTTPClient):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, **kw):
        super().__init__(model, **kw)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")

    def generate(self, system, user, temperature, max_tokens, seed):
        t0 = time.perf_counter()
        d = self._post(self.base_url + "/v1/messages",
                       {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                       {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "system": system,
                        "messages": [{"role": "user", "content": user}]})
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return LLMResponse(text, d.get("model", self.model), u.get("input_tokens"), u.get("output_tokens"),
                           time.perf_counter() - t0, False,
                           raw={"id": d.get("id"), "stop_reason": d.get("stop_reason"), "request_temperature": temperature, "response_model": d.get("model")})


class OpenAICompatClient(_HTTPClient):
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, Gemini AI Studio, Groq, OpenRouter, Ollama...)."""
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, key_env: str | None = "OPENAI_API_KEY",
                 send_seed: bool = True, min_interval_s: float = 0.0, name: str | None = None, rpm: float | None = None,
                 min_temperature: float | None = None, min_max_tokens: int | None = None, **kw):
        super().__init__(model, **kw)
        if name:
            self.name = name
        self.api_key = api_key or (os.environ.get(key_env) if key_env else None)
        if key_env and not self.api_key:
            raise LLMError(f"{key_env} is not set")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.send_seed, self._last = send_seed, 0.0
        self.min_interval_s = max(min_interval_s, 60.0 / rpm if rpm else 0.0)
        self.min_temperature = min_temperature
        self.min_max_tokens = min_max_tokens
        self.merge_system = False

    def list_models(self) -> list[str]:
        """Model ids offered by the endpoint (GET /models). Gemini returns ids like 'models/gemini-...': the prefix is removed."""
        h = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        status, data = self._get(self.base_url + "/models", h)
        if status != 200:
            raise LLMError(f"HTTP {status} listing models: {json.dumps(data)[:300]}")
        return sorted({str(m.get("id", "")).split("/", 1)[1] if str(m.get("id", "")).startswith("models/") else str(m.get("id", ""))
                       for m in (data.get("data") or data.get("models") or []) if m.get("id")})

    def generate(self, system, user, temperature, max_tokens, seed):
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)  # stay under free-tier requests-per-minute limits
        waited = max(wait, 0.0)
        t0 = time.perf_counter()
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.min_temperature is not None:
            temperature = max(temperature, self.min_temperature)  # some providers reject greedy (temperature 0)
        if self.min_max_tokens is not None:
            max_tokens = max(max_tokens, self.min_max_tokens)  # thinking models need a budget for the reply after reasoning
        adjustments = {}
        spec = getattr(self, "spec", None)
        sent_temp, sent_tokens, sent_seed = temperature, max_tokens, (seed if self.send_seed else None)
        if spec is not None:   # capability registry (config/models.yaml): omit unsupported parameters / clamp, and RECORD every change
            sent_temp, sent_tokens, sent_seed, adjustments = adapt_request(spec, temperature, max_tokens, seed if self.send_seed else None)
        def _body():
            msgs = ([{"role": "user", "content": f"{system}\n\n{user}"}] if self.merge_system
                    else [{"role": "system", "content": system}, {"role": "user", "content": user}])
            b = {"model": self.model, "max_tokens": sent_tokens, "messages": msgs}
            if sent_temp is not None:
                b["temperature"] = sent_temp
            if sent_seed is not None and self.send_seed:
                b["seed"] = sent_seed
            b.update(getattr(self, "extra_body", None) or {})     # provider-specific options from the config (e.g. reasoning_effort), recorded below
            return b
        try:
            try:
                d = self._post(self.base_url + "/chat/completions", headers, _body())
            except LLMError as e:
                # some open models (e.g. Gemma behind the Gemini API) reject a system role; send the same text as one user message and record it
                if not self.merge_system and re.search(r"HTTP 400", str(e)) and re.search(r"developer instruction|system (role|instruction|message|prompt)s? (is |are )?(not|un)", str(e), re.I):
                    self.merge_system = True
                    log.warning("model rejected the system role; resending with the system prompt merged into the user message (recorded as system_merged)")
                    d = self._post(self.base_url + "/chat/completions", headers, _body())
                else:
                    raise
        finally:
            self._last = time.monotonic()
        u = d.get("usage") or {}
        if not d.get("choices"):   # some routers answer HTTP 200 with an error body: provider failure, never a model outcome
            raise LLMError(f"unexpected provider reply without 'choices': {json.dumps(d)[:300]}")
        text = (d["choices"][0].get("message", {}).get("content") or "")
        return LLMResponse(text, d.get("model", self.model), u.get("prompt_tokens"), u.get("completion_tokens"),
                           time.perf_counter() - t0, False,
                           raw={"id": d.get("id"), "request_temperature": sent_temp, "request_max_tokens": sent_tokens, "response_model": d.get("model"),
                                "system_fingerprint": d.get("system_fingerprint"), "finish_reason": d["choices"][0].get("finish_reason"),
                                "seed_sent": sent_seed is not None, "params_adjusted": adjustments, "extra_body": getattr(self, "extra_body", None) or {}, "rate_limit_wait_s": round(waited, 3), "system_merged": self.merge_system,
                                "request_latency_s": round(time.perf_counter() - t0, 3)})


# Free-tier presets (limits/model ids change often: check each provider's docs). None of these needs a credit card; Ollama runs locally.
PRESETS = {
    "gemini":     dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai", key_env="GEMINI_API_KEY", send_seed=False, rpm=8),
    "groq":       dict(base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY", send_seed=True, rpm=2),   # free tier: 8000 tokens/min; ~2300 tok/call => ~2 calls/min
    "openrouter": dict(base_url="https://openrouter.ai/api/v1", key_env="OPENROUTER_API_KEY", send_seed=False, rpm=10),
    "ollama":     dict(base_url="http://localhost:11434/v1", key_env=None, send_seed=True, rpm=0),
    "grok":       dict(base_url="https://api.x.ai/v1", key_env="XAI_API_KEY", send_seed=False, rpm=10),   # xAI; OpenAI-compatible; NOT tested by the authors of this repo (no key available)
    "cohere":     dict(base_url="https://api.cohere.ai/compatibility/v1", key_env="COHERE_API_KEY", send_seed=False, rpm=20),  # Cohere trial: 20 req/min, 1000 calls/month
    "requesty":   dict(base_url="https://router.requesty.ai/v1", key_env="REQUESTY_API_KEY", send_seed=False, rpm=3, min_temperature=0.1, min_max_tokens=16000),  # OpenAI-compatible router, e.g. mistral/leanstral-1-5 (rejects greedy temp=0; thinking needs a big budget; free pool rate-limits hard)
}


class NoModelClient(LLMClient):
    """Provider 'none': for runs that contain ONLY model-free conditions (E_classical, F_astar, G_direct_*, and A_llm_only in the dynamic study, where the
    'initial plan' is the stale plan). It is not a stand-in model: any attempt to generate text raises, so a record produced with it can never contain model output."""
    name = "none"
    is_mock = False

    def __init__(self, model: str = "none"):
        self.model = model or "none"

    def generate(self, system, user, temperature, max_tokens, seed):
        raise LLMError("provider 'none' cannot generate text: this run may only contain model-free conditions")

    def config(self) -> dict:
        return {"provider": "none", "model": self.model, "is_mock": False, "note": "no language model; model-free conditions only"}


class _FlakyQuota(LLMClient):
    """Test double: behaves like the wrapped client until `after` calls were made in this process, then raises a daily-quota error."""
    _calls = 0

    def __init__(self, inner, after: int):
        self.inner, self.after = inner, after
        self.name, self.model, self.is_mock = inner.name, inner.model, inner.is_mock

    def generate(self, system, user, temperature, max_tokens, seed):
        if _FlakyQuota._calls >= self.after:
            raise LLMRateLimitError("HTTP 429: daily quota appears exhausted (simulated)", daily=True)
        _FlakyQuota._calls += 1
        return self.inner.generate(system, user, temperature, max_tokens, seed)

    def config(self):
        return self.inner.config()


def make_client(cfg: dict) -> LLMClient:
    p = cfg.get("provider", "mock")
    if p == "none":
        return NoModelClient(cfg.get("model") or "none")
    if p == "mock":
        from src.planner.mock_llm import MockLLM
        m = MockLLM(cfg.get("mock", {}))
        if os.environ.get("CA_CLP_FLAKY_AFTER"):      # TESTS ONLY: simulate a daily free quota that runs out after N calls in this process
            return _FlakyQuota(m, int(os.environ["CA_CLP_FLAKY_AFTER"]))
        return m
    if p == "follower":
        from src.planner.follower_llm import FollowerLLM
        return FollowerLLM()
    if p == "anthropic":
        return AnthropicClient(cfg["model"])
    if p == "openai":
        return OpenAICompatClient(cfg["model"], min_interval_s=cfg.get("request_interval_s", 0.0),
                                  rpm=cfg.get("requests_per_minute"))
    if p in PRESETS:
        pr = PRESETS[p]
        base = os.environ.get("OLLAMA_BASE_URL") if p == "ollama" and os.environ.get("OLLAMA_BASE_URL") else pr["base_url"]
        rpm = cfg.get("requests_per_minute", None)
        if rpm is None:
            rpm = pr["rpm"]
        c = OpenAICompatClient(cfg["model"], base_url=base, key_env=pr["key_env"], send_seed=pr["send_seed"], name=p,
                               min_interval_s=cfg.get("request_interval_s", 0.0), rpm=rpm,
                               min_temperature=pr.get("min_temperature"), min_max_tokens=pr.get("min_max_tokens"),
                               max_retries=cfg.get("max_retries", 5), backoff=5.0)
        c.spec = get_spec(p, cfg["model"])
        c.extra_body = dict(cfg.get("extra_body") or {})
        return c
    raise LLMError(f"unknown provider {p!r}")
