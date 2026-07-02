"""Headroom Proxy Server - Production Ready.

A full-featured LLM proxy with optimization, caching, rate limiting,
and observability.

Features:
- Context optimization (SmartCrusher, CacheAligner — live-zone-only after Phase B)
- Semantic caching (save costs on repeated queries)
- Rate limiting (token bucket)
- Retry with exponential backoff
- Cost tracking and budgets
- Request tagging and metadata
- Provider fallback
- Prometheus metrics
- Full request/response logging

Usage:
    python -m headroom.proxy.server --port 8787

    # With Claude Code:
    ANTHROPIC_BASE_URL=http://localhost:8787 claude
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from ..backends.base import Backend
    from ..cache.compression_cache import CompressionCache
    from ..memory.tracker import MemoryTracker
    from .outcome import RequestOutcome


import httpx

try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from headroom._version import __version__
from headroom.agent_savings import proxy_pipeline_kwargs
from headroom.cache.compression_feedback import get_compression_feedback
from headroom.cache.compression_store import format_retrieval_miss_detail, get_compression_store
from headroom.ccr import (
    CCR_TOOL_NAME,
    # Batch processing
    CCRResponseHandler,
    CCRToolInjector,
    ContextTracker,
    ContextTrackerConfig,
    ResponseHandlerConfig,
    parse_tool_call,
)
from headroom.config import (
    DEFAULT_EXCLUDE_TOOLS,
    CacheAlignerConfig,
    ReadLifecycleConfig,
)
from headroom.dashboard import get_dashboard_html
from headroom.observability import (
    LangfuseTracingConfig,
    OTelMetricsConfig,
    configure_langfuse_tracing,
    configure_otel_metrics,
    get_langfuse_tracing_status,
    get_otel_metrics_status,
    shutdown_headroom_tracing,
    shutdown_otel_metrics,
)
from headroom.pipeline import PipelineExtensionManager, PipelineStage
from headroom.providers.proxy_routes import register_provider_routes
from headroom.providers.registry import (
    DEFAULT_ANTHROPIC_API_URL,
    DEFAULT_CLOUDCODE_API_URL,
    DEFAULT_GEMINI_API_URL,
    DEFAULT_OPENAI_API_URL,
    DEFAULT_VERTEX_API_URL,
    build_proxy_provider_runtime,
    create_proxy_backend,
    format_backend_status,
    resolve_api_targets,
)
from headroom.proxy import runtime_env
from headroom.proxy.auth_mode import should_stamp_codex_client
from headroom.proxy.background_compression import BackgroundCompressor

# =============================================================================
# Extracted modules (re-exported for backward compatibility)
# =============================================================================
from headroom.proxy.cost import (
    _CACHE_ECONOMICS,  # noqa: F401
    CostTracker,  # noqa: F401
    _summarize_transforms,  # noqa: F401
    build_prefix_cache_stats,  # noqa: F401
    build_session_summary,  # noqa: F401
    merge_cost_stats,  # noqa: F401
)
from headroom.proxy.helpers import (
    COMPRESSION_TIMEOUT_SECONDS,  # noqa: F401
    MAX_COMPRESSION_CACHE_SESSIONS,  # noqa: F401
    MAX_MESSAGE_ARRAY_LENGTH,  # noqa: F401
    MAX_REQUEST_BODY_SIZE,  # noqa: F401
    MAX_SSE_BUFFER_SIZE,  # noqa: F401
    _get_context_tool_stats,
    _get_image_compressor,  # noqa: F401
    _get_rtk_stats,  # noqa: F401
    _read_request_json,  # noqa: F401
    _setup_file_logging,  # noqa: F401
    initialize_context_tool_session_baseline,
    is_anthropic_auth,  # noqa: F401
    jitter_delay_ms,
    retry_after_ms,
)
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

# Data models (extracted to headroom/proxy/models.py for maintainability)
from headroom.proxy.models import CacheEntry, ProxyConfig, RateLimitState, RequestLog  # noqa: F401
from headroom.proxy.modes import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    is_token_mode,
    normalize_proxy_mode,
)
from headroom.proxy.probe_recorder import probe_recorder_from_env
from headroom.proxy.project_context import (
    classify_project,
    set_current_project,
    strip_project_path_prefix,
)
from headroom.proxy.prometheus_metrics import PrometheusMetrics  # noqa: F401
from headroom.proxy.rate_limiter import TokenBucketRateLimiter  # noqa: F401
from headroom.proxy.request_logger import RequestLogger  # noqa: F401
from headroom.proxy.savings_tracker import LITELLM_AVAILABLE
from headroom.proxy.semantic_cache import SemanticCache  # noqa: F401
from headroom.proxy.ssl_context import build_httpx_verify
from headroom.proxy.warmup import WarmupRegistry
from headroom.proxy.ws_session_registry import WebSocketSessionRegistry
from headroom.subscription.base import get_quota_registry, reset_quota_registry
from headroom.subscription.codex_rate_limits import get_codex_rate_limit_state
from headroom.subscription.copilot_quota import get_copilot_quota_tracker
from headroom.subscription.tracker import (
    configure_subscription_tracker,
    get_subscription_tracker,
)
from headroom.transforms import (
    CacheAligner,
    CodeAwareCompressor,
    CodeCompressorConfig,
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    TransformPipeline,
    is_tree_sitter_available,
)

AnyLLMBackend: Any = None
LiteLLMBackend: Any = None

fcntl: Any = None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

_build_prefix_cache_stats = build_prefix_cache_stats
_build_session_summary = build_session_summary
_merge_cost_stats = merge_cost_stats


_AGENT_LABELS: dict[str, str] = {
    "claude": "Claude",
    "claude-code": "Claude",
    "claude_cli": "Claude",
    "claude-code-cli": "Claude",
    "codex": "Codex",
    "codex-cli": "Codex",
    "cursor": "Cursor",
    "copilot": "GitHub Copilot",
    "github-copilot": "GitHub Copilot",
    "aider": "Aider",
    "zed": "Zed",
    "opencode": "OpenCode",
    "openclaw": "OpenClaw",
    "gemini": "Gemini",
    "google": "Gemini",
    "vertex:google": "Gemini",
    "anthropic": "Claude",
    "openai": "OpenAI",
    "unknown": "Unidentified",
}

_AGENT_SOURCE_PRIORITY: dict[str, int] = {
    "unknown": 0,
    "provider": 1,
    "model": 2,
    "stack": 3,
    "client": 4,
}


def _normalize_agent_key(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    value = value.replace(" ", "-").replace("_", "-")
    if value.startswith("wrap-"):
        value = value.removeprefix("wrap-")
    if value in {"claude-cli", "claude-code", "claude-code-cli"}:
        return "claude-code"
    if value in {"codex-cli", "codex"}:
        return "codex"
    if value in {"github-copilot", "copilot"}:
        return "copilot"
    if value in {"google", "vertex-google", "vertex:google"}:
        return "gemini"
    return value


def _agent_label(agent_key: str) -> str:
    if agent_key in _AGENT_LABELS:
        return _AGENT_LABELS[agent_key]
    return agent_key.replace("-", " ").replace("_", " ").title()


def _classify_agent_from_log(entry: dict[str, Any]) -> tuple[str, str, str]:
    raw_tags = entry.get("tags")
    tags = raw_tags if isinstance(raw_tags, dict) else {}
    for source, candidate in (
        ("client", tags.get("client")),
        ("stack", tags.get("stack") or tags.get("headroom-stack")),
    ):
        key = _normalize_agent_key(candidate)
        if key:
            return key, _agent_label(key), source

    model = str(entry.get("model") or "").lower()
    if "codex" in model:
        return "codex", _agent_label("codex"), "model"
    if "claude" in model:
        return "claude-code", _agent_label("claude-code"), "model"
    if "gemini" in model:
        return "gemini", _agent_label("gemini"), "model"

    key = _normalize_agent_key(entry.get("provider"))
    if key:
        return key, _agent_label(key), "provider"

    return "unknown", _agent_label("unknown"), "unknown"


def _build_agent_usage_summary(
    logs: list[dict[str, Any]],
    *,
    requests_by_provider: dict[str, int],
    requests_by_model: dict[str, int],
    global_before_tokens: int,
    global_after_tokens: int,
    global_tokens_saved: int,
    global_output_tokens: int,
) -> dict[str, Any]:
    agents: dict[str, dict[str, Any]] = {}

    def _agent_row(agent_key: str, label: str, source: str) -> dict[str, Any]:
        row = agents.setdefault(
            agent_key,
            {
                "agent": agent_key,
                "label": label,
                "source": source,
                "requests": 0,
                "before_tokens": 0,
                "after_tokens": 0,
                "output_tokens": 0,
                "tokens_saved": 0,
                "models": {},
                "providers": {},
                "has_exact_tokens": False,
            },
        )
        if _AGENT_SOURCE_PRIORITY.get(source, 0) > _AGENT_SOURCE_PRIORITY.get(
            str(row.get("source") or "unknown"), 0
        ):
            row["source"] = source
        return row

    for entry in logs:
        agent_key, label, source = _classify_agent_from_log(entry)
        row = _agent_row(agent_key, label, source)
        before = max(0, int(entry.get("input_tokens_original") or 0))
        after = max(0, int(entry.get("input_tokens_optimized") or 0))
        saved = max(0, int(entry.get("tokens_saved") or 0))
        output = max(0, int(entry.get("output_tokens") or 0))
        provider = str(entry.get("provider") or "unknown")
        model = str(entry.get("model") or "unknown")

        row["requests"] += 1
        row["before_tokens"] += before
        row["after_tokens"] += after
        row["output_tokens"] += output
        row["tokens_saved"] += saved
        row["providers"][provider] = int(row["providers"].get(provider, 0)) + 1
        row["models"][model] = int(row["models"].get(model, 0)) + 1
        if before > 0 or after > 0 or saved > 0:
            row["has_exact_tokens"] = True

    if not agents:
        inferred_model_counts: dict[str, int] = {}
        for model, count in requests_by_model.items():
            model_lower = str(model).lower()
            if "codex" in model_lower:
                key = "codex"
            elif "claude" in model_lower:
                key = "claude-code"
            elif "gemini" in model_lower:
                key = "gemini"
            else:
                continue
            inferred_model_counts[str(model)] = int(count)

        provider_request_count = sum(max(0, int(count)) for count in requests_by_provider.values())
        inferred_request_count = sum(max(0, count) for count in inferred_model_counts.values())
        use_model_fallback = (
            inferred_request_count > 0 and inferred_request_count == provider_request_count
        )

        if not use_model_fallback:
            for provider, count in requests_by_provider.items():
                key = _normalize_agent_key(provider) or "unknown"
                row = _agent_row(key, _agent_label(key), "provider")
                row["requests"] += int(count)
                row["providers"][provider] = int(row["providers"].get(provider, 0)) + int(count)
        for model, count in requests_by_model.items():
            model_lower = str(model).lower()
            if "codex" in model_lower:
                key = "codex"
            elif "claude" in model_lower:
                key = "claude-code"
            elif "gemini" in model_lower:
                key = "gemini"
            else:
                continue
            if not use_model_fallback:
                continue
            row = _agent_row(key, _agent_label(key), "model")
            row["requests"] += int(count)
            row["models"][str(model)] = int(row["models"].get(str(model), 0)) + int(count)

    rows: list[dict[str, Any]] = []
    for row in agents.values():
        before = int(row["before_tokens"])
        saved = int(row["tokens_saved"])
        after = int(row["after_tokens"])
        if before == 0 and (after > 0 or saved > 0):
            before = after + saved
        savings_percent = round((saved / before) * 100.0, 2) if before else 0.0
        row["before_tokens"] = before
        row["savings_percent"] = savings_percent
        row["after_percent"] = round((after / before) * 100.0, 2) if before else 0.0
        row["share_of_saved_percent"] = (
            round((saved / global_tokens_saved) * 100.0, 2) if global_tokens_saved else 0.0
        )
        row["share_of_requests_percent"] = 0.0
        rows.append(row)

    total_requests = sum(int(row["requests"]) for row in rows)
    for row in rows:
        row["share_of_requests_percent"] = (
            round((int(row["requests"]) / total_requests) * 100.0, 2) if total_requests else 0.0
        )

    rows.sort(
        key=lambda row: (
            int(row.get("tokens_saved", 0)),
            int(row.get("before_tokens", 0)),
            int(row.get("requests", 0)),
        ),
        reverse=True,
    )

    return {
        "agents": rows,
        "totals": {
            "requests": total_requests,
            "before_tokens": global_before_tokens,
            "after_tokens": global_after_tokens,
            "output_tokens": global_output_tokens,
            "tokens_saved": global_tokens_saved,
            "savings_percent": (
                round((global_tokens_saved / global_before_tokens) * 100.0, 2)
                if global_before_tokens
                else 0.0
            ),
        },
        "coverage": {
            "logged_requests": len(logs),
            "exact_token_rows": sum(1 for row in rows if row.get("has_exact_tokens")),
            "mode": "request_logs" if logs else "aggregate_fallback",
        },
    }


# Suppress "[transformers] PyTorch was not found" warning emitted when
# transformers is imported for availability checks (e.g. kompress ONNX probe).
# PyTorch is optional in headroom; the warning is not actionable for operators.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("headroom.proxy")

_MULTI_WORKER_CONFIG_ENV = "HEADROOM_PROXY_CONFIG_JSON"

# Env var that opts out of the Rust core deployment smoke test (Hotfix-A0).
# Default behavior: hard-fail at startup if `headroom._core` is unimportable
# (Finding #2 in HEADROOM_PROXY_LOG_FINDINGS_2026_05_03.md — production
# deployment was silently running without the Rust extension and degrading
# every compressed request to a Python-only path or a no-op).
#
# Set to the literal string "false" to start the proxy in degraded
# Python-only mode. Any other value (including unset) keeps the
# fail-loud behavior.
_RUST_CORE_REQUIRED_ENV = "HEADROOM_REQUIRE_RUST_CORE"

# sysexits.h(3) — EX_CONFIG. Process supervisors (systemd, k8s, docker)
# treat this as a deliberate configuration failure rather than a crash, so
# they won't restart-loop on a broken deployment.
_EXIT_CONFIG = 78


def _check_rust_core() -> tuple[str, str | None]:
    """Verify the Rust extension `headroom._core` is loadable at startup.

    Returns a `(status, error)` tuple:
      - ``("loaded", None)``     — `headroom._core.hello()` returned the
        expected sentinel.
      - ``("disabled", reason)`` — opt-out env var was set; proxy starts
        in Python-only degraded mode. `reason` carries the underlying
        import error (or ``None`` if the import actually succeeded).
      - ``("missing", reason)``  — never returned: this branch calls
        ``sys.exit(78)`` so the proxy refuses to start. The branch exists
        only as a typed sentinel for callers that want to reason about
        all three states (e.g. health endpoints).

    Behavior is gated by the ``HEADROOM_REQUIRE_RUST_CORE`` env var:
    any value other than ``"false"`` (case-insensitive) keeps the
    fail-loud default.
    """
    require = os.environ.get(_RUST_CORE_REQUIRED_ENV, "true").strip().lower() != "false"
    try:
        from headroom._core import hello as _rust_hello

        marker = _rust_hello()
    except Exception as exc:  # ImportError, but also any init-time PyO3 failure
        reason = f"{type(exc).__name__}: {exc}"
        if not require:
            logger.warning(
                "event=rust_core_disabled reason=%r opt_out_env=%s=false mode=python_only_degraded",
                reason,
                _RUST_CORE_REQUIRED_ENV,
            )
            return ("disabled", reason)
        # Fail loud. Print to stderr in addition to logging so operators
        # see it even if the logging handler is mis-configured.
        msg = (
            f"FATAL: Rust extension `headroom._core` not loadable.\n"
            f"    error: {reason}\n"
            f"    fix:   `make build-wheel && pip install --force-reinstall "
            f"target/wheels/headroom_*.whl`\n"
            f"    opt-out: set {_RUST_CORE_REQUIRED_ENV}=false to start in "
            f"degraded Python-only mode\n"
        )
        logger.error("event=rust_core_missing reason=%r action=exit_78", reason)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(_EXIT_CONFIG)

    # Import succeeded; sanity-check the marker so we catch a stale or
    # mis-linked .so where the symbol name resolves but returns garbage.
    if marker != "headroom-core":
        reason = f"unexpected marker {marker!r}"
        if not require:
            logger.warning(
                "event=rust_core_disabled reason=%r opt_out_env=%s=false",
                reason,
                _RUST_CORE_REQUIRED_ENV,
            )
            return ("disabled", reason)
        msg = (
            f"FATAL: Rust extension `headroom._core` is loaded but the "
            f"marker function returned {marker!r}; expected 'headroom-core'.\n"
            f"    fix:   rebuild: `make build-wheel && pip install "
            f"--force-reinstall target/wheels/headroom_*.whl`\n"
        )
        logger.error("event=rust_core_marker_mismatch marker=%r action=exit_78", marker)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(_EXIT_CONFIG)

    logger.info("event=rust_core_loaded marker=%r", marker)
    return ("loaded", None)


# Compression pipeline timeout in seconds


from headroom.proxy.handlers import (  # noqa: E402
    AnthropicHandlerMixin,
    BatchHandlerMixin,
    BedrockHandlerMixin,
    GeminiHandlerMixin,
    OpenAIHandlerMixin,
    StreamingMixin,
)


def _apply_stateless_persistence(config: ProxyConfig) -> None:
    """When the proxy runs stateless, force global persisters to in-memory so no
    files are written to the workspace.

    Covers TOIN (the always-on serving writer): it keeps learning patterns
    in-memory but never reads or writes ``toin.json``. An empty ``storage_path``
    makes the backend ``None``, which no-ops load/save/auto-save. The savings
    subsystem is handled separately via ``PrometheusMetrics(stateless=...)``.

    Note: setting ``HEADROOM_TOIN_BACKEND=none`` is NOT sufficient on its own —
    ``ToolIntelligenceNetwork`` falls back to ``config.storage_path`` when no
    backend is passed, so we must clear the path explicitly here.

    Concurrency: ``stateless`` is a per-process flag (set once at ``headroom
    proxy`` launch), never a per-request/per-session value — every session a
    process serves shares it, and two proxies with different settings run as
    separate OS processes with independent TOIN singletons. In the rare case
    where two HeadroomProxy instances with different ``stateless`` settings live
    in ONE process (e.g. tests), this fails closed: the reset forces the
    process-global TOIN in-memory, so a stateless proxy never persists (the safe
    direction). A co-resident stateful proxy would then also stop persisting
    TOIN — acceptable, since not-writing can never leak data.
    """
    if not getattr(config, "stateless", False):
        return


class HeadroomProxy(
    StreamingMixin,
    AnthropicHandlerMixin,
    OpenAIHandlerMixin,
    GeminiHandlerMixin,
    BatchHandlerMixin,
    BedrockHandlerMixin,
):
    """Production-ready Headroom optimization proxy."""

    ANTHROPIC_API_URL = DEFAULT_ANTHROPIC_API_URL
    OPENAI_API_URL = DEFAULT_OPENAI_API_URL
    GEMINI_API_URL = DEFAULT_GEMINI_API_URL
    CLOUDCODE_API_URL = DEFAULT_CLOUDCODE_API_URL
    VERTEX_API_URL = DEFAULT_VERTEX_API_URL

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.config.mode = normalize_proxy_mode(self.config.mode)
        # Record process-wide stateless mode so module-level persisters
        # (output-savings recorder, etc.) can skip workspace writes.
        from headroom import paths as _hr_paths

        _hr_paths.set_process_stateless(config.stateless)
        # Stateless: keep TOIN learning in-memory; never touch toin.json.
        _apply_stateless_persistence(self.config)
        pipeline_extensions = list(config.pipeline_extensions or [])
        probe_recorder = probe_recorder_from_env()
        if probe_recorder is not None:
            pipeline_extensions.append(probe_recorder)
        self.pipeline_extensions = PipelineExtensionManager(
            hooks=config.hooks,
            extensions=pipeline_extensions,
            discover=config.discover_pipeline_extensions,
        )

        self.provider_runtime = build_proxy_provider_runtime(config)
        api_targets = self.provider_runtime.api_targets

        # Preserve the long-standing proxy compatibility surface while keeping
        # provider_runtime as the source of truth for resolved upstream targets.
        HeadroomProxy.ANTHROPIC_API_URL = api_targets.anthropic
        HeadroomProxy.OPENAI_API_URL = api_targets.openai
        HeadroomProxy.GEMINI_API_URL = api_targets.gemini
        HeadroomProxy.CLOUDCODE_API_URL = api_targets.cloudcode
        HeadroomProxy.VERTEX_API_URL = api_targets.vertex
        self.anthropic_provider = self.provider_runtime.pipeline_provider("anthropic")
        self.openai_provider = self.provider_runtime.pipeline_provider("openai")

        # `metrics` is hoisted ahead of transform construction so the
        # transforms can receive `self.metrics` as their compression
        # observer at __init__ time. The forcing function for catching
        # silent strategy regressions: per-strategy counters increment
        # only when wired up here, so the wiring is mandatory, not
        # something we patch in later. (See `RUST_DEV.md` audit notes.)
        self.cost_tracker = (
            CostTracker(
                budget_limit_usd=config.budget_limit_usd,
                budget_period=config.budget_period,
            )
            if config.cost_tracking_enabled
            else None
        )
        self.metrics = PrometheusMetrics(cost_tracker=self.cost_tracker, stateless=config.stateless)

        # Initialize transforms based on routing mode.
        #
        # Phase B PR-B1 retired the IntelligentContextManager / RollingWindow
        # message-dropping branch. Live-zone-only compression (PR-B2..B7) does
        # not drop messages — it operates on content blocks within messages —
        # so the proxy no longer needs a "context manager" transform stage.
        # Reported via metrics as `_context_manager_status = "passthrough"`.
        self._context_manager_status = "passthrough"

        # ContentRouter is the single proxy routing surface. Provider handlers
        # normalize their request shapes into messages or CompressionUnits, and
        # the router chooses SmartCrusher, log/search/diff/code, or Kompress.
        profile_kwargs = proxy_pipeline_kwargs(config)
        router_config = ContentRouterConfig(
            enable_code_aware=config.code_aware_enabled,
            tool_profiles=config.tool_profiles,
            read_lifecycle=ReadLifecycleConfig(enabled=config.read_lifecycle),
            smart_crusher_max_items_after_crush=cast(
                int | None,
                profile_kwargs.get("max_items_after_crush"),
            ),
            smart_crusher_with_compaction=cast(
                bool,
                profile_kwargs.get("smart_crusher_with_compaction", True),
            ),
            ccr_inject_marker=config.ccr_inject_marker,
        )
        if config.disable_kompress:
            router_config.enable_kompress = False
            # Opt-in restore of the legacy behaviour: send fall-through content
            # to PASSTHROUGH instead of the default KOMPRESS fallback strategy.
            if config.disable_kompress_fallback:
                router_config.fallback_strategy = CompressionStrategy.PASSTHROUGH
        # `HEADROOM_LOSSLESS_ONLY=1` routes SmartCrusher through strict
        # marker-free mode: lossless tabular compaction still applies, but
        # any path that would emit a `<<ccr:…>>` marker (row-drop or
        # opaque-blob offload) leaves the content uncompacted instead — so
        # the session needs no CCR retrieval round-trips to stay recoverable.
        if "HEADROOM_LOSSLESS_ONLY" in os.environ:
            router_config.smart_crusher_lossless_only = _get_env_bool(
                "HEADROOM_LOSSLESS_ONLY", False
            )
        # A non-None exclude_tools replaces DEFAULT_EXCLUDE_TOOLS in
        # ContentRouter, so merge rather than assign.
        if config.exclude_tools:
            router_config.exclude_tools = set(DEFAULT_EXCLUDE_TOOLS) | config.exclude_tools
        # protect_tool_results: force-merge named tools into the exclude set
        # so their results are never lossy-compressed, regardless of mode.
        if config.protect_tool_results:
            base = (
                router_config.exclude_tools
                if router_config.exclude_tools is not None
                else set(DEFAULT_EXCLUDE_TOOLS)
            )
            router_config.exclude_tools = base | config.protect_tool_results
        # Token mode: allow compression of older excluded-tool results,
        # and emit search results grouped by file (path once per file
        # instead of repeated on every match line).
        if is_token_mode(config.mode):
            router_config.protect_recent_reads_fraction = 0.3
            router_config.search_group_by_file = True
        if config.protect_tool_results:
            router_config.protect_recent_reads_fraction = 0.0
        # `--compress-user-messages` flips the router's default skip rule.
        # Off by default for prefix-cache safety; enabled for workloads where
        # user-message content dominates input (OpenAI/Azure chat with pasted
        # code/RAG context — see issue #454).
        if profile_kwargs.get("compress_user_messages"):
            router_config.skip_user_messages = False
        # Kompress (lossy ML text compression) is resolved per provider. The
        # global `disable_kompress` above is the baseline for both; a per-
        # provider override (disable_kompress_{anthropic,openai}) wins when set.
        # Only `enable_kompress` differs between providers — routing, tool
        # exclusion, and read-protection are identical — so when both resolve
        # the same we reuse ONE ContentRouter instance and the Kompress model
        # still loads once (startup warmup dedupes transforms by id()).
        base_kompress_disabled = not router_config.enable_kompress
        anthropic_kompress_disabled = (
            base_kompress_disabled
            if config.disable_kompress_anthropic is None
            else config.disable_kompress_anthropic
        )
        openai_kompress_disabled = (
            base_kompress_disabled
            if config.disable_kompress_openai is None
            else config.disable_kompress_openai
        )

        def _router_config_for(kompress_disabled: bool) -> ContentRouterConfig:
            if kompress_disabled == base_kompress_disabled:
                return router_config
            return replace(router_config, enable_kompress=not kompress_disabled)

        cache_aligner = CacheAligner(CacheAlignerConfig(enabled=False))
        anthropic_router = ContentRouter(
            _router_config_for(anthropic_kompress_disabled), observer=self.metrics
        )
        openai_router = (
            anthropic_router
            if openai_kompress_disabled == anthropic_kompress_disabled
            else ContentRouter(_router_config_for(openai_kompress_disabled), observer=self.metrics)
        )
        self._code_aware_status = "lazy" if config.code_aware_enabled else "disabled"

        _intercept_prefix: list = []
        if os.environ.get("HEADROOM_INTERCEPT_ENABLED"):
            from headroom.proxy.interceptors import ToolResultInterceptorTransform

            _intercept_prefix = [ToolResultInterceptorTransform()]

        self.anthropic_pipeline = TransformPipeline(
            transforms=[*_intercept_prefix, cache_aligner, anthropic_router],
            provider=self.anthropic_provider,
        )
        self.openai_pipeline = TransformPipeline(
            transforms=[*_intercept_prefix, cache_aligner, openai_router],
            provider=self.openai_provider,
        )

        # Initialize components
        self.cache = (
            SemanticCache(
                max_entries=config.cache_max_entries,
                ttl_seconds=config.cache_ttl_seconds,
            )
            if config.cache_enabled
            else None
        )

        self.rate_limiter = (
            TokenBucketRateLimiter(
                requests_per_minute=config.rate_limit_requests_per_minute,
                tokens_per_minute=config.rate_limit_tokens_per_minute,
            )
            if config.rate_limit_enabled
            else None
        )

        # `cost_tracker` and `metrics` were hoisted to before transforms so
        # ContentRouter / SmartCrusher could take `self.metrics` as their
        # compression observer at __init__ time.

        # Prefix cache tracking: freeze already-cached messages to avoid
        # invalidating the provider's prefix cache with our transforms
        from headroom.cache.prefix_tracker import PrefixFreezeConfig, SessionTrackerStore

        self.session_tracker_store = SessionTrackerStore(
            default_config=PrefixFreezeConfig(
                enabled=config.prefix_freeze_enabled,
                session_ttl_seconds=config.prefix_freeze_session_ttl,
            )
        )

        # Compression cache store for token mode (session-scoped). The dict
        # itself is mutated under `_compression_caches_lock`; the per-session
        # `CompressionCache` instances have their own internal lock guarding
        # `_cache`/`_stable_hashes`/`_first_seen` against concurrent
        # async-dispatched requests for the same session.
        self._compression_caches: dict[str, CompressionCache] = {}
        self._compression_caches_lock = threading.RLock()

        self.logger = (
            RequestLogger(
                log_file=config.log_file,
                log_full_messages=config.log_full_messages,
            )
            if config.log_requests
            else None
        )

        # Enterprise security plugin (loaded dynamically if available + licensed)
        self.security = None

        # HTTP client
        self.http_client: httpx.AsyncClient | None = None
        # HTTP/1.1-only client for ChatGPT passthrough (Cloudflare challenges
        # our HTTP/2 fingerprint on its sensitive account endpoints).
        self.http_client_h1: httpx.AsyncClient | None = None

        # Shared cold-start warmup registry (populated by startup()).
        # Holds typed slots with loaded / loading / null / error status for
        # each preloaded heavy asset. Exposed as ``proxy.warmup`` and
        # serialized by the /debug/warmup route (Unit 5).
        self.warmup: WarmupRegistry = WarmupRegistry()
        # Unit 3: live registry of Codex WS sessions. Populated by
        # ``handle_openai_responses_ws`` on accept; drained in its
        # outermost ``finally``. Consumed by ``/debug/ws-sessions``.
        self.ws_sessions: WebSocketSessionRegistry = WebSocketSessionRegistry()

        # Unit 4: bounded pre-upstream concurrency for the Anthropic HTTP
        # path. Caps how many ``handle_anthropic_messages`` calls may be
        # running deep-copy / first-stage compression / memory-context
        # lookup / upstream connect concurrently. ``/livez``, ``/readyz``,
        # ``/health``, ``/metrics``, ``/stats``, and the Codex WS path are
        # intentionally NOT gated by this semaphore.
        #
        # A value of ``0`` or negative disables the semaphore (unbounded
        # mode); this is useful for the Unit 6 counter-factual where we
        # deliberately reproduce the original starvation. The default is
        # ``max(2, min(8, os.cpu_count() or 4))``.
        _pre_upstream_cfg = config.anthropic_pre_upstream_concurrency
        if _pre_upstream_cfg is None:
            _pre_upstream_resolved = max(2, min(8, os.cpu_count() or 4))
        else:
            _pre_upstream_resolved = _pre_upstream_cfg
        self.anthropic_pre_upstream_concurrency: int = _pre_upstream_resolved
        self.anthropic_pre_upstream_acquire_timeout_seconds = float(
            config.anthropic_pre_upstream_acquire_timeout_seconds
        )
        self.anthropic_pre_upstream_memory_context_timeout_seconds = float(
            config.anthropic_pre_upstream_memory_context_timeout_seconds
        )
        if _pre_upstream_resolved > 0:
            self.anthropic_pre_upstream_sem: asyncio.Semaphore | None = asyncio.Semaphore(
                _pre_upstream_resolved
            )
        else:
            self.anthropic_pre_upstream_sem = None

        # Dedicated compression executor — see C3 in the audit followup.
        # Replaces ``asyncio.to_thread(...)`` for ``pipeline.apply()`` calls
        # so that:
        #   1. Compression work is bounded — CPU-bound Rust runs here, and
        #      bursts cannot starve other ``asyncio.to_thread`` callers
        #      sharing the loop's default executor (file IO, etc.).
        #   2. Tasks that exceed ``COMPRESSION_TIMEOUT_SECONDS`` and complete
        #      *after* the asyncio future was cancelled are counted in the
        #      ``compression_leaked_threads`` gauge — Python cannot preempt
        #      the worker, so this is the only signal that some pool slots
        #      are sitting on stuck work.
        _compression_max_cfg = config.compression_max_workers
        if _compression_max_cfg is None:
            _compression_max = min(32, (os.cpu_count() or 1) * 4)
        else:
            _compression_max = max(1, _compression_max_cfg)
        self.compression_max_workers: int = _compression_max
        self._compression_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_compression_max,
            thread_name_prefix="headroom-compress",
        )
        # Phase 3 (#1171): off-path background compression. When enabled, a
        # cold-start-large request (frozen=0 + large live zone) forwards
        # uncompressed immediately and enqueues the compression here instead of
        # blocking the request thread under the 30s budget (which leaks a
        # non-preemptible worker -> executor saturation -> cascade). Default
        # off (opt-in), fail-open. Per-process, matching _compression_caches.
        self._background_compression_enabled: bool = os.environ.get(
            "HEADROOM_BACKGROUND_COMPRESSION", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        # Traffic Learner (live pattern extraction from proxy traffic)
        # Only activates with --learn flag; requires --memory for backend
        self.traffic_learner: TrafficLearner | None = None
        self.traffic_learning_agent_type: str = config.traffic_learning_agent_type
        if config.traffic_learning_enabled:
            from headroom.memory.traffic_learner import TrafficLearner

            self.traffic_learner = TrafficLearner(
                user_id=os.environ.get("HEADROOM_USER_ID", os.environ.get("USER", "default")),
                agent_type=config.traffic_learning_agent_type,
                min_evidence=config.traffic_learning_min_evidence,
            )

        self.pipeline_extensions.emit(
            PipelineStage.SETUP,
            operation="proxy.setup",
            metadata={
                "mode": self.config.mode,
                "optimize": self.config.optimize,
                "backend": self.config.backend,
                "memory_enabled": self.config.memory_enabled,
            },
        )

    async def _run_compression_in_executor(
        self,
        fn,  # noqa: ANN001 — caller-supplied no-arg sync callable
        *,
        timeout: float,
    ):
        """Run a synchronous compression callable on the bounded executor
        with cancel-aware metrics.

        Replaces ``asyncio.wait_for(asyncio.to_thread(fn), timeout=...)``.

        Why a dedicated executor: the proxy's compression path is CPU-bound
        Rust work that releases the GIL via ``py.allow_threads``. Sharing
        the loop's default executor (used by ``asyncio.to_thread``) means
        a burst of slow compressions can starve unrelated ``to_thread``
        callers (file IO, etc.). The compression executor is sized
        independently via ``config.compression_max_workers``.

        Why "cancel-aware metrics": when ``asyncio.wait_for`` times out, it
        cancels the *asyncio future*. The underlying
        ``concurrent.futures.Future`` from ``run_in_executor`` cannot
        actually cancel a thread that has started — Python has no way to
        preempt running CPython bytecode or in-flight Rust calls. The
        worker keeps running to completion, ignored. We detect this by
        marking the call timed out on the asyncio side and incrementing
        ``_compression_leaked_threads`` from the worker's ``finally``
        block after it eventually finishes. Jobs that time out before a
        worker starts are removed from the queued gauge instead. Operators
        can see leaked-thread rate and queue pressure climbing in
        ``/stats`` before the pool fills up.

        Args:
            fn: A no-arg sync callable that runs the compression. Must not
                raise asyncio Cancellation; if it does, the wrapper still
                decrements the in-flight gauge but the leaked-thread
                counter may double-count.
            timeout: Wall-clock timeout for the asyncio side. The
                executor worker keeps running past this (Python limitation
                — see above), but at least the awaiter unblocks.

        Returns:
            Whatever ``fn()`` returns.

        Raises:
            ``asyncio.TimeoutError`` if the callable doesn't return within
            ``timeout``. Any exception raised by ``fn`` propagates
            unchanged.
        """
        loop = asyncio.get_running_loop()
        queued_at = time.monotonic()
        state = {"queued": True, "timed_out": False}
        with self._compression_metrics_lock:
            self._compression_queued += 1
            if self._compression_queued > self._compression_queued_max:
                self._compression_queued_max = self._compression_queued

        def _wrapped():  # noqa: ANN202
            started_at = time.monotonic()
            queue_wait = started_at - queued_at
            with self._compression_metrics_lock:
                if state["queued"]:
                    self._compression_queued -= 1
                    state["queued"] = False
                self._compression_queue_wait_seconds_total += queue_wait
                if queue_wait > self._compression_queue_wait_seconds_max:
                    self._compression_queue_wait_seconds_max = queue_wait
                self._compression_in_flight += 1
                if self._compression_in_flight > self._compression_in_flight_max:
                    self._compression_in_flight_max = self._compression_in_flight
            try:
                return fn()
            finally:
                elapsed = time.monotonic() - started_at
                with self._compression_metrics_lock:
                    self._compression_in_flight -= 1
                    self._compression_run_seconds_total += elapsed
                    if elapsed > self._compression_run_seconds_max:
                        self._compression_run_seconds_max = elapsed
                    if state["timed_out"]:
                        self._compression_leaked_threads += 1

        future = loop.run_in_executor(self._compression_executor, _wrapped)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            with self._compression_metrics_lock:
                state["timed_out"] = True
                if state["queued"]:
                    self._compression_queued -= 1
                    state["queued"] = False
                    self._compression_queue_timeouts += 1
            raise

    async def _run_compression_background(self, fn):  # noqa: ANN001, ANN201
        """Run a compression callable on the shared executor with NO request-
        coupled deadline (Phase 3 off-path, #1171).

        Unlike ``_run_compression_in_executor`` there is no ``asyncio.wait_for``
        and no leaked-thread accounting: no caller is waiting, so a slow run
        backs up the background queue rather than starving the request executor.
        Runs on the dedicated single-thread background executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._background_compression_executor, fn)

    def _get_compression_cache(self, session_id: str) -> CompressionCache:
        """Get or create a CompressionCache for a session.

        Thread-safe under `_compression_caches_lock`: a concurrent pair of
        `_get_compression_cache(session_id)` calls (e.g. two async requests
        for the same conversation) must return the **same** instance,
        otherwise the per-session cache state splits and the two halves
        diverge across requests.
        """
        with self._compression_caches_lock:
            if session_id not in self._compression_caches:
                from headroom.cache.compression_cache import CompressionCache

                # Evict oldest caches if at capacity
                if len(self._compression_caches) >= MAX_COMPRESSION_CACHE_SESSIONS:
                    # Remove oldest quarter to amortize cleanup cost
                    oldest_keys = list(self._compression_caches.keys())[
                        : MAX_COMPRESSION_CACHE_SESSIONS // 4
                    ]
                    for key in oldest_keys:
                        del self._compression_caches[key]
                    logger.info(
                        "Evicted %d compression caches (exceeded %d max sessions)",
                        len(oldest_keys),
                        MAX_COMPRESSION_CACHE_SESSIONS,
                    )

                self._compression_caches[session_id] = CompressionCache()
            return self._compression_caches[session_id]

    def _setup_code_aware(self, config: ProxyConfig, transforms: list) -> str:
        """Set up code-aware compression if enabled.

        Args:
            config: Proxy configuration
            transforms: Transform list to append to

        Returns:
            Status string for logging: 'enabled', 'disabled', 'available', 'unavailable'
        """
        if config.code_aware_enabled:
            if is_tree_sitter_available():
                code_config = CodeCompressorConfig(
                    preserve_imports=True,
                    preserve_signatures=True,
                    preserve_type_annotations=True,
                )
                # CodeAware runs after the content/structure transforms.
                # Phase B PR-B1 retired the trailing context_manager so we
                # append rather than insert(-1).
                transforms.append(CodeAwareCompressor(code_config))
                return "enabled"
            else:
                logger.warning(
                    "Code-aware compression requested but tree-sitter not installed. "
                    "Install with: pip install headroom-ai[code]"
                )
                return "unavailable"
        else:
            if is_tree_sitter_available():
                return "available"  # Available but not enabled
            return "disabled"

    async def startup(self):
        """Initialize async resources."""
        self.pipeline_extensions.emit(
            PipelineStage.PRE_START,
            operation="proxy.startup",
            metadata={"port": self.config.port, "host": self.config.host},
        )
        # Resolve TLS verification: a custom CA bundle (corporate PKI) if one
        # is configured, else a strict-relaxed default context when
        # HEADROOM_TLS_STRICT=0, else httpx's default strict verification.
        _verify = build_httpx_verify()
        _client_kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(
                connect=self.config.connect_timeout_seconds,
                read=self.config.request_timeout_seconds,
                write=self.config.request_timeout_seconds,
                pool=self.config.connect_timeout_seconds,
            ),
            "limits": httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
                keepalive_expiry=self.config.keepalive_expiry,
            ),
            "verify": _verify,
        }
        self.http_client = httpx.AsyncClient(http2=self.config.http2, **_client_kwargs)
        # Reuse the primary client when HTTP/2 is already off; otherwise keep a
        # dedicated HTTP/1.1 client for ChatGPT passthrough.
        self.http_client_h1 = (
            self.http_client
            if not self.config.http2
            else httpx.AsyncClient(http2=False, **_client_kwargs)
        )
        logger.info("Headroom Proxy started")
        logger.info(f"Optimization: {'ENABLED' if self.config.optimize else 'DISABLED'}")
        self.config.mode = normalize_proxy_mode(self.config.mode)
        logger.info(f"Mode: {self.config.mode}")
        if self.config.mode == PROXY_MODE_TOKEN:
            logger.info("  Prefix freeze: re-freeze after compression")
            logger.info("  Read protection window: 30%% of excluded-tool messages")
            logger.info("  CCR TTL: extended for session lifetime")
            logger.info("  Compression cache: active")
        if self.config.mode == PROXY_MODE_CACHE:
            logger.info("  Prefix freeze: strict (all prior turns immutable)")
            logger.info("  Mutations: latest turn only")
        logger.info(f"Caching: {'ENABLED' if self.config.cache_enabled else 'DISABLED'}")
        logger.info(f"Rate Limiting: {'ENABLED' if self.config.rate_limit_enabled else 'DISABLED'}")
        logger.info(
            f"Connection Pool: max_connections={self.config.max_connections}, "
            f"max_keepalive={self.config.max_keepalive_connections}, "
            f"http2={'ENABLED' if self.config.http2 else 'DISABLED'}"
        )

        # Unit 4 pre-upstream concurrency announcement. Report the resolved
        # value (auto-detected vs. explicit) so operators can correlate
        # ``pre_upstream_wait_ms`` log lines with the configured cap.
        if self.anthropic_pre_upstream_sem is None:
            logger.info("Anthropic pre-upstream concurrency: unbounded (explicitly disabled)")
        else:
            _explicit = self.config.anthropic_pre_upstream_concurrency
            _origin = "auto-detected" if _explicit is None else "explicit"
            logger.info(
                "Anthropic pre-upstream concurrency: %d (%s)",
                self.anthropic_pre_upstream_concurrency,
                _origin,
            )
        logger.info(
            "Anthropic pre-upstream timeouts: acquire=%.1fs compression=%.1fs memory_context=%.1fs",
            self.anthropic_pre_upstream_acquire_timeout_seconds,
            float(COMPRESSION_TIMEOUT_SECONDS),
            self.anthropic_pre_upstream_memory_context_timeout_seconds,
        )

        logger.info("Smart Routing: ENABLED (ContentRouter is always active)")

        # Eagerly load ALL compressors, parsers, and detectors at startup
        # This eliminates cold-start latency spikes on first requests.
        # Iterate BOTH pipelines (Anthropic + OpenAI) and dedupe transforms
        # by id() so shared-transform instances never load twice. The
        # resulting status dict is merged into ``self.warmup`` so /debug/warmup
        # (Unit 5) and /readyz have a single source of truth.
        self._kompress_status = "not installed"
        eager_status: dict[str, str] = {}

        if self.config.optimize:
            logger.info("Pre-loading compressors and parsers...")
            seen_transform_ids: set[int] = set()
            pipelines = (self.anthropic_pipeline, self.openai_pipeline)
            for pipeline in pipelines:
                for transform in pipeline.transforms:
                    if id(transform) in seen_transform_ids:
                        continue
                    seen_transform_ids.add(id(transform))
                    if not hasattr(transform, "eager_load_compressors"):
                        continue
                    try:
                        transform_status = transform.eager_load_compressors()
                    except Exception as exc:
                        logger.warning(
                            "Eager preload failed for %s: %s",
                            type(transform).__name__,
                            exc,
                        )
                        continue
                    if not isinstance(transform_status, dict):
                        continue
                    # Merge: later writers win only if the key wasn't set.
                    # Preload a transform ONCE — if another pipeline also has
                    # ``eager_load_compressors`` it contributes only new keys.
                    for key, value in transform_status.items():
                        eager_status.setdefault(key, value)
                    self.warmup.merge_transform_status(transform_status)

        # Update internal status from eager loading results
        if eager_status.get("kompress") == "enabled":
            self._kompress_status = "enabled"
        if eager_status.get("code_aware") == "enabled":
            self._code_aware_status = "enabled"

        # Log component status
        if self._kompress_status == "enabled":
            logger.info("Kompress: ENABLED (ModernBERT token compressor)")
        elif self.config.optimize:
            logger.info("Kompress: not installed (pip install headroom-ai[ml] for ML compression)")

        if self._code_aware_status == "enabled":
            logger.info("Code-Aware: ENABLED (AST-based compression)")
            if "tree_sitter" in eager_status:
                logger.info(f"Tree-Sitter: {eager_status['tree_sitter']}")
        elif self._code_aware_status == "lazy":
            logger.info("Code-Aware: LAZY (will load when code content detected)")
        elif self._code_aware_status == "available":
            logger.info("Code-Aware: available but disabled (use --code-aware)")
        elif self._code_aware_status == "unavailable":
            logger.info("Code-Aware: not installed (pip install headroom-ai[code])")
        elif self._code_aware_status == "disabled":
            logger.info("Code-Aware: DISABLED")

        if eager_status.get("magika") == "enabled":
            logger.info("Magika: ENABLED (ML content detection)")

        if self.memory_handler:
            if (
                self.config.memory_backend == "qdrant-neo4j"
                and not self.config.memory_neo4j_password
            ):
                logger.warning(
                    "NEO4J password is not set — using default credentials is insecure in production"
                )
            self.warmup.memory_backend.mark_loading()
            memory_status = self.memory_handler.health_status()
            if memory_status.get("initialized"):
                self.warmup.memory_backend.mark_loaded(
                    handle=self.memory_handler,
                    backend=memory_status.get("backend"),
                )
                # Force one embed call so the ONNX graph is compiled now,
                # not lazily during the first request. Best-effort — any
                # failure is swallowed inside warmup_embedder.
                self.warmup.memory_embedder.mark_loading()
                warmed = await self.memory_handler.warmup_embedder()
                if warmed:
                    self.warmup.memory_embedder.mark_loaded()
                else:
                    # Not an error — e.g. qdrant-neo4j has no embedder slot
                    # we can reach, or the backend simply exposes no handle.
                    self.warmup.memory_embedder.mark_null()
            else:
                if self.warmup.memory_backend.status != "error":
                    self.warmup.memory_backend.mark_null()
                self.warmup.memory_embedder.mark_null()
            logger.info(
                "Memory: ENABLED "
                f"(backend={memory_status['backend']}, initialized={memory_status['initialized']})"
            )
        else:
            logger.info("Memory: DISABLED")

        # CCR status
        ccr_features = []
        if self.config.ccr_inject_tool:
            ccr_features.append("tool_injection")
        if self.config.ccr_handle_responses:
            ccr_features.append("response_handling")
        if self.config.ccr_context_tracking:
            ccr_features.append("context_tracking")
        if self.config.ccr_proactive_expansion:
            ccr_features.append("proactive_expansion")
        if ccr_features:
            logger.info(f"CCR (Compress-Cache-Retrieve): ENABLED ({', '.join(ccr_features)})")
        else:
            logger.info("CCR: DISABLED")
        logger.info(f"Savings history: {self.metrics.savings_tracker.storage_path}")

        # Reset and rebuild the quota tracker registry for this server instance.
        # reset_quota_registry() ensures a clean slate when the proxy is restarted
        # (e.g. in tests that spin up multiple app instances in the same process).
        reset_quota_registry()
        registry = get_quota_registry()
        tracker = configure_subscription_tracker(
            poll_interval_s=self.config.subscription_poll_interval_s,
            active_window_s=self.config.subscription_active_window_s,
            enabled=self.config.subscription_tracking_enabled,
        )
        registry.register(tracker)
        registry.register(get_codex_rate_limit_state())
        registry.register(get_copilot_quota_tracker())
        await registry.start_all()

        if self.config.subscription_tracking_enabled:
            logger.info(
                "Subscription tracking: ENABLED "
                f"(poll_interval={self.config.subscription_poll_interval_s}s, "
                f"active_window={self.config.subscription_active_window_s}s)"
            )
        else:
            logger.info("Subscription tracking: DISABLED")

        copilot_tracker = get_copilot_quota_tracker()
        if copilot_tracker.is_available():
            logger.info("GitHub Copilot quota tracking: ENABLED")
        else:
            logger.info(
                "GitHub Copilot quota tracking: DISABLED "
                "(set GITHUB_TOKEN or GITHUB_COPILOT_GITHUB_TOKEN to enable)"
            )

