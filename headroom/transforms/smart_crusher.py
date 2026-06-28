
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from ..ccr.tool_injection import CCR_TOOL_NAME
from ..config import CCRConfig, TransformResult
from ..tokenizer import Tokenizer
from ..utils import compute_short_hash, create_tool_digest_marker, deep_copy_messages
from .base import Transform

logger = logging.getLogger(__name__)


# Lossless-compaction renderers known to the Rust core — mirrors
# `CompactionStage::SUPPORTED_FORMAT_NAMES` in
# `crates/headroom-core/.../compaction/mod.rs`.
_SUPPORTED_COMPACTION_FORMATS = ("csv-schema", "json", "markdown-kv")


# ─── CCR sentinel ─────────────────────────────────────────────────────────
#
# When SmartCrusher's lossy path drops rows, it appends a sentinel object
# `{"_ccr_dropped": "<<ccr:HASH N_rows_offloaded>>"}` to the kept-items
# array. The LLM sees this in the prompt and can ask for the original via
# the CCR retrieval tool. Downstream consumers that iterate the array
# expecting a uniform schema (e.g. `for e in entries: e["level"]`) need
# to skip the sentinel — that's what `strip_ccr_sentinels` is for.

CCR_SENTINEL_KEY = "_ccr_dropped"


def is_ccr_sentinel(item: Any) -> bool:
    """True if `item` is a CCR-dropped sentinel object."""
    return isinstance(item, dict) and CCR_SENTINEL_KEY in item


def strip_ccr_sentinels(items: Any) -> Any:
    """Return `items` with any CCR-dropped sentinel objects filtered out.

    Pass this through any iteration over a compressed array's contents
    when your code expects a uniform-schema list of records. The sentinel
    carries a `<<ccr:HASH ...>>` marker for the LLM and shouldn't be
    confused for a record — it has only the `_ccr_dropped` key.

    Non-list inputs pass through unchanged so callers can wrap whatever
    `json.loads` returned without first checking the shape.
    """
    if not isinstance(items, list):
        return items
    return [x for x in items if not is_ccr_sentinel(x)]


# ─── Tool-name attribution ────────────────────────────────────────────────


def _build_tool_name_index(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map tool_call_id/tool_use_id → tool name across OpenAI + Anthropic formats.

    Skips entries where id or name is missing; those calls still crush, but
    won't contribute a tool-name to the ``smart_crush`` tag.
    """
    index: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            tc_id = tc.get("id")
            name = (tc.get("function") or {}).get("name")
            if tc_id and name:
                index[tc_id] = name
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                bid = block.get("id")
                name = block.get("name")
                if bid and name:
                    index[bid] = name
    return index


def _format_smart_crush_transform(count: int, tool_names: list[str]) -> str:
    """Format ``smart_crush:<count>[:<name1,name2,...>]``.

    Names are included when known so consumers can show what was crushed. Empty
    names fall back to the count-only form for backwards compatibility.
    """
    if tool_names:
        return f"smart_crush:{count}:{','.join(tool_names)}"
    return f"smart_crush:{count}"


# ─── Public dataclasses ───────────────────────────────────────────────────


@dataclass
class CrushResult:
    """Result from `SmartCrusher.crush()`.

    Used by `ContentRouter` when routing JSON arrays to `SmartCrusher`.
    """

    compressed: str
    original: str
    was_modified: bool
    strategy: str = "passthrough"


@dataclass
class SmartCrusherConfig:
    """Configuration for SmartCrusher.

    SCHEMA-PRESERVING: output contains only items from the original
    array. No wrappers, no generated text, no metadata keys.

    Field names + defaults match the Rust `SmartCrusherConfig` byte-for-
    byte; the shim copies these straight into the PyO3 constructor.
    """

    enabled: bool = True
    min_items_to_analyze: int = 5
    min_tokens_to_crush: int = 200
    variance_threshold: float = 2.0
    uniqueness_threshold: float = 0.1
    similarity_threshold: float = 0.8
    max_items_after_crush: int = 15
    preserve_change_points: bool = True
    factor_out_constants: bool = False
    include_summaries: bool = False
    use_feedback_hints: bool = True
    dedup_identical_items: bool = True
    first_fraction: float = 0.3
    last_fraction: float = 0.15
    # Minimum byte-savings ratio for the lossless Table/CSV compaction
    # path to win over the lossy path (0.15, matching the Rust default —
    # the two must stay in lockstep, see config.rs). Lossless output
    # needs no CCR retrieval round-trip when the model wants more rows,
    # so it gets a lower bar than the lossy path. Mainly raised in tests
    # and KV experiments — KV repeats field names per row, so it clears
    # the gate less often than CSV.
    lossless_min_savings_ratio: float = 0.15
    # Strict lossless mode. When True, lossless tabular compaction still
    # applies, but any path that would otherwise emit a CCR marker — the
    # lossy row-drop sentinel AND opaque-blob offload — leaves the content
    # uncompacted instead. The output is always marker-free and fully
    # byte-recoverable: rows are never dropped and opaque cells render
    # inline. Default False (markers allowed). Mirrors the Rust default.
    lossless_only: bool = False

    # Compaction heuristics (mirror Rust CompactConfig; see
    # crates/headroom-core/src/transforms/smart_crusher/compaction/compactor.rs).
    # A field is "core" if present in at least this fraction of rows.
    compaction_core_field_fraction: float = 0.8
    # Below this fraction of core keys, treat the array as heterogeneous
    # and look for a discriminator to bucket by.
    compaction_heterogeneous_core_ratio: float = 0.6
    # Cap on inner-key count for nested-uniform flattening.
    compaction_max_flatten_inner_keys: int = 6
    # Bucket-count bounds for discriminator usefulness.
    compaction_min_buckets: int = 2
    compaction_max_buckets: int = 8


# ─── Rust-backed SmartCrusher ─────────────────────────────────────────────


class SmartCrusher(Transform):
    """Rust-backed `SmartCrusher` (via PyO3 / `headroom._core`).

    Same `__init__` and method shapes as the retired Python class —
    drop-in replacement. The `crush()` and `_smart_crush_content()`
    methods delegate every byte to Rust; `apply()` keeps the
    Transform-protocol orchestration in Python (message walking,
    digest-marker insertion, token counting) since that's mostly glue
    around the per-message compression call.
    """

    name = "smart_crusher"

    def __init__(
        self,
        config: SmartCrusherConfig | None = None,
        relevance_config: Any = None,
        scorer: Any = None,
        ccr_config: CCRConfig | None = None,
        with_compaction: bool = True,
        observer: Any = None,
        compaction_format: str | None = None,
        lossless_only: bool | None = None,
    ):
        # Hard import — no Python fallback. If the wheel is missing the
        # caller must build it (scripts/build_rust_extension.sh) or
        # install a prebuilt one. Failing loudly is better than silent
        # degradation; see feedback memory `feedback_no_silent_fallbacks.md`.
        from headroom._core import (
            SmartCrusher as _RustSmartCrusher,
        )
        from headroom._core import (
            SmartCrusherConfig as _RustSmartCrusherConfig,
        )

        cfg = config or SmartCrusherConfig()
        self.config = cfg
        self._with_compaction = with_compaction
        # Strict lossless mode. An explicit `lossless_only=` kwarg wins
        # over the config field, so callers can flip it without rebuilding
        # a whole config. `crush(..., lossless_only=...)` overrides again
        # per call. getattr fallback: callers may pass the SDK-side
        # `headroom.config.SmartCrusherConfig`, which also carries it.
        self._lossless_only = (
            bool(getattr(cfg, "lossless_only", False))
            if lossless_only is None
            else bool(lossless_only)
        )
        # `observer`: see `headroom.transforms.observability`. The
        # legacy proxy pipeline uses SmartCrusher.apply() directly
        # (no ContentRouter); without an observer here, those
        # compressions would be invisible to per-strategy metrics —
        # exactly the silent-regression class we're guarding against.
        self._observer = observer

        # CCR config is preserved on `self` for callers that read it
        # back (`headroom.proxy.server` does). Both `enabled=False` and
        # `inject_retrieval_marker=False` collapse to the Rust crusher's
        # `enable_ccr_marker=False` gate — when either is off, the
        # lossy row-drop path skips marker emission AND the CCR store
        # write (no point storing a payload nothing in the prompt can
        # reference; storing it under `enabled=False` would also be a
        # surprise side effect the user explicitly disabled).
        #
        # Default falls through to `CCRConfig()` so direct callers
        # (the proxy and tests that don't pass an explicit config) get
        # the documented dataclass defaults (`enabled=True,
        # inject_retrieval_marker=True`). The previous override here
        # set `inject_retrieval_marker=False` as a no-op-intent hack
        # back when the Rust port silently ignored the flag; now that
        # the flag is honored, that override would actively suppress
        # markers + store writes for every caller.
        #
        # Scope: gates ONLY the row-drop sentinel path. Stage-3c.2
        # opaque-string CCR substitutions still emit always — they have
        # no Python equivalent and no production caller has asked for
        # them to be suppressed.
        if ccr_config is None:
            self._ccr_config = CCRConfig()
        else:
            self._ccr_config = ccr_config

        # `relevance_config` and `scorer` remain in the signature for
        # source compatibility, but the Rust port doesn't support
        # overrides yet (it always uses `HybridScorer` from the
        # relevance crate; the Python-bridged constructor surface
        # arrives in Stage 3c.2). Silently dropping a user-supplied
        # scorer would be a textbook silent fallback — if a caller
        # depends on a custom scoring function and we ignore it, the
        # compression they get back is wrong in a way they cannot see.
        # Fail loud instead. See `feedback_no_silent_fallbacks.md`.
        if relevance_config is not None or scorer is not None:
            raise NotImplementedError(
                "SmartCrusher: custom `relevance_config` / `scorer` "
                "overrides are not yet supported by the Rust-backed "
                "implementation. Pass `None` to use the default "
                "HybridScorer. Tracked in RUST_DEV.md; full support "
                "lands with Stage 3c.2's relevance-crate Python bridge."
            )

        # to learn from. Skipping import at __init__ keeps cold-start
        # fast for environments where telemetry is disabled.

        # F2.2: per-request CompressionPolicy, set from
        # ``kwargs["compression_policy"]`` at the start of ``apply()``
        # Defaults to ``None`` so the direct ``crush()`` / ``crush_array_json()``
        # / ``compact_document_json()`` entry points (which don't go
        # writes are not gated. Same pattern as the existing
        # ``_runtime_target_ratio`` / ``_runtime_kompress_model``
        # fields in ContentRouter.
        self._runtime_compression_policy: Any = None

        # Build the Rust crusher with every field from the Python
        # config, plus the relevance_threshold default (0.3) — the
        # Python dataclass doesn't carry that field; it lives on
        # `RelevanceScorerConfig` instead. Kept as a kwargs dict so the
        # per-call `crush(..., lossless_only=...)` override can rebuild an
        # alternate crusher with just that one field flipped.
        self._RustSmartCrusher = _RustSmartCrusher
        self._RustSmartCrusherConfig = _RustSmartCrusherConfig
        self._rust_cfg_kwargs = {
            "enabled": cfg.enabled,
            "min_items_to_analyze": cfg.min_items_to_analyze,
            "min_tokens_to_crush": cfg.min_tokens_to_crush,
            "variance_threshold": cfg.variance_threshold,
            "uniqueness_threshold": cfg.uniqueness_threshold,
            "similarity_threshold": cfg.similarity_threshold,
            "max_items_after_crush": cfg.max_items_after_crush,
            "preserve_change_points": cfg.preserve_change_points,
            "factor_out_constants": cfg.factor_out_constants,
            "include_summaries": cfg.include_summaries,
            "use_feedback_hints": cfg.use_feedback_hints,
            "dedup_identical_items": cfg.dedup_identical_items,
            "first_fraction": cfg.first_fraction,
            "last_fraction": cfg.last_fraction,
            "relevance_threshold": 0.3,
            "enable_ccr_marker": (
                self._ccr_config.enabled and self._ccr_config.inject_retrieval_marker
            ),
            "lossless_only": self._lossless_only,
            # getattr fallbacks: callers may pass the structurally-similar
            # `headroom.config.SmartCrusherConfig` (MCP server, SDK) or a
            # pre-existing config object that predates these fields.
            "lossless_min_savings_ratio": getattr(cfg, "lossless_min_savings_ratio", 0.15),
            "compaction_core_field_fraction": getattr(cfg, "compaction_core_field_fraction", 0.8),
            "compaction_heterogeneous_core_ratio": getattr(
                cfg, "compaction_heterogeneous_core_ratio", 0.6
            ),
            "compaction_max_flatten_inner_keys": getattr(
                cfg, "compaction_max_flatten_inner_keys", 6
            ),
            "compaction_min_buckets": getattr(cfg, "compaction_min_buckets", 2),
            "compaction_max_buckets": getattr(cfg, "compaction_max_buckets", 8),
        }
        # Default: lossless-first compaction (PR4). Lossless wins for
        # cleanly tabular input where it saves ≥ 30% bytes; otherwise
        # falls through to the lossy path with CCR-Dropped retrieval
        # markers. Pass `with_compaction=False` to opt into the
        # pre-PR4 lossy-only path (used by retention-property tests
        # that depend on row-level item preservation).
        #
        # `compaction_format` picks the lossless renderer:
        # "csv-schema" (default), "json", or "markdown-kv" (opt-in
        # trade of tokens for model read accuracy). Falls back to the
        # HEADROOM_COMPACTION_FORMAT env var when the kwarg is None.
        # Ignored when with_compaction=False.
        resolved_format = compaction_format or os.environ.get(
            "HEADROOM_COMPACTION_FORMAT", "csv-schema"
        )
        # Validate even when with_compaction=False: an explicit bogus
        # format (kwarg or env var) is a misconfiguration that should be
        # visible, not silently accepted because the knob happens to be
        # ignored on this path.
        if resolved_format not in _SUPPORTED_COMPACTION_FORMATS:
            raise ValueError(
                f"unknown compaction format {resolved_format!r}; "
                f"expected one of: {', '.join(_SUPPORTED_COMPACTION_FORMATS)}"
            )
        self._compaction_format = resolved_format if with_compaction else None
        self._resolved_compaction_format = resolved_format
        # Cache of Rust crushers keyed by lossless_only, so a per-call
        # override builds the alternate at most once.
        self._rust_by_lossless_only: dict[bool, Any] = {}
        self._rust = self._build_rust(self._lossless_only)

    def _build_rust(self, lossless_only: bool) -> Any:
        """Build (and cache) the Rust crusher for a `lossless_only` value."""
        cached = self._rust_by_lossless_only.get(lossless_only)
        if cached is not None:
            return cached
        kwargs = dict(self._rust_cfg_kwargs)
        kwargs["lossless_only"] = lossless_only
        rust_cfg = self._RustSmartCrusherConfig(**kwargs)
        if not self._with_compaction:
            rust = self._RustSmartCrusher.without_compaction(rust_cfg)
        elif self._resolved_compaction_format == "csv-schema":
            # Keep the `new()` constructor for the default path so its
            # byte-parity coverage stays on the exact production codepath.
            rust = self._RustSmartCrusher(rust_cfg)
        else:
            rust = self._RustSmartCrusher.with_compaction_format(
                rust_cfg, self._resolved_compaction_format
            )
        self._rust_by_lossless_only[lossless_only] = rust
        return rust

    def crush(
        self,
        content: str,
        query: str = "",
        bias: float = 1.0,
        lossless_only: bool | None = None,
    ) -> CrushResult:
        """Crush a single JSON content string.

        Mirrors the retired Python method. Returns a `CrushResult`
        dataclass so call sites that destructure with `asdict()` keep
        working.

        `lossless_only` overrides the configured strict-lossless mode for
        this call only. When `True`, the output is guaranteed marker-free
        and byte-recoverable: lossless tabular compaction still applies,
        but any path that would need a CCR marker (row-drop or
        opaque-blob offload) leaves the content uncompacted instead.
        `None` (default) uses the instance's configured value.
        """
        rust = (
            self._rust
            if lossless_only is None or bool(lossless_only) == self._lossless_only
            else self._build_rust(bool(lossless_only))
        )
        r = rust.crush(content, query, bias)
        # SmartCrusher on the assumption SmartCrusher records its own.
        # Bridging the gap here keeps JSON-array compressions fueling
        # the learning system.
        #
        # Filter on `was_modified AND strategy != "passthrough"`. The
        # Rust crusher sometimes flips `was_modified=True` from pure
        # JSON re-canonicalization (whitespace normalization) without
        # actually compressing — the strategy stays `"passthrough"` in
        # that case, and there's no learning value in recording it.
        return CrushResult(
            compressed=r.compressed,
            original=r.original,
            was_modified=r.was_modified,
            strategy=r.strategy,
        )

    def crush_array_json(
        self,
        items_json: str,
        query: str = "",
        bias: float = 1.0,
    ) -> dict[str, Any]:
        """Crush a JSON array directly and surface the structured result.

        Returns a dict with `items` (kept rows as JSON), `ccr_hash` (12-char
        hash if rows were dropped), `dropped_summary` (the marker text),
        `strategy_info`, `compacted` (rendered bytes when lossless won),
        and `compaction_kind`.

        Used by tests and by the proxy's CCR retrieval flow when it needs
        the hash directly rather than parsing it out of a prompt marker.
        """
        result: dict[str, Any] = self._rust.crush_array_json(items_json, query, bias)
        # Row-drop case: Rust returns the structured `ccr_hash` and has
        # already stashed the canonical in its own store. Mirror that
        # entry into the Python compression_store keyed by the same
        # 12-char SHA-256 hash so /v1/retrieve resolves it.
        ccr_hash = result.get("ccr_hash")
        if ccr_hash:
            self._mirror_single_hash_to_python_store(
                ccr_hash,
                strategy=str(result.get("strategy_info") or "smart_crusher_row_drop"),
                query_context=query,
            )
        return result

    def compact_document_json(self, doc_json: str) -> str:
        """Run the document walker on ``doc_json`` and return compacted JSON.

        Lossless walker pass over objects, arrays, and strings —
        tabular sub-arrays become CSV+schema strings, long opaque
        blobs become ``<<ccr:HASH,KIND,SIZE>>`` markers (originals
        stashed in this crusher's CCR store, so ``ccr_get`` resolves them).

        Use this when callers want pure document-shape compaction
        without per-array lossy crushing.
        """
        result: str = self._rust.compact_document_json(doc_json)
        # Mirror any opaque-blob markers the walker emitted into the
        # Python store so /v1/retrieve resolves them.
        if "<<ccr:" in result:
            self._mirror_ccr_markers_in_text(
                result,
                strategy="smart_crusher_compact_document",
                query_context="",
            )
        return result

    def ccr_get(self, hash_key: str) -> str | None:
        """Look up an original payload by CCR hash from the Rust store.

        Returns the canonical-JSON serialization of the original
        `[item, item, ...]` array that the lossy path stashed before
        emitting `<<ccr:HASH ...>>`. Returns ``None`` if the hash is
        unknown, expired, or no store is configured.

        Used by the proxy's CCR retrieval tool to serve the dropped
        rows back to the LLM on demand.
        """
        result: str | None = self._rust.ccr_get(hash_key)
        return result

    def ccr_len(self) -> int:
        """Number of entries currently held by the Rust CCR store."""
        n: int = self._rust.ccr_len()
        return n

    def _smart_crush_content(
        self,
        content: str,
        query_context: str = "",
        tool_name: str | None = None,
        bias: float = 1.0,
    ) -> tuple[str, bool, str]:
        """Apply smart crushing; return `(crushed, was_modified, info)`.

        Mirrors the retired Python method's tuple shape. `tool_name` is
        tool name is available (e.g. the legacy pipeline doesn't have
        one in scope) the recording uses content-based signature only.
        """
        crushed, was_modified, info = self._rust.smart_crush_content(content, query_context, bias)
        # Same passthrough filter as `crush()` — re-canonicalization of
        # JSON whitespace can flip `was_modified=True` even when the
        # `info` field reports `passthrough` and no compression happened.
        return crushed, was_modified, info

        for h in hashes:
            self._mirror_single_hash_to_python_store(
                h,
                strategy=strategy,
                query_context=query_context,
                tool_name=tool_name,
            )

    @staticmethod
    def _collect_ccr_hashes(value: Any, sink: set[str]) -> None:
        """Recursively walk a parsed-JSON value, appending every CCR
        hash found inside string leaves to `sink`. Never raises."""
        if isinstance(value, str):
            SmartCrusher._collect_ccr_hashes_from_string(value, sink)
            return
        if isinstance(value, dict):
            for v in value.values():
                SmartCrusher._collect_ccr_hashes(v, sink)
            return
        if isinstance(value, list):
            for v in value:
                SmartCrusher._collect_ccr_hashes(v, sink)
            return
        # ints/bools/None/floats — no markers possible

    @staticmethod
    def _collect_ccr_hashes_from_string(s: str, sink: set[str]) -> None:
        """Extract every `<<ccr:HASH...>>` hash from a string by
        substring scan (no regex). The marker grammar is fixed:

            <<ccr:HASH<sep>...>>

        where ``HASH`` is `[0-9a-f]+` and ``<sep>`` is one of the
        delimiters the Rust emitters use today: a single space (the
        row-drop summary, ``<<ccr:abc 100_rows_offloaded>>``) or a
        comma (the opaque-blob marker, ``<<ccr:abc,base64,4.5KB>>``).
        We accept either delimiter and tolerate `>>` as the terminator
        (the case where the marker is just `<<ccr:abc>>` with no
        suffix, used by the bare CCR helpers).
        """
        idx = 0
        prefix = "<<ccr:"
        n = len(s)
        while True:
            start = s.find(prefix, idx)
            if start == -1:
                return
            cursor = start + len(prefix)
            end = cursor
            while end < n and s[end] in "0123456789abcdefABCDEF":
                end += 1
            if end == cursor:
                # No hex chars after `<<ccr:` — not a real marker.
                idx = cursor
                continue
            hash_str = s[cursor:end].lower()
            sink.add(hash_str)
            idx = end

    def _mirror_single_hash_to_python_store(
        self,
        ccr_hash: str,
        strategy: str,
        query_context: str,
        tool_name: str | None,
    ) -> None:
        """Mirror a single Rust-stored CCR entry into the Python
        compression_store, keyed by `ccr_hash`. Best-effort.
        """
        canonical = self._rust.ccr_get(ccr_hash)
        if canonical is None:
            # Rust store doesn't have it — either the marker came from
            # somewhere else (defensive: another transform's marker
            # leaked into our input), or the entry expired between
            # emission and mirror. Either way, nothing to mirror.
            logger.debug(
                "CCR mirror: hash %s not in Rust store (skipped)",
                ccr_hash,
            )
            return
        try:
            from ..cache.compression_store import get_compression_store
        except ImportError:
            # Stripped build without the compression_store module.
            # Mirror is a no-op; Rust side still serves the data.
            logger.debug("CCR mirror: compression_store module unavailable")
            return
        try:
            store = get_compression_store()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("CCR mirror: cannot get compression_store (%s)", e)
            return
        # The TTL on the Python store defaults to 5 minutes — same as
        # the Rust store's `DEFAULT_TTL` (see crates/headroom-core/src/
        # ccr/mod.rs). No need to override.
        try:
            store.store(
                original=canonical,
                # The "compressed" payload for the row-drop case isn't
                # readily available here (the rendered output may be
                # only one of many crushed sub-arrays). Use the marker
                # itself as a placeholder — `/v1/retrieve` returns
                # `original_content` and `compressed` isn't surfaced.
                compressed=f"<<ccr:{ccr_hash}>>",
                tool_name=tool_name,
                query_context=query_context if query_context else None,
                compression_strategy=strategy,
                explicit_hash=ccr_hash,
            )
        except ValueError:
            # explicit_hash validation failed — the marker had a
            # malformed hash (shouldn't happen in practice).
            logger.warning(
                "CCR mirror: invalid hash %r from rendered marker",
                ccr_hash,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("CCR mirror: store.store() raised (%s)", e)

    def _extract_context_from_messages(self, messages: list[dict[str, Any]]) -> str:
        """Build a query string from the last 5 user messages + recent
        assistant tool-call arguments. Used by `apply()` to derive the
        relevance context per-request.

        Pure Python because it walks the message envelope, not the
        compressed payload. The retired implementation lived inline on
        `SmartCrusher`; preserved here unchanged.
        """
        context_parts: list[str] = []
        user_message_count = 0

        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    context_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                context_parts.append(text)

                user_message_count += 1
                if user_message_count >= 5:
                    break

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        args = func.get("arguments", "")
                        if isinstance(args, str) and args:
                            context_parts.append(args)

        return " ".join(context_parts)

    def _notify_observer(self, original_tokens: int, compressed_tokens: int) -> None:
        """Forward a compression event to the configured
        `CompressionObserver` (see `headroom.transforms.observability`).
        No-op when no observer is set; swallows observer exceptions at
        debug level so a buggy metrics impl doesn't break the
        compression that just succeeded.
        """
        if self._observer is None:
            return
    def apply(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> TransformResult:
        """Transform-protocol entry point. Walks every tool/tool_result
        message, applies SmartCrusher to large enough payloads, and
        replaces the message content with `<crushed>\\n<digest_marker>`.

        Pure orchestration — the per-message compression delegates to
        Rust via `_smart_crush_content`.
        """
        tokens_before = tokenizer.count_messages(messages)
        result_messages = deep_copy_messages(messages)
        transforms_applied: list[str] = []
        markers_inserted: list[str] = []
        warnings: list[str] = []

        # F2.2: capture the per-request CompressionPolicy so
        # ContentRouter uses for ``_runtime_target_ratio``. ``None``
        # when the caller didn't pass a policy (e.g. legacy direct-
        # as "no gate", matching pre-F2.2 behaviour.
        self._runtime_compression_policy = kwargs.get("compression_policy")

        query_context = self._extract_context_from_messages(result_messages)
        crushed_count = 0
        frozen_message_count = kwargs.get("frozen_message_count", 0)

        crushed_tool_names: list[str] = []
        seen_tool_names: set[str] = set()
        tool_names_by_id = _build_tool_name_index(result_messages)

        def _record(tool_id: str | None) -> None:
            name = tool_names_by_id.get(tool_id or "")
            if name and name not in seen_tool_names:
                seen_tool_names.add(name)
                crushed_tool_names.append(name)

        for msg_idx, msg in enumerate(result_messages):
            if msg_idx < frozen_message_count:
                continue

            # OpenAI-style: top-level role=tool with string content.
            if msg.get("role") == "tool":
                # #1077: never re-compress headroom_retrieve results — they ARE
                # already-retrieved CCR content; compressing them again creates an
                # unresolvable retrieval loop.
                # ponytail: ceiling is tool_call_id lookup; if the id is missing we
                # compress (conservative: unknown tool names don't get a free pass).
                if tool_names_by_id.get(msg.get("tool_call_id") or "") == CCR_TOOL_NAME:
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    tokens = tokenizer.count_text(content)
                    if tokens > self.config.min_tokens_to_crush:
                        crushed, was_modified, info = self._smart_crush_content(
                            content, query_context
                        )
                        if was_modified:
                            marker = create_tool_digest_marker(compute_short_hash(content))
                            msg["content"] = crushed + "\n" + marker
                            crushed_count += 1
                            _record(msg.get("tool_call_id"))
                            markers_inserted.append(marker)
                            if info:
                                transforms_applied.append(f"smart:{info}")
                            self._notify_observer(tokens, tokenizer.count_text(crushed))

            # Anthropic-style: content is a list of blocks; each tool_result
            # block has a string content field of its own.
            content = msg.get("content")
            if isinstance(content, list):
                for i, block in enumerate(content):
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    # #1077: skip headroom_retrieve results — compressing them
                    # would produce a new <<ccr:hash>> marker the agent cannot
                    # redeem (infinite retrieval loop).
                    # ponytail: ceiling is tool_use_id lookup; unknown ids pass through.
                    if tool_names_by_id.get(block.get("tool_use_id") or "") == CCR_TOOL_NAME:
                        continue
                    tool_content = block.get("content", "")
                    if not isinstance(tool_content, str):
                        continue
                    tokens = tokenizer.count_text(tool_content)
                    if tokens <= self.config.min_tokens_to_crush:
                        continue

                    crushed, was_modified, info = self._smart_crush_content(
                        tool_content, query_context
                    )
                    if was_modified:
                        marker = create_tool_digest_marker(compute_short_hash(tool_content))
                        content[i]["content"] = crushed + "\n" + marker
                        crushed_count += 1
                        _record(block.get("tool_use_id"))
                        markers_inserted.append(marker)
                        if info:
                            transforms_applied.append(f"smart:{info}")
                        self._notify_observer(tokens, tokenizer.count_text(crushed))

        if crushed_count > 0:
            transforms_applied.insert(
                0, _format_smart_crush_transform(crushed_count, crushed_tool_names)
            )

        tokens_after = tokenizer.count_messages(result_messages)

        return TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
            markers_inserted=markers_inserted,
            warnings=warnings,
        )


# ─── Convenience function ─────────────────────────────────────────────────


def smart_crush_tool_output(
    content: str,
    config: SmartCrusherConfig | None = None,
    ccr_config: CCRConfig | None = None,
    with_compaction: bool = True,
    lossless_only: bool | None = None,
) -> tuple[str, bool, str]:
    """Compress a single tool output. Returns `(crushed, was_modified, info)`.

    Convenience wrapper that builds a one-shot `SmartCrusher` per call.
    Defaults to the PR4 lossless-first behavior; pass
    `with_compaction=False` to exercise the legacy lossy-only path
    (still useful for retention-property tests).

    `lossless_only=True` forces strict lossless mode: the output is
    marker-free and byte-recoverable (no row drops, opaque blobs inline).
    """
    crusher = SmartCrusher(
        config=config,
        ccr_config=ccr_config,
        with_compaction=with_compaction,
        lossless_only=lossless_only,
    )
    return crusher._smart_crush_content(content)
