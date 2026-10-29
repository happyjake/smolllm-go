# SmolLLM (Go)

Go port of smolllm: a minimal client for many LLM providers over the OpenAI-compatible wire protocol — one interface, API-key/endpoint balancing, model fallback. Shares its ubiquitous language with the Python lib (`../smolllm/CONTEXT.md` is the canonical copy; keep in sync).

## Language

**Provider**:
A named OpenAI-compatible endpoint (e.g. `openai`, `groq`); credentials and base URL resolve from env by name. Bare model specs have no provider — identity surfaces as empty string.
_Avoid_: vendor, backend.

**Model spec**:
The user-facing model string `provider/model[!effort]`, or bare `model[!effort]` (no `/`) — bare form has no provider and resolves base URL/API key from explicit options only, never env. Comma-separated specs form a fallback chain; may mix both forms. Explicit base URL applies to every leg.

**Fallback chain**:
Ordered or weighted candidate models; on failure the call advances to the next candidate.
_Avoid_: confusing with retry.

**Retry**:
Re-attempt of the *same* model after a transient failure. Distinct from fallback (which switches models).

**Account**:
The provider plus the credential a leg resolves to; every model behind it shares one quota. Identified only by a non-reversible fingerprint.

**Account circuit**:
Process-local breaker state for one account, opened by a structured quota-exhausted error (doom HTTP 429, business code 14019) and skipping the account's legs until a bounded cooldown, then one lazy probe. Distinct from retry (same leg) and fallback (next leg): it decides which untouched legs may start.

**Balancer pair**:
One (API key, base URL) combination for a provider; the least-used pair is chosen per call.

**Estimated usage**:
Token counts derived by heuristic when the provider omits usage; always marked (`~` prefix, `Estimated` flag).

**Reasoning**:
Model thinking text, kept in a channel separate from content.
_Avoid_: mixing reasoning into content.

**FinishReason**:
Verbatim provider string explaining why generation ended; never normalized.

**Request hook**:
Per-attempt observation callback receiving usage or error; the library's only telemetry surface.

**Escape hatch**:
A pass-through (`WithExtraBody`) letting callers set raw request fields the library does not model, merged last so the caller wins. The fields the library machinery reads back (`stream`, `stream_options`, `messages`, `model`) are rejected.
_Avoid_: raw options, extra params.

**Tool call**:
A provider-issued request to run a named function, surfaced verbatim as a `ToolCall` carrying the wire fields with the argument JSON as an opaque string. The caller executes it and replays the assistant and `tool` messages; the library runs no agentic loop and never inspects or repairs the argument JSON.
_Avoid_: function call, tool use, implying smolllm executes anything.
