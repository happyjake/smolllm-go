# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

smolllm-go is the Go port of smolllm (Python, sibling repo `../smolllm`): a minimal client for many LLM providers over the OpenAI-compatible wire protocol — one interface (`Ask`/`Stream`/`Embed`/`Validate`), API-key/endpoint load balancing, model fallback chains, token metering. Consumed by smolllm-server (OpenAI-compatible proxy, sibling repo).

## Design Philosophy

**Extreme minimalism.** Scope is frozen at chat + embeddings over the OpenAI-compat wire: no new modalities, no native provider transports, and no new API surface without a consumer to exercise it. Tool calling and the `WithExtraBody` escape hatch follow the design recorded in the Python repo; JSON mode is absorbed by the escape hatch and needs no code.

Canonical doctrine lives in the Python repo — read before proposing features:
- `../smolllm/docs/adr/0001-extreme-minimalism.md` — scope freeze
- `../smolllm/docs/adr/0002-token-only-accounting.md` — usage stops at tokens; cost is the caller's concern
- `../smolllm/docs/DEFERRED.md` — recorded designs (incl. Go specifics) awaiting a real use case

Domain glossary: [CONTEXT.md](CONTEXT.md).

## Managed consumers

**Dual-host maintenance:** runtime changes consumed by Pio/Jake-doom require the
`pi-proxy-ops` maintenance runbook and running-version/tool gate before rollout
completion. Rebuild both routers from the intended server/library revision pair;
obtain separate approval for live service changes.

## Development

- `make test` — `go test -v -race ./...` (offline: providers faked with `httptest`, no API keys needed)
- `make lint` — golangci-lint v2, strict profile
- Provider map is hand-maintained in `smolllm/providers.go` (the Python repo's `providers.json` is generated; sync manually — no generator yet)
