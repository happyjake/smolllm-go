package smolllm

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// canceledLegWarnings counts the "model failed, trying fallback" warnings — one
// per fallback leg the chain actually walks. This is the production symptom:
// a dead deadline produced one 0ms "context canceled" hop per model instead of
// stopping at the first one.
func fallbackWarnings(logs *bytes.Buffer) int {
	return strings.Count(logs.String(), "trying fallback")
}

// A caller that already gave up must not make the chain walk every remaining
// model. Before the fix, each remaining leg dialed with an already-cancelled
// context, failed in microseconds, and logged a fallback warning — so a
// four-model chain reported three fallback attempts that never really happened
// and surfaced as a generic upstream 502.
func TestCanceledContextDoesNotWalkFallbackChain(t *testing.T) {
	t.Parallel()

	var logs bytes.Buffer
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Error(w, "upstream unavailable", http.StatusInternalServerError)
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // caller already gave up

	_, err := Ask(ctx, PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b,openai/model-c,gemini/model-d"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))),
	)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, int32(0), hits.Load(), "no provider request should be dialed with a dead context")
	assert.LessOrEqual(t, fallbackWarnings(&logs), 1,
		"the chain must abort on the cancelled context, not walk all four models")
}

// The deadline-during-first-leg case is the one seen in production: a slow
// primary, then the caller's deadline lands. Every remaining leg must be
// skipped rather than dialed against a dead context.
func TestDeadlineDuringFirstLegDoesNotWalkFallbackChain(t *testing.T) {
	t.Parallel()

	var logs bytes.Buffer
	var hits atomic.Int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if n == 1 {
			select {
			case <-release:
			case <-r.Context().Done():
			}
			return
		}
		http.Error(w, "should not be reached", http.StatusInternalServerError)
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := Ask(ctx, PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b,openai/model-c,gemini/model-d"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))),
	)
	elapsed := time.Since(start)

	require.Error(t, err)
	assert.Equal(t, int32(1), hits.Load(),
		"only the in-flight leg should reach a provider; remaining fallbacks must be skipped")
	assert.LessOrEqual(t, fallbackWarnings(&logs), 1,
		"the dead deadline must short-circuit the chain, not log one hop per model")
	assert.Less(t, elapsed, 3*time.Second,
		"the chain must surface the cancellation promptly")
}

// The streaming entry point is the path the agent actually uses.
func TestDeadlineDuringFirstStreamLegDoesNotWalkFallbackChain(t *testing.T) {
	t.Parallel()

	var logs bytes.Buffer
	var hits atomic.Int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if n == 1 {
			select {
			case <-release:
			case <-r.Context().Done():
			}
			return
		}
		http.Error(w, "should not be reached", http.StatusInternalServerError)
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := Stream(ctx, PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b,openai/model-c,gemini/model-d"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))),
	)
	elapsed := time.Since(start)

	require.Error(t, err)
	assert.Equal(t, int32(1), hits.Load())
	assert.LessOrEqual(t, fallbackWarnings(&logs), 1,
		"the dead deadline must short-circuit the stream fallback chain")
	assert.Less(t, elapsed, 3*time.Second)
}

// A healthy chain must still fall back — the short-circuit must be scoped to
// a dead context, not disable failover broadly.
func TestHealthyContextStillWalksFallbackChain(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if n == 1 {
			http.Error(w, "upstream unavailable", http.StatusInternalServerError)
			return
		}
		writeChatSuccess(t, w, "fallback answer", "stop")
	}))
	defer srv.Close()

	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	assert.Equal(t, "fallback answer", resp.Text)
	assert.Equal(t, int32(2), hits.Load(), "a live context must still fail over")
}
