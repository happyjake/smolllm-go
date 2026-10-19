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

// The blocking transport disarms the first-output guard as soon as headers
// arrive, and that is a deliberate, load-bearing property rather than an
// oversight: a non-streaming response carries no deltas, so its complete body
// IS its first output. The guard bounds "time until something usable appears";
// on this path nothing can appear before the body ends, so leaving it armed
// would truncate real answers once a provider takes longer to generate than the
// budget.
//
// These tests pin that decision so it cannot be changed by accident. The
// asymmetry it buys is real and worth stating where the behaviour is asserted:
// a blocking leg that stalls AFTER headers is not cut off at the budget, it
// waits out the request timeout. Anyone who wants the guard to keep applying
// mid-body has to add a stall detector, and will have to rewrite these tests
// deliberately — which is the point.

// blockingServer answers every request with a complete non-streaming
// completion. The body is written only when the test releases it, so a caller
// can hold the response in the "headers sent, body pending" state that
// separates the two transports.
func blockingServer(t *testing.T, delay time.Duration, content string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush() // headers are out; the body is not
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"` + content + `"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

// The core assertion: a blocking call whose body takes far longer than the
// first-output budget still returns its content. If the guard were left armed
// for the blocking path this call would be abandoned mid-flight and the answer
// would be a FirstOutputTimeoutError instead of text.
func TestBlockingCallOutlivesFirstOutputBudget(t *testing.T) {
	t.Parallel()

	srv, _ := blockingServer(t, 300*time.Millisecond, "slow but real")

	var logs bytes.Buffer
	res, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/m"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithoutStreaming(),
		// A budget far below the body delay. The guard must not be what
		// determines the outcome here.
		WithFirstOutputTimeout(50*time.Millisecond),
		WithTimeout(10*time.Second),
		WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))),
	)

	require.NoError(t, err, "a slow blocking answer is a success, not a timeout")
	assert.Equal(t, "slow but real", res.Text)
	assert.NotContains(t, logs.String(), "first_output_timeout",
		"the guard must already be disarmed by the time the body is read")
}

// The same property from the other side: because the guard is disarmed on
// headers, a blocking leg that never sends a body is NOT bounded by the
// budget. It is bounded by the request timeout. This test documents the cost
// of the decision — 10s failover becomes a full-timeout wait for this path —
// so that a future change to make blocking safer cannot silently claim the
// guard was already covering it.
func TestBlockingStallIsBoundedByRequestTimeoutNotBudget(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush() // headers out, body never comes
		<-r.Context().Done()
	}))
	defer srv.Close()

	start := time.Now()
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/m"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithoutStreaming(),
		WithFirstOutputTimeout(50*time.Millisecond),
		WithTimeout(700*time.Millisecond),
	)

	require.Error(t, err)
	elapsed := time.Since(start)
	// Well past the 50ms budget: the guard did not fire. The request timeout
	// is what ended this leg.
	assert.Greater(t, elapsed, 500*time.Millisecond,
		"a blocking stall must not be cut off at the first-output budget")
	assert.Less(t, elapsed, 3*time.Second,
		"the request timeout must still end the leg")
	assert.Equal(t, int32(1), hits.Load(), "the stalled leg must not be retried")
}

// Guard the guard: the streaming path must still honour the budget. Without
// this, a future change that disarms unconditionally would keep the two tests
// above green while quietly removing the protection the guard exists for.
func TestStreamingStallIsStillCutOffAtTheBudget(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush() // headers out, body silent
		<-r.Context().Done()
	}))
	defer srv.Close()

	start := time.Now()
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/m"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(80*time.Millisecond),
		WithTimeout(5*time.Second),
	)

	require.Error(t, err)
	assert.Less(t, time.Since(start), 2*time.Second,
		"a silent streaming leg must be abandoned in ~one budget")
	assert.Equal(t, int32(1), hits.Load())
}

// The behaviour a caller with a blocking client actually depends on: headers
// arrive, body is pending, and the answer is still delivered in full. This is
// the scenario the disarm comment in ask.go describes, so it is asserted as a
// silent-to-content transition rather than a timed sleep.
func TestBlockingBodyAfterHeadersIsDeliveredInFull(t *testing.T) {
	t.Parallel()

	payload := strings.Repeat("x", 500)
	srv, _ := blockingServer(t, 200*time.Millisecond, payload)

	res, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/m"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithoutStreaming(),
		WithFirstOutputTimeout(50*time.Millisecond),
		WithTimeout(10*time.Second),
	)

	require.NoError(t, err)
	assert.Equal(t, payload, res.Text, "the full body must survive, not a truncated prefix")
}
