package smolllm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// collectEvents installs a hook and returns the events it saw.
func collectEvents() (*[]RequestEvent, Option) {
	var events []RequestEvent
	return &events, WithHook(func(e RequestEvent) { events = append(events, e) })
}

// A leg that produces nothing in time must be reported as a first-output
// timeout, with its phase and with the wait recorded as TTFT. Without this the
// server cannot distinguish "queued upstream" from "model is broken", and every
// abandoned leg looks like a model failure in the ledger.
func TestFirstOutputTimeoutIsIdentifiableInTheHook(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := silentLeg(t, &hits)
	events, hook := collectEvents()

	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(150*time.Millisecond),
		WithTimeout(10*time.Second),
		hook,
	)
	require.Error(t, err)

	phase, ok := IsFirstOutputTimeout(err)
	require.True(t, ok, "IsFirstOutputTimeout did not recognise %v", err)
	assert.Contains(t, []string{"headers", "stream"}, phase)

	require.Len(t, *events, 1, "one attempt, and a first-output timeout must not be retried")
	e := (*events)[0]
	assert.True(t, e.FirstOutputTimeout, "event.FirstOutputTimeout = false, want true")
	assert.Equal(t, phase, e.FirstOutputPhase)
	// A silent leg waited its whole budget; TTFT=0 would read as "answered
	// instantly", the opposite of what happened.
	assert.GreaterOrEqual(t, e.TTFT, 100*time.Millisecond,
		"TTFT should reflect the wait, got %v", e.TTFT)
	assert.Error(t, e.Error)
}

// An ordinary upstream failure must not be dressed up as a first-output
// timeout, or the metric would count connection errors as queueing.
func TestOrdinaryFailureIsNotAFirstOutputTimeout(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeChatSuccess(t, w, "unused", "stop")
	}))
	defer srv.Close()

	// Point at a closed port: a plain transport error, not silence.
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL + "/"
	dead.Close()
	_ = srv // the healthy server is unused; the failure is what is under test

	events, hook := collectEvents()
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(deadURL),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(2*time.Second),
		WithTimeout(5*time.Second),
		hook,
	)
	require.Error(t, err)
	_, isFOT := IsFirstOutputTimeout(err)
	assert.False(t, isFOT, "a connection error was classified as a first-output timeout")

	require.NotEmpty(t, *events)
	for _, e := range *events {
		assert.False(t, e.FirstOutputTimeout, "plain failure marked as first-output timeout")
		assert.Empty(t, e.FirstOutputPhase)
	}
}

// The guard's error travels back through retry and fallback layers, so a
// wrapped timeout must still be recognised: errors.As, not a type switch.
func TestIsFirstOutputTimeoutSeesThroughWrapping(t *testing.T) {
	t.Parallel()

	inner := &FirstOutputTimeoutError{Model: "m", Phase: "headers"}
	wrapped := fmt.Errorf("chain exhausted: %w", inner)
	phase, ok := IsFirstOutputTimeout(wrapped)
	require.True(t, ok, "wrapped timeout was not recognised")
	assert.Equal(t, "headers", phase)

	_, ok = IsFirstOutputTimeout(nil)
	assert.False(t, ok, "nil must not be a timeout")
}
