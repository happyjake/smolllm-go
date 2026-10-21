package smolllm

import (
	"context"
	"fmt"
	"io"
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

// The streaming path must report a first-output timeout the same way the
// blocking path does. It did not: gateFirstOutput returned the guard's verdict
// bare, so a streamed leg abandoned for silence produced no RequestEvent at
// all and vanished from the ledger -- which is precisely the metric that
// distinguishes "queued upstream" from "model is broken". Found live: the same
// silent upstream recorded one attempt via Ask and zero via Stream.
func TestStreamedFirstOutputTimeoutIsReportedToTheHook(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := silentLeg(t, &hits)
	events, hook := collectEvents()

	st, err := Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(150*time.Millisecond),
		WithTimeout(10*time.Second),
		hook,
	)
	if err == nil {
		// The guard may surface through Wait instead; either surface counts,
		// but exactly one event must exist.
		err = st.Stream.Wait()
	}
	require.Error(t, err)

	require.Len(t, *events, 1,
		"a streamed first-output timeout must produce exactly one attempt event")
	e := (*events)[0]
	assert.True(t, e.FirstOutputTimeout, "event.FirstOutputTimeout = false, want true")
	assert.Equal(t, "stream", e.FirstOutputPhase)
	assert.GreaterOrEqual(t, e.TTFT, 100*time.Millisecond,
		"TTFT should reflect the wait, got %v", e.TTFT)
	assert.Error(t, e.Error)
}

// headerlessLeg accepts the request and never answers -- not even a status
// line -- until the client gives up. silentLeg flushes headers first, so it
// only exercises the "stream" phase; this one exercises "headers".
func headerlessLeg(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body first: net/http only notices a client hang-up (and
		// cancels r.Context) once the request body has been consumed.
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A streamed leg that times out before the upstream sends any headers must be
// reported just like the blocking path reports it. streamOnce returned the
// guard's headers-phase verdict bare, so this outcome -- an upstream that
// accepted the connection and then said nothing -- was missing from the
// ledger for streaming while Ask recorded it. Found live on 2026-10-20: one
// blocking and one streamed request against such an upstream produced a
// first_output_timeout count of 1, not 2.
func TestStreamedHeadersPhaseTimeoutIsReportedToTheHook(t *testing.T) {
	t.Parallel()

	srv := headerlessLeg(t)
	events, hook := collectEvents()

	st, err := Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(150*time.Millisecond),
		WithTimeout(10*time.Second),
		hook,
	)
	if err == nil {
		err = st.Stream.Wait()
	}
	require.Error(t, err)
	phase, ok := IsFirstOutputTimeout(err)
	require.True(t, ok, "IsFirstOutputTimeout did not recognise %v", err)
	assert.Equal(t, "headers", phase)

	require.Len(t, *events, 1,
		"a streamed headers-phase timeout must produce exactly one attempt event")
	e := (*events)[0]
	assert.True(t, e.FirstOutputTimeout, "event.FirstOutputTimeout = false, want true")
	assert.Equal(t, "headers", e.FirstOutputPhase)
	assert.GreaterOrEqual(t, e.TTFT, 100*time.Millisecond,
		"TTFT should reflect the wait, got %v", e.TTFT)
	assert.Error(t, e.Error)
}

// Ask has always reported this case; pin it so the two transports cannot
// drift apart again.
func TestBlockingHeadersPhaseTimeoutIsReportedToTheHook(t *testing.T) {
	t.Parallel()

	srv := headerlessLeg(t)
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
	require.Len(t, *events, 1)
	assert.True(t, (*events)[0].FirstOutputTimeout)
	assert.Equal(t, "headers", (*events)[0].FirstOutputPhase)
}
