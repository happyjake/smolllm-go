package smolllm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
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

// silentLeg is the production failure mode this guard exists for: the upstream
// accepts the request and then says nothing at all. Unlike EmptyStreamError the
// leg does not fail fast — it holds the connection, so without a first-output
// budget the chain waits out the whole request timeout on this one model and
// never reaches a healthy one.
func silentLeg(t *testing.T, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush() // headers are out; body stays silent
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	return srv
}

// readRequestBody consumes the request body so the fake provider can branch on
// which model the chain asked for.
func readRequestBody(t *testing.T, r *http.Request) string {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	require.NoError(t, err)
	return string(body)
}

func guardLogs(buf *bytes.Buffer) Option {
	return WithLogger(slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
}

// The incident: model-a never emits, and the chain must abandon it and answer
// from model-b. This is the whole point of the change — an answer instead of a
// 502 — so it is asserted on the response, not on a log line.
func TestSlowFirstLegFallsThroughToHealthyLeg(t *testing.T) {
	t.Parallel()

	var fastHits atomic.Int32
	// One upstream, behaving per model — closer to the real incident, where a
	// proxy holds one model's request while the next responds normally.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(readRequestBody(t, r), "model-a") {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.(http.Flusher).Flush() // headers out, body silent
			<-r.Context().Done()
			return
		}
		fastHits.Add(1)
		writeChatSuccess(t, w, "from model-b", "stop")
	}))
	defer srv.Close()

	var logs bytes.Buffer
	start := time.Now()
	res, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a,openai/model-b"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(150*time.Millisecond),
		WithTimeout(10*time.Second),
		guardLogs(&logs),
	)

	require.NoError(t, err)
	assert.Equal(t, "from model-b", res.Text)
	assert.GreaterOrEqual(t, fastHits.Load(), int32(1), "the healthy leg must be tried")
	assert.Less(t, time.Since(start), 5*time.Second,
		"the dead leg must be abandoned in ~one budget, not the full request timeout")
	assert.Contains(t, logs.String(), "model failed",
		"the abandoned leg must be reported as the reason the chain advanced")
}

// The guard abandons a leg, so the error must be identifiable and must not be
// retried: retrying it would spend another budget to relearn the same silence.
func TestFirstOutputTimeoutIsReportedAndNotRetried(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := silentLeg(t, &hits)

	var logs bytes.Buffer
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(120*time.Millisecond),
		WithTimeout(10*time.Second),
		guardLogs(&logs),
	)

	require.Error(t, err)
	assert.ErrorIs(t, err, ErrFirstOutputTimeout)
	var timeoutErr *FirstOutputTimeoutError
	require.ErrorAs(t, err, &timeoutErr)
	assert.Equal(t, "openai/model-a", timeoutErr.Model)
	assert.Equal(t, 120*time.Millisecond, timeoutErr.Budget)
	assert.Equal(t, int32(1), hits.Load(),
		"a first-output timeout must not be retried on the same leg")
}

// Once deltas flow the guard is disarmed; a slow but healthy generation must
// run to completion rather than be cut off mid-stream.
func TestGuardDisarmedOnceOutputStarts(t *testing.T) {
	t.Parallel()

	// First delta lands inside the budget and the rest arrives slowly after it:
	// the guard must be disarmed by that first delta and never cut the tail off.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"slow \"}}]}\n\n")
		w.(http.Flusher).Flush()
		time.Sleep(400 * time.Millisecond)
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"but alive\"}}]}\n\n")
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	var logs bytes.Buffer
	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(100*time.Millisecond),
		WithTimeout(10*time.Second),
		guardLogs(&logs),
	)

	require.NoError(t, err)
	assert.Equal(t, "slow but alive", resp.Text,
		"output that started inside the budget must not be truncated by the guard")
}

// A zero budget means "off", which is what an operator reaches for when a
// provider legitimately queues before answering.
func TestZeroBudgetDisablesTheGuard(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := silentLeg(t, &hits)

	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()

	var logs bytes.Buffer
	_, err := Ask(ctx, PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(0),
		WithTimeout(10*time.Second),
		guardLogs(&logs),
	)

	require.Error(t, err)
	assert.False(t, errors.Is(err, ErrFirstOutputTimeout),
		"a disabled guard must never report a first-output timeout")
	assert.GreaterOrEqual(t, hits.Load(), int32(1))
}

// Existing behaviour must survive: a genuinely cancelled caller stays
// cancelled, and is not relabelled as a first-output timeout.
func TestCallerCancellationIsNotFirstOutputTimeout(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	srv := silentLeg(t, &hits)

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)

	var logs bytes.Buffer
	_, err := Ask(ctx, PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(5*time.Second),
		WithTimeout(10*time.Second),
		guardLogs(&logs),
	)

	require.Error(t, err)
	assert.False(t, errors.Is(err, ErrFirstOutputTimeout))
	assert.ErrorIs(t, err, context.Canceled)
}
