package smolllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeDeadOnArrivalStream reproduces the doom/pi-proxy gpt-6.1-sol failure
// captured on 2026-10-16: HTTP 200, one synthesized terminal chunk carrying
// finish_reason=error and all-zero usage, then [DONE].
func writeDeadOnArrivalStream(t *testing.T, w http.ResponseWriter) {
	t.Helper()
	w.Header().Set("Content-Type", "text/event-stream")
	_, err := fmt.Fprint(w,
		"data: {\"id\":\"\",\"model\":\"gpt-6.1-sol\",\"object\":\"chat.completion.chunk\",\"created\":0,"+
			"\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"\","+
			"\"tool_calls\":[]},\"finish_reason\":\"error\"}],"+
			"\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0,\"credit\":0}}\n\n"+
			"data: [DONE]\n\n",
	)
	if err != nil {
		t.Errorf("write fake provider response: %v", err)
	}
}

func writeTextStream(t *testing.T, w http.ResponseWriter, parts ...string) {
	t.Helper()
	w.Header().Set("Content-Type", "text/event-stream")
	var b strings.Builder
	for _, p := range parts {
		fmt.Fprintf(&b, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", p)
	}
	b.WriteString("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
	b.WriteString("data: [DONE]\n\n")
	if _, err := fmt.Fprint(w, b.String()); err != nil {
		t.Errorf("write fake provider response: %v", err)
	}
}

func drain(t *testing.T, resp *StreamResponse) (string, error) {
	t.Helper()
	var sb strings.Builder
	for chunk := range resp.Stream.Chan() {
		sb.WriteString(chunk.Content)
	}
	return sb.String(), resp.Stream.Wait()
}

func TestStreamRetriesDeadOnArrivalSameLeg(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			writeDeadOnArrivalStream(t, w)
			return
		}
		writeTextStream(t, w, "hel", "lo")
	}))
	defer srv.Close()

	var events []RequestEvent
	resp, err := Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithHook(func(e RequestEvent) { events = append(events, e) }),
	)
	require.NoError(t, err)
	text, err := drain(t, resp)
	require.NoError(t, err)
	assert.Equal(t, "hello", text)
	assert.Equal(t, "openai/model-a", resp.Model, "a same-leg retry recovers before the chain moves on")
	assert.Equal(t, int32(2), attempts.Load())

	require.Len(t, events, 2)
	var empty *EmptyStreamError
	require.ErrorAs(t, events[0].Error, &empty, "the dead attempt is recorded as a failure")
	assert.Equal(t, "error", empty.FinishReason)
	assert.NoError(t, events[1].Error)
}

func TestStreamFallsBackWhenLegKeepsArrivingDead(t *testing.T) {
	t.Parallel()

	var deadAttempts, fallbackAttempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		switch model {
		case "model-a":
			deadAttempts.Add(1)
			writeDeadOnArrivalStream(t, w)
		case "model-b":
			fallbackAttempts.Add(1)
			writeTextStream(t, w, "fallback")
		}
	}))
	defer srv.Close()

	resp, err := Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	text, err := drain(t, resp)
	require.NoError(t, err)
	assert.Equal(t, "fallback", text)
	assert.Equal(t, "gemini/model-b", resp.Model)
	assert.Equal(t, int32(defaultMaxRetries), deadAttempts.Load())
	assert.Equal(t, int32(1), fallbackAttempts.Load())
}

func TestStreamSurfacesEmptyStreamOnLastLeg(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeDeadOnArrivalStream(t, w)
	}))
	defer srv.Close()

	_, err := Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	var empty *EmptyStreamError
	require.ErrorAs(t, err, &empty)
	assert.Contains(t, err.Error(), `finish_reason="error"`)
}

func TestStreamGateKeepsToolCallOnlyTurns(t *testing.T) {
	t.Parallel()

	srv := newToolCallServer(t, "tool_calls")
	defer srv.Close()

	resp, err := Stream(context.Background(), PromptFromString("weather?"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	text, err := drain(t, resp)
	require.NoError(t, err)
	assert.Empty(t, text)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "tool_calls", resp.FinishReason)
}

func TestStreamGateReplaysFirstChunkAndKeepsOrder(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeTextStream(t, w, "a", "b", "c", "d", "e")
	}))
	defer srv.Close()

	for range 20 { // the relay must not drop or reorder under scheduling noise
		resp, err := Stream(context.Background(), PromptFromString("hi"),
			WithModel("openai/gpt-5"),
			WithBaseURL(srv.URL+"/"),
			WithAPIKey("test-key"),
		)
		require.NoError(t, err)
		text, err := drain(t, resp)
		require.NoError(t, err)
		assert.Equal(t, "abcde", text)
		assert.Equal(t, "stop", resp.FinishReason)
	}
}

func TestStreamGateCloseMidStreamDoesNotLeak(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		parts := make([]string, 200)
		for i := range parts {
			parts[i] = "x"
		}
		writeTextStream(t, w, parts...)
	}))
	defer srv.Close()

	resp, err := Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	<-resp.Stream.Chan()
	resp.Stream.Close()
	for range resp.Stream.Chan() { // must terminate
	}
	_ = resp.Stream.Wait() // must return, whatever the error
}

func TestAskRetriesEmptyResponseSameLeg(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			writeDeadOnArrivalStream(t, w)
			return
		}
		writeTextStream(t, w, "ok")
	}))
	defer srv.Close()

	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	assert.Equal(t, "ok", resp.Text)
	assert.Equal(t, "openai/model-a", resp.Model)
	assert.Equal(t, int32(2), attempts.Load())
}

func TestStreamDoesNotRetryCanceledContext(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	ctx, cancel := context.WithCancel(context.Background())
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		cancel()
		<-ctx.Done()
	}))
	defer srv.Close()

	_, err := Stream(ctx, PromptFromString("hi"),
		WithModel("openai/model-a"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.Error(t, err)
	var empty *EmptyStreamError
	assert.False(t, errors.As(err, &empty), "a canceled request is not an empty stream: %v", err)
	assert.Equal(t, int32(1), attempts.Load())
}
