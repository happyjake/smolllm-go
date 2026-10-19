package smolllm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeToolFragment emits one SSE chunk whose only payload is a tool-call
// fragment — no content, no reasoning. Once the accumulator takes the fragment,
// the chunk carries nothing that StreamChunk.IsEmpty can see.
func writeToolFragment(w http.ResponseWriter, args string) {
	_, _ = w.Write([]byte("data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[" + args + "]}}]}\n\n"))
}

// toolOnlyLeg streams a tool call and nothing else.
//
// This is the shape the first-output guard cannot see today. Every chunk carries
// its payload in delta.tool_calls, so once the accumulator has taken the
// fragments the chunk has no content and no reasoning — IsEmpty() is true — and
// the receive loop skips it before reaching the firstToken marker. The call
// therefore looks silent to the guard for its whole duration even though the
// upstream is streaming normally.
//
// gap is the delay before the first fragment; tail is the delay between the
// first fragment and the last one, so a test can put the tail beyond the guard
// budget while the head arrives well inside it.
func toolOnlyLeg(t *testing.T, gap, tail time.Duration, fragmentHits *atomic.Int32) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		flusher.Flush()

		select {
		case <-time.After(gap):
		case <-r.Context().Done():
			return
		}

		// The first fragment arrives inside the budget: the leg has proven it
		// delivers, so nothing after this point should be reported as a
		// first-output timeout.
		fragmentHits.Add(1)
		writeToolFragment(w, `{"index":0,"id":"call_1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\""}}`)
		flusher.Flush()

		select {
		case <-time.After(tail):
		case <-r.Context().Done():
			return
		}

		fragmentHits.Add(1)
		writeToolFragment(w, `{"index":0,"function":{"arguments":"/tmp/a.txt\"}"}}`)
		flusher.Flush()

		_, _ = w.Write([]byte("data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestToolOnlyStreamOutlivesFirstOutputGuard is the latent bug the tool-call
// blind spot hides: because a tool-only stream never marks the first output, the
// guard stays armed for the entire call. A model that streams a large tool
// argument slowly — a file write, a long patch — is abandoned mid-answer and
// reported as first_output_timeout while it was healthy the whole time.
//
// The first fragment arrives at 50ms, well inside the 250ms budget. The last
// arrives around 900ms. A guard that can see the first fragment lets the call
// finish; one that cannot fires at 250ms and reports a timeout.
func TestToolOnlyStreamOutlivesFirstOutputGuard(t *testing.T) {
	t.Parallel()

	var fragmentHits atomic.Int32
	srv := toolOnlyLeg(t, 50*time.Millisecond, 850*time.Millisecond, &fragmentHits)

	res, err := Ask(context.Background(), PromptFromString("write a file"),
		WithModel("openai/tool-model"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(250*time.Millisecond),
		WithTimeout(5*time.Second),
	)

	require.NoError(t, err,
		"a healthy tool-only stream must not be abandoned by the first-output guard")
	require.NotNil(t, res)
	require.Len(t, res.ToolCalls, 1, "the streamed tool call must survive intact")
	assert.Equal(t, "write_file", res.ToolCalls[0].Function.Name)
	assert.Equal(t, int32(2), fragmentHits.Load(), "both fragments must have been streamed")
}

// A keep-alive must not be mistaken for generation: an empty slice or an
// all-empty fragment proves the upstream is alive but produced nothing, so
// neither may mark the first output or disarm the guard.
func TestCarriesToolPayloadRejectsEmptyFragments(t *testing.T) {
	t.Parallel()

	empty := ""
	args := "/tmp/a.txt"

	cases := []struct {
		name  string
		input []toolCallDelta
		want  bool
	}{
		{"nil slice", nil, false},
		{"empty slice", []toolCallDelta{}, false},
		{"all-empty fragment", []toolCallDelta{{}}, false},
		{"empty function", []toolCallDelta{{Function: &toolCallFunctionDelta{}}}, false},
		{"empty arguments", []toolCallDelta{{Function: &toolCallFunctionDelta{Arguments: &empty}}}, false},
		{"id only", []toolCallDelta{{ID: "call_1"}}, true},
		{"name only", []toolCallDelta{{Function: &toolCallFunctionDelta{Name: "write_file"}}}, true},
		{"arguments only", []toolCallDelta{{Function: &toolCallFunctionDelta{Arguments: &args}}}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.want, carriesToolPayload(tc.input))
		})
	}
}

// The parser reports evidence per line, and the flag must be false for every
// line that is not a delta carrying tool payload: comments, [DONE], usage-only
// and finish-only. A true here would disarm the guard on traffic that proves
// nothing.
func TestProcessChunkLineReportsToolEvidenceOnlyForPayload(t *testing.T) {
	t.Parallel()

	logger := newDefaultLogger()

	cases := []struct {
		name string
		line string
		want bool
	}{
		{"done", "data: [DONE]", false},
		{"comment", ": keep-alive", false},
		{"empty", "", false},
		{"usage only", `data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`, false},
		{"finish only", `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, false},
		{"empty tool calls", `data: {"choices":[{"index":0,"delta":{"tool_calls":[]}}]}`, false},
		{"content", `data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}`, false},
		{"tool payload", `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f"}}]}}]}`, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var acc toolCallAccumulator
			_, evidence, err := processChunkLineWithMetadata(logger, tc.line, nil, nil, &acc)
			require.NoError(t, err)
			assert.Equal(t, tc.want, evidence)
		})
	}
}

// The behavioural pair that proves the guard is stopped by the right thing:
// a tool-only stream that starts late must still time out, and one that starts
// on time but finishes late must not. Only the second case was broken.
func TestToolOnlyStreamStartingLateStillTimesOut(t *testing.T) {
	t.Parallel()

	var fragmentHits atomic.Int32
	// Nothing at all until 600ms, well past the 250ms budget.
	srv := toolOnlyLeg(t, 600*time.Millisecond, 10*time.Millisecond, &fragmentHits)

	_, err := Ask(context.Background(), PromptFromString("write a file"),
		WithModel("openai/tool-model"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithFirstOutputTimeout(250*time.Millisecond),
		WithTimeout(5*time.Second),
	)

	require.Error(t, err, "a genuinely silent leg must still be abandoned")
	assert.ErrorIs(t, err, ErrFirstOutputTimeout)
	assert.Equal(t, int32(0), fragmentHits.Load(), "no fragment should have been streamed")
}
