package smolllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const weatherArgs = `{"city":"Paris"}`

// writeToolCallStream emulates a provider answering with a tool call and no text:
// the argument JSON arrives fragmented across frames, as every provider streams it.
func writeToolCallStream(t *testing.T, w http.ResponseWriter, finishReason string) {
	t.Helper()

	w.Header().Set("Content-Type", "text/event-stream")
	_, err := fmt.Fprintf(w,
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":"+
			"[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\","+
			"\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"+
			"\"function\":{\"arguments\":\"{\\\"ci\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"+
			"\"function\":{\"arguments\":\"ty\\\":\\\"Paris\\\"}\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":%q}]}\n\n"+
			"data: [DONE]\n\n",
		finishReason,
	)
	if err != nil {
		t.Errorf("write fake provider response: %v", err)
		return
	}
}

func newToolCallServer(t *testing.T, finishReason string) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeToolCallStream(t, w, finishReason)
	}))
}

// ------------------------------------------------------------------ accumulator

func feedLine(t *testing.T, acc *toolCallAccumulator, payload string) {
	t.Helper()
	_, _, err := processChunkLineWithMetadata(newDefaultLogger(), "data: "+payload, nil, nil, acc)
	require.NoError(t, err)
}

func TestToolCallAccumulatorMergesArgumentFragments(t *testing.T) {
	t.Parallel()
	acc := &toolCallAccumulator{}
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",`+
		`"function":{"name":"get_weather","arguments":""}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 1)
	assert.Equal(t, "call_1", calls[0].ID)
	assert.Equal(t, "function", calls[0].Type)
	assert.Equal(t, "get_weather", calls[0].Function.Name)
	assert.JSONEq(t, weatherArgs, calls[0].Function.Arguments)
}

func TestToolCallAccumulatorKeepsParallelCallsOrdered(t *testing.T) {
	t.Parallel()
	acc := &toolCallAccumulator{}
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[`+
		`{"index":1,"id":"b","type":"function","function":{"name":"second"}},`+
		`{"index":0,"id":"a","type":"function","function":{"name":"first"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 2)
	assert.Equal(t, []string{"a", "b"}, []string{calls[0].ID, calls[1].ID})
	assert.Equal(t, "first", calls[0].Function.Name)
	assert.Equal(t, "{}", calls[0].Function.Arguments)
	assert.Empty(t, calls[1].Function.Arguments)
}

func TestToolCallAccumulatorWithoutIndexStartsNewCallOnID(t *testing.T) {
	t.Parallel()
	acc := &toolCallAccumulator{}
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"id":"a","type":"function","function":{"name":"f"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"x\":1}"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"id":"b","type":"function","function":{"name":"g"}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 2)
	assert.Equal(t, []string{"a", "b"}, []string{calls[0].ID, calls[1].ID})
	assert.Equal(t, `{"x":1}`, calls[0].Function.Arguments)
}

func TestToolCallAccumulatorIsEmptyForPlainTextStream(t *testing.T) {
	t.Parallel()
	acc := &toolCallAccumulator{}
	feedLine(t, acc, `{"choices":[{"delta":{"content":"hi"}}]}`)
	assert.Empty(t, acc.result())
}

func TestToolCallAccumulatorKeepsProviderExtras(t *testing.T) {
	t.Parallel()
	acc := &toolCallAccumulator{}
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",`+
		`"function":{"name":"get_weather","arguments":"{}"},`+
		`"extra_content":{"google":{"thought_signature":"sig-abc"}}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 1)
	require.Contains(t, calls[0].Extra, "extra_content")
	assert.JSONEq(t, `{"google":{"thought_signature":"sig-abc"}}`, string(calls[0].Extra["extra_content"]))
}

func TestToolCallRoundTripsExtrasThroughJSON(t *testing.T) {
	t.Parallel()
	raw := `{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"},` +
		`"extra_content":{"google":{"thought_signature":"sig"}}}`

	var call ToolCall
	require.NoError(t, json.Unmarshal([]byte(raw), &call))
	assert.Equal(t, "call_1", call.ID)

	encoded, err := json.Marshal(call)
	require.NoError(t, err)
	assert.JSONEq(t, raw, string(encoded))
}

// ------------------------------------------------------------------ prompt side

func TestPromptValidateAcceptsToolMessages(t *testing.T) {
	t.Parallel()
	prompt := PromptFromMessages([]Message{
		User("weather in Paris?"),
		AssistantToolCalls("", []ToolCall{{
			ID: "call_1", Type: "function",
			Function: ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
			Extra:    nil,
		}}),
		ToolResult("call_1", `{"temp_c":18}`),
	})
	require.NoError(t, prompt.Validate())
}

func TestPromptValidateStillRejectsFunctionRole(t *testing.T) {
	t.Parallel()
	legacy := openai.ChatCompletionFunctionMessageParam{ //nolint:exhaustruct // legacy arm under test
		Content: openai.String("x"),
		Name:    "f",
	}
	msg := Message{OfFunction: &legacy} //nolint:exhaustruct // union arm under test
	err := Prompt{Messages: []Message{msg}}.Validate()
	require.ErrorContains(t, err, "unsupported role")
}

func TestReplayedToolConversationReachesTheWire(t *testing.T) {
	t.Parallel()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		writeChatSuccess(t, w, "It is 18C in Paris.", "stop")
	}))
	defer srv.Close()

	signature := json.RawMessage(`{"google":{"thought_signature":"sig"}}`)
	prompt := PromptFromMessages([]Message{
		User("weather in Paris?"),
		AssistantToolCalls("", []ToolCall{{
			ID: "call_1", Type: "function",
			Function: ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
			Extra:    map[string]json.RawMessage{"extra_content": signature},
		}}),
		ToolResult("call_1", `{"temp_c":18}`),
	})

	resp, err := Ask(context.Background(), prompt,
		WithModel("openai/model-a"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"))
	require.NoError(t, err)
	assert.Equal(t, "It is 18C in Paris.", resp.Text)

	messages, ok := body["messages"].([]any)
	require.True(t, ok)
	require.Len(t, messages, 3)

	assistant, ok := messages[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "assistant", assistant["role"])
	calls, ok := assistant["tool_calls"].([]any)
	require.True(t, ok)
	require.Len(t, calls, 1)
	call, ok := calls[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "call_1", call["id"])
	assert.Equal(t, map[string]any{"google": map[string]any{"thought_signature": "sig"}}, call["extra_content"])

	toolMsg, ok := messages[2].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "call_1", toolMsg["tool_call_id"])
	assert.Equal(t, `{"temp_c":18}`, toolMsg["content"])
}

// ------------------------------------------------------------------ end to end

func TestAskReturnsToolCallsWithoutContent(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "tool_calls")
	defer srv.Close()

	resp, err := Ask(context.Background(), PromptFromString("weather?"),
		WithModel("openai/model-a"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
		WithExtraBody(map[string]any{"tools": []any{}}))
	require.NoError(t, err)

	assert.Empty(t, resp.Text)
	assert.Equal(t, "tool_calls", resp.FinishReason)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "get_weather", resp.ToolCalls[0].Function.Name)
	assert.JSONEq(t, weatherArgs, resp.ToolCalls[0].Function.Arguments)
}

func TestStreamExposesToolCallsAfterWait(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "tool_calls")
	defer srv.Close()

	sr, err := Stream(context.Background(), PromptFromString("weather?"),
		WithModel("openai/model-a"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"))
	require.NoError(t, err)

	chunks := 0
	for range sr.Stream.Chan() {
		chunks++
	}
	require.NoError(t, sr.Stream.Wait())

	assert.Equal(t, 0, chunks, "partial tool-call fragments must never reach consumers")
	assert.Equal(t, "tool_calls", sr.FinishReason)
	require.Len(t, sr.ToolCalls, 1)
	assert.JSONEq(t, weatherArgs, sr.ToolCalls[0].Function.Arguments)
}

func TestAskFailsLegWhenToolCallsAreTruncated(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "length")
	defer srv.Close()

	_, err := Ask(context.Background(), PromptFromString("weather?"),
		WithModel("openai/model-a"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"))
	require.ErrorContains(t, err, "truncated")
}

func TestAskSkipsMinOutputTokensForToolCalls(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "tool_calls")
	defer srv.Close()

	resp, err := Ask(context.Background(), PromptFromString("weather?"),
		WithModel("openai/model-a"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
		WithMinOutputTokens(500))
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
}

func TestAskStillFailsOnEmptyResponseWithoutToolCalls(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+"data: [DONE]\n\n")
		if err != nil {
			t.Errorf("write fake provider response: %v", err)
		}
	}))
	defer srv.Close()

	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"))
	require.ErrorContains(t, err, "empty response")
}
