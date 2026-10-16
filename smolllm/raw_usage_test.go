package smolllm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

// Opaque provider metadata, not a pricing model. Snapshot frames replace, not sum.
func TestRawUsageAcrossTransports(t *testing.T) {
	for _, transport := range []string{"ask-stream", "ask-blocking", "stream", "failed-stream"} {
		t.Run(transport, func(t *testing.T) {
			raw := `{"prompt_tokens":10,"completion_tokens":2,"credit":0,"vendor":{"exact":9007199254740993}}`
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if transport == "ask-blocking" {
					fmt.Fprintf(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":%s}`, raw)
				} else {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
					fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":%s}\n\n", raw)
					if transport == "failed-stream" {
						fmt.Fprint(w, "data: BROKEN\n\n")
					} else {
						fmt.Fprint(w, "data: [DONE]\n\n")
					}
				}
			}))
			defer srv.Close()
			var event RequestEvent
			opts := []Option{WithModel("openai/gpt-5"), WithBaseURL(srv.URL + "/"), WithAPIKey("k"), WithHook(func(e RequestEvent) { event = e })}
			if transport == "ask-blocking" {
				opts = append(opts, WithoutStreaming())
			}
			if transport == "stream" || transport == "failed-stream" {
				resp, err := Stream(context.Background(), PromptFromString("hi"), opts...)
				require.NoError(t, err)
				for range resp.Stream.Chan() {
				}
				err = resp.Stream.Wait()
				if transport == "failed-stream" {
					require.Error(t, err)
				} else {
					require.NoError(t, err)
				}
				require.JSONEq(t, raw, string(resp.Usage.RawUsage))
			} else {
				resp, err := Ask(context.Background(), PromptFromString("hi"), opts...)
				require.NoError(t, err)
				require.JSONEq(t, raw, string(resp.Usage.RawUsage))
			}
			require.JSONEq(t, raw, string(event.RawUsage))
			require.Contains(t, string(event.RawUsage), "9007199254740993")
		})
	}
}
