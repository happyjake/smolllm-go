package smolllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type headerRecorder struct {
	mu   sync.Mutex
	seen []http.Header
}

func (r *headerRecorder) record(h http.Header) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen = append(r.seen, h.Clone())
}

func (r *headerRecorder) all() []http.Header {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]http.Header(nil), r.seen...)
}

// Custom headers must reach EVERY outbound request path. The retry sub-test is
// the one that matters: retryWithoutStreamUsage builds a brand-new
// http.Request, so a fix applied only where the first request is built passes
// every other test here and still loses the header on the retry. The embed
// sub-test covers the third path, which historically never had headers at all.
func TestCustomHeadersReachEveryRequestPath(t *testing.T) {
	t.Parallel()

	sse := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}

	t.Run("ask", func(t *testing.T) {
		t.Parallel()
		rec := &headerRecorder{}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			rec.record(r.Header)
			sse(w)
		}))
		defer srv.Close()

		_, err := Ask(context.Background(), PromptFromString("hi"),
			WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
			WithHeader("X-Custom", "yes"))
		require.NoError(t, err)
		assertEveryRequestCarried(t, rec, "X-Custom", "yes")
	})

	t.Run("stream_options_retry", func(t *testing.T) {
		t.Parallel()
		rec := &headerRecorder{}
		var calls int
		var mu sync.Mutex
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			rec.record(r.Header)
			mu.Lock()
			calls++
			first := calls == 1
			mu.Unlock()
			if first {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"stream_options is not supported"}}`))
				return
			}
			sse(w)
		}))
		defer srv.Close()

		_, err := Ask(context.Background(), PromptFromString("hi"),
			WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
			WithHeader("X-Custom", "yes"))
		require.NoError(t, err)
		require.GreaterOrEqual(t, len(rec.all()), 2, "expected an initial request and a retry")
		assertEveryRequestCarried(t, rec, "X-Custom", "yes")
	})

	t.Run("stream", func(t *testing.T) {
		t.Parallel()
		rec := &headerRecorder{}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			rec.record(r.Header)
			sse(w)
		}))
		defer srv.Close()

		stream, err := Stream(context.Background(), PromptFromString("hi"),
			WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
			WithHeader("X-Custom", "yes"))
		require.NoError(t, err)
		for range stream.Stream.Chan() {
		}
		require.NoError(t, stream.Stream.Wait())
		assertEveryRequestCarried(t, rec, "X-Custom", "yes")
	})

	t.Run("embed", func(t *testing.T) {
		t.Parallel()
		rec := &headerRecorder{}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			rec.record(r.Header)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1]}],"model":"m","usage":{"prompt_tokens":1,"total_tokens":1}}`))
		}))
		defer srv.Close()

		_, err := Embed(context.Background(), []string{"hi"},
			WithModel("openai/text-embedding-3-small"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
			WithHeader("X-Custom", "yes"))
		require.NoError(t, err)
		assertEveryRequestCarried(t, rec, "X-Custom", "yes")
	})
}

func assertEveryRequestCarried(t *testing.T, rec *headerRecorder, name, value string) {
	t.Helper()
	seen := rec.all()
	require.NotEmpty(t, seen, "provider received no requests")
	for i, h := range seen {
		assert.Equalf(t, value, h.Get(name), "request #%d did not carry %s", i+1, name)
		assert.Equalf(t, "Bearer k", h.Get("Authorization"), "request #%d lost Authorization", i+1)
		assert.Equalf(t, "application/json", h.Get("Content-Type"), "request #%d lost Content-Type", i+1)
	}
}

func TestWithHeaderRejectsReservedNames(t *testing.T) {
	t.Parallel()
	assert.Panics(t, func() { WithHeader("Authorization", "Bearer nope") })
	assert.Panics(t, func() { WithHeader("content-type", "text/plain") })
	assert.Panics(t, func() { WithHeader("  ", "x") })
	assert.NotPanics(t, func() { WithHeader("X-Title", "smolllm") })
}

// Prompt-cache usage arrives in three mutually-exclusive dialects. This must
// hold on BOTH transports, which is only true because they share usageChunk —
// two independent parsers would drift.
func TestUsageCacheTokensAcrossWireShapes(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		usageJSON string
		wantRead  int
		wantWrite int
	}{
		{"openai_details", `{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}`, 80, 0},
		{"deepseek_hit_miss", `{"prompt_tokens":100,"completion_tokens":5,"prompt_cache_hit_tokens":80,"prompt_cache_miss_tokens":20}`, 80, 0},
		{"anthropic_style", `{"prompt_tokens":100,"completion_tokens":5,"cache_read_input_tokens":70,"cache_creation_input_tokens":30}`, 70, 30},
		{"no_cache_fields", `{"prompt_tokens":100,"completion_tokens":5}`, 0, 0},
	}

	for _, tc := range cases {
		t.Run(tc.name+"/streaming", func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
				_, _ = w.Write([]byte(fmt.Sprintf("data: {\"choices\":[],\"usage\":%s}\n\n", tc.usageJSON)))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
			}))
			defer srv.Close()

			var event RequestEvent
			resp, err := Ask(context.Background(), PromptFromString("hi"),
				WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
				WithHook(func(e RequestEvent) { event = e }))
			require.NoError(t, err)
			assert.Equal(t, tc.wantRead, resp.Usage.CacheReadTokens)
			assert.Equal(t, tc.wantWrite, resp.Usage.CacheWriteTokens)
			// A DeepSeek miss is ordinary uncached input, never a cache write.
			assert.Equal(t, tc.wantRead, event.CacheReadTokens, "cache tokens must ride RequestEvent")
			assert.Equal(t, tc.wantWrite, event.CacheWriteTokens)
		})

		t.Run(tc.name+"/blocking", func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(fmt.Sprintf(
					`{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":%s}`, tc.usageJSON)))
			}))
			defer srv.Close()

			resp, err := Ask(context.Background(), PromptFromString("hi"),
				WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
				WithoutStreaming())
			require.NoError(t, err)
			assert.Equal(t, "hi", resp.Text)
			assert.Equal(t, tc.wantRead, resp.Usage.CacheReadTokens)
			assert.Equal(t, tc.wantWrite, resp.Usage.CacheWriteTokens)
		})
	}
}

// The blocking transport must not send stream_options, which also makes the
// stream_options-rejection retry a structural no-op rather than a live path.
func TestWithoutStreamingSendsPlainRequest(t *testing.T) {
	t.Parallel()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
		WithoutStreaming())
	require.NoError(t, err)

	assert.Equal(t, false, body["stream"])
	assert.NotContains(t, body, "stream_options")
	assert.Equal(t, "hi", resp.Text)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.False(t, resp.Usage.Estimated)
	// Without a stream the first token arrives with the last.
	assert.Equal(t, resp.Usage.Duration, resp.Usage.TTFT)
}

func TestWithoutStreamingIncompatibleOptions(t *testing.T) {
	t.Parallel()
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"), WithBaseURL("http://127.0.0.1:1/"), WithAPIKey("k"),
		WithoutStreaming(), WithStreamHandler(func(context.Context, string) error { return nil }))
	require.ErrorContains(t, err, "WithStreamHandler")

	_, err = Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"), WithBaseURL("http://127.0.0.1:1/"), WithAPIKey("k"),
		WithoutStreaming())
	require.ErrorContains(t, err, "Stream")
}
