package smolllm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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
			WithHeaderForAllProviders("X-Custom", "yes"))
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
			WithHeaderForAllProviders("X-Custom", "yes"))
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
			WithHeaderForAllProviders("X-Custom", "yes"))
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
			WithHeaderForAllProviders("X-Custom", "yes"))
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
	assert.Panics(t, func() { WithHeaderForAllProviders("Authorization", "Bearer nope") })
	assert.Panics(t, func() { WithHeaderForAllProviders("content-type", "text/plain") })
	assert.Panics(t, func() { WithHeaderForAllProviders("  ", "x") })
	assert.NotPanics(t, func() { WithHeaderForAllProviders("X-Title", "smolllm") })
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

// A fallback chain reaches a different provider on each attempt, so a
// provider-scoped header must never travel to the leg that follows it.
func TestProviderHeadersDoNotLeakAcrossFallbackLegs(t *testing.T) {
	// No t.Parallel: t.Setenv is incompatible with it.

	var mu sync.Mutex
	seen := map[string]http.Header{}
	record := func(who string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			seen[who] = r.Header.Clone()
			mu.Unlock()
			if who == "first" {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		}
	}
	first := httptest.NewServer(record("first"))
	defer first.Close()
	second := httptest.NewServer(record("second"))
	defer second.Close()

	t.Setenv("ALPHA_BASE_URL", first.URL)
	t.Setenv("ALPHA_API_KEY", "alpha-key")
	t.Setenv("BETA_BASE_URL", second.URL)
	t.Setenv("BETA_API_KEY", "beta-key")

	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("alpha/m,beta/m"),
		WithProviderHeader("alpha", "X-Api-Key", "alpha-secret"),
		WithHeaderForAllProviders("X-Title", "shared"))
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Contains(t, seen, "first")
	require.Contains(t, seen, "second")
	assert.Equal(t, "alpha-secret", seen["first"].Get("X-Api-Key"))
	assert.Empty(t, seen["second"].Get("X-Api-Key"),
		"alpha's credential must not reach beta after fallback")
	assert.Equal(t, "shared", seen["second"].Get("X-Title"), "unscoped headers still apply everywhere")
}

// The dangerous form is the verbosely-named one, and a scoped header is
// available for every provider identity including bare models.
func TestHeaderOptionShape(t *testing.T) {
	t.Parallel()
	assert.Panics(t, func() { WithHeaderForAllProviders("Authorization", "x") })
	assert.Panics(t, func() { WithProviderHeader("p", "Content-Type", "x") })
	// No name-based refusals any more: Idempotency-Key and X-Token-Count are
	// safe globally, and guessing secrecy from a name never worked.
	assert.NotPanics(t, func() { WithHeaderForAllProviders("Idempotency-Key", "abc") })
	assert.NotPanics(t, func() { WithHeaderForAllProviders("X-Token-Count", "5") })
	// A bare model resolves to the empty provider identity and must still be
	// able to carry a scoped credential.
	assert.NotPanics(t, func() { WithProviderHeader("", "X-Api-Key", "secret") })
}

// Options is exported, so a custom Option can write ExtraHeaders directly and
// never pass WithHeader's guard. The request builder must refuse reserved names
// too, or Add would append a second Authorization beside the real one.
func TestReservedHeadersRejectedAtRequestBuild(t *testing.T) {
	t.Parallel()
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	sneaky := func(o *Options) {
		o.ExtraHeaders = http.Header{}
		o.ExtraHeaders.Set("Authorization", "Bearer stolen")
		o.ExtraHeaders.Set("Content-Type", "text/plain")
	}
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("real"), sneaky)
	require.NoError(t, err)

	assert.Equal(t, []string{"Bearer real"}, got.Values("Authorization"))
	assert.Equal(t, []string{"application/json"}, got.Values("Content-Type"))
}

// An explicitly reported zero outranks a lower-priority dialect: the provider
// said nothing came from cache, and reporting another field's number as fact
// would be worse than reporting nothing.
func TestCacheReadHonoursExplicitZero(t *testing.T) {
	t.Parallel()
	eighty, zero := 80, 0
	u := usageChunk{
		PromptTokensDetails: &struct {
			CachedTokens *int `json:"cached_tokens"`
		}{CachedTokens: &zero},
		CacheReadInputTokens: &eighty,
	}
	assert.Equal(t, 0, u.cacheRead())
	assert.True(t, u.reportedCacheRead())

	// A details object carrying only unrelated keys says nothing about caching,
	// so it must not mask a lower-priority dialect that does.
	masked := usageChunk{
		PromptTokensDetails: &struct {
			CachedTokens *int `json:"cached_tokens"`
		}{},
		CacheReadInputTokens: &eighty,
	}
	assert.Equal(t, 80, masked.cacheRead())
	assert.True(t, masked.reportedCacheRead())

	// Write-only reporting says nothing about reads.
	writeOnly := usageChunk{CacheCreationInputTokens: &eighty}
	assert.False(t, writeOnly.reportedCacheRead())
	assert.True(t, writeOnly.reportedCacheWrite())

	silent := usageChunk{PromptTokens: 10}
	assert.Equal(t, 0, silent.cacheRead())
	assert.False(t, silent.reportedCacheRead(), "a provider that said nothing must not read as a reported zero")
}

// A blocking response carrying usage but no choices is an error, but its token
// counts are real and must still reach the failure hook — the streaming path
// records usage from a frame whose choices array is empty.
func TestBlockingEmptyChoicesStillReportsUsage(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":0,"cache_read_input_tokens":80}}`))
	}))
	defer srv.Close()

	var event RequestEvent
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
		WithoutStreaming(), WithHook(func(e RequestEvent) { event = e }))
	require.Error(t, err)
	assert.Equal(t, 100, event.InputTokens, "real prompt tokens must not be replaced by an estimate")
	assert.Equal(t, 80, event.CacheReadTokens)
	assert.False(t, event.Estimated)
}

// The extra-body merge decodes the whole payload, so it must not round integers
// that no longer fit a float64 on their way to the provider.
func TestExtraBodyMergePreservesLargeIntegers(t *testing.T) {
	t.Parallel()
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"), WithBaseURL(srv.URL+"/"), WithAPIKey("k"),
		WithSeed(9007199254740993),
		WithExtraBody(map[string]any{"metadata": map[string]any{"k": "v"}}))
	require.NoError(t, err)
	assert.Contains(t, string(raw), "9007199254740993",
		"seed was rounded through the extra-body merge")
}
