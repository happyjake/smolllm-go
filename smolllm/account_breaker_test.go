package smolllm

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// quotaExhaustedBody is the structured doom/aitoken account-quota rejection.
// The message is deliberately neutral: classification must come from the
// nested business code, never from the localized text.
const quotaExhaustedBody = `{"error":{"data":{"code":14019,"msg":"irrelevant text","requestId":"r1"}}}`

// fakeClock is a manually advanced clock for cooldown tests.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{mu: sync.Mutex{}, now: time.Date(2026, 10, 29, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// scriptedUpstream is one fake OpenAI-compatible server whose behavior is
// scripted per model name, counting every request each model receives.
type scriptedUpstream struct {
	mu      sync.Mutex
	scripts map[string]http.HandlerFunc
	hits    map[string]*atomic.Int32
	srv     *httptest.Server
}

func newScriptedUpstream(t *testing.T) *scriptedUpstream {
	t.Helper()
	u := &scriptedUpstream{
		mu:      sync.Mutex{},
		scripts: map[string]http.HandlerFunc{},
		hits:    map[string]*atomic.Int32{},
		srv:     nil,
	}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		u.mu.Lock()
		script, ok := u.scripts[model]
		counter := u.counter(model)
		u.mu.Unlock()
		counter.Add(1)
		if !ok {
			http.Error(w, "unexpected model", http.StatusBadRequest)
			return
		}
		script(w, r)
	}))
	t.Cleanup(u.srv.Close)
	return u
}

// counter must be called with mu held.
func (u *scriptedUpstream) counter(model string) *atomic.Int32 {
	c, ok := u.hits[model]
	if !ok {
		c = &atomic.Int32{}
		u.hits[model] = c
	}
	return c
}

func (u *scriptedUpstream) script(model string, fn http.HandlerFunc) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.scripts[model] = fn
}

func (u *scriptedUpstream) calls(model string) int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return int(u.counter(model).Load())
}

func (u *scriptedUpstream) options(model string, extra ...Option) []Option {
	opts := []Option{
		WithModel(model),
		WithBaseURL(u.srv.URL + "/"),
		WithAPIKey("account-key"),
		WithLogger(testLogger()),
	}
	return append(opts, extra...)
}

func quotaExhausted(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusTooManyRequests)
	_, _ = w.Write([]byte(quotaExhaustedBody))
}

func answer(t *testing.T, text string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, _ *http.Request) { writeChatSuccess(t, w, text, "stop") }
}

const botChain = "doom/a,doom/b,zai/c"

func TestQuotaExhaustionSkipsRemainingSameAccountLegsInAsk(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))
	breaker := newAccountBreaker(newFakeClock().Now)

	resp, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain, WithAccountBreaker(breaker))...)
	require.NoError(t, err)
	assert.Equal(t, "other provider", resp.Text)
	assert.Equal(t, 1, up.calls("a"))
	assert.Equal(t, 0, up.calls("b"), "the second model on the exhausted account must never be called")
	assert.Equal(t, 1, up.calls("c"))
}

func TestOpenAccountIsSkippedOnLaterRequestsUntilCooldownProbeSucceeds(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	ask := func() string {
		t.Helper()
		resp, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain, WithAccountBreaker(breaker))...)
		require.NoError(t, err)
		return resp.Text
	}

	assert.Equal(t, "other provider", ask())
	clock.Advance(4 * time.Minute)
	assert.Equal(t, "other provider", ask())
	assert.Equal(t, 1, up.calls("a"), "an open account must not be attempted before its cooldown")
	assert.Equal(t, 0, up.calls("b"))

	// Quota restored: the first real request after cooldown probes the account
	// and, on success, the account is preferred again.
	up.script("a", answer(t, "recovered"))
	clock.Advance(time.Minute)
	assert.Equal(t, "recovered", ask())
	assert.Equal(t, "recovered", ask())
	assert.Equal(t, 3, up.calls("a"))
	assert.Equal(t, 2, up.calls("c"))
}

func TestOrdinaryRateLimitStaysModelLocal(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	// A 429 without the structured code — even one whose text says "quota" —
	// keeps today's behavior: fall through to the next leg, open nothing.
	up.script("a", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"quota exhausted","code":"rate_limit_exceeded"}}`))
	})
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))
	breaker := newAccountBreaker(newFakeClock().Now)

	for range 2 {
		resp, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain, WithAccountBreaker(breaker))...)
		require.NoError(t, err)
		assert.Equal(t, "same account", resp.Text)
	}
	assert.Equal(t, 2, up.calls("a"))
	assert.Equal(t, 2, up.calls("b"))
	assert.Equal(t, 0, up.calls("c"))
}

func TestQuotaCodeOnlyCountsOnHTTP429(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(quotaExhaustedBody))
	})
	up.script("b", answer(t, "same account"))
	breaker := newAccountBreaker(newFakeClock().Now)

	opts := up.options("doom/a,doom/b", WithAccountBreaker(breaker))
	resp, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	assert.Equal(t, "same account", resp.Text)
}

func TestRepeatedProbeFailureBacksOffExponentiallyToOneHour(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("c", answer(t, "other provider"))
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	ask := func() {
		t.Helper()
		opts := up.options("doom/a,zai/c", WithAccountBreaker(breaker))
		resp, err := Ask(context.Background(), PromptFromString("hi"), opts...)
		require.NoError(t, err)
		assert.Equal(t, "other provider", resp.Text)
	}

	ask() // opens with the base cooldown
	calls := 1
	for _, cooldown := range []time.Duration{
		5 * time.Minute, 10 * time.Minute, 20 * time.Minute, 40 * time.Minute, 60 * time.Minute, 60 * time.Minute,
	} {
		clock.Advance(cooldown - time.Second)
		ask()
		require.Equal(t, calls, up.calls("a"), "probed before the %v cooldown elapsed", cooldown)
		clock.Advance(time.Second)
		ask()
		calls++
		require.Equal(t, calls, up.calls("a"), "no probe once the %v cooldown elapsed", cooldown)
	}
}

func TestHalfOpenAdmitsOneProbeWhileConcurrentRequestsSkip(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	opts := up.options(botChain, WithAccountBreaker(breaker))

	_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	clock.Advance(accountCooldownBase)

	probeArrived := make(chan struct{})
	release := make(chan struct{})
	up.script("a", func(w http.ResponseWriter, _ *http.Request) {
		close(probeArrived)
		<-release
		writeChatSuccess(t, w, "recovered", "stop")
	})

	probe := make(chan string, 1)
	go func() {
		resp, probeErr := Ask(context.Background(), PromptFromString("hi"), opts...)
		if probeErr != nil {
			probe <- probeErr.Error()
			return
		}
		probe <- resp.Text
	}()
	<-probeArrived

	const concurrent = 5
	var wg sync.WaitGroup
	texts := make([]string, concurrent)
	for i := range concurrent {
		wg.Go(func() {
			resp, askErr := Ask(context.Background(), PromptFromString("hi"), opts...)
			if askErr != nil {
				texts[i] = askErr.Error()
				return
			}
			texts[i] = resp.Text
		})
	}
	wg.Wait()
	for _, text := range texts {
		assert.Equal(t, "other provider", text, "a request must not join an in-flight probe")
	}
	assert.Equal(t, 2, up.calls("a"), "exactly one probe reaches the account")
	assert.Equal(t, 0, up.calls("b"))

	close(release)
	assert.Equal(t, "recovered", <-probe)
}

func TestAllOpenChainFailsOpen(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", quotaExhausted)
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	chain := WithModel("doom/a,zai/b")
	opts := append(up.options(""), chain, WithAccountBreaker(breaker))

	_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.Error(t, err)
	assert.Equal(t, 1, up.calls("a"))
	assert.Equal(t, 1, up.calls("b"))

	// Both accounts are open. Skipping both would turn a recovered account into
	// a guaranteed failure, so the chain is attempted as if there were no breaker.
	up.script("b", answer(t, "recovered early"))
	clock.Advance(time.Minute)
	resp, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	assert.Equal(t, "recovered early", resp.Text)
	assert.Equal(t, 2, up.calls("a"))
	assert.Equal(t, 2, up.calls("b"))
}

func TestFailOpenStillSkipsAccountExhaustedInThisRequest(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", answer(t, "same account"))
	up.script("c", quotaExhausted)
	breaker := newAccountBreaker(newFakeClock().Now)

	_, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain, WithAccountBreaker(breaker))...)
	require.Error(t, err)
	assert.True(t, isQuotaExhausted(err), "the last real failure surfaces, got %v", err)
	assert.Equal(t, 1, up.calls("a"))
	assert.Equal(t, 0, up.calls("b"))
	assert.Equal(t, 1, up.calls("c"))
}

func TestQuotaExhaustionSkipsSameAccountLegsInStreamAndLaterStreams(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", func(w http.ResponseWriter, _ *http.Request) { writeTextStream(t, w, "same account") })
	up.script("c", func(w http.ResponseWriter, _ *http.Request) { writeTextStream(t, w, "other ", "provider") })
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	stream := func() string {
		t.Helper()
		opts := up.options(botChain, WithAccountBreaker(breaker))
		resp, err := Stream(context.Background(), PromptFromString("hi"), opts...)
		require.NoError(t, err)
		text, err := drain(t, resp)
		require.NoError(t, err)
		return text
	}

	assert.Equal(t, "other provider", stream())
	assert.Equal(t, "other provider", stream())
	assert.Equal(t, 1, up.calls("a"))
	assert.Equal(t, 0, up.calls("b"), "the second model on the exhausted account must never be called")
	assert.Equal(t, 2, up.calls("c"))
}

// A streamed probe succeeds at its first output: the account closes even though
// the stream later dies, and the dying stream is not replayed on another leg.
func TestStreamProbeSucceedsAtFirstOutputAndNeverFallsBackAfterOutput(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("c", func(w http.ResponseWriter, _ *http.Request) { writeTextStream(t, w, "other provider") })
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	opts := up.options("doom/a,zai/c", WithAccountBreaker(breaker))

	resp, err := Stream(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	_, err = drain(t, resp)
	require.NoError(t, err)

	clock.Advance(accountCooldownBase)
	up.script("a", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {not json\n\n"))
	})
	resp, err = Stream(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	text, err := drain(t, resp)
	require.Error(t, err, "a stream that dies after output surfaces through Wait")
	assert.Equal(t, "partial", text)
	assert.Equal(t, 1, up.calls("c"), "no fallback after streamed output")

	// First output closed the circuit: the account is first choice again.
	up.script("a", func(w http.ResponseWriter, _ *http.Request) { writeTextStream(t, w, "recovered") })
	resp, err = Stream(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	text, err = drain(t, resp)
	require.NoError(t, err)
	assert.Equal(t, "recovered", text)
	assert.Equal(t, 1, up.calls("c"))
}

func TestBreakerKeepsSameLegRetryForTransientErrors(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	var attempts atomic.Int32
	up.script("a", func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			http.Error(w, "upstream unavailable", http.StatusServiceUnavailable)
			return
		}
		writeTextStream(t, w, "retried")
	})
	up.script("b", func(w http.ResponseWriter, _ *http.Request) { writeTextStream(t, w, "same account") })
	breaker := newAccountBreaker(newFakeClock().Now)

	opts := up.options("doom/a,doom/b", WithAccountBreaker(breaker))
	resp, err := Stream(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	text, err := drain(t, resp)
	require.NoError(t, err)
	assert.Equal(t, "retried", text)
	assert.Equal(t, 2, up.calls("a"))
	assert.Equal(t, 0, up.calls("b"))
}

func TestAccountIsScopedToTheCredentialNotTheProviderName(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("c", answer(t, "other provider"))
	breaker := newAccountBreaker(newFakeClock().Now)

	_, err := Ask(context.Background(), PromptFromString("hi"), up.options("doom/a,zai/c", WithAccountBreaker(breaker))...)
	require.NoError(t, err)

	up.script("a", answer(t, "second account"))
	resp, err := Ask(context.Background(), PromptFromString("hi"),
		up.options("doom/a,zai/c", WithAccountBreaker(breaker), WithAPIKey("another-account-key"))...)
	require.NoError(t, err)
	assert.Equal(t, "second account", resp.Text)
}

func TestBreakerLogsNeverCarryTheCredential(t *testing.T) {
	t.Parallel()
	const secret = "sk-very-secret-account-credential"
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("c", answer(t, "other provider"))
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	var logs bytes.Buffer
	handlerOpts := &slog.HandlerOptions{AddSource: false, Level: slog.LevelDebug, ReplaceAttr: nil}
	logger := slog.New(slog.NewTextHandler(&logs, handlerOpts))
	opts := up.options("doom/a,doom/b,zai/c", WithAccountBreaker(breaker), WithAPIKey(secret), WithLogger(logger))

	for range 2 {
		_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
		require.NoError(t, err)
	}
	clock.Advance(accountCooldownBase)
	up.script("a", answer(t, "recovered"))
	_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)

	out := logs.String()
	require.Contains(t, out, "account circuit opened")
	require.Contains(t, out, "account circuit closed")
	assert.NotContains(t, out, secret)
	// The fingerprint is a hash of provider+credential; neither the whole
	// secret nor its distinctive middle may appear.
	assert.NotContains(t, out, "very-secret-account")
}

// doom has been seen emitting the business object both bare and wrapped in
// "error"; either shape is the same structured signal.
func TestQuotaCodeIsRecognisedWithoutTheErrorWrapper(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"data":{"code":14019,"msg":"x"}}`))
	})
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))
	breaker := newAccountBreaker(newFakeClock().Now)

	resp, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain, WithAccountBreaker(breaker))...)
	require.NoError(t, err)
	assert.Equal(t, "other provider", resp.Text)
	assert.Equal(t, 0, up.calls("b"))
}

func TestWithoutBreakerQuotaExhaustionKeepsPlainFallback(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))

	resp, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain)...)
	require.NoError(t, err)
	assert.Equal(t, "same account", resp.Text)
}
