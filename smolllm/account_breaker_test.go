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

func TestQuotaExhaustionDoesNotSkipSiblingModelsOnTheSameAccount(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))
	breaker := newAccountBreaker(newFakeClock().Now)

	// The quota code is reused by the upstream for account-wide and
	// model-local causes alike; free-tier models on the same credential keep
	// serving. So the sibling model must be attempted and can succeed.
	for range 2 {
		resp, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain, WithAccountBreaker(breaker))...)
		require.NoError(t, err)
		assert.Equal(t, "same account", resp.Text)
	}
	assert.Equal(t, 1, up.calls("a"), "only model a itself is circuit-open after its quota rejection")
	assert.Equal(t, 2, up.calls("b"), "the sibling model on the same account must not be skipped")
	assert.Equal(t, 0, up.calls("c"))
}

// A probe whose attempt panics (here: the caller's stream handler) must still
// release the half-open lease. Otherwise the account stays half-open with a
// probe "in flight" forever and is never attempted again.
func TestPanickingProbeReleasesTheLeaseForALaterProbe(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("c", answer(t, "other provider"))
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	opts := up.options("doom/a,zai/c", WithAccountBreaker(breaker))

	_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	clock.Advance(accountCooldownBase)

	up.script("a", answer(t, "recovered"))
	panicHandler := WithStreamHandler(func(context.Context, string) error { panic("caller bug") })
	panicking := up.options("doom/a,zai/c", WithAccountBreaker(breaker), panicHandler)
	assert.PanicsWithValue(t, "caller bug", func() {
		_, _ = Ask(context.Background(), PromptFromString("hi"), panicking...)
	})
	require.Equal(t, 2, up.calls("a"), "the panicking request was the probe")

	resp, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	assert.Equal(t, "recovered", resp.Text, "a later request must be able to probe again")
	assert.Equal(t, 3, up.calls("a"))
}

func TestOpenPairIsSkippedOnLaterRequestsUntilCooldownProbeSucceeds(t *testing.T) {
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

	// Model a is circuit-open, but its sibling b on the same account keeps
	// serving, so the chain answers from b throughout the cooldown.
	assert.Equal(t, "same account", ask())
	clock.Advance(4 * time.Minute)
	assert.Equal(t, "same account", ask())
	assert.Equal(t, 1, up.calls("a"), "an open pair must not be attempted before its cooldown")
	assert.Equal(t, 2, up.calls("b"))

	// Quota restored: the first real request after cooldown probes model a and,
	// on success, the pair is preferred again.
	up.script("a", answer(t, "recovered"))
	clock.Advance(time.Minute)
	assert.Equal(t, "recovered", ask())
	assert.Equal(t, "recovered", ask())
	assert.Equal(t, 3, up.calls("a"))
	assert.Equal(t, 2, up.calls("b"))
	assert.Equal(t, 0, up.calls("c"))
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
		assert.Equal(t, "same account", text, "a request must not join an in-flight probe; it falls to the sibling, not off the account")
	}
	assert.Equal(t, 2, up.calls("a"), "exactly one probe reaches the open pair")
	assert.Equal(t, concurrent+1, up.calls("b"), "concurrent requests fall to the sibling model")

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

func TestFailOpenStillSkipsPairExhaustedInThisRequest(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", quotaExhausted)
	up.script("c", quotaExhausted)
	breaker := newAccountBreaker(newFakeClock().Now)

	_, err := Ask(context.Background(), PromptFromString("hi"), up.options(botChain, WithAccountBreaker(breaker))...)
	require.Error(t, err)
	assert.True(t, isQuotaExhausted(err), "the last real failure surfaces, got %v", err)
	assert.Equal(t, 1, up.calls("a"))
	assert.Equal(t, 1, up.calls("b"), "the sibling model is attempted: quota rejections are per model")
	assert.Equal(t, 1, up.calls("c"))
}

func TestQuotaExhaustedModelIsSkippedInStreamAndLaterStreams(t *testing.T) {
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

	assert.Equal(t, "same account", stream())
	assert.Equal(t, "same account", stream())
	assert.Equal(t, 1, up.calls("a"), "the quota-rejected model is circuit-open after its first rejection")
	assert.Equal(t, 2, up.calls("b"), "the sibling model on the same account keeps serving")
	assert.Equal(t, 0, up.calls("c"))
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
	up.script("b", answer(t, "same account"))
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
	assert.Equal(t, "same account", resp.Text)
	assert.Equal(t, 1, up.calls("b"), "the sibling model is attempted: quota rejections are per model")
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

// A chain may repeat the same leg (the selector does not deduplicate). The
// second occurrence must be skipped as exhausted-in-request, and when the
// whole chain fails, fail-open must not retry it: the rejection is fresh
// evidence from this very request, not a stale circuit.
func TestDuplicatePairIsSkippedWithinOneRequest(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("c", answer(t, "other provider"))
	breaker := newAccountBreaker(newFakeClock().Now)
	var log eventLog
	opts := up.options("doom/a,doom/a,zai/c", WithAccountBreaker(breaker), WithAccountEventHook(log.record))

	resp, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	assert.Equal(t, "other provider", resp.Text)
	assert.Equal(t, 1, up.calls("a"), "the repeated leg must not be attempted twice")
	assert.Equal(t, []eventShape{
		{Kind: AccountOpened, Provider: "doom", ModelName: "a"},
		{Kind: AccountSkipped, Provider: "doom", ModelName: "a", SkipReason: SkipExhaustedInRequest},
	}, shapes(log.take()))

	// All-fail variant: fail-open retries refused (circuit-skipped) legs, but
	// the pair that reported quota exhaustion in this request stays skipped.
	up2 := newScriptedUpstream(t)
	up2.script("a", quotaExhausted)
	breaker2 := newAccountBreaker(newFakeClock().Now)
	_, err = Ask(context.Background(), PromptFromString("hi"), up2.options("doom/a,doom/a", WithAccountBreaker(breaker2))...)
	require.Error(t, err)
	assert.True(t, isQuotaExhausted(err))
	assert.Equal(t, 1, up2.calls("a"), "fail-open must not retry a pair exhausted in this request")
}

// The breaker map is bounded independently of the model names requests send:
// once full of working circuits, further pairs get no circuit (attempted every
// time, like no breaker), and lapsed circuits are evicted to make room.
func TestBreakerCapsAccountModelEntries(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	for _, m := range []string{"m1", "m2", "m3"} {
		up.script(m, quotaExhausted)
	}
	up.script("ok", answer(t, "fine"))
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	breaker.maxCircuits = 2
	ask := func(model string) error {
		_, err := Ask(context.Background(), PromptFromString("hi"), up.options("doom/"+model, WithAccountBreaker(breaker))...)
		return err
	}

	require.Error(t, ask("m1")) // circuit 1
	require.Error(t, ask("m2")) // circuit 2
	require.Error(t, ask("m3")) // map full, both circuits in cooldown: no circuit for m3
	assert.Len(t, breaker.Circuits(), 2, "the map never exceeds its bound")

	// An unguarded pair is attempted on every request instead of being skipped.
	require.Error(t, ask("m3"))
	assert.Equal(t, 2, up.calls("m3"), "a pair without a circuit must stay attemptable")
	assert.Len(t, breaker.Circuits(), 2)

	// After every cooldown lapses, opening a new circuit evicts idle ones to
	// make room; the map stays within its bound.
	clock.Advance(accountCooldownMax + time.Minute)
	require.Error(t, ask("m3"))
	circuits := breaker.Circuits()
	assert.LessOrEqual(t, len(circuits), 2, "the bound is an invariant, not a one-shot")
	names := map[string]bool{}
	for _, c := range circuits {
		names[c.ModelName] = true
	}
	assert.False(t, names["m1"], "a lapsed circuit is evicted to make room")
	assert.True(t, names["m3"], "the fresh circuit is kept")
}

func TestSplitAccountHandlesNulModelNames(t *testing.T) {
	t.Parallel()
	key := "doom:abcdef0123456789\x00weird\x00model"
	provider, label, model := splitAccount(key)
	assert.Equal(t, "doom", provider)
	assert.Equal(t, "abcdef012345", label)
	assert.Equal(t, "weird\x00model", model, "the model name may itself contain NULs")
	assert.Equal(t, "doom:abcdef012345", shortFingerprint(key))

	provider, label, model = splitAccount("doom:abcdef0123456789")
	assert.Equal(t, "doom", provider)
	assert.Equal(t, "abcdef012345", label)
	assert.Empty(t, model, "a bare legacy account key has no model half")
}
