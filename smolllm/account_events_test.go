package smolllm

import (
	"context"
	"net/http"
	"regexp"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// eventLog collects account events from the request-scoped hook.
type eventLog struct {
	mu     sync.Mutex
	events []AccountEvent
}

func (l *eventLog) record(e AccountEvent) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, e)
}

// take returns and clears the collected events.
func (l *eventLog) take() []AccountEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := l.events
	l.events = nil
	return out
}

// shape strips the fingerprint so expectations stay literal.
type eventShape struct {
	Kind       AccountEventKind
	Provider   string
	ModelName  string
	SkipReason string
}

func shapes(events []AccountEvent) []eventShape {
	out := make([]eventShape, 0, len(events))
	for _, e := range events {
		out = append(out, eventShape{Kind: e.Kind, Provider: e.Provider, ModelName: e.ModelName, SkipReason: e.SkipReason})
	}
	return out
}

var shortHex = regexp.MustCompile(`^[0-9a-f]{12}$`)

func TestAccountEventsReportTransitionsSkipsAndOpenState(t *testing.T) {
	t.Parallel()
	const secret = "sk-account-event-secret"
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", answer(t, "same account"))
	up.script("c", answer(t, "other provider"))
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	var log eventLog
	opts := up.options(botChain, WithAccountBreaker(breaker), WithAPIKey(secret), WithAccountEventHook(log.record))
	ask := func() {
		t.Helper()
		_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
		require.NoError(t, err)
	}

	ask()
	first := log.take()
	assert.Equal(t, []eventShape{
		{Kind: AccountOpened, Provider: "doom", ModelName: "a", SkipReason: ""},
		{Kind: AccountSkipped, Provider: "doom", ModelName: "b", SkipReason: SkipExhaustedInRequest},
	}, shapes(first))
	doom := first[0].Account
	assert.Regexp(t, shortHex, doom, "the account label is a short hash, never the credential")
	assert.Equal(t, doom, first[1].Account, "legs on one credential share one account")
	assert.Equal(t, []AccountCircuit{{Provider: "doom", Account: doom, State: CircuitOpen}}, breaker.Circuits())

	ask()
	assert.Equal(t, []eventShape{
		{Kind: AccountSkipped, Provider: "doom", ModelName: "a", SkipReason: SkipCircuitOpen},
		{Kind: AccountSkipped, Provider: "doom", ModelName: "b", SkipReason: SkipCircuitOpen},
	}, shapes(log.take()))

	clock.Advance(accountCooldownBase)
	assert.Equal(t, []AccountCircuit{{Provider: "doom", Account: doom, State: CircuitHalfOpen}}, breaker.Circuits())

	up.script("a", answer(t, "recovered"))
	ask()
	assert.Equal(t, []eventShape{
		{Kind: AccountProbing, Provider: "doom", ModelName: "a", SkipReason: ""},
		{Kind: AccountClosed, Provider: "doom", ModelName: "a", SkipReason: ""},
	}, shapes(log.take()))
	assert.Empty(t, breaker.Circuits(), "a closed account leaves no circuit behind")
}

func TestAccountEventsReportFailOpenAttempts(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", quotaExhausted)
	up.script("b", quotaExhausted)
	clock := newFakeClock()
	breaker := newAccountBreaker(clock.Now)
	var log eventLog
	opts := up.options("doom/a,zai/b", WithAccountBreaker(breaker), WithAccountEventHook(log.record))

	_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.Error(t, err)
	log.take()

	up.script("b", answer(t, "recovered early"))
	clock.Advance(time.Minute)
	_, err = Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	assert.Equal(t, []eventShape{
		{Kind: AccountSkipped, Provider: "doom", ModelName: "a", SkipReason: SkipCircuitOpen},
		{Kind: AccountSkipped, Provider: "zai", ModelName: "b", SkipReason: SkipCircuitOpen},
		{Kind: AccountFailOpen, Provider: "doom", ModelName: "a", SkipReason: ""},
		{Kind: AccountFailOpen, Provider: "zai", ModelName: "b", SkipReason: ""},
		{Kind: AccountClosed, Provider: "zai", ModelName: "b", SkipReason: ""},
	}, shapes(log.take()))
}

// A structured 429 the breaker does not recognize stays model-local but is
// reported, so a new account-scoped code shows up before anyone parses text.
// A plain-text 429 is ordinary rate limiting and reports nothing.
func TestAccountEventsFlagStructured429WithoutRecognizedCode(t *testing.T) {
	t.Parallel()
	up := newScriptedUpstream(t)
	up.script("a", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"data":{"code":14020,"msg":"x"}}}`))
	})
	up.script("b", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "slow down", http.StatusTooManyRequests)
	})
	up.script("c", answer(t, "other provider"))
	breaker := newAccountBreaker(newFakeClock().Now)
	var log eventLog

	opts := up.options(botChain, WithAccountBreaker(breaker), WithAccountEventHook(log.record))
	_, err := Ask(context.Background(), PromptFromString("hi"), opts...)
	require.NoError(t, err)
	assert.Equal(t, []eventShape{
		{Kind: AccountUnrecognized429, Provider: "doom", ModelName: "a", SkipReason: ""},
	}, shapes(log.take()))
	assert.Equal(t, 1, up.calls("b"), "an unrecognized code must not open the account")
	assert.Empty(t, breaker.Circuits())
}
