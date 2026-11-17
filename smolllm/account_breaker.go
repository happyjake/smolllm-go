package smolllm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	accountCooldownBase = 5 * time.Minute
	accountCooldownMax  = 60 * time.Minute
	// quotaExhaustedCode is the doom/aitoken business code the upstream reuses
	// for every quota-shaped rejection. Observed 2026-11-17 behind one shared
	// credential it can mean "global quota exhausted" for billed models while
	// free-tier models (credit 0) keep answering 200, and it also fires for
	// unregistered model names. The upstream does not separate those scopes,
	// so the least wrong assumption is per model: open the circuit for the
	// account+model that reported it, never for sibling models on the account.
	quotaExhaustedCode = 14019
)

// maxAccountCircuits bounds the breaker map. Because the key carries the
// model name and model names can arrive from requests (not only from
// configured chains — unregistered names also report quotaExhaustedCode),
// the map is bounded independently of configuration: when it is full of
// circuits still doing their job, new pairs simply get no circuit, which
// degrades to no-breaker behavior for them (they are attempted every time)
// and never to less availability. 4096 pairs is far above any configured
// chain fan-out and caps the map at a few hundred kilobytes.
const maxAccountCircuits = 4096

// maxBreakerModelLen bounds one entry's byte size: a model name longer than
// this gets no circuit (attempted every request, like no breaker) instead of
// parking a request-sized string in the breaker map. Real names are short.
const maxBreakerModelLen = 256

// maxBreakerProviderLen caps the other variable part of a key: the provider
// name arrives in the same request-borne model string ("provider/model") and
// would otherwise bypass the model-length bound.
const maxBreakerProviderLen = 64

// breakerKey returns the account+model circuit key for the leg, or ok=false
// when the leg must run unguarded: a variable part exceeds its length cap
// (the map bound counts entries, not bytes), or the provider name contains a
// NUL, which would corrupt key parsing (the fingerprint is provider plus hex
// and otherwise NUL-free, so the first NUL is the separator).
func (leg legAccount) breakerKey() (string, bool) {
	if leg.fingerprint == "" || len(leg.modelName) > maxBreakerModelLen {
		return "", false
	}
	if len(leg.provider) > maxBreakerProviderLen || strings.IndexByte(leg.provider, 0) >= 0 {
		return "", false
	}
	return leg.fingerprint + "\x00" + leg.modelName, true
}

// AccountBreaker is a process-local circuit breaker keyed by provider account
// plus model (provider name, a fingerprint of the configured credential, and
// the model name). Once a leg reports structured quota exhaustion, that
// account+model pair is skipped — for the rest of the current request and for
// later requests sharing the breaker — until a bounded cooldown elapses. The
// next real request then probes it once; concurrent requests keep skipping it
// while that probe is in flight. Sibling models on the same account are not
// affected: the upstream quota code does not distinguish account-wide
// exhaustion from model-local causes, and free-tier models sharing the
// credential stay available while billed ones are rejected.
//
// Create one per process and pass it to every call with WithAccountBreaker.
// It holds no credentials, persists nothing and runs no background traffic.
// The zero value is not usable; call NewAccountBreaker.
type AccountBreaker struct {
	mu       sync.Mutex
	now      func() time.Time
	accounts map[string]*accountCircuit
	// maxCircuits bounds len(accounts); a field so tests can shrink it.
	// Defaults to maxAccountCircuits.
	maxCircuits int
}

// accountCircuit exists only while an account is not closed.
type accountCircuit struct {
	openUntil time.Time
	failures  int
	probing   bool
}

// NewAccountBreaker returns an empty breaker in which every account is closed.
func NewAccountBreaker() *AccountBreaker {
	return newAccountBreaker(time.Now)
}

func newAccountBreaker(now func() time.Time) *AccountBreaker {
	return &AccountBreaker{now: now, accounts: map[string]*accountCircuit{}, maxCircuits: maxAccountCircuits}
}

// makeRoom evicts idle half-open circuits so a new pair can be inserted. A
// circuit is evictable only when its cooldown has lapsed and it holds no
// probe lease; circuits still working are never dropped. It reports whether
// the map now has room.
func (b *AccountBreaker) makeRoom() bool {
	if len(b.accounts) < b.maxCircuits {
		return true
	}
	now := b.now()
	for key, c := range b.accounts {
		if now.Before(c.openUntil) || c.probing {
			continue
		}
		delete(b.accounts, key)
		if len(b.accounts) < b.maxCircuits {
			return true
		}
	}
	return len(b.accounts) < b.maxCircuits
}

// admit decides whether a leg on account may start. probe reports that the
// caller holds the half-open lease and must settle it with record.
func (b *AccountBreaker) admit(account string) (ok, probe bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	c, exists := b.accounts[account]
	if !exists {
		return true, false
	}
	if b.now().Before(c.openUntil) || c.probing {
		return false, false
	}
	c.probing = true
	return true, true
}

type legOutcome int

const (
	legInconclusive legOutcome = iota
	legSucceeded
	legQuotaExhausted
)

// record settles one attempted leg on account and reports the transition it
// caused: AccountOpened, AccountClosed, or "" when the circuit did not move.
func (b *AccountBreaker) record(logger *slog.Logger, account string, probe bool, outcome legOutcome) AccountEventKind {
	b.mu.Lock()
	defer b.mu.Unlock()
	c, exists := b.accounts[account]
	switch outcome {
	case legSucceeded:
		if exists {
			delete(b.accounts, account)
			logger.Info("account circuit closed", "account", shortFingerprint(account))
			return AccountClosed
		}
	case legQuotaExhausted:
		// Only a closed account or the probe holder moves the circuit. A
		// straggler admitted before another request opened it, or a fail-open
		// attempt, must not escalate the cooldown it merely re-confirms.
		if exists && !probe {
			return ""
		}
		if !exists {
			if !b.makeRoom() {
				// The map is full of circuits still in cooldown or probing.
				// This pair gets no circuit: it is attempted on every request,
				// which is exactly the no-breaker behavior for it.
				logger.Warn("account circuit map full, leaving pair unguarded",
					"account", shortFingerprint(account))
				return ""
			}
			c = &accountCircuit{openUntil: time.Time{}, failures: 0, probing: false}
			b.accounts[account] = c
		}
		c.failures++
		c.probing = false
		cooldown := accountCooldown(c.failures)
		c.openUntil = b.now().Add(cooldown)
		logger.Warn("account circuit opened: quota exhausted",
			"account", shortFingerprint(account), "failures", c.failures, "cooldown", cooldown)
		return AccountOpened
	case legInconclusive:
		if exists && probe {
			c.probing = false
		}
	}
	return ""
}

// CircuitState is the state of an account circuit that is not closed.
type CircuitState string

const (
	// CircuitOpen means the account+model pair is skipped until its cooldown elapses.
	CircuitOpen CircuitState = "open"
	// CircuitHalfOpen means the cooldown elapsed; the next real request probes
	// that pair (or one already is).
	CircuitHalfOpen CircuitState = "half_open"
)

// AccountCircuit is a read-only view of one account+model circuit that is not closed.
type AccountCircuit struct {
	Provider string
	// Account is the short credential fingerprint also carried by AccountEvent.
	Account string
	// ModelName is the model whose quota rejection opened this circuit.
	ModelName string
	State     CircuitState
}

// Circuits returns every account+model circuit that is not closed, sorted by
// provider, account, then model. A closed pair has no circuit and is not listed.
func (b *AccountBreaker) Circuits() []AccountCircuit {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.now()
	out := make([]AccountCircuit, 0, len(b.accounts))
	for key, c := range b.accounts {
		state := CircuitHalfOpen
		if now.Before(c.openUntil) {
			state = CircuitOpen
		}
		provider, label, model := splitAccount(key)
		out = append(out, AccountCircuit{Provider: provider, Account: label, ModelName: model, State: state})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		if out[i].Account != out[j].Account {
			return out[i].Account < out[j].Account
		}
		return out[i].ModelName < out[j].ModelName
	})
	return out
}

func accountCooldown(failures int) time.Duration {
	d := accountCooldownBase
	for i := 1; i < failures && d < accountCooldownMax; i++ {
		d *= 2
	}
	return min(d, accountCooldownMax)
}

// legAccount is the account serving one leg, as the chain sees it.
type legAccount struct {
	provider  string
	modelName string
	// fingerprint is "" when the leg cannot be resolved; it then runs
	// unguarded and fails on its own terms.
	fingerprint string
}

// resolveLegAccount identifies the account serving model. The credential is
// hashed, never stored, so neither memory nor logs hold it.
func resolveLegAccount(opts Options, model string) legAccount {
	spec, _ := parseModelSpec(model)
	prov, modelName, err := parseModelString(spec)
	if err != nil {
		return legAccount{provider: "", modelName: "", fingerprint: ""}
	}
	leg := legAccount{provider: prov.Name, modelName: modelName, fingerprint: ""}
	key, err := resolveAPIKey(prov, modelName, opts.APIKey)
	if err != nil {
		return leg
	}
	sum := sha256.Sum256([]byte(prov.Name + "\x00" + key))
	leg.fingerprint = prov.Name + ":" + hex.EncodeToString(sum[:])
	return leg
}

// accountLabelLen bounds the fingerprint exposed to events and metrics: enough
// to tell a handful of configured credentials apart, far too little to matter
// as a hash of a secret.
const accountLabelLen = 12

// splitAccount returns the provider, the short fingerprint label and the
// model name of an account+model breaker key. The credential hash is hex and
// provider names come from configuration, so neither contains '\x00': the
// first NUL is the separator and the model name may safely contain anything,
// including further NULs. The last ':' before the fingerprint separates it
// from the provider. An old bare account key (no model) yields an empty model.
func splitAccount(key string) (provider, label, model string) {
	if i := strings.IndexByte(key, 0); i >= 0 {
		model = key[i+1:]
		key = key[:i]
	}
	i := strings.LastIndexByte(key, ':')
	if i < 0 {
		return "", "", model
	}
	label = key[i+1:]
	if len(label) > accountLabelLen {
		label = label[:accountLabelLen]
	}
	return key[:i], label, model
}

// shortFingerprint keeps logs correlatable without carrying the whole hash;
// the model half of a breaker key never reaches it (logs carry model=), and
// the first NUL is the separator for the same reason as in splitAccount.
func shortFingerprint(key string) string {
	const keep = 12
	if i := strings.IndexByte(key, 0); i >= 0 {
		key = key[:i]
	}
	for i := 0; i < len(key); i++ {
		if key[i] == ':' && len(key) > i+1+keep {
			return key[:i+1+keep]
		}
	}
	return key
}

// isQuotaExhausted reports a structured quota rejection: HTTP 429 whose JSON
// body carries business code 14019 at data.code or error.data.code. The code
// is reused by the upstream for account-wide and model-local causes alike
// (including unregistered model names), so it is treated as scoped to the
// account+model that reported it. The message text is never consulted — it
// is localized and free to change.
func isQuotaExhausted(err error) bool {
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusTooManyRequests {
		return false
	}
	var body businessErrorBody
	if json.Unmarshal([]byte(httpErr.Body), &body) != nil {
		return false
	}
	if body.Data.is(quotaExhaustedCode) {
		return true
	}
	return body.Error != nil && body.Error.Data.is(quotaExhaustedCode)
}

// isUnrecognizedStructured429 reports an HTTP 429 with a JSON object body that
// is not a recognized account-scoped code. It stays model-local; it is surfaced
// so operators can spot account-scoped codes the breaker does not know yet.
func isUnrecognizedStructured429(err error) bool {
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusTooManyRequests {
		return false
	}
	var body map[string]json.RawMessage
	if json.Unmarshal([]byte(httpErr.Body), &body) != nil {
		return false
	}
	return !isQuotaExhausted(err)
}

type businessErrorBody struct {
	Data  *businessErrorData `json:"data"`
	Error *struct {
		Data *businessErrorData `json:"data"`
	} `json:"error"`
}

// businessErrorData decodes only the numeric code. A string or absent code
// fails to match rather than being coerced.
type businessErrorData struct {
	Code *int `json:"code"`
}

func (d *businessErrorData) is(code int) bool {
	return d != nil && d.Code != nil && *d.Code == code
}
