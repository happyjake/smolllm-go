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
	// quotaExhaustedCode is the doom/aitoken business code for "this account
	// has no quota left". It is account-scoped: every model behind the same
	// credential fails the same way until quota is restored.
	quotaExhaustedCode = 14019
)

// AccountBreaker is a process-local circuit breaker keyed by provider account
// (provider name plus a fingerprint of the configured credential). Once an
// account reports structured quota exhaustion, every leg served by that
// account is skipped — for the rest of the current request and for later
// requests sharing the breaker — until a bounded cooldown elapses. The next
// real request then probes the account once; concurrent requests keep
// skipping it while that probe is in flight.
//
// Create one per process and pass it to every call with WithAccountBreaker.
// It holds no credentials, persists nothing and runs no background traffic.
// The zero value is not usable; call NewAccountBreaker.
type AccountBreaker struct {
	mu       sync.Mutex
	now      func() time.Time
	accounts map[string]*accountCircuit
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
	return &AccountBreaker{mu: sync.Mutex{}, now: now, accounts: map[string]*accountCircuit{}}
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
	// CircuitOpen means the account is skipped until its cooldown elapses.
	CircuitOpen CircuitState = "open"
	// CircuitHalfOpen means the cooldown elapsed; the next real request probes the
	// account (or one already is).
	CircuitHalfOpen CircuitState = "half_open"
)

// AccountCircuit is a read-only view of one account circuit that is not closed.
type AccountCircuit struct {
	Provider string
	// Account is the short credential fingerprint also carried by AccountEvent.
	Account string
	State   CircuitState
}

// Circuits returns every account circuit that is not closed, sorted by
// provider then account. A closed account has no circuit and is not listed.
func (b *AccountBreaker) Circuits() []AccountCircuit {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.now()
	out := make([]AccountCircuit, 0, len(b.accounts))
	for account, c := range b.accounts {
		state := CircuitHalfOpen
		if now.Before(c.openUntil) {
			state = CircuitOpen
		}
		provider, label := splitAccount(account)
		out = append(out, AccountCircuit{Provider: provider, Account: label, State: state})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		return out[i].Account < out[j].Account
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

// splitAccount returns the provider and short fingerprint label of an account
// key. The hash is hex, so the last ':' separates it from the provider.
func splitAccount(account string) (provider, label string) {
	i := strings.LastIndexByte(account, ':')
	if i < 0 {
		return "", ""
	}
	label = account[i+1:]
	if len(label) > accountLabelLen {
		label = label[:accountLabelLen]
	}
	return account[:i], label
}

// shortFingerprint keeps logs correlatable without carrying the whole hash.
func shortFingerprint(account string) string {
	const keep = 12
	for i := 0; i < len(account); i++ {
		if account[i] == ':' && len(account) > i+1+keep {
			return account[:i+1+keep]
		}
	}
	return account
}

// isQuotaExhausted reports a structured account-quota rejection: HTTP 429 whose
// JSON body carries business code 14019 at data.code or error.data.code. The
// message text is never consulted — it is localized and free to change.
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
