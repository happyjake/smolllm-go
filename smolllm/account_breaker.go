package smolllm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
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

// record settles one attempted leg on account.
func (b *AccountBreaker) record(logger *slog.Logger, account string, probe bool, outcome legOutcome) {
	b.mu.Lock()
	defer b.mu.Unlock()
	c, exists := b.accounts[account]
	switch outcome {
	case legSucceeded:
		if exists {
			delete(b.accounts, account)
			logger.Info("account circuit closed", "account", shortFingerprint(account))
		}
	case legQuotaExhausted:
		// Only a closed account or the probe holder moves the circuit. A
		// straggler admitted before another request opened it, or a fail-open
		// attempt, must not escalate the cooldown it merely re-confirms.
		if exists && !probe {
			return
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
	case legInconclusive:
		if exists && probe {
			c.probing = false
		}
	}
}

func accountCooldown(failures int) time.Duration {
	d := accountCooldownBase
	for i := 1; i < failures && d < accountCooldownMax; i++ {
		d *= 2
	}
	return min(d, accountCooldownMax)
}

// accountFingerprint identifies the account serving model, or "" when the leg
// cannot be resolved (it then runs unguarded and fails on its own terms). The
// credential is hashed, never stored, so neither memory nor logs hold it.
func accountFingerprint(opts Options, model string) string {
	spec, _ := parseModelSpec(model)
	prov, modelName, err := parseModelString(spec)
	if err != nil {
		return ""
	}
	key, err := resolveAPIKey(prov, modelName, opts.APIKey)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256([]byte(prov.Name + "\x00" + key))
	return prov.Name + ":" + hex.EncodeToString(sum[:])
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
