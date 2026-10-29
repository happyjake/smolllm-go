package smolllm

// AccountEventKind names one account-breaker decision.
type AccountEventKind string

const (
	// AccountOpened means a quota-exhausted leg opened (or a failed probe reopened)
	// its account circuit.
	AccountOpened AccountEventKind = "opened"
	// AccountClosed means a leg on an open or half-open account succeeded.
	AccountClosed AccountEventKind = "closed"
	// AccountProbing means this request holds the half-open lease and is attempting
	// the account after its cooldown.
	AccountProbing AccountEventKind = "probing"
	// AccountSkipped means a leg was not attempted; SkipReason says why.
	AccountSkipped AccountEventKind = "skipped"
	// AccountFailOpen means no admitted leg succeeded, so a leg on an open account
	// is attempted anyway as a last resort.
	AccountFailOpen AccountEventKind = "fail_open"
	// AccountUnrecognized429 means a leg failed with an HTTP 429 carrying a JSON
	// body that is not a recognized account-scoped code. It stays model-local;
	// the event exists so a new account-scoped code can be spotted.
	AccountUnrecognized429 AccountEventKind = "unrecognized_429"
)

// Skip reasons carried by AccountSkipped.
const (
	// SkipCircuitOpen: the shared account circuit is open, or half-open with a
	// probe already in flight.
	SkipCircuitOpen = "circuit_open"
	// SkipExhaustedInRequest: another leg on the same account reported quota
	// exhaustion earlier in this request.
	SkipExhaustedInRequest = "exhausted_in_request"
)

// AccountEvent reports one account-breaker decision on one leg. It carries no
// credential: Account is a short fingerprint of provider plus credential,
// stable for the process and safe to use as a metric label.
type AccountEvent struct {
	Kind      AccountEventKind
	Provider  string
	Account   string
	ModelName string
	// SkipReason is set only for AccountSkipped.
	SkipReason string
}

// WithAccountEventHook receives the account-breaker decisions made while
// serving this call. It fires only for legs guarded by an AccountBreaker, is
// called synchronously from the chain, and is separate from WithHook: a
// skipped leg is not an upstream attempt and must not look like one.
func WithAccountEventHook(fn func(AccountEvent)) Option {
	return func(o *Options) {
		o.AccountEventHook = fn
	}
}
