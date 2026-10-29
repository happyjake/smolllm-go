package smolllm

import (
	"context"
	"fmt"
)

// runChain walks the fallback chain, running attempt (with same-leg retry) on
// each leg until one succeeds. It is the one place Ask and Stream decide which
// untouched upstream attempt may start; what happens after a leg succeeds —
// including streamed output that can no longer be retracted — is the
// caller's.
//
// With an AccountBreaker, a leg whose account circuit is open is skipped, and
// an account that reports quota exhaustion is skipped for the rest of this
// request even if the shared breaker closes concurrently. When the chain ends
// without a success — every leg refused (all accounts open), or the admitted
// legs all failed — it fails open and attempts the refused legs as a last
// resort, so the breaker only ever reorders attempts and never makes
// availability worse than having no breaker. Legs on an account that reported
// quota exhaustion in this very request stay skipped: that is fresh evidence,
// not a stale circuit.
func runChain[T any](ctx context.Context, opts Options, attempt func(model string) (T, error)) (T, error) {
	var zero T
	selector, err := createSelector(opts)
	if err != nil {
		return zero, err
	}

	breaker := opts.AccountBreaker
	exhausted := map[string]bool{}
	var refused []string
	var lastErr error

	runLeg := func(model string, failOpen bool) (T, bool, error) {
		leg := legAccount{provider: "", modelName: "", fingerprint: ""}
		if breaker != nil {
			leg = resolveLegAccount(opts, model)
		}
		account := leg.fingerprint
		probe := false
		emit := func(kind AccountEventKind, skipReason string) {
			if opts.AccountEventHook == nil || kind == "" {
				return
			}
			_, label := splitAccount(account)
			opts.AccountEventHook(AccountEvent{
				Kind: kind, Provider: leg.provider, Account: label, ModelName: leg.modelName, SkipReason: skipReason,
			})
		}
		if account != "" {
			if exhausted[account] {
				opts.Logger.Warn("skipping model: account quota exhausted earlier in this request",
					"model", model, "account", shortFingerprint(account))
				emit(AccountSkipped, SkipExhaustedInRequest)
				return zero, false, nil
			}
			if failOpen {
				emit(AccountFailOpen, "")
			} else {
				var ok bool
				ok, probe = breaker.admit(account)
				if !ok {
					opts.Logger.Warn("skipping model: account circuit open",
						"model", model, "account", shortFingerprint(account))
					emit(AccountSkipped, SkipCircuitOpen)
					refused = append(refused, model)
					return zero, false, nil
				}
				if probe {
					opts.Logger.Info("probing account after cooldown", "model", model, "account", shortFingerprint(account))
					emit(AccountProbing, "")
				}
			}
		}

		// Settled by defer so an attempt that panics (a caller's stream handler,
		// say) still releases a half-open lease as inconclusive; otherwise the
		// account would look probed-in-flight forever and never be tried again.
		outcome := legInconclusive
		if account != "" {
			defer func() { emit(breaker.record(opts.Logger, account, probe, outcome), "") }()
		}
		resp, err := withRetry(ctx, opts.Logger, model, func() (T, error) { return attempt(model) })
		if account != "" {
			switch {
			case err == nil:
				outcome = legSucceeded
			case isQuotaExhausted(err):
				outcome = legQuotaExhausted
				exhausted[account] = true
			case isUnrecognizedStructured429(err):
				emit(AccountUnrecognized429, "")
			}
		}
		return resp, true, err
	}

	handle := func(model string, err error, more bool) error {
		lastErr = err
		// A dead caller context is not a per-model failure: every remaining
		// leg would dial with a cancelled context, fail in microseconds and
		// log a fallback warning that never represented a real attempt.
		// Stop here so the cancellation surfaces as itself.
		if ctxErr := ctx.Err(); ctxErr != nil {
			opts.Logger.Warn("aborting fallback chain: caller context is done", "model", model, "error", ctxErr.Error())
			return ctxErr
		}
		if more {
			opts.Logger.Warn("model failed, trying fallback", "model", model, "error", err.Error())
		} else {
			opts.Logger.Warn("model failed", "model", model, "error", err.Error())
		}
		return nil
	}

	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}
		resp, ran, err := runLeg(model, false)
		if !ran {
			continue
		}
		if err == nil {
			return resp, nil
		}
		if stop := handle(model, err, selector.HasMore()); stop != nil {
			return zero, stop
		}
	}

	if len(refused) > 0 {
		opts.Logger.Warn("no admitted model succeeded; failing open to open-circuit models", "models", len(refused))
		for i, model := range refused {
			resp, ran, err := runLeg(model, true)
			if !ran {
				continue
			}
			if err == nil {
				return resp, nil
			}
			if stop := handle(model, err, i < len(refused)-1); stop != nil {
				return zero, stop
			}
		}
	}

	if lastErr != nil {
		return zero, lastErr
	}
	return zero, fmt.Errorf("no models were attempted")
}
