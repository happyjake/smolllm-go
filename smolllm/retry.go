package smolllm

import (
	"context"
	"errors"
	"log/slog"
	"time"
)

const (
	defaultMaxRetries = 3
	retryBaseDelay    = 1 * time.Second
	// defaultFirstOutputTimeout is generous enough for a reasoning model to
	// think before its first delta and short enough that a queued leg cannot
	// eat the chain's budget. Three such legs stay well inside the 300s header
	// timeout the bundled pi client applies (DEFAULT_HTTP_IDLE_TIMEOUT_MS).
	defaultFirstOutputTimeout = 30 * time.Second
	retryMaxDelay     = 30 * time.Second
	retryBackoffScale = 2
)

// isRetryableHTTPStatus reports whether the given HTTP status should be retried
// against the SAME upstream model. 429 (rate limit) is intentionally excluded:
// it should fall through to the next model in the fallback chain immediately
// rather than waste seconds waiting on the same exhausted quota.
func isRetryableHTTPStatus(code int) bool {
	switch code {
	case 500, 502, 503, 529:
		return true
	}
	return false
}

func isRetryableError(err error) bool {
	// An empty stream emitted nothing to the caller and, on the providers that
	// behave this way, billed nothing (zero-usage terminal chunk), so a
	// same-leg re-attempt is cheap against a failure mode that comes and goes
	// between requests.
	var empty *EmptyStreamError
	if errors.As(err, &empty) {
		return true
	}
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		return isRetryableHTTPStatus(httpErr.StatusCode)
	}
	return false
}

func retryDelay(attempt int) time.Duration {
	d := retryBaseDelay
	for range attempt {
		d *= retryBackoffScale
	}
	if d > retryMaxDelay {
		d = retryMaxDelay
	}
	return d
}

// withRetry runs fn up to maxRetries times, retrying only on transient HTTP errors.
func withRetry[T any](ctx context.Context, logger *slog.Logger, model string, fn func() (T, error)) (T, error) {
	var zero T
	var lastErr error
	for attempt := range defaultMaxRetries {
		if attempt > 0 {
			// Delays: attempt=1 -> 1s, attempt=2 -> 2s, attempt=3 -> 4s, ...
			delay := retryDelay(attempt - 1)
			logger.Warn("retrying after transient error", "model", model, "attempt", attempt+1, "delay", delay, "err", lastErr)
			t := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				t.Stop()
				return zero, lastErr
			case <-t.C:
			}
		}
		result, err := fn()
		if err == nil {
			if attempt > 0 {
				logger.Info("recovered after retry", "model", model, "attempt", attempt+1, "prev_err", lastErr)
			}
			return result, nil
		}
		lastErr = err
		if !isRetryableError(err) {
			return zero, err
		}
	}
	return zero, lastErr
}
