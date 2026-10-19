package smolllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// Stream returns a streaming response channel for the given prompt.
func Stream(ctx context.Context, prompt Prompt, opts ...Option) (*StreamResponse, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}

	options := applyOptions(opts...)
	if options.DisableStreaming {
		return nil, errors.New("WithoutStreaming is incompatible with Stream; use Ask")
	}

	selector, err := createSelector(options)
	if err != nil {
		return nil, err
	}

	var lastErr error
	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}
		resp, err := withRetry(ctx, options.Logger, model, func() (*StreamResponse, error) {
			return streamOnce(ctx, prompt, options, model)
		})
		if err != nil {
			lastErr = err
			// A dead caller context is not a per-model failure: every remaining
			// leg would dial with a cancelled context, fail in microseconds and
			// log a fallback warning that never represented a real attempt.
			// Stop here so the cancellation surfaces as itself.
			if ctxErr := ctx.Err(); ctxErr != nil {
				options.Logger.Warn("aborting fallback chain: caller context is done", "model", model, "error", ctxErr.Error())
				return nil, ctxErr
			}
			if selector.HasMore() {
				options.Logger.Warn("model failed, trying fallback", "model", model, "error", err.Error())
			} else {
				options.Logger.Warn("model failed", "model", model, "error", err.Error())
			}
			continue
		}
		return resp, nil
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, fmt.Errorf("no models were attempted")
}

func streamOnce(ctx context.Context, prompt Prompt, opts Options, model string) (*StreamResponse, error) {
	exec, err := newCallExecution(ctx, prompt, opts, model)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*StreamResponse, error) {
		// No usage can exist yet: this fail path runs before the stream is
		// established. Usage that arrives and is then followed by an error is
		// preserved by the forwarder, which builds metrics unconditionally.
		emitFailureHook(opts.Hook, exec.call, err, exec.start, nil)
		return nil, err
	}

	resp, err := exec.do("starting stream")
	if err != nil {
		if guardErr := exec.firstOutputError("headers"); guardErr != nil {
			exec.cancel()
			return nil, guardErr
		}
		exec.cancel()
		return fail(err)
	}

	if resp.StatusCode >= http.StatusBadRequest {
		retryResp, retried, retryErr := exec.retryWithoutStreamUsage(resp)
		if retryErr != nil {
			exec.cancel()
			return fail(retryErr)
		}
		if retried {
			resp = retryResp
		}
	}

	if resp.StatusCode >= http.StatusBadRequest {
		defer func() { _ = resp.Body.Close() }()
		exec.cancel()
		return fail(httpError(resp))
	}

	chunks, done := startStreamForwarder(exec.requestContext(), opts.Logger, resp, exec.call, exec.cancel, exec.start)

	sr := &StreamResponse{
		Stream: DeltaStream{ //nolint:exhaustruct // populated below
		},
		Reasoning:    "",
		FinishReason: "",
		ToolCalls:    nil,
		Model:        exec.call.Model,
		ModelName:    exec.call.ModelName,
		Provider:     exec.call.Provider.Name,
		Usage: Usage{
			Provider:     exec.call.Provider.Name,
			Model:        exec.call.Model,
			ModelName:    exec.call.ModelName,
			APIKeyHint:   previewAPIKey(exec.call.APIKey),
			InputTokens:  exec.call.InputTokens,
			OutputTokens: 0,
			Duration:     0,
			TTFT:         0,
			Estimated:    true,
		},
	}
	sr.Stream = DeltaStream{
		ch:           chunks,
		done:         done,
		cancel:       exec.cancel,
		logger:       opts.Logger,
		reasoning:    &sr.Reasoning,
		finishReason: &sr.FinishReason,
		toolCalls:    &sr.ToolCalls,
		usage:        &sr.Usage,
		hook:         opts.Hook,
		streaming:    exec.call.Streaming,
	}
	if err := gateFirstOutput(exec, opts, sr); err != nil {
		return nil, err
	}
	return sr, nil
}

// gateFirstOutput holds a freshly established stream until it proves it will
// deliver an assistant turn, so a dead-on-arrival stream fails the leg before
// anything reaches the caller — and therefore while the fallback chain can
// still route around it.
//
// It waits for the first usable chunk (content or reasoning) and hands the
// stream back with that chunk re-injected. A stream that ends before any
// chunk carries a complete turn only when tool calls accumulated; anything
// else is an EmptyStreamError. The doom/pi-proxy gpt-6.1-sol leg returns HTTP
// 200 with a single synthesized finish_reason=error chunk and zero usage on a
// fraction of calls; without this gate that surfaced to clients as a
// "successful" stream whose only content was a terminal error frame.
//
// Streams that delivered output and then died keep their existing behavior:
// bytes already handed to the caller cannot be taken back, so their terminal
// error still surfaces through Wait().
func gateFirstOutput(exec *callExecution, opts Options, sr *StreamResponse) error {
	s := &sr.Stream
	first, ok := <-s.ch
	if ok {
		// A delta arrived, so this leg has proven it will deliver: stop the
		// first-output guard before anything is forwarded. If it already fired
		// the leg was abandoned mid-wait and this chunk must not leak out.
		if !exec.disarmFirstOutput() {
			exec.cancel()
			s.absorb(<-s.done)
			return exec.firstOutputError("stream")
		}
		replay := make(chan StreamChunk, 1)
		replay <- first
		// A dedicated stop signal, not the request context: the forwarder
		// itself cancels that context as it exits, which would race the final
		// flush of a stream that completed normally. Only the stream's own
		// Close() stops the relay, so a consumer that keeps reading receives
		// every chunk even while the upstream tears down.
		stop := make(chan struct{})
		s.cancel = func() {
			close(stop)
			exec.cancel()
		}
		go relayStreamChunks(s.ch, replay, stop)
		s.ch = replay
		return nil
	}
	// The chunk channel closed without ever carrying output. Terminal state
	// decides between a legitimate tool-call-only turn and a dead stream.
	result := <-s.done
	if result.err == nil && len(result.toolCalls) > 0 {
		reseat := make(chan streamCompletion, 1)
		reseat <- result
		s.done = reseat
		return nil
	}
	s.absorb(result)
	// A leg abandoned for silence is not merely empty: report the guard's own
	// verdict so the chain's logs and metrics separate queueing from failures.
	if guardErr := exec.firstOutputError("stream"); guardErr != nil {
		exec.cancel()
		return guardErr
	}
	// Context failures are not transient emptiness — re-attempting them would
	// replay the same timeout against the same dead leg.
	if result.err != nil && (errors.Is(result.err, context.Canceled) || errors.Is(result.err, context.DeadlineExceeded)) {
		exec.cancel()
		return result.err
	}
	result.err = &EmptyStreamError{
		Model:        exec.call.Model,
		FinishReason: result.finishReason,
		Err:          result.err,
	}
	// One stable, greppable line per dead-on-arrival leg: the raw material for
	// measuring the empty rate per model and whether retries recover it.
	opts.Logger.Warn("empty stream",
		"model", exec.call.Model,
		"finish_reason", result.finishReason,
		"input_tokens", exec.call.InputTokens,
		"elapsed", time.Since(exec.start).Round(time.Millisecond),
		"stream_err", errString(result.err),
	)
	// Route the takeover back through Wait()'s bookkeeping: metrics log,
	// usage, and a failure hook so the ledger records the attempt.
	reseat := make(chan streamCompletion, 1)
	reseat <- result
	s.done = reseat
	_ = s.Wait()
	exec.cancel()
	return result.err
}

// relayStreamChunks forwards a source chunk channel into dst, which has been
// seeded with a chunk the gate already consumed. When the consumer stops
// reading (stop closed), the source is drained so the forwarder goroutine is
// never blocked on a send nobody will receive.
func relayStreamChunks(src <-chan StreamChunk, dst chan<- StreamChunk, stop <-chan struct{}) {
	defer close(dst)
	for chunk := range src {
		select {
		case dst <- chunk:
		case <-stop:
			for range src {
			}
			return
		}
	}
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
