package smolllm

import (
	"context"
	"errors"
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

	// A leg counts as a success once the gate has seen its first output, so
	// the breaker settles there; a stream that dies afterwards cannot be
	// retracted and is the caller's to handle through Wait().
	return runChain(ctx, options, func(model string) (*StreamResponse, error) {
		return streamOnce(ctx, prompt, options, model)
	})
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
		// The guard firing before headers arrive is still an attempt: report
		// its verdict exactly as Ask does, or an upstream that accepts the
		// connection and then says nothing vanishes from the ledger whenever
		// the caller streams.
		if guardErr := exec.firstOutputError("headers"); guardErr != nil {
			exec.cancel()
			return fail(guardErr)
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
			return reportAbandonedLeg(opts.Hook, exec, "stream")
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
		return reportAbandonedLeg(opts.Hook, exec, "stream")
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

// reportAbandonedLeg records a stream leg the first-output guard wrote off, and
// returns the guard's verdict.
//
// The gate runs after the stream is established, so nothing downstream will
// emit a hook for it: without this the attempt leaves no trace in the ledger
// and a streamed first-output timeout reads as a request that never happened —
// the one outcome that separates "queued upstream" from "model is broken".
// Scoped to the guard's own verdict on purpose. An empty stream also fails the
// gate, but it already reports itself, and hooking it here too would double
// count every dead-on-arrival leg.
func reportAbandonedLeg(hook func(RequestEvent), exec *callExecution, phase string) error {
	guardErr := exec.firstOutputError(phase)
	if guardErr == nil {
		return nil
	}
	emitFailureHook(hook, exec.call, guardErr, exec.start, nil)
	return guardErr
}
