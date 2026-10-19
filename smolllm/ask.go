package smolllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Ask performs a single prompt/response exchange, streaming under the hood before returning the full text.
func Ask(ctx context.Context, prompt Prompt, opts ...Option) (*Response, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}

	options := applyOptions(opts...)
	// Calling the handler once with the finished text would counterfeit the
	// handler's contract, which is that it receives streamed deltas.
	if options.DisableStreaming && options.StreamHandler != nil {
		return nil, errors.New("WithoutStreaming is incompatible with WithStreamHandler")
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
		resp, err := withRetry(ctx, options.Logger, model, func() (*Response, error) {
			return askOnce(ctx, prompt, options, model)
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

func askOnce(ctx context.Context, prompt Prompt, opts Options, model string) (*Response, error) {
	exec, err := newCallExecution(ctx, prompt, opts, model)
	if err != nil {
		return nil, err
	}
	defer exec.cancel()
	fail := func(err error, reported *reportedUsage) (*Response, error) {
		emitFailureHook(opts.Hook, exec.call, err, exec.start, reported)
		return nil, err
	}

	resp, err := exec.do("sending request")
	if err != nil {
		if guardErr := exec.firstOutputError("headers"); guardErr != nil {
			return fail(guardErr, nil)
		}
		return fail(err, nil)
	}
	defer func(resp *http.Response) { _ = resp.Body.Close() }(resp)

	if resp.StatusCode >= http.StatusBadRequest {
		retryResp, retried, retryErr := exec.retryWithoutStreamUsage(resp)
		if retryErr != nil {
			return fail(retryErr, nil)
		}
		if retried {
			resp = retryResp
			defer func(resp *http.Response) { _ = resp.Body.Close() }(resp)
		}
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return fail(httpError(resp), nil)
	}

	var cr consumeResult
	if opts.DisableStreaming {
		// A non-streaming response carries no deltas: the complete body is the
		// first output, so the guard has nothing left to bound once headers have
		// arrived and must be disarmed or it would truncate a real answer.
		exec.disarmFirstOutput()
		cr, err = consumeBlocking(resp.Body, exec.start)
	} else {
		cr, err = consumeStream(exec.requestContext(), opts.Logger, resp.Body, opts.StreamHandler, exec.start,
			func() error {
				if exec.disarmFirstOutput() {
					return nil
				}
				return exec.firstOutputError("stream")
			})
	}
	if err != nil {
		// The guard can also fire mid-body on a non-streaming call, in which case
		// the transport reports a bare cancellation; attribute it before the
		// generic handling below relabels an abandoned leg as a plain failure.
		if guardErr := exec.firstOutputError("stream"); guardErr != nil {
			return fail(guardErr, &cr.usage)
		}
		var empty *EmptyStreamError
		if errors.As(err, &empty) && empty.Model == "" {
			empty.Model = exec.call.Model
		}
		return fail(err, &cr.usage)
	}

	content := cr.content
	if opts.RemoveBackticks {
		content = stripBackticks(content)
	}
	if strings.TrimSpace(content) == "" && strings.TrimSpace(cr.reasoning) == "" && len(cr.toolCalls) == 0 {
		// Typed so withRetry treats it as transient: the provider emitted
		// nothing usable, and this failure mode (see EmptyStreamError) comes and
		// goes between identical requests. Same policy as the Stream gate.
		return fail(&EmptyStreamError{
			Model:        exec.call.Model,
			FinishReason: cr.finishReason,
		}, &cr.usage)
	}

	// Reasoning spent the whole output budget before any answer was emitted.
	// An empty completion is useless to every caller, so fail the leg and let
	// the chain route around it (same policy as 429). Truncated-but-non-empty
	// content still passes through: partial output may be usable and a retry
	// elsewhere would double-spend tokens.
	if cr.finishReason == "length" && strings.TrimSpace(content) == "" && len(cr.toolCalls) == 0 {
		return fail(fmt.Errorf("model %q truncated before any content (finish_reason=length, reasoning consumed the output budget)", exec.call.Model), &cr.usage)
	}

	// Tool calls cut off mid-argument are unusable to every caller: the argument
	// JSON no longer parses. Fail the leg rather than hand back a broken call.
	if cr.finishReason == "length" && len(cr.toolCalls) > 0 {
		return fail(
			fmt.Errorf("model %q truncated mid tool call (finish_reason=length)", exec.call.Model),
			&cr.usage,
		)
	}

	// Treat suspiciously short output as empty — likely context window overflow.
	// Only enabled when MinOutputTokens is set by the caller.
	outTok := estimateTokens(content)
	if cr.usage.reported {
		outTok = cr.usage.outputTokens
	}
	// A tool-call-only turn is legitimately tiny — the guard would reject every one.
	if opts.MinOutputTokens > 0 && len(cr.toolCalls) == 0 &&
		outTok < opts.MinOutputTokens && exec.call.InputTokens > 1000 {
		return fail(
			fmt.Errorf("model %q returned suspiciously short response (%d output tokens for %d input tokens, min=%d)",
				exec.call.Model, outTok, exec.call.InputTokens, opts.MinOutputTokens),
			&cr.usage,
		)
	}

	total := time.Since(exec.start)
	if opts.DisableStreaming {
		// On a blocking transport the first usable token arrives with the last,
		// so TTFT is the whole duration by definition. Pin them to the same
		// measurement rather than two time.Since calls a microsecond apart.
		cr.ttft = total
	}
	outputTokens := estimateTokens(content + cr.reasoning)
	inputTokens := exec.call.InputTokens
	estimated := true
	if cr.usage.reported {
		inputTokens = cr.usage.inputTokens
		outputTokens = cr.usage.outputTokens
		estimated = false
	}
	opts.Logger.Info(
		formatMetrics(exec.call.ModelName, inputTokens, outputTokens, total, cr.ttft),
		"model", exec.call.ModelName,
	)

	usage := Usage{
		RawUsage:     cr.usage.rawUsage,
		Provider:     exec.call.Provider.Name,
		Model:        exec.call.Model,
		ModelName:    exec.call.ModelName,
		APIKeyHint:   previewAPIKey(exec.call.APIKey),
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		// Cache counts are provider-reported only; an estimated usage has none.
		CacheReadTokens:    cr.usage.cacheReadTokens,
		CacheWriteTokens:   cr.usage.cacheWriteTokens,
		CacheReadReported:  cr.usage.cacheReadReported,
		CacheWriteReported: cr.usage.cacheWriteReported,
		Duration:           total,
		TTFT:               cr.ttft,
		Estimated:          estimated,
	}
	if opts.Hook != nil {
		opts.Hook(RequestEvent{Usage: usage, Error: nil, Timestamp: time.Now().UTC()})
	}

	return &Response{
		Text:         content,
		Reasoning:    cr.reasoning,
		FinishReason: cr.finishReason,
		Model:        exec.call.Model,
		ModelName:    exec.call.ModelName,
		Provider:     exec.call.Provider.Name,
		Usage:        usage,
		ToolCalls:    cr.toolCalls,
	}, nil
}
