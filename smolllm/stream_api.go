package smolllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
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
	}
	return sr, nil
}
