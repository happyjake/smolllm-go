package smolllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"
)

type preparedCall struct {
	URL         string
	Body        []byte
	Provider    provider
	Model       string
	ModelName   string
	APIKey      string
	InputTokens int
	// Streaming records the transport mode chosen for this call. It is stamped
	// once here, from the same expression that sets the request's stream flag,
	// so the event that reports it can never drift from what was actually sent.
	Streaming bool
	// ExtraHeaders is resolved per leg: a fallback chain reaches a different
	// provider on each attempt, so headers must not be captured once globally.
	ExtraHeaders http.Header
}

func prepareLLMCall(prompt Prompt, opts Options, model string) (*preparedCall, error) {
	modelSpec, effortOverride := parseModelSpec(model)
	prov, modelName, err := parseModelString(modelSpec)
	if err != nil {
		return nil, err
	}

	base, err := resolveBaseURL(prov, modelName, opts.BaseURL)
	if err != nil {
		return nil, err
	}

	apiKey, err := resolveAPIKey(prov, modelName, opts.APIKey)
	if err != nil {
		return nil, err
	}

	chosenKey, chosenURL, err := balancer.choosePair(apiKey, base)
	if err != nil {
		return nil, err
	}

	reasoningEffort := opts.ReasoningEffort
	if effortOverride != nil {
		reasoningEffort = effortOverride
	}

	url, body, inputTokens, err := buildRequestPayload(
		prompt,
		opts.SystemPrompt,
		modelName,
		prov.Name,
		chosenURL,
		opts.ImagePaths,
		chatPayloadOptions{
			Temperature:        opts.Temperature,
			TopP:               opts.TopP,
			ReasoningEffort:    reasoningEffort,
			MaxTokens:          opts.MaxTokens,
			Stop:               opts.Stop,
			Seed:               opts.Seed,
			Stream:             !opts.DisableStreaming,
			IncludeStreamUsage: !opts.DisableStreaming,
			ExtraBody:          opts.ExtraBody,
		},
	)
	if err != nil {
		return nil, err
	}

	return &preparedCall{
		URL:          url,
		Body:         body,
		Provider:     prov,
		Model:        modelSpec,
		ModelName:    modelName,
		APIKey:       chosenKey,
		InputTokens:  inputTokens,
		Streaming:    !opts.DisableStreaming,
		ExtraHeaders: resolveExtraHeaders(prov, opts.ExtraHeaders, opts.ProviderHeaders, opts.Logger),
	}, nil
}

func resolveBaseURL(prov provider, modelName, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	// Bare model (no provider): explicit option only — never derive env keys
	// from the empty provider name.
	if prov.Name == "" {
		return "", fmt.Errorf(
			"bare model %q requires a base URL. Provide WithBaseURL or use provider/model format", modelName,
		)
	}

	envKey := providerEnvKey(prov.Name, "BASE_URL")
	value := os.Getenv(envKey)
	if strings.TrimSpace(value) != "" {
		return value, nil
	}

	if strings.TrimSpace(prov.BaseURL) == "" {
		return "", fmt.Errorf("base URL not found. set %s or provide WithBaseURL", envKey)
	}
	return prov.BaseURL, nil
}

func resolveAPIKey(prov provider, modelName, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	// Bare model (no provider): explicit option only — never derive env keys
	// from the empty provider name. The ollama literal-key fallback below does
	// not apply either.
	if prov.Name == "" {
		return "", fmt.Errorf(
			"bare model %q requires an API key. Provide WithAPIKey or use provider/model format", modelName,
		)
	}

	envKey := providerEnvKey(prov.Name, "API_KEY")
	value := os.Getenv(envKey)
	if strings.TrimSpace(value) != "" {
		return value, nil
	}

	if prov.Name == providerOllama {
		return providerOllama, nil
	}
	return "", fmt.Errorf("API key not found. set %s or provide WithAPIKey", envKey)
}

// newProviderRequest is the ONLY place an outbound provider request is built.
// It owns Content-Type, Authorization, and the caller's extra headers.
//
// Every request path must go through it — the chat call, the
// stream_options-rejection retry, and embeddings. Do not call
// http.NewRequestWithContext for a provider call anywhere else: when headers
// lived at each call site, the embeddings path silently never got them, and a
// naive fix that patches the chat path alone still loses them on the retry,
// which builds a brand-new request.
func newProviderRequest(
	ctx context.Context, url string, body []byte, apiKey string, extra http.Header,
) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	// Reserved names are revalidated here, not merely at option-construction
	// time: Options is exported, so a custom Option can write ExtraHeaders
	// directly and never pass through WithHeader's guard. Add would append a
	// second Authorization rather than replace it, which is worse than either
	// value alone.
	for key, values := range extra {
		if isReservedHeader(http.CanonicalHeaderKey(key)) {
			continue
		}
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	return req, nil
}

// resolveExtraHeaders merges the per-provider env convention with the caller's
// explicit headers, lowest precedence first, matching how base URL and API key
// already resolve (explicit > env > builtin).
//
// ${PROVIDER}_EXTRA_HEADERS holds newline-separated "Name: Value" lines. Header
// values legally contain commas and semicolons, so a newline is the only
// delimiter that cannot collide with a real value.
//
// A reserved header in the env value is skipped with a warning rather than a
// panic: crashing a deployment over its own configuration is hostile, whereas
// the option form is programmer error and does panic.
func resolveExtraHeaders(
	prov provider, explicit http.Header, perProvider map[string]http.Header, logger *slog.Logger,
) http.Header {
	merged := http.Header{}
	if prov.Name != "" {
		for _, line := range strings.Split(os.Getenv(providerEnvKey(prov.Name, "EXTRA_HEADERS")), "\n") {
			name, value, ok := strings.Cut(line, ":")
			name = http.CanonicalHeaderKey(strings.TrimSpace(name))
			if !ok || name == "" {
				continue
			}
			if isReservedHeader(name) {
				if logger != nil {
					logger.Warn("ignoring reserved header from env",
						"env", providerEnvKey(prov.Name, "EXTRA_HEADERS"), "header", name)
				}
				continue
			}
			merged.Add(name, strings.TrimSpace(value))
		}
	}
	for name, values := range explicit {
		merged.Del(name) // explicit wins outright rather than appending to env
		for _, value := range values {
			merged.Add(name, value)
		}
	}
	// Provider-scoped headers apply only on this leg, and outrank the global
	// ones, so a chain never carries one provider's header to another.
	for name, values := range perProvider[prov.Name] {
		merged.Del(name)
		for _, value := range values {
			merged.Add(name, value)
		}
	}
	if len(merged) == 0 {
		return nil
	}
	return merged
}

func providerEnvKey(providerName, suffix string) string {
	base := strings.ToUpper(strings.ReplaceAll(providerName, "-", "_"))
	return base + "_" + suffix
}

type callExecution struct {
	call   *preparedCall
	client *http.Client
	req    *http.Request
	cancel context.CancelFunc
	start  time.Time
	logger *slog.Logger
	hook   func(RequestEvent)

	// attemptCtx is the per-attempt context, cancelled by cancel.
	attemptCtx context.Context
	// firstOutputTimer fires when no delta has arrived in time. Nil when the
	// check is disabled.
	firstOutputTimer  *time.Timer
	firstOutputBudget time.Duration
	startedAt         time.Time
}

func newCallExecution(ctx context.Context, prompt Prompt, opts Options, model string) (*callExecution, error) {
	call, err := prepareLLMCall(prompt, opts, model)
	if err != nil {
		return nil, err
	}

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	attemptCtx, cancel := deriveContext(ctx, opts.Timeout)
	// A cancel-with-cause lets the first-output guard stop the wait without
	// disturbing an attempt that is already streaming, which a plain deadline
	// context could not do.
	reqCtx, cancelCause := context.WithCancelCause(attemptCtx)

	req, err := newProviderRequest(reqCtx, call.URL, call.Body, call.APIKey, call.ExtraHeaders)
	if err != nil {
		cancelCause(nil)
		cancel()
		return nil, err
	}

	exec := &callExecution{
		call:       call,
		client:     client,
		req:        req,
		cancel:     func() { cancelCause(nil); cancel() },
		start:      time.Time{},
		logger:     opts.Logger,
		hook:       opts.Hook,
		attemptCtx: attemptCtx,
	}

	if budget := opts.FirstOutputTimeout; budget > 0 {
		exec.firstOutputBudget = budget
		exec.startedAt = time.Now()
		exec.firstOutputTimer = time.AfterFunc(budget, func() {
			cancelCause(ErrFirstOutputTimeout)
		})
	}

	return exec, nil
}

// disarmFirstOutput stops the first-output guard once a leg has proven it will
// deliver. Reports false when the timer already fired, in which case the leg
// must be treated as abandoned and nothing may be forwarded.
func (c *callExecution) disarmFirstOutput() bool {
	if c.firstOutputTimer == nil {
		return true
	}
	return c.firstOutputTimer.Stop()
}

// firstOutputError reports the guard's verdict for err, or nil when the guard
// did not fire. Callers check the attempt context's cause rather than the
// error itself: the HTTP client reports its own cancellation as
// context.Canceled, which is indistinguishable from a caller hang-up.
func (c *callExecution) firstOutputError(phase string) error {
	if !errors.Is(context.Cause(c.req.Context()), ErrFirstOutputTimeout) {
		return nil
	}
	return &FirstOutputTimeoutError{
		Model:   c.call.Model,
		Budget:  c.firstOutputBudget,
		Elapsed: time.Since(c.startedAt),
		Phase:   phase,
	}
}

func (c *callExecution) do(event string) (*http.Response, error) {
	c.logger.Info(
		event,
		"url", c.call.URL,
		"model", c.call.ModelName,
		"api_key", previewAPIKey(c.call.APIKey),
		"approx_tokens", c.call.InputTokens,
	)

	c.start = time.Now().UTC()
	resp, err := c.client.Do(c.req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (c *callExecution) retryWithoutStreamUsage(resp *http.Response) (*http.Response, bool, error) {
	if resp.StatusCode != http.StatusBadRequest {
		return resp, false, nil
	}

	body, ok, err := requestBodyWithoutStreamUsage(c.call.Body)
	if err != nil {
		return resp, false, err
	}
	if !ok {
		return resp, false, nil
	}

	originalBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp, false, fmt.Errorf("read original stream_options rejection body: %w", err)
	}
	_ = resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(originalBody))

	req, err := newProviderRequest(c.req.Context(), c.call.URL, body, c.call.APIKey, c.call.ExtraHeaders)
	if err != nil {
		return nil, false, err
	}

	// The rejected HTTP request is a distinct attempt, even if no usage was
	// reported. Callers must not mistake the retry's credit for a complete total.
	emitFailureHook(c.hook, c.call, &HTTPError{StatusCode: resp.StatusCode, Body: string(originalBody)}, c.start, nil)
	c.call.Body = body
	c.call.InputTokens = estimateTokens(string(body))
	c.req = req
	retryResp, err := c.do("retrying request without stream usage")
	if err != nil {
		return nil, false, fmt.Errorf(
			"retry without stream_options failed after original HTTP 400 response %q: %w",
			string(originalBody),
			err,
		)
	}
	return retryResp, true, nil
}

func emitFailureHook(hook func(RequestEvent), call *preparedCall, err error, start time.Time, reported *reportedUsage) {
	if hook == nil || call == nil || err == nil {
		return
	}
	duration := time.Duration(0)
	if !start.IsZero() {
		duration = time.Since(start)
	}
	// Some providers report actual usage alongside an HTTP error. Preserve it
	// just like usage arriving before a streaming failure, without interpreting it.
	if reported == nil {
		var httpErr *HTTPError
		if errors.As(err, &httpErr) {
			cr, _ := consumeBlocking(strings.NewReader(httpErr.Body), start)
			if cr.usage.reported {
				reported = &cr.usage
			}
		}
	}
	inputTokens := call.InputTokens
	outputTokens := 0
	cacheReadTokens := 0
	cacheWriteTokens := 0
	cacheReadReported := false
	cacheWriteReported := false
	estimated := true
	if reported != nil && reported.reported {
		inputTokens = reported.inputTokens
		outputTokens = reported.outputTokens
		cacheReadTokens = reported.cacheReadTokens
		cacheWriteTokens = reported.cacheWriteTokens
		cacheReadReported = reported.cacheReadReported
		cacheWriteReported = reported.cacheWriteReported
		estimated = false
	}
	hook(RequestEvent{
		Usage: Usage{
			RawUsage:           failureRawUsage(reported),
			Provider:           call.Provider.Name,
			Model:              call.Model,
			ModelName:          call.ModelName,
			APIKeyHint:         previewAPIKey(call.APIKey),
			InputTokens:        inputTokens,
			OutputTokens:       outputTokens,
			CacheReadTokens:    cacheReadTokens,
			CacheWriteTokens:   cacheWriteTokens,
			CacheReadReported:  cacheReadReported,
			CacheWriteReported: cacheWriteReported,
			Duration:           duration,
			// A first-output timeout IS a TTFT measurement: the attempt waited
			// this long and got nothing. Reporting 0 would make the one number
			// operators want look like "answered instantly".
			TTFT:      firstOutputWait(err),
			Estimated: estimated,
		},
		Error:              err,
		Timestamp:          time.Now().UTC(),
		FirstOutputTimeout: isFirstOutputTimeout(err),
		FirstOutputPhase:   firstOutputPhase(err),
		Streaming:          call.Streaming,
	})
}

func failureRawUsage(reported *reportedUsage) json.RawMessage {
	if reported == nil {
		return nil
	}
	return reported.rawUsage
}

func requestBodyWithoutStreamUsage(body []byte) ([]byte, bool, error) {
	// UseNumber for the same reason mergeExtraBody needs it: this rebuilds the
	// whole payload, so without it the retry silently rounds any integer above
	// 2^53 that survived the first request.
	var payload map[string]any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&payload); err != nil {
		return nil, false, fmt.Errorf("decode request body for stream_options retry: %w", err)
	}
	if _, ok := payload["stream_options"]; !ok {
		return body, false, nil
	}
	delete(payload, "stream_options")
	next, err := json.Marshal(payload)
	if err != nil {
		return nil, false, fmt.Errorf("encode request body for stream_options retry: %w", err)
	}
	return next, true, nil
}

func (c *callExecution) requestContext() context.Context {
	return c.req.Context()
}

func deriveContext(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func previewAPIKey(key string) string {
	if len(key) <= 9 {
		return key
	}
	return key[:5] + "..." + key[len(key)-4:]
}

// HTTPError represents an HTTP error response from an LLM provider.
type HTTPError struct {
	StatusCode int
	Body       string
}

// EmptyStreamError marks a leg whose stream ended before delivering any
// usable assistant output — no content, no reasoning, no tool calls. The
// doom/pi-proxy upstream does this on a fraction of gpt-6.1-sol calls: HTTP
// 200, a single synthesized terminal chunk carrying finish_reason=error with
// all-zero usage, then [DONE]. Nothing reached the caller and nothing was
// billed, so the same leg is worth re-attempting before the chain moves on.
// ErrFirstOutputTimeout reports that a leg produced no content or reasoning
// delta within Options.FirstOutputTimeout, so the chain abandoned it. It is
// deliberately not retryable: the leg is merely queueing upstream, and a
// same-leg retry would spend the budget again to learn the same thing.
var ErrFirstOutputTimeout = errors.New("smolllm: no output before first-output timeout")

// FirstOutputTimeoutError carries which leg was abandoned and how long it had
// been waiting, so logs and metrics can separate queueing from real failures.
type FirstOutputTimeoutError struct {
	Model   string
	Budget  time.Duration
	Elapsed time.Duration
	// Phase records where the wait happened: "headers" when the upstream had not
	// returned response headers yet, "stream" when headers arrived but no delta
	// followed. A proxy that holds the connection open while queueing shows up
	// as headers phase in the incident that motivated this check.
	//
	// There are exactly two phases, and both streaming and non-streaming calls
	// use them: a non-streaming body that never completes is the same silence,
	// so reporting it under a third name ("body") would split one phenomenon
	// across two metric labels and hide the trend.
	Phase string
}

func (e *FirstOutputTimeoutError) Error() string {
	return fmt.Sprintf("no output from model %q within %s (waited %s, phase=%s)",
		e.Model, e.Budget, e.Elapsed.Round(time.Millisecond), e.Phase)
}

// Is makes errors.Is(err, ErrFirstOutputTimeout) work through wrap chains.
func (e *FirstOutputTimeoutError) Is(target error) bool { return target == ErrFirstOutputTimeout }

type EmptyStreamError struct {
	Model        string
	FinishReason string
	Err          error // underlying stream error when the stream did not end cleanly
}

func (e *EmptyStreamError) Error() string {
	detail := ""
	if e.FinishReason != "" {
		detail = fmt.Sprintf(" (upstream finish_reason=%q)", e.FinishReason)
	}
	if e.Err != nil {
		return fmt.Sprintf("empty response: stream from model %q ended before any output%s: %v", e.Model, detail, e.Err)
	}
	return fmt.Sprintf("empty response: stream from model %q ended before any output%s", e.Model, detail)
}

func (e *EmptyStreamError) Unwrap() error { return e.Err }

func (e *HTTPError) Error() string {
	return fmt.Sprintf("http error %d: %s", e.StatusCode, e.Body)
}

func httpError(resp *http.Response) error {
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return &HTTPError{StatusCode: resp.StatusCode, Body: fmt.Sprintf("read body: %v", readErr)}
	}
	message := strings.TrimSpace(string(body))
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	return &HTTPError{StatusCode: resp.StatusCode, Body: message}
}

// isFirstOutputTimeout reports whether err means "this leg produced nothing in
// time" rather than "this leg is broken". The distinction matters downstream:
// a queued leg must not be counted against the model's health.
func isFirstOutputTimeout(err error) bool {
	if err == nil {
		return false
	}
	var target *FirstOutputTimeoutError
	return errors.As(err, &target)
}

func firstOutputPhase(err error) string {
	phase, ok := IsFirstOutputTimeout(err)
	if !ok {
		return ""
	}
	return phase
}

// firstOutputWait exposes how long a timed-out attempt waited, so the hook can
// record it as TTFT instead of the zero that means "answered instantly".
func firstOutputWait(err error) time.Duration {
	var target *FirstOutputTimeoutError
	if errors.As(err, &target) {
		return target.Elapsed
	}
	return 0
}
