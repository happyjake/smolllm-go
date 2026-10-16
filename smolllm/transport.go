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

	reqCtx, cancel := deriveContext(ctx, opts.Timeout)

	req, err := newProviderRequest(reqCtx, call.URL, call.Body, call.APIKey, call.ExtraHeaders)
	if err != nil {
		cancel()
		return nil, err
	}

	return &callExecution{
		call:   call,
		client: client,
		req:    req,
		cancel: cancel,
		start:  time.Time{},
		logger: opts.Logger,
		hook:   opts.Hook,
	}, nil
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
			TTFT:               0,
			Estimated:          estimated,
		},
		Error:     err,
		Timestamp: time.Now().UTC(),
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
