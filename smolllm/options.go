package smolllm

import (
	"context"
	"log/slog"
	"math"
	"net/http"
	"os"
	"strings"
	"time"
)

// Option mutates the configuration for Ask or Stream.
type Option func(*Options)

// Options bundles optional arguments for Ask and Stream.
type Options struct {
	// SystemPrompt becomes the leading system message when populated.
	SystemPrompt string
	// Model accepts provider/model strings; comma-separate multiple entries to try them in order.
	Model string
	// Selector provides custom model selection strategy. Takes precedence over Model.
	Selector ModelSelector
	// Temperature controls sampling randomness.
	Temperature *float64
	// TopP applies nucleus sampling cutoff.
	TopP *float64
	// MaxTokens limits output length for providers that support it.
	MaxTokens *int
	// Stop sequences terminate generation for providers that support them.
	Stop []string
	// Seed asks providers that support deterministic sampling to use this seed.
	Seed *int
	// ReasoningEffort controls how much thinking a reasoning model does. Passed through to the provider as-is.
	ReasoningEffort *string
	// APIKey overrides env lookup. Comma-separated values enable automatic rotation across calls.
	APIKey string
	// BaseURL overrides the inferred endpoint for the provider.
	BaseURL string
	// ImagePaths embeds local files or data URLs into the first user message.
	ImagePaths []string
	// Timeout bounds one upstream attempt, not the whole chain: each fallback
	// leg gets its own budget, so the worst case is legs x retries x Timeout.
	// A caller that needs an overall bound must impose it on the context.
	Timeout time.Duration
	// FirstOutputTimeout bounds how long one leg may take to produce its first
	// content or reasoning delta. A leg that exceeds it is abandoned and the
	// chain advances, which is what a merely-slow provider needs: Timeout alone
	// would let such a leg consume the whole budget and never fall back. It
	// guards only the first output — once deltas flow, the bytes cannot be
	// retracted, so a mid-stream stall is bounded by Timeout instead.
	// Zero disables the check.
	FirstOutputTimeout time.Duration
	// RemoveBackticks toggles best-effort markdown fence stripping post-response.
	RemoveBackticks bool
	// StreamHandler receives streamed deltas when streaming is enabled.
	StreamHandler func(context.Context, string) error
	// HTTPClient allows injecting a custom HTTP client implementation.
	HTTPClient *http.Client
	// Logger captures structured logs. Must not be nil.
	Logger *slog.Logger
	// Hook is called after each LLM call attempt with usage and error details.
	Hook func(RequestEvent)
	// MinOutputTokens rejects responses shorter than this (estimated tokens).
	// Helps detect context window overflow where models return near-empty output.
	// Only applies when input > 1000 tokens. 0 = disabled (default).
	MinOutputTokens int
	// Dimensions truncates embedding vectors to the given length (Embed only).
	// Requires a model that supports MRL (e.g. text-embedding-3-*, qwen3-embedding).
	// 0 = use the model's native dimensionality (default).
	Dimensions int
	// ExtraBody carries raw request fields the library does not model (e.g. tools,
	// response_format). Merged into the payload last, so the caller wins.
	ExtraBody map[string]any
	// ExtraHeaders are added to every outbound provider request of every leg,
	// after the library sets Content-Type and Authorization. Only for headers
	// that are safe to show every provider in a fallback chain.
	ExtraHeaders http.Header
	// ProviderHeaders are added only on legs served by the named provider, so a
	// provider-specific credential never reaches a different provider when the
	// chain falls through.
	ProviderHeaders map[string]http.Header
	// DisableStreaming makes Ask issue a plain stream:false request.
	DisableStreaming bool
	// AccountBreaker, when set, skips fallback legs whose provider account
	// reported quota exhaustion. Nil (the default) keeps plain fallback.
	AccountBreaker *AccountBreaker
	// AccountEventHook receives AccountBreaker decisions; see WithAccountEventHook.
	AccountEventHook func(AccountEvent)
}

func defaultOptions() Options {
	return Options{
		SystemPrompt:       "",
		Model:              "",
		Selector:           nil,
		Temperature:        nil,
		TopP:               nil,
		MaxTokens:          nil,
		Stop:               nil,
		Seed:               nil,
		ReasoningEffort:    nil,
		APIKey:             "",
		BaseURL:            "",
		ImagePaths:         nil,
		Timeout:            600 * time.Second,
		FirstOutputTimeout: defaultFirstOutputTimeout,
		RemoveBackticks:    false,
		StreamHandler:      nil,
		HTTPClient:         nil,
		Logger:             newDefaultLogger(),
		Hook:               nil,
		MinOutputTokens:    0,
		Dimensions:         0,
		ExtraBody:          nil,
		AccountBreaker:     nil,
		AccountEventHook:   nil,
	}
}

// applyOptions creates default options and applies all provided option functions.
func applyOptions(opts ...Option) Options {
	options := defaultOptions()
	for _, opt := range opts {
		opt(&options)
	}
	return options
}

// WithSystemPrompt sets the system prompt.
func WithSystemPrompt(system string) Option {
	return func(o *Options) {
		o.SystemPrompt = system
	}
}

// WithModel explicitly selects a provider/model string. Provide comma-separated
// entries (e.g. "gemini/flash,openai/gpt-4o-mini") to list ordered fallbacks.
func WithModel(model string) Option {
	return func(o *Options) {
		o.Model = model
	}
}

// WithModelSet selects randomly from the provided models with equal probability.
// On failure, retries remaining models until exhausted.
func WithModelSet(models ...string) Option {
	if len(models) == 0 {
		panic("WithModelSet: at least one model required")
	}
	return func(o *Options) {
		o.Selector = NewRandomSelector(models, nil)
	}
}

// WithModelWeights selects randomly using the provided weights.
// Higher weights increase selection probability. Weights must be positive.
// On failure, retries remaining models with re-normalized weights.
func WithModelWeights(weights map[string]float64) Option {
	if len(weights) == 0 {
		panic("WithModelWeights: at least one model required")
	}
	models := make([]string, 0, len(weights))
	for m, w := range weights {
		if math.IsNaN(w) || w <= 0 {
			panic("WithModelWeights: weights must be positive and not NaN")
		}
		models = append(models, m)
	}
	return func(o *Options) {
		o.Selector = NewRandomSelector(models, weights)
	}
}

// WithTemperature sets the sampling temperature. Valid range is [0, 2].
func WithTemperature(value float64) Option {
	if math.IsNaN(value) || value < 0 || value > 2 {
		panic("WithTemperature: value must be between 0 and 2 inclusive")
	}
	return func(o *Options) {
		v := value
		o.Temperature = &v
	}
}

// WithTopP sets the nucleus sampling probability mass. Valid range is [0, 1].
func WithTopP(value float64) Option {
	if math.IsNaN(value) || value < 0 || value > 1 {
		panic("WithTopP: value must be between 0 and 1 inclusive")
	}
	return func(o *Options) {
		v := value
		o.TopP = &v
	}
}

// WithMaxTokens sets the maximum number of output tokens.
func WithMaxTokens(value int) Option {
	if value <= 0 {
		panic("WithMaxTokens: value must be positive")
	}
	return func(o *Options) {
		v := value
		o.MaxTokens = &v
	}
}

// WithStop sets one or more stop sequences.
func WithStop(stops ...string) Option {
	if len(stops) == 0 {
		panic("WithStop: at least one stop sequence required")
	}
	copied := make([]string, len(stops))
	for i, stop := range stops {
		trimmed := strings.TrimSpace(stop)
		if trimmed == "" {
			panic("WithStop: stop sequences must not be empty")
		}
		copied[i] = stop
	}
	return func(o *Options) {
		o.Stop = copied
	}
}

// WithSeed sets a deterministic sampling seed for providers that support it.
func WithSeed(value int) Option {
	return func(o *Options) {
		v := value
		o.Seed = &v
	}
}

// WithReasoningEffort sets the reasoning effort for reasoning models. Value is passed through to the provider as-is.
func WithReasoningEffort(value string) Option {
	v := strings.ToLower(strings.TrimSpace(value))
	if v == "" {
		panic("WithReasoningEffort: value must not be empty")
	}
	return func(o *Options) {
		o.ReasoningEffort = &v
	}
}

// WithAPIKey overrides the resolved API key. Multiple comma-separated keys are
// balanced automatically just like the environment variable format.
func WithAPIKey(key string) Option {
	return func(o *Options) {
		o.APIKey = key
	}
}

// WithBaseURL overrides the resolved base URL.
func WithBaseURL(url string) Option {
	return func(o *Options) {
		o.BaseURL = url
	}
}

// WithImagePaths attaches user images to the request.
func WithImagePaths(paths ...string) Option {
	copied := make([]string, len(paths))
	copy(copied, paths)
	return func(o *Options) {
		o.ImagePaths = copied
	}
}

// WithTimeout defines the hard timeout for one upstream attempt.
func WithTimeout(timeout time.Duration) Option {
	return func(o *Options) {
		o.Timeout = timeout
	}
}

// WithFirstOutputTimeout bounds how long a leg may wait for its first content
// or reasoning delta before the chain abandons it and tries the next one. A
// non-positive value disables the check. Defaults to
// defaultFirstOutputTimeout.
func WithFirstOutputTimeout(timeout time.Duration) Option {
	return func(o *Options) {
		o.FirstOutputTimeout = timeout
	}
}

// WithAccountBreaker routes the fallback chain through a shared, process-local
// account circuit breaker. Pass the same breaker to every call: its state is
// what lets later requests skip an account an earlier one found exhausted.
func WithAccountBreaker(breaker *AccountBreaker) Option {
	return func(o *Options) {
		o.AccountBreaker = breaker
	}
}

// WithBacktickRemoval strips enclosing markdown fences once complete.
func WithBacktickRemoval() Option {
	return func(o *Options) {
		o.RemoveBackticks = true
	}
}

// WithStreamHandler registers a callback for streamed deltas.
func WithStreamHandler(handler func(context.Context, string) error) Option {
	return func(o *Options) {
		o.StreamHandler = handler
	}
}

// WithHTTPClient injects a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(o *Options) {
		o.HTTPClient = client
	}
}

// WithLogger overrides the logger. Logger must not be nil.
func WithLogger(logger *slog.Logger) Option {
	if logger == nil {
		panic("WithLogger: logger must not be nil")
	}
	return func(o *Options) {
		o.Logger = logger
	}
}

// WithHook registers a callback invoked after each LLM call attempt.
func WithHook(fn func(RequestEvent)) Option {
	return func(o *Options) {
		o.Hook = fn
	}
}

// WithMinOutputTokens rejects responses shorter than minTokens (estimated).
// Useful for detecting context window overflow where a model returns near-empty output.
// Only enforced when input exceeds 1000 tokens to allow short replies on small prompts.
func WithMinOutputTokens(minTokens int) Option {
	return func(o *Options) {
		o.MinOutputTokens = minTokens
	}
}

// reservedExtraBodyKeys are read back by the library machinery: the stream parser,
// usage collection and routing all depend on them, so a caller override would
// silently break them.
var reservedExtraBodyKeys = []string{"stream", "stream_options", "messages", "model"}

// reservedHeaders are set by the library itself on every provider request.
//
// Authorization is deliberately not overridable: the value is chosen per
// attempt by the key balancer and per leg by the fallback chain, so a header
// carried on Options would ride EVERY leg and send one provider's credential
// to another. A provider needing a differently-named auth header (x-api-key,
// api-key) adds it with WithHeader alongside Authorization; one needing a
// different Authorization scheme needs real per-provider auth support, not an
// override that silently defeats key rotation and makes APIKeyHint a lie.
//
// Content-Type is not overridable because the library encodes the body and it
// is always JSON; overriding could only misdescribe the wire.
var reservedHeaders = []string{"Authorization", "Content-Type"}

func isReservedHeader(canonicalName string) bool {
	for _, reserved := range reservedHeaders {
		if canonicalName == reserved {
			return true
		}
	}
	return false
}

// WithoutStreaming makes Ask perform a plain (stream:false) request instead of
// streaming and reassembling.
//
// Streaming remains the default: existing deployments depend on its timeout
// behaviour through proxies that idle out a long blocking response, and both
// WithStreamHandler and a meaningful TTFT only exist under streaming.
//
// Use it when a provider only reports — or only applies — prompt caching on
// non-streaming requests. It is incompatible with WithStreamHandler and with
// Stream(); both report an error rather than silently degrading.
func WithoutStreaming() Option {
	return func(o *Options) { o.DisableStreaming = true }
}

// WithProviderHeader adds one header to requests served by the named provider
// only — "openrouter" in "openrouter/z-ai/glm-5.3". This is the correct home for
// anything provider-specific, a credential above all: a fallback chain reaches a
// different provider on each attempt, and a scoped header never travels to the
// wrong one.
//
// An empty provider name targets bare models (a model string with no "/"), which
// resolve to the empty provider identity and would otherwise have no way to send
// an x-api-key style header at all.
func WithProviderHeader(providerName, key, value string) Option {
	provider := strings.TrimSpace(providerName)
	canonical := http.CanonicalHeaderKey(strings.TrimSpace(key))
	if canonical == "" {
		panic("WithProviderHeader: header name must not be empty")
	}
	if isReservedHeader(canonical) {
		panic("WithProviderHeader: may not set " + canonical)
	}
	return func(o *Options) {
		if o.ProviderHeaders == nil {
			o.ProviderHeaders = map[string]http.Header{}
		}
		if o.ProviderHeaders[provider] == nil {
			o.ProviderHeaders[provider] = http.Header{}
		}
		o.ProviderHeaders[provider].Add(canonical, value)
	}
}

// WithHeaderForAllProviders adds one header to EVERY leg of a fallback chain.
//
// The name is deliberately unwieldy: this is the dangerous form, and the danger
// is not detectable from the header's name. An earlier version tried to refuse
// "credential-looking" names by substring, which was both porous (an
// authenticating X-Signature sailed through) and overbroad (it rejected
// Idempotency-Key and X-Token-Count, which are perfectly safe to send
// everywhere). Guessing intent from a name cannot work; making the blast radius
// visible at the call site can.
//
// Use it for headers every provider may see — attribution and tracing, such as
// X-Title or HTTP-Referer. For anything that must not cross a provider boundary,
// use WithProviderHeader.
func WithHeaderForAllProviders(key, value string) Option {
	canonical := http.CanonicalHeaderKey(strings.TrimSpace(key))
	if canonical == "" {
		panic("WithHeaderForAllProviders: header name must not be empty")
	}
	if isReservedHeader(canonical) {
		panic("WithHeaderForAllProviders: may not set " + canonical)
	}
	return func(o *Options) {
		if o.ExtraHeaders == nil {
			o.ExtraHeaders = http.Header{}
		}
		o.ExtraHeaders.Add(canonical, value)
	}
}

// WithExtraBody sets raw request fields the library does not model, merged into
// the payload last so they win over library defaults. Panics when the caller sets
// a field the library machinery reads back.
func WithExtraBody(fields map[string]any) Option {
	var reserved []string
	for _, key := range reservedExtraBodyKeys {
		if _, ok := fields[key]; ok {
			reserved = append(reserved, key)
		}
	}
	if len(reserved) > 0 {
		panic("WithExtraBody: may not set " + strings.Join(reserved, ", "))
	}
	// Copy so later caller mutations cannot reach an in-flight request.
	copied := make(map[string]any, len(fields))
	for key, value := range fields {
		copied[key] = value
	}
	return func(o *Options) {
		o.ExtraBody = copied
	}
}

// WithDimensions truncates embedding vectors to the given length. Only applies
// to Embed calls and requires a model that supports MRL (Matryoshka Representation
// Learning), e.g. OpenAI text-embedding-3-* or qwen3-embedding.
func WithDimensions(dimensions int) Option {
	if dimensions <= 0 {
		panic("WithDimensions: dimensions must be positive")
	}
	return func(o *Options) {
		o.Dimensions = dimensions
	}
}

func newDefaultLogger() *slog.Logger {
	level := slog.LevelInfo
	switch strings.ToUpper(os.Getenv("LOG_LEVEL")) {
	case "DEBUG":
		level = slog.LevelDebug
	case "WARN", "WARNING":
		level = slog.LevelWarn
	case "ERROR":
		level = slog.LevelError
	}

	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		AddSource: false,
		Level:     level,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.TimeKey {
				return slog.Attr{
					Key:   attr.Key,
					Value: slog.StringValue(attr.Value.Time().UTC().Format(time.RFC3339)),
				}
			}
			return attr
		},
	})

	return slog.New(handler)
}
