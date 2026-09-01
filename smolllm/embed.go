package smolllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"time"
)

// EmbeddingResponse holds the result of an Embed call.
type EmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
	Model      string      `json:"model"`
	ModelName  string      `json:"model_name"`
	Provider   string      `json:"provider"`
	Usage      Usage       `json:"usage"`
}

type embeddingRequest struct {
	Model           string  `json:"model"`
	Input           any     `json:"input"`
	Dimensions      int     `json:"dimensions,omitempty"`
	ReasoningEffort *string `json:"reasoning_effort,omitempty"`
}

type embeddingDataItem struct {
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type embeddingAPIResponse struct {
	Data  []embeddingDataItem `json:"data"`
	Model string              `json:"model"`
	Usage *struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// Embed generates embeddings for the given input strings using an OpenAI-compatible endpoint.
func Embed(ctx context.Context, input []string, opts ...Option) (*EmbeddingResponse, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}
	if len(input) == 0 {
		return nil, errors.New("input must not be empty")
	}

	options := applyOptions(opts...)

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
		resp, err := withRetry(ctx, options.Logger, model, func() (*EmbeddingResponse, error) {
			return embedOnce(ctx, input, options, model)
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

func embedOnce(ctx context.Context, input []string, opts Options, model string) (*EmbeddingResponse, error) {
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

	url := resolveEndpointURL(chosenURL, prov.Name, "embeddings")

	// Single string input for len==1, slice otherwise.
	var inputPayload any
	if len(input) == 1 {
		inputPayload = input[0]
	} else {
		inputPayload = input
	}

	reasoningEffort := opts.ReasoningEffort
	if effortOverride != nil {
		reasoningEffort = effortOverride
	}
	normalizedReasoningEffort, err := normalizeReasoningEffort(reasoningEffort, prov.Name)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(embeddingRequest{
		Model:           modelName,
		Input:           inputPayload,
		Dimensions:      opts.Dimensions,
		ReasoningEffort: normalizedReasoningEffort,
	})
	if err != nil {
		return nil, fmt.Errorf("encode embedding request: %w", err)
	}
	inputTokens := estimateTokens(string(body))
	fail := func(err error, start time.Time) (*EmbeddingResponse, error) {
		if opts.Hook != nil {
			duration := time.Duration(0)
			if !start.IsZero() {
				duration = time.Since(start)
			}
			opts.Hook(RequestEvent{
				Usage: Usage{
					Provider:     prov.Name,
					Model:        modelSpec,
					ModelName:    modelName,
					APIKeyHint:   previewAPIKey(chosenKey),
					InputTokens:  inputTokens,
					OutputTokens: 0,
					Duration:     duration,
					TTFT:         0,
					Estimated:    true,
				},
				Error:     err,
				Timestamp: time.Now().UTC(),
			})
		}
		return nil, err
	}

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	reqCtx, cancel := deriveContext(ctx, opts.Timeout)
	defer cancel()

	req, err := newProviderRequest(reqCtx, url, body, chosenKey,
		resolveExtraHeaders(prov, opts.ExtraHeaders, opts.ProviderHeaders, opts.Logger))
	if err != nil {
		return fail(err, time.Time{})
	}

	opts.Logger.Info("sending embedding request",
		"url", url,
		"model", modelName,
		"api_key", previewAPIKey(chosenKey),
		"inputs", len(input),
	)

	start := time.Now().UTC()
	resp, err := client.Do(req)
	if err != nil {
		return fail(err, start)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= http.StatusBadRequest {
		return fail(httpError(resp), start)
	}

	var apiResp embeddingAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return fail(fmt.Errorf("decode embedding response: %w", err), start)
	}

	if len(apiResp.Data) != len(input) {
		return fail(fmt.Errorf("embedding response returned %d vectors for %d inputs", len(apiResp.Data), len(input)), start)
	}

	// Sort by index to guarantee input-order alignment.
	sort.Slice(apiResp.Data, func(i, j int) bool {
		return apiResp.Data[i].Index < apiResp.Data[j].Index
	})

	embeddings := make([][]float64, len(apiResp.Data))
	for i, item := range apiResp.Data {
		embeddings[i] = item.Embedding
	}

	total := time.Since(start)
	promptTokens := 0
	estimated := true
	if apiResp.Usage != nil {
		promptTokens = apiResp.Usage.PromptTokens
		estimated = false
	}
	opts.Logger.Info(
		formatMetrics(modelName, promptTokens, 0, total, 0),
		"model", modelName,
	)

	usage := Usage{
		Provider:     prov.Name,
		Model:        modelSpec,
		ModelName:    modelName,
		APIKeyHint:   previewAPIKey(chosenKey),
		InputTokens:  promptTokens,
		OutputTokens: 0,
		Duration:     total,
		TTFT:         0,
		Estimated:    estimated,
	}
	if opts.Hook != nil {
		opts.Hook(RequestEvent{Usage: usage, Error: nil, Timestamp: time.Now().UTC()})
	}

	return &EmbeddingResponse{
		Embeddings: embeddings,
		Model:      modelSpec,
		ModelName:  modelName,
		Provider:   prov.Name,
		Usage:      usage,
	}, nil
}
