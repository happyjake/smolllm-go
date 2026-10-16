package smolllm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

type streamDelta struct {
	Content          *string         `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"` // DeepSeek, vLLM, LiteLLM
	Reasoning        *string         `json:"reasoning"`         // Ollama
	ToolCalls        []toolCallDelta `json:"tool_calls"`
}

type streamChoice struct {
	Delta        *streamDelta `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type streamChunk struct {
	Choices []streamChoice `json:"choices"`
	Usage   *usageChunk    `json:"usage"`
}

// usageChunk decodes the provider's usage object. OpenAI-compatible endpoints
// report prompt caching in three mutually-exclusive dialects, so all three are
// decoded here and normalized by cacheRead.
type usageChunk struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails *struct {
		// Pointer: the details object also carries unrelated keys such as
		// audio_tokens, so the object's presence does not mean the provider
		// said anything about caching.
		CachedTokens *int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"` // OpenAI, OpenRouter, z.ai, vLLM
	// Pointers, so an explicitly reported zero is distinguishable from an
	// absent field. A provider that says cached_tokens:0 is asserting "nothing
	// came from cache"; treating that as "unset" and falling through to a
	// lower-priority dialect would report someone else's number as fact.
	PromptCacheHitTokens     *int `json:"prompt_cache_hit_tokens"`     // DeepSeek
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`     // Anthropic-style
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"` // Anthropic-style
}

// cacheRead returns the prompt tokens served from cache, whichever dialect the
// provider used. Precedence is fixed rather than "whichever is set", so a
// provider that emits two of these (they carry the same number) decodes
// deterministically.
//
// DeepSeek's prompt_cache_miss_tokens is deliberately not mapped: a miss is
// ordinary uncached input, not a billed cache write, and it is derivable as
// InputTokens - CacheReadTokens.
func (u *usageChunk) cacheRead() int {
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens != nil {
		return *u.PromptTokensDetails.CachedTokens
	}
	if u.PromptCacheHitTokens != nil {
		return *u.PromptCacheHitTokens
	}
	if u.CacheReadInputTokens != nil {
		return *u.CacheReadInputTokens
	}
	return 0
}

// cacheWrite reports tokens written to cache. Only Anthropic-style endpoints
// report writes at all.
func (u *usageChunk) cacheWrite() int {
	if u.CacheCreationInputTokens != nil {
		return *u.CacheCreationInputTokens
	}
	return 0
}

// reportedCacheRead and reportedCacheWrite are tracked separately: a provider
// reporting only cache_creation_input_tokens has said nothing about reads, and
// folding the two together would let a consumer count an unknown read as a
// known miss.
func (u *usageChunk) reportedCacheRead() bool {
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens != nil {
		return true
	}
	return u.PromptCacheHitTokens != nil || u.CacheReadInputTokens != nil
}

func (u *usageChunk) reportedCacheWrite() bool {
	return u.CacheCreationInputTokens != nil
}

type reportedUsage struct {
	inputTokens        int
	outputTokens       int
	cacheReadTokens    int
	cacheWriteTokens   int
	cacheReadReported  bool
	cacheWriteReported bool
	reported           bool
}

func startStreamForwarder(
	reqCtx context.Context,
	logger *slog.Logger,
	resp *http.Response,
	call *preparedCall,
	cancel context.CancelFunc,
	start time.Time,
) (chan StreamChunk, chan streamCompletion) {
	chunks := make(chan StreamChunk)
	done := make(chan streamCompletion, 1)

	go func() {
		defer cancel()
		defer func() { _ = resp.Body.Close() }()
		defer close(chunks)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

		var (
			firstToken       time.Time
			contentBuilder   strings.Builder
			reasoningBuilder strings.Builder
			thinkFilter      ThinkTagFilter
			usage            reportedUsage
			finishReason     string
			tools            toolCallAccumulator
			err              error
		)

	Loop:
		for scanner.Scan() {
			select {
			case <-reqCtx.Done():
				err = reqCtx.Err()
				break Loop
			default:
			}

			line := scanner.Text()
			var chunk StreamChunk
			chunk, err = processChunkLineWithMetadata(logger, line, &usage, &finishReason, &tools)
			if err != nil {
				break
			}

			chunk = thinkFilter.Feed(chunk)
			if chunk.IsEmpty() {
				continue
			}

			if firstToken.IsZero() {
				firstToken = time.Now().UTC()
			}

			contentBuilder.WriteString(chunk.Content)
			reasoningBuilder.WriteString(chunk.Reasoning)

			select {
			case <-reqCtx.Done():
				err = reqCtx.Err()
				break Loop
			case chunks <- chunk:
			}
		}

		// Flush any buffered think-tag content.
		if final := thinkFilter.Flush(); !final.IsEmpty() {
			contentBuilder.WriteString(final.Content)
			reasoningBuilder.WriteString(final.Reasoning)
			select {
			case <-reqCtx.Done():
			case chunks <- final:
			}
		}

		if err == nil {
			err = scanner.Err()
		}

		total := time.Since(start)
		ttft := computeTTFT(firstToken, start)

		completion := streamCompletion{
			err:          err,
			metrics:      nil,
			reasoning:    reasoningBuilder.String(),
			finishReason: finishReason,
			toolCalls:    tools.result(),
		}

		// Metrics are built even on a hard error: a usage frame that already
		// arrived carries real token counts, and discarding it makes the failure
		// hook and the ledger record an estimate for a call the provider
		// actually accounted for.
		combined := contentBuilder.String() + reasoningBuilder.String()
		inputTokens := call.InputTokens
		outputTokens := estimateTokens(combined)
		cacheReadTokens := 0
		cacheWriteTokens := 0
		cacheReadReported := false
		cacheWriteReported := false
		estimated := true
		if usage.reported {
			inputTokens = usage.inputTokens
			outputTokens = usage.outputTokens
			cacheReadTokens = usage.cacheReadTokens
			cacheWriteTokens = usage.cacheWriteTokens
			cacheReadReported = usage.cacheReadReported
			cacheWriteReported = usage.cacheWriteReported
			estimated = false
		}
		completion.metrics = &streamMetrics{
			modelName:          call.ModelName,
			inputTokens:        inputTokens,
			outputTokens:       outputTokens,
			cacheReadTokens:    cacheReadTokens,
			cacheWriteTokens:   cacheWriteTokens,
			cacheReadReported:  cacheReadReported,
			cacheWriteReported: cacheWriteReported,
			total:              total,
			ttft:               ttft,
			estimated:          estimated,
		}

		done <- completion
		close(done)
	}()

	return chunks, done
}

type consumeResult struct {
	content      string
	reasoning    string
	ttft         time.Duration
	usage        reportedUsage
	finishReason string
	toolCalls    []ToolCall
}

func consumeStream(
	ctx context.Context,
	logger *slog.Logger,
	reader io.Reader,
	handler func(context.Context, string) error,
	start time.Time,
) (consumeResult, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	var thinkFilter ThinkTagFilter
	var firstToken time.Time
	var usage reportedUsage
	var finishReason string
	var tools toolCallAccumulator

	// failed carries whatever the provider already told us out with the error.
	// A usage frame that arrived before a malformed line still holds real token
	// counts, and returning an empty result makes the failure hook record an
	// estimate for a call the provider accounted for. The unconditional-metrics
	// change covered Stream; this is the Ask path.
	failed := func(err error) (consumeResult, error) {
		return consumeResult{usage: usage, finishReason: finishReason, toolCalls: tools.result()}, err
	}

	for scanner.Scan() {
		if ctx.Err() != nil {
			return failed(ctx.Err())
		}

		line := scanner.Text()
		chunk, err := processChunkLineWithMetadata(logger, line, &usage, &finishReason, &tools)
		if err != nil {
			return failed(err)
		}

		chunk = thinkFilter.Feed(chunk)
		if chunk.IsEmpty() {
			continue
		}

		if firstToken.IsZero() {
			firstToken = time.Now().UTC()
		}

		if handler != nil {
			if err := handler(ctx, chunk.Content); err != nil {
				return failed(err)
			}
		}
		contentBuilder.WriteString(chunk.Content)
		reasoningBuilder.WriteString(chunk.Reasoning)
	}

	// Flush any buffered think-tag content.
	if final := thinkFilter.Flush(); !final.IsEmpty() {
		contentBuilder.WriteString(final.Content)
		reasoningBuilder.WriteString(final.Reasoning)
	}

	if err := scanner.Err(); err != nil {
		return failed(err)
	}

	content := strings.TrimSpace(contentBuilder.String())
	reasoning := strings.TrimSpace(reasoningBuilder.String())
	ttft := computeTTFT(firstToken, start)
	toolCalls := tools.result()
	result := consumeResult{
		content: content, reasoning: reasoning, finishReason: finishReason,
		ttft: ttft, usage: usage, toolCalls: toolCalls,
	}
	// An assistant turn that only requests tool calls carries no text, but it is
	// a complete, useful response.
	if content == "" && reasoning == "" && len(toolCalls) == 0 {
		return result, &EmptyStreamError{FinishReason: finishReason}
	}

	return result, nil
}

func processChunkLine(logger *slog.Logger, line string) (StreamChunk, error) {
	return processChunkLineWithMetadata(logger, line, nil, nil, nil)
}

func processChunkLineWithMetadata(
	logger *slog.Logger,
	line string,
	usage *reportedUsage,
	finishReason *string,
	tools *toolCallAccumulator,
) (StreamChunk, error) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" || trimmed == "data: [DONE]" {
		return StreamChunk{Content: "", Reasoning: ""}, nil
	}
	if !strings.HasPrefix(trimmed, "data:") {
		return StreamChunk{Content: "", Reasoning: ""}, nil
	}
	payload := strings.TrimSpace(trimmed[len("data:"):])
	if payload == "" {
		return StreamChunk{Content: "", Reasoning: ""}, nil
	}

	var chunk streamChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		logger.Error("malformed streaming chunk", "error", err)
		return StreamChunk{Content: "", Reasoning: ""}, fmt.Errorf("malformed streaming chunk: %w", err)
	}

	if usage != nil && chunk.Usage != nil {
		usage.inputTokens = chunk.Usage.PromptTokens
		usage.outputTokens = chunk.Usage.CompletionTokens
		usage.cacheReadTokens = chunk.Usage.cacheRead()
		usage.cacheWriteTokens = chunk.Usage.cacheWrite()
		usage.cacheReadReported = chunk.Usage.reportedCacheRead()
		usage.cacheWriteReported = chunk.Usage.reportedCacheWrite()
		usage.reported = true
	}
	if finishReason != nil && len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != nil {
		*finishReason = *chunk.Choices[0].FinishReason
	}

	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		logger.Debug("stream chunk missing delta")
		return StreamChunk{Content: "", Reasoning: ""}, nil
	}

	delta := chunk.Choices[0].Delta
	// Tool-call fragments feed the accumulator; they are never forwarded as
	// chunks, so a caller only ever sees complete calls after the stream ends.
	if tools != nil && len(delta.ToolCalls) > 0 {
		tools.feed(delta.ToolCalls)
	}
	content := ""
	if delta.Content != nil {
		content = *delta.Content
	}
	reasoning := extractReasoning(delta)

	if content == "" && reasoning == "" {
		return StreamChunk{Content: "", Reasoning: ""}, nil
	}
	return StreamChunk{Content: content, Reasoning: reasoning}, nil
}

func extractReasoning(d *streamDelta) string {
	if d.ReasoningContent != nil && *d.ReasoningContent != "" {
		return *d.ReasoningContent
	}
	if d.Reasoning != nil && *d.Reasoning != "" {
		return *d.Reasoning
	}
	return ""
}

func computeTTFT(firstToken time.Time, start time.Time) time.Duration {
	if firstToken.IsZero() {
		return -1
	}
	ttft := firstToken.Sub(start)
	if ttft < 0 {
		return 0
	}
	return ttft
}
