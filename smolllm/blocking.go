package smolllm

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// completionResponse decodes a plain (stream:false) chat completion.
//
// It reuses usageChunk, so the three prompt-cache dialects are normalized
// identically on both transports rather than by two copies that can drift.
type completionResponse struct {
	Choices []struct {
		Message struct {
			Content          *string    `json:"content"`
			ReasoningContent *string    `json:"reasoning_content"`
			Reasoning        *string    `json:"reasoning"`
			ToolCalls        []ToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usageChunk `json:"usage"`
}

// consumeBlocking reads a non-streaming response into the same consumeResult
// the streaming path produces, so every caller downstream — the empty-response
// check, the length-truncation guards, MinOutputTokens, usage, hooks — works
// unchanged.
func consumeBlocking(body io.Reader, start time.Time) (consumeResult, error) {
	var decoded completionResponse
	if err := json.NewDecoder(body).Decode(&decoded); err != nil {
		return consumeResult{}, fmt.Errorf("decode completion response: %w", err)
	}
	// Usage is parsed before the choices check so a response carrying usage but
	// no choices still reports real token counts to the failure hook, matching
	// the streaming path — where the usage frame itself has an empty choices
	// array and is recorded regardless.
	var usage reportedUsage
	if decoded.Usage != nil {
		usage = reportedUsage{
			inputTokens:      decoded.Usage.PromptTokens,
			outputTokens:     decoded.Usage.CompletionTokens,
			cacheReadTokens:  decoded.Usage.cacheRead(),
			cacheWriteTokens: decoded.Usage.cacheWrite(),
			cacheReported:    decoded.Usage.reportedCache(),
			reported:         true,
		}
	}
	if len(decoded.Choices) == 0 {
		return consumeResult{usage: usage}, fmt.Errorf("completion response contained no choices")
	}
	choice := decoded.Choices[0]

	content := derefString(choice.Message.Content)
	reasoning := derefString(choice.Message.ReasoningContent)
	if reasoning == "" {
		reasoning = derefString(choice.Message.Reasoning)
	}

	// Run the same think-tag reclassification the streaming path applies, so a
	// <think>-tagged model splits content and reasoning identically on both
	// transports.
	var thinkFilter ThinkTagFilter
	filtered := thinkFilter.Feed(StreamChunk{Content: content, Reasoning: reasoning})
	final := thinkFilter.Flush()

	result := consumeResult{
		content:      strings.TrimSpace(filtered.Content + final.Content),
		reasoning:    strings.TrimSpace(filtered.Reasoning + final.Reasoning),
		finishReason: derefString(choice.FinishReason),
		toolCalls:    choice.Message.ToolCalls,
		// Provisional: askOnce overwrites this with the call's total duration so
		// TTFT == Duration exactly. The "no token seen" sentinel would instead
		// make every blocking call look like a dead stream on a dashboard.
		ttft: time.Since(start),
	}
	result.usage = usage
	return result, nil
}

func derefString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
