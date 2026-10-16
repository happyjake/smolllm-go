package smolllm

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared/constant"
)

// Message is compatible with the OpenAI Go SDK chat completion message union.
type Message = openai.ChatCompletionMessageParamUnion

// Prompt is the request payload passed to Ask/Stream.
type Prompt struct {
	Messages []Message
}

// PromptFromString creates a single user message prompt.
func PromptFromString(text string) Prompt {
	msg := openai.UserMessage(text)
	ensureRole(&msg)
	return Prompt{
		Messages: []Message{msg},
	}
}

// PromptFromMessages constructs a prompt from an existing slice.
func PromptFromMessages(messages []Message) Prompt {
	cp := make([]Message, len(messages))
	copy(cp, messages)
	for i := range cp {
		ensureRole(&cp[i])
	}
	return Prompt{Messages: cp}
}

// System returns a system role chat message.
func System(content string) Message {
	msg := openai.SystemMessage(content)
	ensureRole(&msg)
	return msg
}

// User returns a user role chat message.
func User(content string) Message {
	msg := openai.UserMessage(content)
	ensureRole(&msg)
	return msg
}

// Assistant returns an assistant role chat message.
func Assistant(content string) Message {
	msg := openai.AssistantMessage(content)
	ensureRole(&msg)
	return msg
}

// Developer returns a developer role chat message.
func Developer(content string) Message {
	msg := openai.DeveloperMessage(content)
	ensureRole(&msg)
	return msg
}

// AssistantToolCalls returns an assistant message replaying the tool calls the
// model asked for. Pass empty text when the turn carried no content.
func AssistantToolCalls(text string, calls []ToolCall) Message {
	params := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(calls))
	for _, call := range calls {
		fn := openai.ChatCompletionMessageFunctionToolCallParam{ //nolint:exhaustruct // Type defaults to "function"
			ID: call.ID,
			Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
				Name:      call.Function.Name,
				Arguments: call.Function.Arguments,
			},
		}
		// Provider extras (e.g. Gemini thought signatures) must survive replay.
		if extra := call.extraFields(); extra != nil {
			fn.SetExtraFields(extra)
		}
		params = append(params, openai.ChatCompletionMessageToolCallUnionParam{ //nolint:exhaustruct // union arm
			OfFunction: &fn,
		})
	}
	assistant := openai.ChatCompletionAssistantMessageParam{ //nolint:exhaustruct // optional fields stay unset
		ToolCalls: params,
	}
	if text != "" {
		assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{ //nolint:exhaustruct // text arm
			OfString: openai.String(text),
		}
	}
	msg := Message{OfAssistant: &assistant} //nolint:exhaustruct // union arm
	ensureRole(&msg)
	return msg
}

// ToolResult returns a tool role message carrying the output of one tool call.
func ToolResult(toolCallID, content string) Message {
	msg := openai.ToolMessage(content, toolCallID)
	ensureRole(&msg)
	return msg
}

// Validate ensures the prompt is well formed.
func (p Prompt) Validate() error {
	if len(p.Messages) == 0 {
		return errors.New("prompt must contain at least one message")
	}
	for i, msg := range p.Messages {
		if _, ok := messageRole(msg); !ok {
			return fmt.Errorf("prompt message #%d must set role", i)
		}

		// The legacy `function` role is deprecated upstream and stays rejected;
		// `tool` is how a caller replays a tool result.
		if role, _ := messageRole(msg); role == "function" {
			return fmt.Errorf("prompt message #%d uses unsupported role %q", i, role)
		}

		if content := msg.GetContent().AsAny(); content != nil {
			continue
		}

		if msg.OfTool != nil {
			// A tool result may legitimately be an empty string.
			continue
		}

		if toolCalls := msg.GetToolCalls(); len(toolCalls) > 0 {
			continue
		}

		if msg.GetFunctionCall() != nil {
			continue
		}

		return fmt.Errorf("prompt message #%d must set content", i)
	}
	return nil
}

func messageRole(msg Message) (string, bool) {
	if role := msg.GetRole(); role != nil {
		if trimmed := strings.TrimSpace(*role); trimmed != "" {
			return trimmed, true
		}
	}
	switch {
	case msg.OfDeveloper != nil:
		return "developer", true
	case msg.OfSystem != nil:
		return "system", true
	case msg.OfUser != nil:
		return "user", true
	case msg.OfAssistant != nil:
		return "assistant", true
	case msg.OfTool != nil:
		return "tool", true
	case msg.OfFunction != nil:
		return "function", true
	default:
		return "", false
	}
}

func ensureRole(msg *Message) {
	if msg == nil {
		return
	}
	switch {
	case msg.OfDeveloper != nil:
		msg.OfDeveloper.Role = constant.ValueOf[constant.Developer]()
	case msg.OfSystem != nil:
		msg.OfSystem.Role = constant.ValueOf[constant.System]()
	case msg.OfUser != nil:
		msg.OfUser.Role = constant.ValueOf[constant.User]()
	case msg.OfAssistant != nil:
		msg.OfAssistant.Role = constant.ValueOf[constant.Assistant]()
	case msg.OfTool != nil:
		msg.OfTool.Role = constant.ValueOf[constant.Tool]()
	}
}

// StreamChunk carries a single streamed delta with optional reasoning.
type StreamChunk struct {
	Content   string
	Reasoning string
}

// String returns only the content portion (backward-compatible with fmt.Fprint).
func (c StreamChunk) String() string { return c.Content }

// IsEmpty reports whether the chunk carries no content and no reasoning.
func (c StreamChunk) IsEmpty() bool { return c.Content == "" && c.Reasoning == "" }

// Usage captures per-call metrics and routing details.
type Usage struct {
	Provider     string `json:"provider"`
	Model        string `json:"model"`
	ModelName    string `json:"model_name"`
	APIKeyHint   string `json:"api_key_hint"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	// CacheReadTokens is the provider-reported count of prompt tokens served
	// from a prompt cache. Only ever provider-reported: it is 0 whenever
	// Estimated is true, and 0 also means "the provider reported no cache
	// data", which is indistinguishable from a genuine miss by design.
	CacheReadTokens int `json:"cache_read_tokens,omitempty"`
	// CacheWriteTokens is the provider-reported count of prompt tokens written
	// to cache on this call. Only Anthropic-style endpoints report writes;
	// elsewhere 0 means "not reported", not "nothing was cached".
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	// CacheReadReported and CacheWriteReported record whether the provider spoke
	// about reads and writes respectively. Without them a zero is ambiguous, and
	// averaging a reported zero with an unreported one yields a confidently wrong
	// hit rate. They are separate because a provider reporting only cache writes
	// has said nothing about reads.
	CacheReadReported  bool          `json:"cache_read_reported,omitempty"`
	CacheWriteReported bool          `json:"cache_write_reported,omitempty"`
	Duration           time.Duration `json:"duration"`
	TTFT               time.Duration `json:"ttft"`
	Estimated          bool          `json:"estimated"`
}

// RequestEvent is emitted after each LLM call attempt (success or failure).
type RequestEvent struct {
	Usage
	Error     error     `json:"-"`
	Timestamp time.Time `json:"timestamp"`
}

// Response holds a full LLM response.
type Response struct {
	Text         string `json:"text"`
	Reasoning    string `json:"reasoning"`
	FinishReason string `json:"finish_reason"`
	Model        string `json:"model"`
	ModelName    string `json:"model_name"`
	Provider     string `json:"provider"`
	Usage        Usage  `json:"usage"`
	// ToolCalls is empty unless the model answered with tool calls. Executing
	// them and replaying the result is the caller's job.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// DeltaStream represents a streaming LLM response.
type DeltaStream struct {
	ch           <-chan StreamChunk
	done         <-chan streamCompletion
	cancel       func()
	logger       *slog.Logger
	reasoning    *string // populated by Wait() from streamCompletion
	finishReason *string
	toolCalls    *[]ToolCall
	usage        *Usage
	hook         func(RequestEvent)
}

type streamCompletion struct {
	err          error
	metrics      *streamMetrics
	reasoning    string
	finishReason string
	toolCalls    []ToolCall
}

type streamMetrics struct {
	modelName          string
	inputTokens        int
	outputTokens       int
	cacheReadTokens    int
	cacheWriteTokens   int
	cacheReadReported  bool
	cacheWriteReported bool
	total              time.Duration
	ttft               time.Duration
	estimated          bool
}

// Chan exposes the underlying channel of chunks.
func (s DeltaStream) Chan() <-chan StreamChunk {
	return s.ch
}

// Close cancels the stream.
func (s DeltaStream) Close() {
	if s.cancel != nil {
		s.cancel()
	}
}

// Wait blocks until the stream finishes and returns the terminal error.
// After Wait returns, the parent StreamResponse.Reasoning and Usage are populated.
func (s DeltaStream) Wait() error {
	if s.done == nil {
		return nil
	}
	result := <-s.done
	s.absorb(result)
	if s.hook != nil && s.usage != nil {
		s.hook(RequestEvent{
			Usage:     *s.usage,
			Error:     result.err,
			Timestamp: time.Now().UTC(),
		})
	}
	return result.err
}

// absorb folds a stream's terminal state into the response fields: reasoning,
// finish reason, tool calls, the metrics log line, and usage. Split out of
// Wait so a caller that takes over the terminal state before the response is
// handed to a consumer (see gateFirstOutput) runs the same bookkeeping.
func (s DeltaStream) absorb(result streamCompletion) {
	if s.reasoning != nil {
		*s.reasoning = result.reasoning
	}
	if s.finishReason != nil {
		*s.finishReason = result.finishReason
	}
	if s.toolCalls != nil {
		*s.toolCalls = result.toolCalls
	}
	if result.metrics == nil {
		return
	}
	if s.logger != nil {
		s.logger.Info(
			formatMetrics(
				result.metrics.modelName,
				result.metrics.inputTokens,
				result.metrics.outputTokens,
				result.metrics.total,
				result.metrics.ttft,
			),
			"model", result.metrics.modelName,
		)
	}
	if s.usage != nil {
		s.usage.InputTokens = result.metrics.inputTokens
		s.usage.OutputTokens = result.metrics.outputTokens
		s.usage.CacheReadTokens = result.metrics.cacheReadTokens
		s.usage.CacheWriteTokens = result.metrics.cacheWriteTokens
		s.usage.CacheReadReported = result.metrics.cacheReadReported
		s.usage.CacheWriteReported = result.metrics.cacheWriteReported
		s.usage.Duration = result.metrics.total
		s.usage.TTFT = result.metrics.ttft
		s.usage.Estimated = result.metrics.estimated
	}
}

// StreamResponse wraps streaming metadata.
type StreamResponse struct {
	Stream       DeltaStream `json:"-"`
	Reasoning    string      `json:"reasoning"`
	FinishReason string      `json:"finish_reason"`
	Model        string      `json:"model"`
	ModelName    string      `json:"model_name"`
	Provider     string      `json:"provider"`
	Usage        Usage       `json:"usage"`
	// ToolCalls is populated by Stream.Wait(), like Reasoning and Usage:
	// partial argument JSON is never pushed to consumers.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}
