package smolllm

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func chatOptions(includeStreamUsage bool) chatPayloadOptions {
	return chatPayloadOptions{
		Temperature:        nil,
		TopP:               nil,
		ReasoningEffort:    nil,
		MaxTokens:          nil,
		Stop:               nil,
		Seed:               nil,
		IncludeStreamUsage: includeStreamUsage,
		// Streaming is what prepareLLMCall sets for every ordinary call; the
		// blocking transport is opt-in and covered by its own tests.
		Stream:    true,
		ExtraBody: nil,
	}
}

func TestBuildRequestPayloadBasic(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	url, body, tokens, err := buildRequestPayload(
		prompt, "", "gpt-4o-mini", "openai", "https://api.openai.com", nil, chatOptions(true),
	)
	require.NoError(t, err)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", url)
	assert.Positive(t, tokens)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, "gpt-4o-mini", payload["model"])
	assert.Equal(t, true, payload["stream"])
	assert.Equal(t, map[string]any{"include_usage": true}, payload["stream_options"])
}

func TestBuildRequestPayloadWithSamplingParams(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	temp := 0.4
	topP := 0.85
	options := chatOptions(true)
	options.Temperature = &temp
	options.TopP = &topP
	_, body, _, err := buildRequestPayload(
		prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, options,
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.InDelta(t, temp, payload["temperature"], 1e-9)
	assert.InDelta(t, topP, payload["top_p"], 1e-9)
}

func TestBuildRequestPayloadWithCommonGenerationParams(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	maxTokens := 128
	seed := 42
	stop := []string{"END", "STOP"}
	options := chatOptions(true)
	options.MaxTokens = &maxTokens
	options.Stop = stop
	options.Seed = &seed
	_, body, _, err := buildRequestPayload(
		prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, options,
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.InDelta(t, 128, payload["max_tokens"], 1e-9)
	assert.Equal(t, []any{"END", "STOP"}, payload["stop"])
	assert.InDelta(t, 42, payload["seed"], 1e-9)
}

func TestBuildRequestPayloadCanOmitStreamOptions(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	_, body, _, err := buildRequestPayload(
		prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, chatOptions(false),
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.NotContains(t, payload, "stream_options")
}

func TestComposeMessagesRejectsImageOnAssistant(t *testing.T) {
	t.Parallel()
	prompt := PromptFromMessages([]Message{Assistant("hi")})
	_, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.Error(t, err)
}

func TestComposeMessagesRejectsMultipleMessagesWithImages(t *testing.T) {
	t.Parallel()
	prompt := PromptFromMessages([]Message{User("one"), User("two")})
	_, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.Error(t, err)
}

func TestBuildRequestPayloadRequiresBaseURL(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	_, _, _, err := buildRequestPayload(prompt, "", "gpt-4o", "openai", "", nil, chatOptions(true))
	require.Error(t, err)
}

func TestBuildRequestPayloadRejectsInvalidTemperature(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	temp := 3.0
	options := chatOptions(true)
	options.Temperature = &temp
	_, _, _, err := buildRequestPayload(
		prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, options,
	)
	require.Error(t, err)
}

func TestBuildRequestPayloadRejectsInvalidTopP(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	topP := -0.1
	options := chatOptions(true)
	options.TopP = &topP
	_, _, _, err := buildRequestPayload(
		prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, options,
	)
	require.Error(t, err)
}

func TestBuildRequestPayloadWithReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	effort := "medium"
	options := chatOptions(true)
	options.ReasoningEffort = &effort
	_, body, _, err := buildRequestPayload(
		prompt, "", "o3", "openai", "https://api.openai.com", nil, options,
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, "medium", payload["reasoning_effort"])
}

func TestBuildRequestPayloadNormalizesReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	effort := " Minimal "
	options := chatOptions(true)
	options.ReasoningEffort = &effort
	_, body, _, err := buildRequestPayload(
		prompt, "", "o3", "openai", "https://api.openai.com", nil, options,
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, "minimal", payload["reasoning_effort"])
}

func TestBuildRequestPayloadRejectsEmptyReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	effort := "  "
	options := chatOptions(true)
	options.ReasoningEffort = &effort
	_, _, _, err := buildRequestPayload(
		prompt, "", "o3", "openai", "https://api.openai.com", nil, options,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reasoning_effort")
}

func TestBuildRequestPayloadRejectsUnsupportedReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	effort := "minimum"
	options := chatOptions(true)
	options.ReasoningEffort = &effort
	_, _, _, err := buildRequestPayload(
		prompt, "", "o3", "openai", "https://api.openai.com", nil, options,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reasoning_effort")
	assert.Contains(t, err.Error(), "none, minimal, low, medium, high, xhigh")
}

func TestBuildRequestPayloadRejectsOllamaUnsupportedReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	effort := "minimal"
	options := chatOptions(true)
	options.ReasoningEffort = &effort
	_, _, _, err := buildRequestPayload(
		prompt, "", "llama", "ollama", "http://localhost:11434", nil, options,
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reasoning_effort")
	assert.Contains(t, err.Error(), "none, low, medium, high")
}

func TestBuildRequestPayloadAcceptsOpenAICompatibleReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	for _, v := range []string{"none", "minimal", "low", "medium", "high", "xhigh"} {
		effort := v
		options := chatOptions(true)
		options.ReasoningEffort = &effort
		_, body, _, err := buildRequestPayload(
			prompt, "", "o3", "openai", "https://api.openai.com", nil, options,
		)
		require.NoError(t, err)

		var payload map[string]any
		require.NoError(t, json.Unmarshal(body, &payload))
		assert.Equal(t, v, payload["reasoning_effort"])
	}
}

func TestComposeMessagesAllowsSystemWithImagesAndSingleUser(t *testing.T) {
	t.Parallel()
	// This tests the fix for the CLI issue where --system with --images was rejected
	prompt := PromptFromMessages([]Message{System("analyze this"), User("what do you see?")})
	messages, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.NoError(t, err, "system + single user message with images should be allowed")
	assert.Len(t, messages, 2, "should have system and user messages")
}

func TestComposeMessagesRejectsMultipleUserMessagesWithImages(t *testing.T) {
	t.Parallel()
	// Even with system prompt, multiple user messages should still be rejected
	prompt := PromptFromMessages([]Message{System("sys"), User("one"), User("two")})
	_, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.Error(t, err, "multiple user messages with images should be rejected")
}
