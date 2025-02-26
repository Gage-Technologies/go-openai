package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/internal/test/checks"
	"github.com/sashabaranov/go-openai/jsonschema"
)

const (
	xCustomHeader      = "X-CUSTOM-HEADER"
	xCustomHeaderValue = "test"
)

var rateLimitHeaders = map[string]any{
	"x-ratelimit-limit-requests":     60,
	"x-ratelimit-limit-tokens":       150000,
	"x-ratelimit-remaining-requests": 59,
	"x-ratelimit-remaining-tokens":   149984,
	"x-ratelimit-reset-requests":     "1s",
	"x-ratelimit-reset-tokens":       "6m0s",
}

func TestChatCompletionsWrongModel(t *testing.T) {
	config := openai.DefaultConfig("whatever")
	config.BaseURL = "http://localhost/v1"
	client := openai.NewClientWithConfig(config)
	ctx := context.Background()

	req := openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     "ada",
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	}
	_, err := client.CreateChatCompletion(ctx, req)
	msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
	checks.ErrorIs(t, err, openai.ErrChatCompletionInvalidModel, msg)
}

func TestO1ModelsChatCompletionsDeprecatedFields(t *testing.T) {
	tests := []struct {
		name          string
		in            openai.ChatCompletionRequest
		expectedError error
	}{
		{
			name: "o1-preview_MaxTokens_deprecated",
			in: openai.ChatCompletionRequest{
				MaxTokens: 5,
				Model:     openai.O1Preview,
			},
			expectedError: openai.ErrReasoningModelMaxTokensDeprecated,
		},
		{
			name: "o1-mini_MaxTokens_deprecated",
			in: openai.ChatCompletionRequest{
				MaxTokens: 5,
				Model:     openai.O1Mini,
			},
			expectedError: openai.ErrReasoningModelMaxTokensDeprecated,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := openai.DefaultConfig("whatever")
			config.BaseURL = "http://localhost/v1"
			client := openai.NewClientWithConfig(config)
			ctx := context.Background()

			_, err := client.CreateChatCompletion(ctx, tt.in)
			checks.HasError(t, err)
			msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
			checks.ErrorIs(t, err, tt.expectedError, msg)
		})
	}
}

func TestO1ModelsChatCompletionsBetaLimitations(t *testing.T) {
	tests := []struct {
		name          string
		in            openai.ChatCompletionRequest
		expectedError error
	}{
		{
			name: "log_probs_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				LogProbs:            true,
				Model:               openai.O1Preview,
			},
			expectedError: openai.ErrReasoningModelLimitationsLogprobs,
		},
		{
			name: "set_temperature_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(2),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_top_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_n_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(1),
				N:           2,
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_presence_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				PresencePenalty: float32(1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_frequency_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O1Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				FrequencyPenalty: float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := openai.DefaultConfig("whatever")
			config.BaseURL = "http://localhost/v1"
			client := openai.NewClientWithConfig(config)
			ctx := context.Background()

			_, err := client.CreateChatCompletion(ctx, tt.in)
			checks.HasError(t, err)
			msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
			checks.ErrorIs(t, err, tt.expectedError, msg)
		})
	}
}

func TestO3ModelsChatCompletionsBetaLimitations(t *testing.T) {
	tests := []struct {
		name          string
		in            openai.ChatCompletionRequest
		expectedError error
	}{
		{
			name: "log_probs_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				LogProbs:            true,
				Model:               openai.O3Mini,
			},
			expectedError: openai.ErrReasoningModelLimitationsLogprobs,
		},
		{
			name: "set_temperature_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(2),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_top_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_n_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(1),
				N:           2,
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_presence_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				PresencePenalty: float32(1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_frequency_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.O3Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				FrequencyPenalty: float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := openai.DefaultConfig("whatever")
			config.BaseURL = "http://localhost/v1"
			client := openai.NewClientWithConfig(config)
			ctx := context.Background()

			_, err := client.CreateChatCompletion(ctx, tt.in)
			checks.HasError(t, err)
			msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
			checks.ErrorIs(t, err, tt.expectedError, msg)
		})
	}
}

func TestGPT5ModelsChatCompletionsBetaLimitations(t *testing.T) {
	tests := []struct {
		name          string
		in            openai.ChatCompletionRequest
		expectedError error
	}{
		{
			name: "log_probs_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				LogProbs:            true,
				Model:               openai.GPT5,
			},
			expectedError: openai.ErrReasoningModelLimitationsLogprobs,
		},
		{
			name: "set_temperature_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.GPT5Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(2),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_top_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.GPT5Nano,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_n_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.GPT5ChatLatest,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				Temperature: float32(1),
				TopP:        float32(1),
				N:           2,
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_presence_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.GPT5,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				PresencePenalty: float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
		{
			name: "set_frequency_penalty_unsupported",
			in: openai.ChatCompletionRequest{
				MaxCompletionTokens: 1000,
				Model:               openai.GPT5Mini,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleUser,
					},
					{
						Role: openai.ChatMessageRoleAssistant,
					},
				},
				FrequencyPenalty: float32(0.1),
			},
			expectedError: openai.ErrReasoningModelLimitationsOther,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := openai.DefaultConfig("whatever")
			config.BaseURL = "http://localhost/v1"
			client := openai.NewClientWithConfig(config)
			ctx := context.Background()

			_, err := client.CreateChatCompletion(ctx, tt.in)
			checks.HasError(t, err)
			msg := fmt.Sprintf("CreateChatCompletion should return wrong model error, returned: %s", err)
			checks.ErrorIs(t, err, tt.expectedError, msg)
		})
	}
}

func TestChatRequestOmitEmpty(t *testing.T) {
	data, err := json.Marshal(openai.ChatCompletionRequest{
		// We set model b/c it's required, so omitempty doesn't make sense
		Model: "gpt-4",
	})
	checks.NoError(t, err)

	// messages is also required so isn't omitted
	const expected = `{"model":"gpt-4","messages":null}`
	if string(data) != expected {
		t.Errorf("expected JSON with all empty fields to be %v but was %v", expected, string(data))
	}
}

func TestChatCompletionsWithStream(t *testing.T) {
	config := openai.DefaultConfig("whatever")
	config.BaseURL = "http://localhost/v1"
	client := openai.NewClientWithConfig(config)
	ctx := context.Background()

	req := openai.ChatCompletionRequest{
		Stream: true,
	}
	_, err := client.CreateChatCompletion(ctx, req)
	checks.ErrorIs(t, err, openai.ErrChatCompletionStreamNotSupported, "unexpected error")
}

// TestCompletions Tests the completions endpoint of the API using the mocked server.
func TestChatCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")
}

// TestCompletions Tests the completions endpoint of the API using the mocked server.
func TestO1ModelChatCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model:               openai.O1Preview,
		MaxCompletionTokens: 1000,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")
}

func TestO3ModelChatCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model:               openai.O3Mini,
		MaxCompletionTokens: 1000,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")
}

func TestDeepseekR1ModelChatCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleDeepseekR1ChatCompletionEndpoint)
	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model:               "deepseek-reasoner",
		MaxCompletionTokens: 100,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")
}

// TestCompletions Tests the completions endpoint of the API using the mocked server.
func TestChatCompletionsWithHeaders(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")

	a := resp.Header().Get(xCustomHeader)
	_ = a
	if resp.Header().Get(xCustomHeader) != xCustomHeaderValue {
		t.Errorf("expected header %s to be %s", xCustomHeader, xCustomHeaderValue)
	}
}

// TestChatCompletionsWithRateLimitHeaders Tests the completions endpoint of the API using the mocked server.
func TestChatCompletionsWithRateLimitHeaders(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateChatCompletion error")

	headers := resp.GetRateLimitHeaders()
	resetRequests := headers.ResetRequests.String()
	if resetRequests != rateLimitHeaders["x-ratelimit-reset-requests"] {
		t.Errorf("expected resetRequests %s to be %s", resetRequests, rateLimitHeaders["x-ratelimit-reset-requests"])
	}
	resetRequestsTime := headers.ResetRequests.Time()
	if resetRequestsTime.Before(time.Now()) {
		t.Errorf("unexpected reset requests: %v", resetRequestsTime)
	}

	bs1, _ := json.Marshal(headers)
	bs2, _ := json.Marshal(rateLimitHeaders)
	if string(bs1) != string(bs2) {
		t.Errorf("expected rate limit header %s to be %s", bs2, bs1)
	}
}

// TestChatCompletionsFunctions tests including a function call.
func TestChatCompletionsFunctions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	t.Run("bytes", func(t *testing.T) {
		//nolint:lll
		msg := json.RawMessage(`{"properties":{"count":{"type":"integer","description":"total number of words in sentence"},"words":{"items":{"type":"string"},"type":"array","description":"list of words in sentence"}},"type":"object","required":["count","words"]}`)
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name:       "test",
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("struct", func(t *testing.T) {
		type testMessage struct {
			Count int      `json:"count"`
			Words []string `json:"words"`
		}
		msg := testMessage{
			Count: 2,
			Words: []string{"hello", "world"},
		}
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name:       "test",
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("JSONSchemaDefinition", func(t *testing.T) {
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name: "test",
				Parameters: &jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"count": {
							Type:        jsonschema.Number,
							Description: "total number of words in sentence",
						},
						"words": {
							Type:        jsonschema.Array,
							Description: "list of words in sentence",
							Items: &jsonschema.Definition{
								Type: jsonschema.String,
							},
						},
						"enumTest": {
							Type: jsonschema.String,
							Enum: []string{"hello", "world"},
						},
					},
				},
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("JSONSchemaDefinitionWithFunctionDefine", func(t *testing.T) {
		// this is a compatibility check
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefine{{
				Name: "test",
				Parameters: &jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"count": {
							Type:        jsonschema.Number,
							Description: "total number of words in sentence",
						},
						"words": {
							Type:        jsonschema.Array,
							Description: "list of words in sentence",
							Items: &jsonschema.Definition{
								Type: jsonschema.String,
							},
						},
						"enumTest": {
							Type: jsonschema.String,
							Enum: []string{"hello", "world"},
						},
					},
				},
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("StructuredOutputs", func(t *testing.T) {
		type testMessage struct {
			Count int      `json:"count"`
			Words []string `json:"words"`
		}
		msg := testMessage{
			Count: 2,
			Words: []string{"hello", "world"},
		}
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			MaxTokens: 5,
			Model:     openai.GPT3Dot5Turbo0613,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []openai.FunctionDefinition{{
				Name:       "test",
				Strict:     true,
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
}

func TestAzureChatCompletions(t *testing.T) {
	client, server, teardown := setupAzureTestServer()
	defer teardown()
	server.RegisterHandler("/openai/deployments/*", handleChatCompletionEndpoint)

	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	})
	checks.NoError(t, err, "CreateAzureChatCompletion error")
}

func TestMultipartChatCompletions(t *testing.T) {
	client, server, teardown := setupAzureTestServer()
	defer teardown()
	server.RegisterHandler("/openai/deployments/*", handleChatCompletionEndpoint)

	_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		MaxTokens: 5,
		Model:     openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role: openai.ChatMessageRoleUser,
				MultiContent: []openai.ChatMessagePart{
					{
						Type: openai.ChatMessagePartTypeText,
						Text: "Hello!",
					},
					{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL:    "URL",
							Detail: openai.ImageURLDetailLow,
						},
					},
				},
			},
		},
	})
	checks.NoError(t, err, "CreateAzureChatCompletion error")
}

func TestMultipartChatMessageSerialization(t *testing.T) {
	jsonText := `[{"role":"system","content":"system-message"},` +
		`{"role":"user","content":[{"type":"text","text":"nice-text"},` +
		`{"type":"image_url","image_url":{"url":"URL","detail":"high"}}]}]`

	var msgs []openai.ChatCompletionMessage
	err := json.Unmarshal([]byte(jsonText), &msgs)
	if err != nil {
		t.Fatalf("Expected no error: %s", err)
	}
	if len(msgs) != 2 {
		t.Errorf("unexpected number of messages")
	}
	if msgs[0].Role != "system" || msgs[0].Content != "system-message" || msgs[0].MultiContent != nil {
		t.Errorf("invalid user message: %v", msgs[0])
	}
	if msgs[1].Role != "user" || msgs[1].Content != "" || len(msgs[1].MultiContent) != 2 {
		t.Errorf("invalid user message")
	}
	parts := msgs[1].MultiContent
	if parts[0].Type != "text" || parts[0].Text != "nice-text" {
		t.Errorf("invalid text part: %v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL.URL != "URL" || parts[1].ImageURL.Detail != "high" {
		t.Errorf("invalid image_url part")
	}

	s, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("Expected no error: %s", err)
	}
	res := strings.ReplaceAll(string(s), " ", "")
	if res != jsonText {
		t.Fatalf("invalid message: %s", string(s))
	}

	invalidMsg := []openai.ChatCompletionMessage{
		{
			Role:    "user",
			Content: "some-text",
			MultiContent: []openai.ChatMessagePart{
				{
					Type: "text",
					Text: "nice-text",
				},
			},
		},
	}
	_, err = json.Marshal(invalidMsg)
	if !errors.Is(err, openai.ErrContentFieldsMisused) {
		t.Fatalf("Expected error: %s", err)
	}

	err = json.Unmarshal([]byte(`["not-a-message"]`), &msgs)
	if err == nil {
		t.Fatalf("Expected error")
	}

	emptyMultiContentMsg := openai.ChatCompletionMessage{
		Role:         "user",
		MultiContent: []openai.ChatMessagePart{},
	}
	s, err = json.Marshal(emptyMultiContentMsg)
	if err != nil {
		t.Fatalf("Unexpected error")
	}
	res = strings.ReplaceAll(string(s), " ", "")
	if res != `{"role":"user"}` {
		t.Fatalf("invalid message: %s", string(s))
	}
}

// handleChatCompletionEndpoint Handles the ChatGPT completion endpoint by the test server.
func handleChatCompletionEndpoint(w http.ResponseWriter, r *http.Request) {
	var err error
	var resBytes []byte

	// completions only accepts POST requests
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
	var completionReq openai.ChatCompletionRequest
	if completionReq, err = getChatCompletionBody(r); err != nil {
		http.Error(w, "could not read request", http.StatusInternalServerError)
		return
	}
	res := openai.ChatCompletionResponse{
		ID:      strconv.Itoa(int(time.Now().Unix())),
		Object:  "test-object",
		Created: time.Now().Unix(),
		// would be nice to validate Model during testing, but
		// this may not be possible with how much upkeep
		// would be required / wouldn't make much sense
		Model: completionReq.Model,
	}
	// create completions
	n := completionReq.N
	if n == 0 {
		n = 1
	}
	for i := 0; i < n; i++ {
		// if there are functions, include them
		if len(completionReq.Functions) > 0 {
			var fcb []byte
			b := completionReq.Functions[0].Parameters
			fcb, err = json.Marshal(b)
			if err != nil {
				http.Error(w, "could not marshal function parameters", http.StatusInternalServerError)
				return
			}

			res.Choices = append(res.Choices, openai.ChatCompletionChoice{
				Message: openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleFunction,
					// this is valid json so it should be fine
					FunctionCall: &openai.FunctionCall{
						Name:      completionReq.Functions[0].Name,
						Arguments: string(fcb),
					},
				},
				Index: i,
			})
			continue
		}
		// generate a random string of length completionReq.Length
		completionStr := strings.Repeat("a", completionReq.MaxTokens)

		res.Choices = append(res.Choices, openai.ChatCompletionChoice{
			Message: openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: completionStr,
			},
			Index: i,
		})
	}
	inputTokens := numTokens(completionReq.Messages[0].Content) * n
	completionTokens := completionReq.MaxTokens * n
	res.Usage = openai.Usage{
		PromptTokens:     inputTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      inputTokens + completionTokens,
	}
	resBytes, _ = json.Marshal(res)
	w.Header().Set(xCustomHeader, xCustomHeaderValue)
	for k, v := range rateLimitHeaders {
		switch val := v.(type) {
		case int:
			w.Header().Set(k, strconv.Itoa(val))
		default:
			w.Header().Set(k, fmt.Sprintf("%s", v))
		}
	}
	fmt.Fprintln(w, string(resBytes))
}

func handleDeepseekR1ChatCompletionEndpoint(w http.ResponseWriter, r *http.Request) {
	var err error
	var resBytes []byte

	// completions only accepts POST requests
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
	var completionReq openai.ChatCompletionRequest
	if completionReq, err = getChatCompletionBody(r); err != nil {
		http.Error(w, "could not read request", http.StatusInternalServerError)
		return
	}
	res := openai.ChatCompletionResponse{
		ID:      strconv.Itoa(int(time.Now().Unix())),
		Object:  "test-object",
		Created: time.Now().Unix(),
		// would be nice to validate Model during testing, but
		// this may not be possible with how much upkeep
		// would be required / wouldn't make much sense
		Model: completionReq.Model,
	}
	// create completions
	n := completionReq.N
	if n == 0 {
		n = 1
	}
	if completionReq.MaxCompletionTokens == 0 {
		completionReq.MaxCompletionTokens = 1000
	}
	for i := 0; i < n; i++ {
		reasoningContent := "User says hello! And I need to reply"
		completionStr := strings.Repeat("a", completionReq.MaxCompletionTokens-numTokens(reasoningContent))
		res.Choices = append(res.Choices, openai.ChatCompletionChoice{
			Message: openai.ChatCompletionMessage{
				Role:             openai.ChatMessageRoleAssistant,
				ReasoningContent: reasoningContent,
				Content:          completionStr,
			},
			Index: i,
		})
	}
	inputTokens := numTokens(completionReq.Messages[0].Content) * n
	completionTokens := completionReq.MaxTokens * n
	res.Usage = openai.Usage{
		PromptTokens:     inputTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      inputTokens + completionTokens,
	}
	resBytes, _ = json.Marshal(res)
	w.Header().Set(xCustomHeader, xCustomHeaderValue)
	for k, v := range rateLimitHeaders {
		switch val := v.(type) {
		case int:
			w.Header().Set(k, strconv.Itoa(val))
		default:
			w.Header().Set(k, fmt.Sprintf("%s", v))
		}
	}
	fmt.Fprintln(w, string(resBytes))
}

// getChatCompletionBody Returns the body of the request to create a completion.
func getChatCompletionBody(r *http.Request) (openai.ChatCompletionRequest, error) {
	completion := openai.ChatCompletionRequest{}
	// read the request body
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		return openai.ChatCompletionRequest{}, err
	}
	err = json.Unmarshal(reqBody, &completion)
	if err != nil {
		return openai.ChatCompletionRequest{}, err
	}
	return completion, nil
}

func TestFinishReason(t *testing.T) {
	c := &openai.ChatCompletionChoice{
		FinishReason: openai.FinishReasonNull,
	}
	resBytes, _ := json.Marshal(c)
	if !strings.Contains(string(resBytes), `"finish_reason":null`) {
		t.Error("null should not be quoted")
	}

	c.FinishReason = ""

	resBytes, _ = json.Marshal(c)
	if !strings.Contains(string(resBytes), `"finish_reason":null`) {
		t.Error("null should not be quoted")
	}

	otherReasons := []openai.FinishReason{
		openai.FinishReasonStop,
		openai.FinishReasonLength,
		openai.FinishReasonFunctionCall,
		openai.FinishReasonContentFilter,
	}
	for _, r := range otherReasons {
		c.FinishReason = r
		resBytes, _ = json.Marshal(c)
		if !strings.Contains(string(resBytes), fmt.Sprintf(`"finish_reason":"%s"`, r)) {
			t.Errorf("%s should be quoted", r)
		}
	}
}

func TestChatCompletionResponseFormatJSONSchema_UnmarshalJSON(t *testing.T) {
	type args struct {
		data []byte
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			"",
			args{
				data: []byte(`{
      "name":   "math_response",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "steps": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "explanation": { "type": "string" },
                "output":      { "type": "string" }
              },
              "required": ["explanation","output"],
              "additionalProperties": false
            }
          },
          "final_answer": { "type": "string" }
        },
        "required": ["steps","final_answer"],
        "additionalProperties": false
      }
  }`),
			},
			false,
		},
		{
			"",
			args{
				data: []byte(`{
      "name":   "math_response",
      "strict": true,
      "schema": null
  }`),
			},
			false,
		},
		{
			"",
			args{
				data: []byte(`[123,456]`),
			},
			true,
		},
		{
			"",
			args{
				data: []byte(`{
      "name":   "math_response",
      "strict": true,
      "schema": 123456
  }`),
			},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r openai.ChatCompletionResponseFormatJSONSchema
			err := r.UnmarshalJSON(tt.args.data)
			if (err != nil) != tt.wantErr {
				t.Errorf("UnmarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestChatCompletionRequest_UnmarshalJSON(t *testing.T) {
	type args struct {
		bs []byte
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			"",
			args{bs: []byte(`{
  "model": "llama3-1b",
  "messages": [
    { "role": "system", "content": "You are a helpful math tutor." },
    { "role": "user",   "content": "solve 8x + 31 = 2" }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name":   "math_response",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "steps": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "explanation": { "type": "string" },
                "output":      { "type": "string" }
              },
              "required": ["explanation","output"],
              "additionalProperties": false
            }
          },
          "final_answer": { "type": "string" }
        },
        "required": ["steps","final_answer"],
        "additionalProperties": false
      }
    }
  }
}`)},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var m openai.ChatCompletionRequest
			err := json.Unmarshal(tt.args.bs, &m)
			if err != nil {
				t.Errorf("UnmarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestChatCompletionsWithAdditionalParameters(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionWithAdditionalParamsEndpoint)

	// Test with additional request parameters
	req := openai.ChatCompletionRequest{
		Model: openai.GPT3Dot5Turbo,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
		AdditionalParameters: map[string]interface{}{
			"custom_param": "custom_value",
			"nested_param": map[string]interface{}{
				"nested_field": 42,
			},
		},
	}

	resp, err := client.CreateChatCompletion(context.Background(), req)
	checks.NoError(t, err, "CreateChatCompletion with additional parameters error")

	// Verify the response's additional parameters
	customValue, exists := resp.AdditionalParameters["echo_param"]
	if !exists {
		t.Error("Expected 'echo_param' in additional parameters but it wasn't there")
	} else if customValue != "custom_value" {
		t.Errorf("Expected 'echo_param' to be 'custom_value', got %v", customValue)
	}

	nestedValue, exists := resp.AdditionalParameters["nested_echo"]
	if !exists {
		t.Error("Expected 'nested_echo' in additional parameters but it wasn't there")
	} else {
		// Type assertion to access nested map
		nestedMap, ok := nestedValue.(map[string]interface{})
		if !ok {
			t.Errorf("Expected 'nested_echo' to be a map, got %T", nestedValue)
		} else if nestedField, ok := nestedMap["echo_field"]; !ok || nestedField != float64(42) {
			t.Errorf("Expected 'nested_echo.echo_field' to be 42, got %v", nestedField)
		}
	}
}

// handleChatCompletionWithAdditionalParamsEndpoint Handles the ChatGPT completion endpoint with additional parameters
func handleChatCompletionWithAdditionalParamsEndpoint(w http.ResponseWriter, r *http.Request) {
	var err error

	// completions only accepts POST requests
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}

	// Read and decode the request body to check for additional parameters
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "could not read request", http.StatusInternalServerError)
		return
	}
	r.Body.Close()

	// Parse the JSON to verify additional parameters are included
	var requestData map[string]interface{}
	if err = json.Unmarshal(body, &requestData); err != nil {
		http.Error(w, "could not parse request", http.StatusInternalServerError)
		return
	}

	// Create basic response
	responseData := map[string]interface{}{
		"id":      strconv.Itoa(int(time.Now().Unix())),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   requestData["model"],
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "Test response",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     10,
			"completion_tokens": 10,
			"total_tokens":      20,
		},
		// Add additional response parameters
		"echo_param": requestData["custom_param"],
		"nested_echo": map[string]interface{}{
			"echo_field": requestData["nested_param"].(map[string]interface{})["nested_field"],
		},
	}

	// Serialize and return the response
	responseBytes, _ := json.Marshal(responseData)
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintln(w, string(responseBytes))
}

func TestChatMessageWithAdditionalParameters(t *testing.T) {
	// Test marshaling message with additional parameters
	message := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "Hello!",
		AdditionalParameters: map[string]interface{}{
			"custom_message_param": "custom_value",
			"nested_message_param": map[string]interface{}{
				"nested_field": 42,
			},
		},
	}

	// Marshal the message
	data, err := json.Marshal(message)
	checks.NoError(t, err, "Error marshaling message with additional parameters")

	// Parse the JSON to verify additional parameters are included
	var parsedData map[string]interface{}
	err = json.Unmarshal(data, &parsedData)
	checks.NoError(t, err, "Error parsing marshaled message")

	// Check standard fields
	if parsedData["role"] != "user" {
		t.Errorf("Expected role to be 'user', got %v", parsedData["role"])
	}
	if parsedData["content"] != "Hello!" {
		t.Errorf("Expected content to be 'Hello!', got %v", parsedData["content"])
	}

	// Check additional parameters
	customValue, exists := parsedData["custom_message_param"]
	if !exists {
		t.Error("Expected 'custom_message_param' in marshaled message but it wasn't there")
	} else if customValue != "custom_value" {
		t.Errorf("Expected 'custom_message_param' to be 'custom_value', got %v", customValue)
	}

	nestedParam, exists := parsedData["nested_message_param"]
	if !exists {
		t.Error("Expected 'nested_message_param' in marshaled message but it wasn't there")
	} else {
		nestedMap, ok := nestedParam.(map[string]interface{})
		if !ok {
			t.Errorf("Expected 'nested_message_param' to be a map, got %T", nestedParam)
		} else if nestedField, ok := nestedMap["nested_field"]; !ok || nestedField != float64(42) {
			t.Errorf("Expected 'nested_message_param.nested_field' to be 42, got %v", nestedField)
		}
	}

	// Test unmarshaling message with additional parameters
	jsonData := `{
		"role": "assistant", 
		"content": "I'm an AI", 
		"custom_response_param": "response_value",
		"nested_response_param": {"response_field": 24}
	}`

	var unmarshaledMessage openai.ChatCompletionMessage
	err = json.Unmarshal([]byte(jsonData), &unmarshaledMessage)
	checks.NoError(t, err, "Error unmarshaling message with additional parameters")

	// Check standard fields
	if unmarshaledMessage.Role != "assistant" {
		t.Errorf("Expected role to be 'assistant', got %v", unmarshaledMessage.Role)
	}
	if unmarshaledMessage.Content != "I'm an AI" {
		t.Errorf("Expected content to be 'I'm an AI', got %v", unmarshaledMessage.Content)
	}

	// Check additional parameters
	customResponseValue, exists := unmarshaledMessage.AdditionalParameters["custom_response_param"]
	if !exists {
		t.Error("Expected 'custom_response_param' in unmarshaled message but it wasn't there")
	} else if customResponseValue != "response_value" {
		t.Errorf("Expected 'custom_response_param' to be 'response_value', got %v", customResponseValue)
	}

	nestedResponseParam, exists := unmarshaledMessage.AdditionalParameters["nested_response_param"]
	if !exists {
		t.Error("Expected 'nested_response_param' in unmarshaled message but it wasn't there")
	} else {
		nestedMap, ok := nestedResponseParam.(map[string]interface{})
		if !ok {
			t.Errorf("Expected 'nested_response_param' to be a map, got %T", nestedResponseParam)
		} else if responseField, ok := nestedMap["response_field"]; !ok || responseField != float64(24) {
			t.Errorf("Expected 'nested_response_param.response_field' to be 24, got %v", responseField)
		}
	}

	// Test with MultiContent
	multiContentMessage := openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleUser,
		MultiContent: []openai.ChatMessagePart{
			{
				Type: openai.ChatMessagePartTypeText,
				Text: "Hello with image",
			},
		},
		AdditionalParameters: map[string]interface{}{
			"multi_content_param": "multi_value",
		},
	}

	multiData, err := json.Marshal(multiContentMessage)
	checks.NoError(t, err, "Error marshaling message with MultiContent and additional parameters")

	var parsedMultiData map[string]interface{}
	err = json.Unmarshal(multiData, &parsedMultiData)
	checks.NoError(t, err, "Error parsing marshaled multi-content message")

	// Check additional parameters in multi-content message
	multiValue, exists := parsedMultiData["multi_content_param"]
	if !exists {
		t.Error("Expected 'multi_content_param' in marshaled multi-content message but it wasn't there")
	} else if multiValue != "multi_value" {
		t.Errorf("Expected 'multi_content_param' to be 'multi_value', got %v", multiValue)
	}
}
