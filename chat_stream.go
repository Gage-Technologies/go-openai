package openai

import (
	"context"
	"encoding/json"
	"net/http"
)

type ChatCompletionStreamChoiceDelta struct {
	Content      string        `json:"content,omitempty"`
	Role         string        `json:"role,omitempty"`
	FunctionCall *FunctionCall `json:"function_call,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	Refusal      string        `json:"refusal,omitempty"`

	// This property is used for the "reasoning" feature supported by deepseek-reasoner
	// which is not in the official documentation.
	// the doc from deepseek:
	// - https://api-docs.deepseek.com/api/create-chat-completion#responses
	ReasoningContent string `json:"reasoning_content,omitempty"`

	// AdditionalParameters contains any additional parameters returned by the API server
	// that are not explicitly defined in this struct
	AdditionalParameters map[string]interface{} `json:"-"`
}

// UnmarshalJSON provides a custom unmarshaller for ChatCompletionStreamChoiceDelta to capture additional fields
func (d *ChatCompletionStreamChoiceDelta) UnmarshalJSON(data []byte) error {
	type Alias ChatCompletionStreamChoiceDelta
	aux := struct {
		*Alias
	}{
		Alias: (*Alias)(d),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	// Capture all fields in a map
	var mapData map[string]interface{}
	if err := json.Unmarshal(data, &mapData); err != nil {
		return err
	}

	// Remove the standard fields from the map to identify additional fields
	delete(mapData, "content")
	delete(mapData, "role")
	delete(mapData, "function_call")
	delete(mapData, "tool_calls")
	delete(mapData, "refusal")
	delete(mapData, "reasoning_content")

	// Store remaining fields in AdditionalParameters
	if len(mapData) > 0 {
		d.AdditionalParameters = mapData
	}

	return nil
}

type ChatCompletionStreamChoiceLogprobs struct {
	Content []ChatCompletionTokenLogprob `json:"content,omitempty"`
	Refusal []ChatCompletionTokenLogprob `json:"refusal,omitempty"`
}

type ChatCompletionTokenLogprob struct {
	Token       string                                 `json:"token"`
	Bytes       []int64                                `json:"bytes,omitempty"`
	Logprob     float64                                `json:"logprob,omitempty"`
	TopLogprobs []ChatCompletionTokenLogprobTopLogprob `json:"top_logprobs"`
}

type ChatCompletionTokenLogprobTopLogprob struct {
	Token   string  `json:"token"`
	Bytes   []int64 `json:"bytes"`
	Logprob float64 `json:"logprob"`
}

type ChatCompletionStreamChoice struct {
	Index                int                                 `json:"index"`
	Delta                ChatCompletionStreamChoiceDelta     `json:"delta"`
	Logprobs             *ChatCompletionStreamChoiceLogprobs `json:"logprobs,omitempty"`
	FinishReason         FinishReason                        `json:"finish_reason"`
	ContentFilterResults ContentFilterResults                `json:"content_filter_results,omitempty"`
}

type PromptFilterResult struct {
	Index                int                  `json:"index"`
	ContentFilterResults ContentFilterResults `json:"content_filter_results,omitempty"`
}

type ChatCompletionStreamResponse struct {
	ID                  string                       `json:"id"`
	Object              string                       `json:"object"`
	Created             int64                        `json:"created"`
	Model               string                       `json:"model"`
	Choices             []ChatCompletionStreamChoice `json:"choices"`
	SystemFingerprint   string                       `json:"system_fingerprint"`
	PromptAnnotations   []PromptAnnotation           `json:"prompt_annotations,omitempty"`
	PromptFilterResults []PromptFilterResult         `json:"prompt_filter_results,omitempty"`
	// An optional field that will only be present when you set stream_options: {"include_usage": true} in your request.
	// When present, it contains a null value except for the last chunk which contains the token usage statistics
	// for the entire request.
	Usage *Usage `json:"usage,omitempty"`
	// AdditionalParameters contains any additional parameters returned by the API server
	// that are not explicitly defined in this struct
	AdditionalParameters map[string]interface{} `json:"-"`
}

// UnmarshalJSON provides a custom unmarshaller for ChatCompletionStreamResponse to capture additional fields
func (r *ChatCompletionStreamResponse) UnmarshalJSON(data []byte) error {
	type Alias ChatCompletionStreamResponse
	aux := struct {
		*Alias
	}{
		Alias: (*Alias)(r),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	// Capture all fields in a map
	var mapData map[string]interface{}
	if err := json.Unmarshal(data, &mapData); err != nil {
		return err
	}

	// Remove the standard fields from the map to identify additional fields
	delete(mapData, "id")
	delete(mapData, "object")
	delete(mapData, "created")
	delete(mapData, "model")
	delete(mapData, "choices")
	delete(mapData, "system_fingerprint")
	delete(mapData, "prompt_annotations")
	delete(mapData, "prompt_filter_results")
	delete(mapData, "usage")

	// Store remaining fields in AdditionalParameters
	if len(mapData) > 0 {
		r.AdditionalParameters = mapData
	}

	return nil
}

// ChatCompletionStream
// Note: Perhaps it is more elegant to abstract Stream using generics.
type ChatCompletionStream struct {
	*streamReader[ChatCompletionStreamResponse]
}

// CreateChatCompletionStream — API call to create a chat completion w/ streaming
// support. It sets whether to stream back partial progress. If set, tokens will be
// sent as data-only server-sent events as they become available, with the
// stream terminated by a data: [DONE] message.
func (c *Client) CreateChatCompletionStream(
	ctx context.Context,
	request ChatCompletionRequest,
) (stream *ChatCompletionStream, err error) {
	urlSuffix := chatCompletionsSuffix
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrChatCompletionInvalidModel
		return
	}

	request.Stream = true
	reasoningValidator := NewReasoningValidator()
	if err = reasoningValidator.Validate(request); err != nil {
		return
	}

	req, err := c.newRequest(
		ctx,
		http.MethodPost,
		c.fullURL(urlSuffix, withModel(request.Model)),
		withBody(request),
	)
	if err != nil {
		return nil, err
	}

	resp, err := sendRequestStream[ChatCompletionStreamResponse](c, req)
	if err != nil {
		return
	}
	stream = &ChatCompletionStream{
		streamReader: resp,
	}
	return
}
