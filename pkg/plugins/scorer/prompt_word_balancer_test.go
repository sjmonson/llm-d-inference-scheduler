package scorer

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

// Test helper functions for PromptWordBalancer

func newWordBalancerTestPod(name string) *types.PodMetrics {
	return &types.PodMetrics{
		Pod:          &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		MetricsState: &backendmetrics.MetricsState{},
	}
}

func newCompletionsRequest(id, prompt string) *types.LLMRequest {
	return &types.LLMRequest{
		RequestId: id,
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: prompt,
			},
		},
	}
}

func newChatCompletionsRequest(id string, messages []types.Message) *types.LLMRequest {
	return &types.LLMRequest{
		RequestId: id,
		Body: &types.LLMRequestBody{
			ChatCompletions: &types.ChatCompletionsRequest{
				Messages: messages,
			},
		},
	}
}

func (s *PromptWordBalancer) getWordCount(podName string) int64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return s.podWordCounts[podName]
}

func (s *PromptWordBalancer) hasWordCount(podName string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	_, exists := s.podWordCounts[podName]
	return exists
}

// Factory Tests

func TestPromptWordBalancer_ParameterParsing(t *testing.T) {
	ctx := utils.NewTestContext(t)

	t.Run("Default parameters", func(t *testing.T) {
		scorer := NewPromptWordBalancer(ctx, nil)
		require.NotNil(t, scorer)
	})

	t.Run("Custom request timeout", func(t *testing.T) {
		params := &PromptWordBalancerParameters{
			RequestTimeout: "5m",
		}
		scorer := NewPromptWordBalancer(ctx, params)
		assert.NotNil(t, scorer)
	})

	t.Run("Invalid request timeout uses default", func(t *testing.T) {
		params := &PromptWordBalancerParameters{
			RequestTimeout: "invalid",
		}
		scorer := NewPromptWordBalancer(ctx, params)
		assert.NotNil(t, scorer)
	})
}

// Word Extraction Tests

func TestPromptWordBalancer_ExtractWordCount(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewPromptWordBalancer(ctx, nil)

	t.Run("Completions with simple prompt", func(t *testing.T) {
		request := newCompletionsRequest("test-1", "Hello world from AI")
		wordCount, err := scorer.extractWordCount(request)
		require.NoError(t, err)
		assert.Equal(t, int64(4), wordCount) // "Hello", "world", "from", "AI"
	})

	t.Run("Completions with multi-line prompt", func(t *testing.T) {
		prompt := "This is line one.\nThis is line two.\nThis is line three."
		request := newCompletionsRequest("test-2", prompt)
		wordCount, err := scorer.extractWordCount(request)
		require.NoError(t, err)
		assert.Equal(t, int64(12), wordCount) // 12 total words
	})

	t.Run("Completions with extra whitespace", func(t *testing.T) {
		request := newCompletionsRequest("test-3", "  Multiple   spaces    between   words  ")
		wordCount, err := scorer.extractWordCount(request)
		require.NoError(t, err)
		assert.Equal(t, int64(4), wordCount) // strings.Fields handles extra whitespace
	})

	t.Run("ChatCompletions with single message", func(t *testing.T) {
		request := newChatCompletionsRequest("test-4", []types.Message{
			{Role: "user", Content: types.Content{Raw: "Hello how are you"}},
		})
		wordCount, err := scorer.extractWordCount(request)
		require.NoError(t, err)
		assert.Equal(t, int64(4), wordCount)
	})

	t.Run("ChatCompletions with multiple messages", func(t *testing.T) {
		request := newChatCompletionsRequest("test-5", []types.Message{
			{Role: "user", Content: types.Content{Raw: "Hello world"}},
			{Role: "assistant", Content: types.Content{Raw: "Hi there"}},
			{Role: "user", Content: types.Content{Raw: "How are you today"}},
		})
		wordCount, err := scorer.extractWordCount(request)
		require.NoError(t, err)
		assert.Equal(t, int64(8), wordCount) // 2 + 2 + 4 = 8
	})

	t.Run("ChatCompletions with empty messages", func(t *testing.T) {
		request := newChatCompletionsRequest("test-6", []types.Message{
			{Role: "user", Content: types.Content{Raw: ""}},
		})
		wordCount, err := scorer.extractWordCount(request)
		require.NoError(t, err)
		assert.Equal(t, int64(0), wordCount)
	})

	t.Run("Nil request", func(t *testing.T) {
		_, err := scorer.extractWordCount(nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "request or body is nil")
	})

	t.Run("Nil body", func(t *testing.T) {
		request := &types.LLMRequest{RequestId: "test-7", Body: nil}
		_, err := scorer.extractWordCount(request)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "request or body is nil")
	})

	t.Run("No valid prompt", func(t *testing.T) {
		request := &types.LLMRequest{
			RequestId: "test-8",
			Body:      &types.LLMRequestBody{}, // Empty body
		}
		_, err := scorer.extractWordCount(request)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no valid prompt found")
	})
}

// Scoring Tests

func TestPromptWordBalancer_Score(t *testing.T) {
	podA := newWordBalancerTestPod("pod-a")
	podB := newWordBalancerTestPod("pod-b")
	podC := newWordBalancerTestPod("pod-c")

	tests := []struct {
		name       string
		setupCache func(*PromptWordBalancer)
		input      []types.Pod
		wantScores map[types.Pod]float64
	}{
		{
			name: "no pods in cache - all score 1.0",
			setupCache: func(_ *PromptWordBalancer) {
				// Cache is empty
			},
			input: []types.Pod{podA, podB, podC},
			wantScores: map[types.Pod]float64{
				podA: 1.0,
				podB: 1.0,
				podC: 1.0,
			},
		},
		{
			name: "all pods with different word counts",
			setupCache: func(s *PromptWordBalancer) {
				s.mutex.Lock()
				s.podWordCounts["default/pod-a"] = 100
				s.podWordCounts["default/pod-b"] = 0
				s.podWordCounts["default/pod-c"] = 200
				s.mutex.Unlock()
			},
			input: []types.Pod{podA, podB, podC},
			wantScores: map[types.Pod]float64{
				podA: 0.5, // 1.0 - (100/200)
				podB: 1.0, // 1.0 - (0/200)
				podC: 0.0, // 1.0 - (200/200)
			},
		},
		{
			name: "some pods in cache",
			setupCache: func(s *PromptWordBalancer) {
				s.mutex.Lock()
				s.podWordCounts["default/pod-a"] = 150
				s.podWordCounts["default/pod-c"] = 50
				// pod-b not in cache
				s.mutex.Unlock()
			},
			input: []types.Pod{podA, podB, podC},
			wantScores: map[types.Pod]float64{
				podA: 0.0,                  // 1.0 - (150/150) = 0
				podB: 1.0,                  // new pod
				podC: 1.0 - (50.0 / 150.0), // ~0.667
			},
		},
		{
			name: "all pods at zero words",
			setupCache: func(s *PromptWordBalancer) {
				s.mutex.Lock()
				s.podWordCounts["default/pod-a"] = 0
				s.podWordCounts["default/pod-b"] = 0
				s.mutex.Unlock()
			},
			input: []types.Pod{podA, podB},
			wantScores: map[types.Pod]float64{
				podA: 1.0,
				podB: 1.0,
			},
		},
		{
			name: "equal word counts",
			setupCache: func(s *PromptWordBalancer) {
				s.mutex.Lock()
				s.podWordCounts["default/pod-a"] = 100
				s.podWordCounts["default/pod-b"] = 100
				s.podWordCounts["default/pod-c"] = 100
				s.mutex.Unlock()
			},
			input: []types.Pod{podA, podB, podC},
			wantScores: map[types.Pod]float64{
				podA: 0.0,
				podB: 0.0,
				podC: 0.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)

			scorer := NewPromptWordBalancer(ctx, nil)
			tt.setupCache(scorer)

			got := scorer.Score(ctx, nil, nil, tt.input)

			// Compare with floating point tolerance
			assert.Equal(t, len(tt.wantScores), len(got), "Number of scored pods should match")
			for pod, expectedScore := range tt.wantScores {
				actualScore, exists := got[pod]
				assert.True(t, exists, "Pod should be in results")
				assert.InDelta(t, expectedScore, actualScore, 0.0001, "Score should match within tolerance")
			}
		})
	}
}

// PreRequest Tests

func TestPromptWordBalancer_PreRequest(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewPromptWordBalancer(ctx, nil)

	podA := newWordBalancerTestPod("pod-a")
	podB := newWordBalancerTestPod("pod-b")

	testProfile := "test-profile"

	t.Run("First request to single pod", func(t *testing.T) {
		request := newCompletionsRequest("test-request-1", "Hello world this is a test")
		schedulingResult := &types.SchedulingResult{
			ProfileResults: map[string]*types.ProfileRunResult{
				testProfile: {TargetPods: []types.Pod{podA}},
			},
		}

		scorer.PreRequest(ctx, request, schedulingResult)

		assert.True(t, scorer.requestCache.Has(request.RequestId), "Expected request to be in cache")
		assert.Equal(t, int64(6), scorer.getWordCount(podA.GetPod().NamespacedName.String()))
	})

	t.Run("Second request to same pod accumulates words", func(t *testing.T) {
		request := newCompletionsRequest("test-request-2", "Another short prompt")
		schedulingResult := &types.SchedulingResult{
			ProfileResults: map[string]*types.ProfileRunResult{
				testProfile: {TargetPods: []types.Pod{podA}},
			},
		}

		scorer.PreRequest(ctx, request, schedulingResult)

		assert.True(t, scorer.requestCache.Has(request.RequestId))
		// Previous: 6, Current: 3, Total: 9
		assert.Equal(t, int64(9), scorer.getWordCount(podA.GetPod().NamespacedName.String()))
	})

	t.Run("Request to multiple pods (P/D disaggregation)", func(t *testing.T) {
		request := newCompletionsRequest("test-request-3", "Prefill and decode test")
		schedulingResult := &types.SchedulingResult{
			ProfileResults: map[string]*types.ProfileRunResult{
				"prefill": {TargetPods: []types.Pod{podA}},
				"decode":  {TargetPods: []types.Pod{podB}},
			},
		}

		scorer.PreRequest(ctx, request, schedulingResult)

		assert.True(t, scorer.requestCache.Has(request.RequestId))
		// Both pods should have accumulated the word count
		// podA was at 9, now 9 + 4 = 13
		assert.Equal(t, int64(13), scorer.getWordCount(podA.GetPod().NamespacedName.String()))
		assert.Equal(t, int64(4), scorer.getWordCount(podB.GetPod().NamespacedName.String()))
	})

	t.Run("Request with extraction error is skipped", func(t *testing.T) {
		request := &types.LLMRequest{RequestId: "error-request", Body: nil}
		schedulingResult := &types.SchedulingResult{
			ProfileResults: map[string]*types.ProfileRunResult{
				testProfile: {TargetPods: []types.Pod{podA}},
			},
		}

		initialCount := scorer.getWordCount(podA.GetPod().NamespacedName.String())
		scorer.PreRequest(ctx, request, schedulingResult)

		// Should not be added to cache
		assert.False(t, scorer.requestCache.Has(request.RequestId))
		// Word count should not change
		assert.Equal(t, initialCount, scorer.getWordCount(podA.GetPod().NamespacedName.String()))
	})

	t.Run("Request with zero word count is skipped", func(t *testing.T) {
		request := newCompletionsRequest("zero-request", "")
		schedulingResult := &types.SchedulingResult{
			ProfileResults: map[string]*types.ProfileRunResult{
				testProfile: {TargetPods: []types.Pod{podA}},
			},
		}

		initialCount := scorer.getWordCount(podA.GetPod().NamespacedName.String())
		scorer.PreRequest(ctx, request, schedulingResult)

		// Should not be added to cache
		assert.False(t, scorer.requestCache.Has(request.RequestId))
		// Word count should not change
		assert.Equal(t, initialCount, scorer.getWordCount(podA.GetPod().NamespacedName.String()))
	})
}

// ResponseComplete Tests

func TestPromptWordBalancer_ResponseComplete(t *testing.T) {
	t.Run("Word count removed on completion", func(t *testing.T) {
		ctx := utils.NewTestContext(t)
		scorer := NewPromptWordBalancer(ctx, nil)

		podA := newWordBalancerTestPod("pod-a")
		request := newCompletionsRequest("test-request-1", "This is a test prompt")

		// Setup: add request
		schedulingResult := &types.SchedulingResult{
			ProfileResults: map[string]*types.ProfileRunResult{
				"test-profile": {TargetPods: []types.Pod{podA}},
			},
		}
		scorer.PreRequest(ctx, request, schedulingResult)
		assert.Equal(t, int64(5), scorer.getWordCount("default/pod-a"))

		// Complete request
		scorer.ResponseComplete(ctx, request, &requestcontrol.Response{}, podA.GetPod())

		// Word count should be removed
		assert.False(t, scorer.requestCache.Has(request.RequestId), "Request should be removed from cache")
		assert.False(t, scorer.hasWordCount("default/pod-a"), "Word count should be removed on completion")
	})

	t.Run("Multiple pods (P/D disaggregation)", func(t *testing.T) {
		ctx := utils.NewTestContext(t)
		scorer := NewPromptWordBalancer(ctx, nil)

		podA := newWordBalancerTestPod("pod-a")
		podB := newWordBalancerTestPod("pod-b")
		request := newCompletionsRequest("test-request-2", "Five word test prompt here")

		// Setup: add request to both pods
		schedulingResult := &types.SchedulingResult{
			ProfileResults: map[string]*types.ProfileRunResult{
				"prefill": {TargetPods: []types.Pod{podA}},
				"decode":  {TargetPods: []types.Pod{podB}},
			},
		}
		scorer.PreRequest(ctx, request, schedulingResult)
		assert.Equal(t, int64(5), scorer.getWordCount("default/pod-a"))
		assert.Equal(t, int64(5), scorer.getWordCount("default/pod-b"))

		// Complete request
		scorer.ResponseComplete(ctx, request, &requestcontrol.Response{}, podA.GetPod())

		// Both pods should have word counts removed
		assert.False(t, scorer.requestCache.Has(request.RequestId))
		assert.False(t, scorer.hasWordCount("default/pod-a"))
		assert.False(t, scorer.hasWordCount("default/pod-b"))
	})

	t.Run("Request not in cache (already completed or TTL expired)", func(t *testing.T) {
		ctx := utils.NewTestContext(t)
		scorer := NewPromptWordBalancer(ctx, nil)

		podA := newWordBalancerTestPod("pod-a")
		request := newCompletionsRequest("unknown-request", "Test")

		// Call ResponseComplete without PreRequest
		scorer.ResponseComplete(ctx, request, &requestcontrol.Response{}, podA.GetPod())

		// Should handle gracefully without errors
		assert.False(t, scorer.requestCache.Has(request.RequestId))
	})

	t.Run("Nil targetPod is skipped", func(t *testing.T) {
		ctx := utils.NewTestContext(t)
		scorer := NewPromptWordBalancer(ctx, nil)

		request := newCompletionsRequest("test-request-3", "Test prompt")

		// Call ResponseComplete with nil targetPod
		scorer.ResponseComplete(ctx, request, &requestcontrol.Response{}, nil)

		// Should return early without errors
	})
}

// TTL and Cache Tests

func TestPromptWordBalancer_TTLExpiration(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// Use very short timeout for test
	params := &PromptWordBalancerParameters{
		RequestTimeout: "1s",
	}
	scorer := NewPromptWordBalancer(ctx, params)

	podA := newWordBalancerTestPod("pod-a")
	request := newCompletionsRequest("ttl-test-request", "Word word word word word")
	schedulingResult := &types.SchedulingResult{
		ProfileResults: map[string]*types.ProfileRunResult{
			"test-profile": {TargetPods: []types.Pod{podA}},
		},
	}

	// Add request
	scorer.PreRequest(ctx, request, schedulingResult)

	// Verify request is added
	require.Equal(t, int64(5), scorer.getWordCount("default/pod-a"))

	// Wait for TTL expiration
	time.Sleep(2 * time.Second)

	// Trigger cleanup
	scorer.requestCache.DeleteExpired()

	// Check that pod word count is decremented due to TTL expiration
	assert.False(t, scorer.hasWordCount("default/pod-a"),
		"Word count should be removed after TTL expiration")
}

// Metadata and Plugin Interface Tests

func TestPromptWordBalancer_TypedName(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewPromptWordBalancer(ctx, nil)

	assert.Equal(t, PromptWordBalancerType, scorer.TypedName().Type)
}

func TestPromptWordBalancer_WithName(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewPromptWordBalancer(ctx, nil)
	testName := "custom-word-balancer"

	scorer = scorer.WithName(testName)

	assert.Equal(t, testName, scorer.TypedName().Name)
}

// Thread Safety Test

func TestPromptWordBalancer_ConcurrentAccess(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewPromptWordBalancer(ctx, nil)

	podA := newWordBalancerTestPod("pod-a")
	podB := newWordBalancerTestPod("pod-b")

	// This test should be run with -race flag to detect race conditions
	// go test -race ./pkg/plugins/scorer

	done := make(chan bool)

	// Concurrent PreRequest calls
	for i := 0; i < 10; i++ {
		go func(id int) {
			request := newCompletionsRequest(fmt.Sprintf("concurrent-%d", id), "concurrent test words here")
			var pod types.Pod
			if id%2 == 0 {
				pod = podA
			} else {
				pod = podB
			}
			schedulingResult := &types.SchedulingResult{
				ProfileResults: map[string]*types.ProfileRunResult{
					"test": {TargetPods: []types.Pod{pod}},
				},
			}
			scorer.PreRequest(ctx, request, schedulingResult)
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Verify word counts are correct (5 requests * 4 words each = 20 per pod)
	assert.Equal(t, int64(20), scorer.getWordCount("default/pod-a"))
	assert.Equal(t, int64(20), scorer.getWordCount("default/pod-b"))
}
