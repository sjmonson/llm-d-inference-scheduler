package scorer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// PromptWordBalancerType is the type of the PromptWordBalancer scorer.
	PromptWordBalancerType = "prompt-word-balancer"

	// defaultWordBalancerRequestTimeout defines the default timeout for open requests to be
	// considered stale and removed from the cache.
	defaultWordBalancerRequestTimeout = 2 * time.Minute

	// defaultDecayFactor defines the default decay factor applied to word counts on ResponseComplete.
	// A value of 1.0 means no decay (counts persist), 0.0 means full reset.
	defaultDecayFactor = 1.0
)

// PromptWordBalancerParameters defines the parameters for the
// PromptWordBalancer scorer.
type PromptWordBalancerParameters struct {
	// DecayFactor defines how much to reduce word counts on ResponseComplete.
	// Range: 0.0-1.0. Default: 1.0 (no decay).
	// Example: 0.95 means counts are multiplied by 0.95 on each completion.
	DecayFactor float64 `json:"decayFactor"`

	// RequestTimeout defines the timeout for requests in seconds.
	// Once the request is "in-flight" for this duration, it is considered to
	// be timed out and dropped.
	// This field accepts duration strings like "30s", "1m", "2h".
	RequestTimeout string `json:"requestTimeout"`
}

// wordBalancerRequestEntry represents a single request in the cache
type wordBalancerRequestEntry struct {
	PodNames  []string
	RequestID string
	WordCount int64
	Timestamp time.Time
}

// String returns a string representation of the request entry.
func (r wordBalancerRequestEntry) String() string {
	return fmt.Sprintf("%s:%s (words=%d, at=%s)",
		r.RequestID, strings.Join(r.PodNames, "."), r.WordCount, r.Timestamp.Format(time.RFC3339))
}

// compile-time type assertion
var _ framework.Scorer = &PromptWordBalancer{}
var _ requestcontrol.PreRequest = &PromptWordBalancer{}
var _ requestcontrol.ResponseComplete = &PromptWordBalancer{}

// PromptWordBalancerFactory defines the factory function for the PromptWordBalancer scorer.
func PromptWordBalancerFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	parameters := PromptWordBalancerParameters{
		DecayFactor: defaultDecayFactor,
	}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", PromptWordBalancerType, err)
		}
	}

	// Validate decay factor
	if parameters.DecayFactor < 0.0 || parameters.DecayFactor > 1.0 {
		return nil, fmt.Errorf("decay factor must be between 0.0 and 1.0, got %f", parameters.DecayFactor)
	}

	return NewPromptWordBalancer(handle.Context(), &parameters).WithName(name), nil
}

// NewPromptWordBalancer creates a new PromptWordBalancer scorer.
func NewPromptWordBalancer(ctx context.Context, params *PromptWordBalancerParameters) *PromptWordBalancer {
	requestTimeout := defaultWordBalancerRequestTimeout
	decayFactor := defaultDecayFactor
	logger := log.FromContext(ctx)

	if params != nil {
		if params.RequestTimeout != "" {
			paramsRequestTimeout, err := time.ParseDuration(params.RequestTimeout)
			if err != nil || paramsRequestTimeout <= 0 {
				logger.Error(err, "Invalid request timeout duration, using default request timeout")
			} else {
				requestTimeout = paramsRequestTimeout
				logger.Info("Using request timeout", "requestTimeout", requestTimeout)
			}
		}

		// DecayFactor validation already done in factory
		decayFactor = params.DecayFactor
		logger.Info("Using decay factor", "decayFactor", decayFactor)
	}

	// cache for individual requests with their own TTL
	requestCache := ttlcache.New[string, *wordBalancerRequestEntry](
		ttlcache.WithTTL[string, *wordBalancerRequestEntry](requestTimeout),
		ttlcache.WithDisableTouchOnHit[string, *wordBalancerRequestEntry](),
	)

	scorer := &PromptWordBalancer{
		typedName:     plugins.TypedName{Type: PromptWordBalancerType},
		requestCache:  requestCache,
		podWordCounts: make(map[string]int64),
		decayFactor:   decayFactor,
		mutex:         &sync.RWMutex{},
	}

	// callback to decrement word count when requests expire
	// most requests will be handled in ResponseComplete, but this ensures
	// that we don't leak pod word counts if ResponseComplete is not called
	requestCache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
		item *ttlcache.Item[string, *wordBalancerRequestEntry]) {
		if reason == ttlcache.EvictionReasonExpired {
			entry := item.Value()
			if entry != nil {
				for _, podName := range entry.PodNames {
					scorer.decrementWordCount(podName, entry.WordCount)
				}
			}
		}
	})

	go cleanWordBalancerCachePeriodically(ctx, requestCache, requestTimeout)

	return scorer
}

// PromptWordBalancer keeps track of cumulative prompt word counts
// per pod to enable balanced distribution.
type PromptWordBalancer struct {
	typedName plugins.TypedName

	// requestCache stores individual request entries with unique composite keys (podName.requestID)
	requestCache *ttlcache.Cache[string, *wordBalancerRequestEntry]

	// podWordCounts maintains cumulative word counts per pod
	podWordCounts map[string]int64

	// decayFactor is applied to word counts on ResponseComplete (0.0-1.0)
	decayFactor float64

	mutex *sync.RWMutex
}

// TypedName returns the typed name of the plugin.
func (s *PromptWordBalancer) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *PromptWordBalancer) WithName(name string) *PromptWordBalancer {
	s.typedName.Name = name
	return s
}

// Score scores the given pods based on their cumulative prompt word counts.
// Pods with lower word counts receive higher scores.
// The score is normalized to a range of 0-1.
func (s *PromptWordBalancer) Score(ctx context.Context, _ *types.CycleState, _ *types.LLMRequest,
	pods []types.Pod) map[types.Pod]float64 {
	scoredPods := make(map[string]int64)
	var maxCount int64

	s.mutex.RLock()
	for podName, count := range s.podWordCounts {
		scoredPods[podName] = count
		if count > maxCount {
			maxCount = count
		}
	}
	s.mutex.RUnlock()

	log.FromContext(ctx).V(logutil.DEBUG).Info("Prompt word balancer counts",
		"podWordCounts", scoredPods, "maxCount", maxCount)

	scoredPodsMap := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		podName := pod.GetPod().NamespacedName.String()
		if count, exists := scoredPods[podName]; exists {
			if maxCount == 0 {
				scoredPodsMap[pod] = 1.0 // all at zero means highest score
			} else {
				// Higher word count = lower score
				// score = 1.0 - (count / maxCount)
				scoredPodsMap[pod] = 1.0 - (float64(count) / float64(maxCount))
			}
		} else {
			// New pod gets highest score
			scoredPodsMap[pod] = 1.0
		}
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("Scored pods by word balance", "scores", scoredPodsMap)
	return scoredPodsMap
}

// PreRequest is called before a request is sent to the target pod.
// It extracts the word count from the prompt, increments the pod's
// cumulative word count, and creates a cache entry for tracking.
func (s *PromptWordBalancer) PreRequest(
	ctx context.Context,
	request *types.LLMRequest,
	schedulingResult *types.SchedulingResult,
) {
	debugLogger := log.FromContext(ctx).V(logutil.DEBUG)

	// Extract word count from the request
	wordCount, err := s.extractWordCount(request)
	if err != nil {
		debugLogger.Error(err, "Failed to extract word count, skipping tracking")
		return
	}

	if wordCount <= 0 {
		debugLogger.Info("Word count is zero or negative, skipping tracking", "wordCount", wordCount)
		return
	}

	podNames := make([]string, 0, len(schedulingResult.ProfileResults))
	for profileName, profileResult := range schedulingResult.ProfileResults {
		if profileResult == nil || len(profileResult.TargetPods) == 0 {
			continue
		}

		podName := profileResult.TargetPods[0].GetPod().NamespacedName.String()
		podNames = append(podNames, podName)
		s.incrementWordCount(podName, wordCount)
		debugLogger.Info(
			"Added request word count to pod",
			"requestId", request.RequestId,
			"podName", podName,
			"profileName", profileName,
			"wordCount", wordCount,
		)
	}

	// add to request cache
	s.requestCache.Set(request.RequestId, &wordBalancerRequestEntry{
		PodNames:  podNames,
		RequestID: request.RequestId,
		WordCount: wordCount,
		Timestamp: time.Now(),
	}, 0) // Use default TTL
}

// ResponseComplete is called after a response is sent to the client.
// It removes the request entry from the cache and optionally applies
// decay to the word counts of involved pods.
func (s *PromptWordBalancer) ResponseComplete(
	ctx context.Context,
	request *types.LLMRequest,
	_ *requestcontrol.Response,
	targetPod *backend.Pod,
) {
	debugLogger := log.FromContext(ctx).V(logutil.DEBUG).WithName("PromptWordBalancer.ResponseComplete")
	if targetPod == nil {
		debugLogger.Info("Skipping ResponseComplete because targetPod is nil")
		return
	}

	item, found := s.requestCache.GetAndDelete(request.RequestId)
	if !found {
		debugLogger.Info("Request not found in cache", "requestId", request.RequestId)
		return
	}

	entry := item.Value()
	if entry == nil {
		debugLogger.Info("Request entry value is nil", "requestId", request.RequestId)
		return
	}

	// Apply decay factor if configured (decay < 1.0)
	if s.decayFactor > 0 && s.decayFactor < 1.0 {
		for _, podName := range entry.PodNames {
			s.applyDecayToPod(podName, s.decayFactor)
		}
		debugLogger.Info("Applied decay to pods", "requestEntry", entry.String(), "decayFactor", s.decayFactor)
	} else if s.decayFactor == 0.0 {
		// Full reset: decrement the original word count
		for _, podName := range entry.PodNames {
			s.decrementWordCount(podName, entry.WordCount)
		}
		debugLogger.Info("Removed request word count from pods (full reset)", "requestEntry", entry.String())
	} else {
		// No decay (decayFactor == 1.0): counts persist
		debugLogger.Info("Request completed, no decay applied", "requestEntry", entry.String())
	}
}

// extractWordCount extracts the word count from the request prompt.
// For Completions: counts words in the prompt string.
// For ChatCompletions: counts words across all message contents.
func (s *PromptWordBalancer) extractWordCount(request *types.LLMRequest) (int64, error) {
	if request == nil || request.Body == nil {
		return 0, errors.New("request or body is nil")
	}

	var totalWords int64

	// Handle chat completions
	if request.Body.ChatCompletions != nil {
		if request.Body.Completions != nil {
			// Defensive: both present, prioritize chat (shouldn't happen)
			log.Log.V(logutil.DEBUG).Info("Both ChatCompletions and Completions present; using ChatCompletions")
		}

		for _, msg := range request.Body.ChatCompletions.Messages {
			totalWords += int64(len(strings.Fields(msg.Content.Raw)))
		}
		return totalWords, nil
	}

	// Handle text completions
	if request.Body.Completions != nil {
		totalWords = int64(len(strings.Fields(request.Body.Completions.Prompt)))
		return totalWords, nil
	}

	return 0, errors.New("no valid prompt found in request")
}

// incrementWordCount increments the word count for a pod.
func (s *PromptWordBalancer) incrementWordCount(podName string, wordCount int64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.podWordCounts[podName] += wordCount
}

// decrementWordCount decrements the word count for a pod and removes
// the entry if count reaches zero or below.
func (s *PromptWordBalancer) decrementWordCount(podName string, wordCount int64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if count, exists := s.podWordCounts[podName]; exists {
		newCount := count - wordCount
		if newCount <= 0 {
			delete(s.podWordCounts, podName)
		} else {
			s.podWordCounts[podName] = newCount
		}
	}
}

// applyDecayToPod applies a decay factor to the word count of a pod.
func (s *PromptWordBalancer) applyDecayToPod(podName string, factor float64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if count, exists := s.podWordCounts[podName]; exists {
		newCount := int64(float64(count) * factor)
		if newCount <= 0 {
			delete(s.podWordCounts, podName)
		} else {
			s.podWordCounts[podName] = newCount
		}
	}
}

// cleanWordBalancerCachePeriodically periodically cleans up expired entries from the cache.
func cleanWordBalancerCachePeriodically[K comparable, V any](ctx context.Context, cache *ttlcache.Cache[K, V], requestTimeout time.Duration) {
	ticker := time.NewTicker(requestTimeout)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cache.DeleteExpired()
		}
	}
}
