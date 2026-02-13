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
	// PromptWordScorerType is the type of the PromptWordScorer scorer.
	PromptWordScorerType = "prompt-word-scorer"

	// defaultWordScorerRequestTimeout defines the default timeout for open requests to be
	// considered stale and removed from the cache.
	defaultWordScorerRequestTimeout = 2 * time.Minute

	// promptWordScorerLogLevel defines the verbosity level for PromptWordScorer logs.
	// Set to logutil.DEFAULT to always show these logs at normal verbosity.
	promptWordScorerLogLevel = logutil.DEBUG
)

// PromptWordScorerParameters defines the parameters for the
// PromptWordScorer scorer.
type PromptWordScorerParameters struct {
	// RequestTimeout defines the timeout for requests in seconds.
	// Once the request is "in-flight" for this duration, it is considered to
	// be timed out and dropped.
	// This field accepts duration strings like "30s", "1m", "2h".
	RequestTimeout string `json:"requestTimeout"`
}

// wordScorerRequestEntry represents a single request in the cache
type wordScorerRequestEntry struct {
	PodNames  []string
	RequestID string
	WordCount int64
	Timestamp time.Time
}

// String returns a string representation of the request entry.
func (r wordScorerRequestEntry) String() string {
	return fmt.Sprintf("%s:%s (words=%d, at=%s)",
		r.RequestID, strings.Join(r.PodNames, "."), r.WordCount, r.Timestamp.Format(time.RFC3339))
}

// compile-time type assertion
var _ framework.Scorer = &PromptWordScorer{}
var _ requestcontrol.PreRequest = &PromptWordScorer{}
var _ requestcontrol.ResponseComplete = &PromptWordScorer{}

// PromptWordScorerFactory defines the factory function for the PromptWordScorer scorer.
func PromptWordScorerFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	parameters := PromptWordScorerParameters{}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", PromptWordScorerType, err)
		}
	}

	return NewPromptWordScorer(handle.Context(), &parameters).WithName(name), nil
}

// NewPromptWordScorer creates a new PromptWordScorer scorer.
func NewPromptWordScorer(ctx context.Context, params *PromptWordScorerParameters) *PromptWordScorer {
	requestTimeout := defaultWordScorerRequestTimeout
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
	}

	// cache for individual requests with their own TTL
	requestCache := ttlcache.New[string, *wordScorerRequestEntry](
		ttlcache.WithTTL[string, *wordScorerRequestEntry](requestTimeout),
		ttlcache.WithDisableTouchOnHit[string, *wordScorerRequestEntry](),
	)

	scorer := &PromptWordScorer{
		typedName:     plugins.TypedName{Type: PromptWordScorerType},
		requestCache:  requestCache,
		podWordCounts: make(map[string]int64),
		mutex:         &sync.RWMutex{},
	}

	// callback to decrement word count when requests expire
	// most requests will be handled in ResponseComplete, but this ensures
	// that we don't leak pod word counts if ResponseComplete is not called
	requestCache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
		item *ttlcache.Item[string, *wordScorerRequestEntry]) {
		if reason == ttlcache.EvictionReasonExpired {
			entry := item.Value()
			if entry != nil {
				for _, podName := range entry.PodNames {
					scorer.decrementWordCount(podName, entry.WordCount)
				}
			}
		}
	})

	go cleanWordScorerCachePeriodically(ctx, requestCache, requestTimeout)

	return scorer
}

// PromptWordScorer keeps track of in-flight prompt word counts
// per pod to enable scoring based on load.
type PromptWordScorer struct {
	typedName plugins.TypedName

	// requestCache stores individual request entries with unique composite keys (podName.requestID)
	requestCache *ttlcache.Cache[string, *wordScorerRequestEntry]

	// podWordCounts maintains in-flight word counts per pod
	podWordCounts map[string]int64

	mutex *sync.RWMutex
}

// TypedName returns the typed name of the plugin.
func (s *PromptWordScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *PromptWordScorer) WithName(name string) *PromptWordScorer {
	s.typedName.Name = name
	return s
}

// Score scores the given pods based on their cumulative prompt word counts.
// Pods with lower word counts receive higher scores.
// The score is normalized to a range of 0-1.
func (s *PromptWordScorer) Score(ctx context.Context, _ *types.CycleState, _ *types.LLMRequest,
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

	log.FromContext(ctx).V(promptWordScorerLogLevel).Info("Prompt word scorer counts",
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

	log.FromContext(ctx).V(promptWordScorerLogLevel).Info("Scored pods by word count", "scores", scoredPodsMap)
	return scoredPodsMap
}

// PreRequest is called before a request is sent to the target pod.
// It extracts the word count from the prompt, increments the pod's
// cumulative word count, and creates a cache entry for tracking.
func (s *PromptWordScorer) PreRequest(
	ctx context.Context,
	request *types.LLMRequest,
	schedulingResult *types.SchedulingResult,
) {
	debugLogger := log.FromContext(ctx).V(promptWordScorerLogLevel)

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
	s.requestCache.Set(request.RequestId, &wordScorerRequestEntry{
		PodNames:  podNames,
		RequestID: request.RequestId,
		WordCount: wordCount,
		Timestamp: time.Now(),
	}, 0) // Use default TTL
}

// ResponseComplete is called after a response is sent to the client.
// It removes the request entry from the cache and decrements the word counts
// of involved pods since the request is no longer in-flight.
func (s *PromptWordScorer) ResponseComplete(
	ctx context.Context,
	request *types.LLMRequest,
	_ *requestcontrol.Response,
	targetPod *backend.Pod,
) {
	debugLogger := log.FromContext(ctx).V(promptWordScorerLogLevel).WithName("PromptWordScorer.ResponseComplete")
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

	// Remove word count from all involved pods
	for _, podName := range entry.PodNames {
		s.decrementWordCount(podName, entry.WordCount)
	}
	debugLogger.Info("Removed request word count from pods", "requestEntry", entry.String())
}

// extractWordCount extracts the word count from the request prompt.
// For Completions: counts words in the prompt string.
// For ChatCompletions: counts words across all message contents.
func (s *PromptWordScorer) extractWordCount(request *types.LLMRequest) (int64, error) {
	if request == nil || request.Body == nil {
		return 0, errors.New("request or body is nil")
	}

	var totalWords int64

	// Handle chat completions
	if request.Body.ChatCompletions != nil {
		if request.Body.Completions != nil {
			// Defensive: both present, prioritize chat (shouldn't happen)
			log.Log.V(promptWordScorerLogLevel).Info("Both ChatCompletions and Completions present; using ChatCompletions")
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
func (s *PromptWordScorer) incrementWordCount(podName string, wordCount int64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.podWordCounts[podName] += wordCount
}

// decrementWordCount decrements the word count for a pod and removes
// the entry if count reaches zero or below.
func (s *PromptWordScorer) decrementWordCount(podName string, wordCount int64) {
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

// cleanWordScorerCachePeriodically periodically cleans up expired entries from the cache.
func cleanWordScorerCachePeriodically[K comparable, V any](ctx context.Context, cache *ttlcache.Cache[K, V], requestTimeout time.Duration) {
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
