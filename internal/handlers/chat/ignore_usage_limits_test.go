package chat

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"9router/proxy/internal/models"
	"9router/proxy/internal/providers"
)

func TestIgnoreUsageLimitsGate(t *testing.T) {
	tests := []struct {
		name       string
		providerID string
		flag       int
		want       bool
	}{
		{"flagged compatible node", "openai-compatible-v", 1, true},
		{"flagged anthropic node", "anthropic-compatible-x", 1, true},
		{"flagged embedding node", "custom-embedding-emb", 1, true},
		{"flagged oauth provider is inert", "antigravity", 1, false},
		{"flagged registry provider is inert", "claude", 1, false},
		{"flag off on compatible node", "openai-compatible-v", 0, false},
		{"case-insensitive compatible prefix", "OpenAI-Compatible-v", 1, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := providers.IgnoreUsageLimitsActive(tt.providerID, tt.flag); got != tt.want {
				t.Errorf("flagsActive(%q, %d) = %v, want %v", tt.providerID, tt.flag, got, tt.want)
			}
		})
	}
}

func TestIsUsageLimitRefusal(t *testing.T) {
	tests := []struct {
		name string
		ue   *upstreamError
		want bool
	}{
		{"402 always fallback", &upstreamError{StatusCode: http.StatusPaymentRequired, Body: []byte(`{"error":{"message":"insufficient balance"}}`)}, true},
		{"429 quota", &upstreamError{StatusCode: http.StatusTooManyRequests, Body: []byte(`{"error":{"message":"rate limit exceeded"}}`)}, true},
		{"429 unmatched client 4xx is still fallback", &upstreamError{StatusCode: http.StatusTooManyRequests, Body: []byte(`{"error":{"message":"random 429"}}`)}, true},
		{"403 is not a usage-limit refusal", &upstreamError{StatusCode: http.StatusForbidden, Body: []byte(`{"error":{"message":"forbidden"}}`)}, false},
		{"500 is not a usage-limit refusal", &upstreamError{StatusCode: http.StatusInternalServerError, Body: []byte(`{"error":{"message":"boom"}}`)}, false},
		{"nil is not a usage-limit refusal", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isUsageLimitRefusal(tt.ue); got != tt.want {
				t.Errorf("isUsageLimitRefusal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func connPointer(id string, flag int) *models.ProviderConnection {
	return &models.ProviderConnection{ID: id, Provider: "openai-compatible-v", IsActive: 1, IgnoreUsageLimits: flag}
}

func TestUsageLimitRotator(t *testing.T) {
	t.Run("no flagged connections", func(t *testing.T) {
		r := newUsageLimitRotator([]*models.ProviderConnection{connPointer("a", 0), connPointer("b", 0)}, "a")
		if got := r.next(); got != nil {
			t.Fatalf("expected nil, got %s", got.ID)
		}
	})
	t.Run("single flagged connection repeats", func(t *testing.T) {
		r := newUsageLimitRotator([]*models.ProviderConnection{connPointer("a", 1), connPointer("b", 0)}, "a")
		for i := 0; i < 3; i++ {
			if got := r.next(); got == nil || got.ID != "a" {
				t.Fatalf("iteration %d: got %v, want a", i, got)
			}
		}
	})
	t.Run("two flagged connections alternate starting after the last tried", func(t *testing.T) {
		all := []*models.ProviderConnection{connPointer("a", 1), connPointer("b", 1)}
		// First refusal came from "a" → next must be "b", then "a", then "b".
		r := newUsageLimitRotator(all, "a")
		want := []string{"b", "a", "b"}
		for i, w := range want {
			if got := r.next(); got == nil || got.ID != w {
				t.Fatalf("step %d: got %v, want %s", i, got, w)
			}
		}
	})
	t.Run("last tried absent still alternates", func(t *testing.T) {
		r := newUsageLimitRotator([]*models.ProviderConnection{connPointer("a", 1), connPointer("b", 1)}, "zz")
		for i := 0; i < 2; i++ {
			if got := r.next(); got == nil {
				t.Fatalf("step %d: unexpected nil", i)
			}
		}
	})
}

func TestMaybeRecordIgnoreUsageAttemptThrottles(t *testing.T) {
	ctx := WithFallbackAttemptRecorder(context.Background(), NewFallbackAttemptRecorder())
	last := time.Now().Add(-ignoreUsageLimitLedgerInterval - time.Second)

	// Interval elapsed → records, returning a fresh window start.
	next := maybeRecordIgnoreUsageAttempt(ctx, last, "c1", "a1", "openai-compatible-v", "gpt-x", errors.New("429"), 1)
	if got := len(FallbackAttemptRecorderFrom(ctx).Snapshot()); got != 1 {
		t.Fatalf("expected 1 attempt recorded, got %d", got)
	}
	// Inside the window → no row, timestamp unchanged.
	if got := maybeRecordIgnoreUsageAttempt(ctx, next, "c1", "a1", "openai-compatible-v", "gpt-x", errors.New("429"), 1); !got.Equal(next) {
		t.Fatalf("expected unchanged window start, got %v", got)
	}
	if got := len(FallbackAttemptRecorderFrom(ctx).Snapshot()); got != 1 {
		t.Fatalf("expected still 1 attempt recorded, got %d", got)
	}
	// A nil error is never recorded and does not advance the window.
	stale := time.Now().Add(-time.Hour)
	if got := maybeRecordIgnoreUsageAttempt(ctx, stale, "c1", "a1", "openai-compatible-v", "gpt-x", nil, 1); !got.Equal(stale) {
		t.Fatalf("nil error must not be recorded or advance the window, got %v", got)
	}
	if got := len(FallbackAttemptRecorderFrom(ctx).Snapshot()); got != 1 {
		t.Fatalf("expected still 1 attempt recorded, got %d", got)
	}
}

func TestAsUpstreamError(t *testing.T) {
	if got := asUpstreamError(nil); got != nil {
		t.Fatalf("nil must unwrap to nil")
	}
	ue := &upstreamError{StatusCode: http.StatusTooManyRequests}
	if got := asUpstreamError(errors.New("plain")); got != nil {
		t.Fatalf("plain error must unwrap to nil, got %v", got)
	}
	if got := asUpstreamError(ue); got != ue {
		t.Fatalf("wanted the same *upstreamError back")
	}
}
