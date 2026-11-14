package chat

import (
	"context"
	"database/sql"
	json "encoding/json/v2"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"9router/proxy/internal/db"
	"9router/proxy/internal/models"
)

// customConnData builds the data payload for a custom OpenAI-compatible
// connection: the apiKey is the account identity (sent as the Authorization
// bearer), and the baseUrl points at the fake upstream. No providerNodes row
// is needed — a conn whose data carries its own baseUrl wins in
// getProviderConfig.
func customConnData(apiKey, baseURL string) string {
	raw, _ := json.Marshal(map[string]any{"apiKey": apiKey, "baseUrl": baseURL})
	return string(raw)
}

const loopTestKeyA = "sk-test-a"
const loopTestKeyB = "sk-test-b"

// seedIgnoreUsageConns seeds two custom-compatible, flag-ON connections pointed
// at the given upstream.
func seedIgnoreUsageConns(t *testing.T, database *sql.DB, baseURL string) {
	t.Helper()
	for _, c := range []struct{ id, name, key string }{
		{"conn-a", "Custom A", loopTestKeyA},
		{"conn-b", "Custom B", loopTestKeyB},
	} {
		dump := customConnData(c.key, baseURL)
		if _, err := database.Exec(`INSERT INTO providerConnections (id, provider, authType, name, priority, isActive, data, createdAt, updatedAt, ignoreUsageLimits)
			VALUES (?, 'openai-compatible-v', 'apikey', ?, 1, 1, ?, '2026-07-18T00:00:00Z', '2026-07-18T00:00:00Z', 1)`, c.id, c.name, dump); err != nil {
			t.Fatalf("seed conn %s: %v", c.id, err)
		}
	}
}

// bearerKey returns the account a request hit, by the Authorization header.
func bearerKey(r *http.Request) string {
	return r.Header.Get("Authorization")
}

// TestRetryUsageLimitForever_SurvivesRefusalsAndNeverParks: upstream answers
// 402 to the first two attempts, then a real completion. The loop must keep
// retrying past the refusals, reach success, and write no
// rateLimitedUntil/accountLockedUntil/model lock (PRD AC-1 §"skip semua
// locking"). Both flag-ON accounts observe at least one attempt (AC-2).
func TestRetryUsageLimitForever_SurvivesRefusalsAndNeverParks(t *testing.T) {
	orig := ignoreUsageLimitRetryDelay
	ignoreUsageLimitRetryDelay = time.Millisecond
	t.Cleanup(func() { ignoreUsageLimitRetryDelay = orig })

	okBody := `{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"gpt-x",
		"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

	var refusals atomic.Int32
	var (
		mu      sync.Mutex
		hitKeys []string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		hitKeys = append(hitKeys, bearerKey(r))
		mu.Unlock()
		if refusals.Add(1) <= 2 {
			w.WriteHeader(http.StatusPaymentRequired)
			_, _ = w.Write([]byte(`{"error":{"message":"insufficient balance"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okBody))
	}))
	defer srv.Close()

	database, cleanup := setupChatTestDB(t)
	defer cleanup()
	dropSeededConn(t, database)
	seedIgnoreUsageConns(t, database, srv.URL)

	repo := db.NewRepo(database)
	h := NewChatHandler(repo)
	allConns, err := repo.GetProviderConnections("openai-compatible-v", true)
	if err != nil || len(allConns) != 2 {
		t.Fatalf("load connections: %v, n=%d", err, len(allConns))
	}
	first := allConns[0]
	t.Logf("priority order: first=%s second=%s", first.ID, otherConnID(allConns, first.ID))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rec := httptest.NewRecorder()
	err = h.retryUsageLimitForever(ctx, forwardRequestParams{
		Ctx: ctx, W: rec, Provider: "openai-compatible-v", Model: "gpt-x",
		ConnectionID: first.ID, ConnName: connObjName(first),
		Body: []byte(`{"model":"gpt-x","messages":[{"role":"user","content":"hi"}]}`), Attempts: 1,
	}, allConns, first)
	if err != nil {
		t.Fatalf("loop must survive two 402s and succeed, got: %v", err)
	}

	mu.Lock()
	seen := hitKeys
	mu.Unlock()
	var sawA, sawB bool
	for _, k := range seen {
		sawA = sawA || k == "Bearer "+loopTestKeyA
		sawB = sawB || k == "Bearer "+loopTestKeyB
	}
	if !sawA || !sawB {
		t.Errorf("both flag-ON accounts must be tried (AC-2), hits=%v", seen)
	}

	// No park writer may have run (AC-1).
	for _, id := range []string{"conn-a", "conn-b"} {
		conn, err := repo.GetProviderConnectionByID(id)
		if err != nil || conn == nil {
			t.Fatalf("reload %s: %v", id, err)
		}
		if _, ok := db.ConnectionCooldownUntil(conn.Data); ok {
			t.Errorf("%s: rateLimitedUntil must not be written, data=%s", id, conn.Data)
		}
		if _, ok := db.ConnectionAccountLockUntil(conn.Data); ok {
			t.Errorf("%s: accountLockedUntil must not be written", id)
		}
		locked, err := repo.IsConnectionModelLocked(id, "gpt-x")
		if err != nil {
			t.Fatalf("lock check %s: %v", id, err)
		}
		if locked {
			t.Errorf("%s: model lock must not be written", id)
		}
	}
}

func otherConnID(conns []*models.ProviderConnection, id string) string {
	for _, c := range conns {
		if c.ID != id {
			return c.ID
		}
	}
	return ""
}

// TestRetryUsageLimitForever_StopsOnClientDisconnect: cancel the context while
// the loop is between attempts, i.e. inside its retry sleep. The loop must exit
// with context.Canceled rather than firing another forward (PRD §0 "stop on
// ctx.Done()"). Cancelling mid-upstream-call is deliberately not exercised
// here — that path races with goroutine shutdown and has no deterministic
// observer.
func TestRetryUsageLimitForever_StopsOnClientDisconnect(t *testing.T) {
	orig := ignoreUsageLimitRetryDelay
	ignoreUsageLimitRetryDelay = time.Hour
	t.Cleanup(func() { ignoreUsageLimitRetryDelay = orig })

	var refusals atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		refusals.Add(1)
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"error":{"message":"insufficient balance"}}`))
	}))
	defer srv.Close()

	database, cleanup := setupChatTestDB(t)
	defer cleanup()
	dropSeededConn(t, database)
	seedIgnoreUsageConns(t, database, srv.URL)

	repo := db.NewRepo(database)
	h := NewChatHandler(repo)
	allConns, err := repo.GetProviderConnections("openai-compatible-v", true)
	if err != nil || len(allConns) != 2 {
		t.Fatalf("load connections: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- h.retryUsageLimitForever(ctx, forwardRequestParams{
			Ctx: ctx, W: httptest.NewRecorder(), Provider: "openai-compatible-v", Model: "gpt-x",
			ConnectionID: allConns[0].ID, ConnName: connObjName(allConns[0]),
			Body: []byte(`{"model":"gpt-x","messages":[{"role":"user","content":"hi"}]}`), Attempts: 1,
		}, allConns, allConns[0])
	}()

	// Wait until the first refusal was observed and the loop is sleeping for
	// the next retry; then cancel. With the delay at 1h the sleep outlives the
	// test unless cancellation cuts it short.
	deadline := time.After(10 * time.Second)
	for refusals.Load() == 0 {
		select {
		case <-deadline:
			t.Fatal("loop never reached upstream")
		case <-time.After(2 * time.Millisecond):
		}
	}
	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("loop did not stop after ctx cancellation")
	}
	// Cancel happened during the sleep: no second attempt may have been fired.
	if n := refusals.Load(); n > 1 {
		t.Errorf("expected exactly 1 upstream attempt before cancel, got %d", n)
	}
}

// TestRetryUsageLimitForever_LedgerThrottle asserts PRD AC-6: a long retry spin
// records at most one ledger row per interval plus the final one, not one per
// attempt. The recorder lives on the ctx, so it is inspectable without SQL.
func TestRetryUsageLimitForever_LedgerThrottle(t *testing.T) {
	orig := ignoreUsageLimitRetryDelay
	ignoreUsageLimitRetryDelay = time.Millisecond
	t.Cleanup(func() { ignoreUsageLimitRetryDelay = orig })
	origInterval := ignoreUsageLimitLedgerInterval
	ignoreUsageLimitLedgerInterval = 5 * time.Millisecond
	t.Cleanup(func() { ignoreUsageLimitLedgerInterval = origInterval })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"error":{"message":"too slow try later"}}`))
	}))
	defer srv.Close()

	database, cleanup := setupChatTestDB(t)
	defer cleanup()
	dropSeededConn(t, database)
	seedIgnoreUsageConns(t, database, srv.URL)

	repo := db.NewRepo(database)
	h := NewChatHandler(repo)
	allConns, err := repo.GetProviderConnections("openai-compatible-v", true)
	if err != nil || len(allConns) != 2 {
		t.Fatalf("load connections: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	recorder := NewFallbackAttemptRecorder()
	ctx = WithFallbackAttemptRecorder(ctx, recorder)

	done := make(chan error, 1)
	go func() {
		done <- h.retryUsageLimitForever(ctx, forwardRequestParams{
			Ctx: ctx, W: httptest.NewRecorder(), Provider: "openai-compatible-v", Model: "gpt-x",
			Body: []byte(`{"model":"gpt-x","messages":[{"role":"user","content":"hi"}]}`), Attempts: 1,
		}, allConns, allConns[0])
	}()
	time.Sleep(40 * time.Millisecond)
	cancel()
	<-done

	// 40ms of spinning at 1ms/attempt would naively log ~30 rows; the throttle
	// caps it to ~1 per 5ms window plus the final cancel row. Also allow one
	// row from the pre-loop fallback attempt. The bound is generous to avoid
	// flakes on slow CI, but far below one-row-per-attempt.
	if n := len(recorder.Snapshot()); n > 16 {
		t.Errorf("ledger throttle failed: %d rows for a 40ms spin, want <= ~12", n)
	}
	if n := len(recorder.Snapshot()); n < 2 {
		t.Errorf("expected at least a first-attempt + final row, got %d", n)
	}
}
