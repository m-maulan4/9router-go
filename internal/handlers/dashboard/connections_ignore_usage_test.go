package dashboard

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestUpdateConnection_IgnoreUsageLimitsToggle verifies the flag flips on its
// own column, survives a reload, and is forced off for non-compatible providers
// (PRD §7 guard).
func TestUpdateConnection_IgnoreUsageLimitsToggle(t *testing.T) {
	repo, cleanup := setupTestDB(t)
	defer cleanup()
	router := setupTestRouter(repo)

	connID := "conn-iul-node"
	if err := repo.CreateProviderConnection(connID, "openai-compatible-v", "apikey", "Custom Node", "sk-node"); err != nil {
		t.Fatalf("failed to create connection: %v", err)
	}

	// Toggle on.
	req := httptest.NewRequest(http.MethodPut, "/api/connections/"+connID, bytes.NewReader([]byte(`{"ignoreUsageLimits": true}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	conn, err := repo.GetProviderConnectionByID(connID)
	if err != nil || conn == nil {
		t.Fatalf("failed to fetch: %v", err)
	}
	if conn.IgnoreUsageLimits != 1 {
		t.Errorf("expected ignoreUsageLimits=1 after toggle, got %d", conn.IgnoreUsageLimits)
	}

	// Serializer round-trip.
	sanitized := sanitizeProviderConnection(conn)
	if v, ok := sanitized["ignoreUsageLimits"].(int); !ok || v != 1 {
		t.Errorf("sanitizer must expose ignoreUsageLimits, got %v", sanitized["ignoreUsageLimits"])
	}

	// Toggle off.
	w2 := httptest.NewRecorder()
	router.ServeHTTP(w2, httptest.NewRequest(http.MethodPut, "/api/connections/"+connID, bytes.NewReader([]byte(`{"ignoreUsageLimits": false}`))))
	if w2.Code != http.StatusOK {
		t.Fatalf("expected 200 turning off, got %d: %s", w2.Code, w2.Body.String())
	}
	conn2, err := repo.GetProviderConnectionByID(connID)
	if err != nil || conn2 == nil {
		t.Fatalf("failed to fetch: %v", err)
	}
	if conn2.IgnoreUsageLimits != 0 {
		t.Errorf("expected ignoreUsageLimits=0 after toggle, got %d", conn2.IgnoreUsageLimits)
	}
}

func TestUpdateConnection_IgnoreUsageLimitsForcedOffForRegistryProvider(t *testing.T) {
	repo, cleanup := setupTestDB(t)
	defer cleanup()
	router := setupTestRouter(repo)

	connID := "conn-iul-registry"
	if err := repo.CreateProviderConnection(connID, "deepseek", "apikey", "Registry Acct", "sk-deepseek"); err != nil {
		t.Fatalf("failed to create connection: %v", err)
	}
	req := httptest.NewRequest(http.MethodPut, "/api/connections/"+connID, bytes.NewReader([]byte(`{"ignoreUsageLimits": true}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	conn, err := repo.GetProviderConnectionByID(connID)
	if err != nil || conn == nil {
		t.Fatalf("failed to fetch: %v", err)
	}
	if conn.IgnoreUsageLimits != 0 {
		t.Errorf("expected ignoreUsageLimits forced to 0 on a registry provider, got %d", conn.IgnoreUsageLimits)
	}
}

func TestCreateConnection_IgnoreUsageLimitsHonouredOnlyOnCompatibleNode(t *testing.T) {
	repo, cleanup := setupTestDB(t)
	defer cleanup()
	router := setupTestRouter(repo)

	// Compatible node create: flag persists.
	// Compatible connections resolve their node row, so one must exist.
	node, err := repo.CreateProviderNode("anthropic-compatible-my", "anthropic-compatible",
		"ANode", `{"baseUrl":"https://example.test/v1","prefix":"an","apiType":"messages"}`)
	if err != nil {
		t.Fatalf("failed to seed node: %v", err)
	}
	_ = node
	body := `{
		"provider": "anthropic-compatible-my",
		"authType": "apikey",
		"name": "ANode",
		"apiKey": "sk-an",
		"ignoreUsageLimits": true
	}`
	rec := postConnection(t, router, body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	storedNode, err := repo.GetProviderConnectionByName("anthropic-compatible-my", "apikey", "ANode")
	if err != nil || storedNode == nil {
		t.Fatalf("failed to reload node: %v", err)
	}
	if storedNode.IgnoreUsageLimits != 1 {
		t.Errorf("expected ignoreUsageLimits=1 on compatible node create, got %d", storedNode.IgnoreUsageLimits)
	}

	// Registry provider create: flag ignored, stays 0.
	body2 := `{
		"provider": "deepseek",
		"authType": "apikey",
		"name": "Registry Acct",
		"apiKey": "sk-d",
		"ignoreUsageLimits": true
	}`
	rec2 := postConnection(t, router, body2)
	if rec2.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec2.Code, rec2.Body.String())
	}
	reg, err := repo.GetProviderConnectionByName("deepseek", "apikey", "Registry Acct")
	if err != nil || reg == nil {
		t.Fatalf("failed to reload registry connection: %v", err)
	}
	if reg.IgnoreUsageLimits != 0 {
		t.Errorf("expected ignoreUsageLimits=0 on registry provider create, got %d", reg.IgnoreUsageLimits)
	}
}
