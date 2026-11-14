package chat

import (
	"context"
	"errors"
	"net/http"
	"time"

	"9router/proxy/internal/log"
	"9router/proxy/internal/models"
	"9router/proxy/internal/providers"
)

// PRD: Ignore Usage Limits (custom providers). An operator can mark a
// user-defined compatible-node account (providers.IsCustomCompatible) so a
// 402/429 quota/credit refusal never parks it: no model/account/rate-limit
// lock, no ban escalation, and no eviction — the request keeps retrying the
// flag-ON accounts every ignoreUsageLimitRetryDelay until the client disconnects.

// ignoreUsageLimitRetryDelay is the fixed wait between retries of a flag-ON
// account (PRD §0: unlimited retries, fixed 10s, stop on ctx.Done()). A var so
// the retry-loop tests can shorten it (antigravityProbeDelay pattern).
var ignoreUsageLimitRetryDelay = 10 * time.Second

// ignoreUsageLimitLedgerInterval throttles fallback-attempt ledger rows to one
// per minute per request while the retry loop spins (PRD §5). A var so the
// retry-loop tests can collapse the window.
var ignoreUsageLimitLedgerInterval = time.Minute

// isUsageLimitRefusal reports whether an upstream error is a quota/credit
// refusal a flag-ON account should retry forever. It is the 402/429 subset of
// the classifier's fallback verdict.
//
// The PRD exclusions are structurally enforced at the call sites, which reach
// this function only after: the opencode free-tier refusal branch
// (isOpencodeFreeTierRefusal — a custom provider can never be one, its switch
// only matches opencode/zen/go), the node-state refusal branch
// (providerLimitRefusal), and the custom-JSON business-endpoint branch
// (isCustomJSONConn). No manual string matching is added here — the existing
// classifier decides.
func isUsageLimitRefusal(ue *upstreamError) bool {
	if ue == nil {
		return false
	}
	switch ue.StatusCode {
	case http.StatusPaymentRequired, http.StatusTooManyRequests:
	default:
		return false
	}
	cls := providers.ClassifyErrorForProvider(ue.StatusCode, extractErrorText(ue.Body), 0, "")
	return cls.ShouldFallback
}

// usageLimitRotator round-robins over the flag-ON connections of one request.
// State is per-request (an index into a cloned slice), never global.
type usageLimitRotator struct {
	conns []*models.ProviderConnection
	idx   int
}

// newUsageLimitRotator keeps only the ignore-usage-limits connections of
// allConns, positioned so the first retry tries the connection AFTER
// lastTriedID — with two flag-ON accounts the sequence alternates A→B→A
// instead of hammering the account that just refused (PRD §5 AC-2).
func newUsageLimitRotator(allConns []*models.ProviderConnection, lastTriedID string) *usageLimitRotator {
	flagged := make([]*models.ProviderConnection, 0, len(allConns))
	for _, c := range allConns {
		if c != nil && c.IgnoreUsageLimits == 1 {
			flagged = append(flagged, c)
		}
	}
	idx := 0
	if len(flagged) > 1 {
		idx = -1
		for i, c := range flagged {
			if c.ID == lastTriedID {
				idx = i
				break
			}
		}
	}
	return &usageLimitRotator{conns: flagged, idx: idx}
}

// next returns the next connection in the round-robin, or nil when no flag-ON
// connection exists.
func (r *usageLimitRotator) next() *models.ProviderConnection {
	if len(r.conns) == 0 {
		return nil
	}
	if len(r.conns) == 1 {
		return r.conns[0]
	}
	r.idx = (r.idx + 1) % len(r.conns)
	return r.conns[r.idx]
}

// retryUsageLimitForever retries a request against every flag-ON connection of
// the provider until one succeeds or the client disconnects. No account/model
// state is written for a refused attempt: the upstream refusal is transient by
// operator decree, so parking the account would hide it behind a quota it does
// not own (PRD §5).
//
// Fallback-attempt ledger rows are throttled to one per interval plus the final
// outcome — the retry counter lives in this call's frames, so a three-minute
// spin records ~4 rows instead of 18.
func (h *ChatHandler) retryUsageLimitForever(ctx context.Context, f forwardRequestParams, allConns []*models.ProviderConnection, firstConn *models.ProviderConnection) error {
	rot := newUsageLimitRotator(allConns, firstConn.ID)
	var lastLedger time.Time
	for {
		conn := rot.next()
		if conn == nil {
			return errors.New("ignore usage limits: no flagged connection available")
		}
		connObj, connData, err := h.getBestConnectionWithContext(ctx, f.Provider, conn.ID, nil, f.Model)
		if err != nil || connObj == nil {
			if err == nil {
				err = errors.New("connection unresolvable")
			}
			log.Warn("fallback", "ignore usage limits: connection unresolvable, rotating",
				"provider", f.Provider, "model", f.Model, "conn", conn.ID, "error", err)
			lastLedger = maybeRecordIgnoreUsageAttempt(ctx, lastLedger, conn.ID, "", f.Provider, f.Model, err, 0)
			if !h.waitForRetry(ctx) {
				return ctx.Err()
			}
			continue
		}
		attemptStart := time.Now()
		fwdErr := h.tryForwardWithConnection(forwardRequestParams{
			Ctx: ctx, W: f.W, Provider: f.Provider, Model: f.Model,
			RequestedModel: f.RequestedModel, ComboName: f.ComboName, Attempts: attemptCount(ctx),
			ConnectionID: connObj.ID, ConnName: connObjName(connObj), ConnEmail: connObjEmail(connObj),
			ConnData: connData, Body: f.Body, IsStream: f.IsStream,
			TranslateResponse: f.TranslateResponse, Endpoint: f.Endpoint,
		})
		if fwdErr == nil {
			RecordAttemptFromContext(ctx, connObj.ID, connObjName(connObj), f.Provider, f.Model, http.StatusOK, nil, time.Since(attemptStart).Milliseconds())
			log.Info("fallback", "ignore usage limits: request succeeded after retries",
				"provider", f.Provider, "model", f.Model, "conn", connObj.ID)
			return nil
		}
		if !isUsageLimitRefusal(asUpstreamError(fwdErr)) {
			// A non-quota failure ends the retry loop with the classified answer,
			// so a hardware 5xx or a request bug does not spin forever.
			RecordAttemptFromContext(ctx, connObj.ID, connObjName(connObj), f.Provider, f.Model, 0, fwdErr, time.Since(attemptStart).Milliseconds())
			return fwdErr
		}
		lastLedger = maybeRecordIgnoreUsageAttempt(ctx, lastLedger, connObj.ID, connObjName(connObj), f.Provider, f.Model, fwdErr, time.Since(attemptStart).Milliseconds())
		if !h.waitForRetry(ctx) {
			RecordAttemptFromContext(ctx, connObj.ID, connObjName(connObj), f.Provider, f.Model, 0, ctx.Err(), time.Since(attemptStart).Milliseconds())
			return ctx.Err()
		}
	}
}

// waitForRetry sleeps ignoreUsageLimitRetryDelay or returns false when the
// client disconnected.
func (h *ChatHandler) waitForRetry(ctx context.Context) bool {
	select {
	case <-time.After(ignoreUsageLimitRetryDelay):
		return true
	case <-ctx.Done():
		return false
	}
}

// maybeRecordIgnoreUsageAttempt writes one fallback-attempt ledger row when the
// throttle window has elapsed since the previous write, returning the new
// window start.
func maybeRecordIgnoreUsageAttempt(ctx context.Context, lastLedger time.Time, connID, account, provider, model string, err error, latencyMs int64) time.Time {
	if err != nil && (lastLedger.IsZero() || time.Since(lastLedger) >= ignoreUsageLimitLedgerInterval) {
		RecordAttemptFromContext(ctx, connID, account, provider, model, 0, err, latencyMs)
		return time.Now()
	}
	return lastLedger
}

// asUpstreamError unwraps an error chain to an *upstreamError, or nil.
func asUpstreamError(err error) *upstreamError {
	var ue *upstreamError
	if errors.As(err, &ue) {
		return ue
	}
	return nil
}
