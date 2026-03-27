package openapigateway

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	log "github.com/sirupsen/logrus"
)

const (
	// Mirror the headers used by CLIProxyAPI's own Codex executor. The Codex backend is sensitive
	// to request shape and client identity; a generic Go UA tends to trigger mitigations.
	codexUserAgent  = "codex_cli_rs/0.116.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464"
	codexOriginator = "codex_cli_rs"
)

type Handler struct {
	cfg *config.Config

	httpClient        *http.Client
	httpClientProxied *http.Client

	// Simple in-process per-tenant limiter (concurrency + QPS).
	mu       sync.Mutex
	tenants  map[string]*tenantLimiter
	now      func() time.Time
	randHex8 func() string
}

type tenantLimiter struct {
	concurrentLimit int
	inFlight        int

	// naive token bucket
	qpsLimit int
	tokens   float64
	lastRef  time.Time
}

type probeRequest struct {
	AccountID         string `json:"accountId"`
	Model             string `json:"model"`
	Prompt            string `json:"prompt"`
	CredentialMaterial struct {
		AccessToken  string  `json:"accessToken"`
		RefreshToken *string `json:"refreshToken"`
		AccountID    *string `json:"accountId"`
		AccountEmail *string `json:"accountEmail"`
	} `json:"credentialMaterial"`
}

type publicAuthContext struct {
	TenantID        string   `json:"tenantId"`
	OwnerUserID     string   `json:"ownerUserId"`
	KeyID           string   `json:"keyId"`
	QPSLimit        int      `json:"qpsLimit"`
	ConcurrentLimit int      `json:"concurrentLimit"`
	ModelAllowlist  []string `json:"modelAllowlist"`
}

type internalAuthResponse struct {
	OK   bool `json:"ok"`
	Auth struct {
		TenantID        string   `json:"tenantId"`
		OwnerUserID     string   `json:"ownerUserId"`
		KeyID           string   `json:"keyId"`
		QPSLimit        int      `json:"qpsLimit"`
		ConcurrentLimit int      `json:"concurrentLimit"`
		ModelAllowlist  []string `json:"modelAllowlist"`
	} `json:"auth"`
	Error string `json:"error"`
}

type executionLeaseResponse struct {
	OK           bool    `json:"ok"`
	DenyReason   string  `json:"denyReason"`
	RetryAfter   *int    `json:"retryAfterMs"`
	Detail       *string `json:"detail"`
	ObservedAt   string  `json:"observedAt"`
	RequestID    string  `json:"requestId"`
	GrantedAt    string  `json:"grantedAt"`
	GrantedCount int     `json:"grantedLeaseCount"`
	Snapshot     struct {
		SnapshotID   string `json:"snapshotId"`
		DispatchEpoch string `json:"dispatchEpoch"`
		Leases []struct {
			LeaseID   string `json:"leaseId"`
			TokenID   string `json:"tokenId"`
			AccountID string `json:"accountId"`
			Lane      string `json:"lane"`
			Provider  string `json:"provider"`
		} `json:"leases"`
	} `json:"snapshot"`
}

type credentialMaterial struct {
	OK          bool    `json:"ok"`
	TokenID     string  `json:"tokenId"`
	AccountID   *string `json:"accountId"`
	AccessToken string  `json:"accessToken"`
	Error       string  `json:"error"`
}

func New(cfg *config.Config) *Handler {
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	transport := &http.Transport{
		Proxy:                 nil,
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          200,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
	baseClient := &http.Client{Transport: transport, Timeout: 0}

	var proxiedClient *http.Client
	if cfg.OpenAPIGateway.UpstreamProxyURL != "" {
		proxyURL, err := url.Parse(cfg.OpenAPIGateway.UpstreamProxyURL)
		if err == nil {
			proxyTransport := transport.Clone()
			proxyTransport.Proxy = http.ProxyURL(proxyURL)
			proxiedClient = &http.Client{Transport: proxyTransport, Timeout: 0}
		}
	}
	if proxiedClient == nil {
		proxiedClient = baseClient
	}

	return &Handler{
		cfg:               cfg,
		httpClient:        baseClient,
		httpClientProxied: proxiedClient,
		tenants:           make(map[string]*tenantLimiter),
		now:               time.Now,
		randHex8: func() string {
			var b [4]byte
			_, _ = rand.Read(b[:])
			return hex.EncodeToString(b[:])
		},
	}
}

func (h *Handler) Models(c *gin.Context) {
	auth, ok := h.authenticate(c)
	if !ok {
		return
	}
	allow := auth.ModelAllowlist
	if len(allow) == 0 {
		allow = []string{"gpt-5.4"}
	}
	data := make([]gin.H, 0, len(allow))
	for _, id := range allow {
		data = append(data, gin.H{"id": id, "object": "model", "owned_by": "openapi"})
	}
	c.JSON(200, gin.H{"object": "list", "data": data})
}

func (h *Handler) ProbeBasic(c *gin.Context) {
	if !h.authenticateInternal(c) {
		return
	}
	var req probeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"ok": false, "error": "invalid_json_body"})
		return
	}
	access := strings.TrimSpace(req.CredentialMaterial.AccessToken)
	accountID := strings.TrimSpace(req.AccountID)
	if access == "" {
		c.JSON(400, gin.H{"ok": false, "error": "access_token_required"})
		return
	}
	if accountID == "" && req.CredentialMaterial.AccountID != nil {
		accountID = strings.TrimSpace(*req.CredentialMaterial.AccountID)
	}

	body := []byte(`{"model":"gpt-5.4","input":"OPENAPI_BASIC_PROBE","stream":false}`)
	status, ct, peek, err := h.callUpstream(c.Request.Context(), access, accountID, false, body)
	if err != nil {
		c.JSON(503, gin.H{"ok": false, "error": "upstream_fetch_failed"})
		return
	}
	if status >= 400 && strings.Contains(strings.ToLower(ct), "text/html") {
		c.JSON(424, gin.H{"ok": false, "error": "upstream_html_blocked", "peek": peek})
		return
	}
	if status >= 200 && status < 300 {
		c.JSON(200, gin.H{"ok": true})
		return
	}
	c.Data(status, "application/json; charset=utf-8", []byte(peek))
}

func (h *Handler) ProbePreflight(c *gin.Context) {
	if !h.authenticateInternal(c) {
		return
	}
	var req probeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"ok": false, "error": "invalid_json_body"})
		return
	}
	access := strings.TrimSpace(req.CredentialMaterial.AccessToken)
	accountID := strings.TrimSpace(req.AccountID)
	if access == "" {
		c.JSON(400, gin.H{"ok": false, "error": "access_token_required"})
		return
	}
	if accountID == "" && req.CredentialMaterial.AccountID != nil {
		accountID = strings.TrimSpace(*req.CredentialMaterial.AccountID)
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = "gpt-5.4"
	}
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		prompt = "reply with exactly: OPENAPI_VALIDATOR_OK"
	}
	payloadObj := map[string]any{
		"model":  model,
		"input":  prompt,
		"stream": false,
	}
	body, _ := json.Marshal(payloadObj)
	status, ct, peek, err := h.callUpstream(c.Request.Context(), access, accountID, false, body)
	if err != nil {
		c.JSON(503, gin.H{"ok": false, "error": "upstream_fetch_failed"})
		return
	}
	if status >= 400 && strings.Contains(strings.ToLower(ct), "text/html") {
		c.JSON(424, gin.H{"ok": false, "error": "upstream_html_blocked", "peek": peek})
		return
	}
	if status >= 200 && status < 300 {
		c.JSON(200, gin.H{"ok": true})
		return
	}
	c.Data(status, "application/json; charset=utf-8", []byte(peek))
}

func (h *Handler) Responses(c *gin.Context) {
	start := h.now()
	auth, ok := h.authenticate(c)
	if !ok {
		return
	}

	requestID := "req-" + h.randHex8()
	traceID := fmt.Sprintf("%s:%s:%s:responses", auth.TenantID, auth.KeyID, requestID)

	if !h.acquire(auth) {
		h.ingestUsage(auth, "", requestID, traceID, nil, nil, time.Since(start), false, "server_or_cliproxy_429", 429, "rate_limited", "rate_limited", false, nil)
		c.JSON(429, gin.H{"ok": false, "error": "rate_limited"})
		return
	}
	defer h.release(auth)

	bodyBytes, err := io.ReadAll(io.LimitReader(c.Request.Body, 32*1024*1024))
	if err != nil {
		c.JSON(400, gin.H{"ok": false, "error": "invalid_json_body"})
		return
	}
	var bodyObj map[string]any
	if err := json.Unmarshal(bodyBytes, &bodyObj); err != nil {
		c.JSON(400, gin.H{"ok": false, "error": "invalid_json_body"})
		return
	}
	streamRequested, _ := bodyObj["stream"].(bool)

	requestedModel := "gpt-5.4"
	if m, ok := bodyObj["model"].(string); ok && strings.TrimSpace(m) != "" {
		requestedModel = strings.TrimSpace(m)
	}
	if len(auth.ModelAllowlist) > 0 && !contains(auth.ModelAllowlist, requestedModel) {
		c.JSON(403, gin.H{"ok": false, "error": "model_not_allowed", "model": requestedModel})
		return
	}

	lease, leaseErr := h.requestLease(c.Request.Context(), requestedModel)
	if leaseErr != nil {
		elapsed := time.Since(start)
		h.ingestUsage(auth, requestedModel, requestID, traceID, nil, nil, elapsed, false, "no_dispatchable_route_ready", 503, "no_dispatchable_route_ready", leaseErr.Error(), false, nil)
		c.JSON(503, gin.H{"ok": false, "error": "no_dispatchable_route_ready"})
		return
	}
	tokenID := lease.TokenID
	accountID := lease.AccountID
	defer h.releaseLease(context.Background(), lease)

	cred, credErr := h.getCredential(c.Request.Context(), tokenID)
	if credErr != nil {
		elapsed := time.Since(start)
		h.ingestUsage(auth, requestedModel, requestID, traceID, &tokenID, &accountID, elapsed, false, "unauthorized", 503, "credential_material_not_found", credErr.Error(), false, nil)
		c.JSON(503, gin.H{"ok": false, "error": "credential_material_not_found"})
		return
	}

	// Note: the Codex backend uses `/backend-api/codex/responses` (no `/v1` segment).
	upstreamURL := strings.TrimRight(h.cfg.OpenAPIGateway.UpstreamBaseURL, "/") + "/responses"
	upReq, _ := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, upstreamURL, bytes.NewReader(bodyBytes))
	upReq.Header.Set("Content-Type", "application/json")
	upReq.Header.Set("Authorization", "Bearer "+cred.AccessToken)
	upReq.Header.Set("User-Agent", codexUserAgent)
	if streamRequested {
		upReq.Header.Set("Accept", "text/event-stream")
	} else {
		upReq.Header.Set("Accept", "application/json")
	}
	upReq.Header.Set("Connection", "Keep-Alive")
	upReq.Header.Set("Originator", codexOriginator)
	if strings.TrimSpace(accountID) != "" {
		upReq.Header.Set("Chatgpt-Account-Id", accountID)
	}
	upReq.Header.Set("Session_id", "openapi-"+h.randHex8()+"-"+h.randHex8())
	upReq.Header.Set("X-Client-Request-Id", "openapi-"+h.randHex8())

	// propagate a few headers that matter for clients
	if v := c.GetHeader("OpenAI-Organization"); v != "" {
		upReq.Header.Set("OpenAI-Organization", v)
	}
	if v := c.GetHeader("OpenAI-Project"); v != "" {
		upReq.Header.Set("OpenAI-Project", v)
	}

	resp, err := h.httpClientProxied.Do(upReq)
	if err != nil {
		elapsed := time.Since(start)
		h.ingestUsage(auth, requestedModel, requestID, traceID, &tokenID, &accountID, elapsed, false, "unknown", 503, "upstream_fetch_failed", err.Error(), false, nil)
		h.reportExecutionFailure(auth, lease, requestedModel, requestID, traceID, start, time.Now().UTC(), "network_transient", 503, "upstream_fetch_failed", err.Error(), true)
		c.JSON(503, gin.H{"ok": false, "error": "upstream_fetch_failed"})
		return
	}
	defer resp.Body.Close()

	// Cloudflare (and similar) mitigation often returns HTML. Never pass HTML to API clients.
	// Instead, convert to a stable JSON error and record the failure for ops/usage.
	if resp.StatusCode >= 400 && strings.Contains(strings.ToLower(resp.Header.Get("content-type")), "text/html") {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
		elapsed := time.Since(start)
		h.ingestUsage(
			auth,
			requestedModel,
			requestID,
			traceID,
			&tokenID,
			&accountID,
			elapsed,
			false,
			"cloudflare_challenge",
			503,
			"upstream_blocked_cloudflare",
			"cloudflare_challenge",
			false,
			nil,
		)
		h.reportExecutionFailure(auth, lease, requestedModel, requestID, traceID, start, time.Now().UTC(), "policy_denied", 503, "upstream_blocked_cloudflare", "cloudflare_challenge", true)
		c.JSON(503, gin.H{
			"ok":    false,
			"error": "upstream_blocked_cloudflare",
			"peek":  strings.TrimSpace(string(payload[:minInt(len(payload), 200)])),
		})
		return
	}

	if !streamRequested {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, 16*1024*1024))
		for k, vv := range resp.Header {
			for _, v := range vv {
				c.Writer.Header().Add(k, v)
			}
		}
		c.Status(resp.StatusCode)
		_, _ = c.Writer.Write(payload)

		var parsed any
		_ = json.Unmarshal(payload, &parsed)
		usage := extractUsage(parsed)
		elapsed := time.Since(start)
		success := resp.StatusCode >= 200 && resp.StatusCode < 300
		failureReason := classifyFailure(resp.StatusCode, false)
		h.ingestUsage(auth, requestedModel, requestID, traceID, &tokenID, &accountID, elapsed, success, failureReason, resp.StatusCode, "", "", success, usage)
		h.reportFromUpstreamResult(auth, lease, requestedModel, requestID, traceID, start, time.Now().UTC(), resp.StatusCode, usage, success, false, "")
		return
	}

	// Streaming: proxy bytes immediately, while parsing response.completed in-band.
	for k, vv := range resp.Header {
		for _, v := range vv {
			c.Writer.Header().Add(k, v)
		}
	}
	c.Status(resp.StatusCode)
	flusher, _ := c.Writer.(http.Flusher)

	type streamObs struct {
		streamCompleted bool
		usage           map[string]int
	}
	obs := &streamObs{streamCompleted: false, usage: nil}

	reader := bufio.NewReader(resp.Body)
	var buf bytes.Buffer
	buf.Grow(64 * 1024)

	// scan by line to detect response.completed event and capture its data JSON
	var lastEvent string
	for {
		line, readErr := reader.ReadBytes('\n')
		if len(line) > 0 {
			_, _ = c.Writer.Write(line)
			if flusher != nil {
				flusher.Flush()
			}
			// keep a small tail buffer for parsing
			if buf.Len() < 256*1024 {
				_, _ = buf.Write(line)
			}
			text := strings.TrimSpace(string(line))
			if strings.HasPrefix(text, "event:") {
				lastEvent = strings.TrimSpace(strings.TrimPrefix(text, "event:"))
			} else if strings.HasPrefix(text, "data:") && lastEvent == "response.completed" {
				raw := strings.TrimSpace(strings.TrimPrefix(text, "data:"))
				var parsed any
				if json.Unmarshal([]byte(raw), &parsed) == nil {
					obs.usage = extractUsage(parsed)
					obs.streamCompleted = true
				}
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			break
		}
	}

	elapsed := time.Since(start)
	streamCompleted := obs.streamCompleted
	success := resp.StatusCode >= 200 && resp.StatusCode < 300 && streamCompleted
	failureReason := classifyFailure(resp.StatusCode, streamCompleted)
	h.ingestUsage(auth, requestedModel, requestID, traceID, &tokenID, &accountID, elapsed, success, failureReason, resp.StatusCode, "", "", streamCompleted, obs.usage)
	h.reportFromUpstreamResult(auth, lease, requestedModel, requestID, traceID, start, time.Now().UTC(), resp.StatusCode, obs.usage, success, streamCompleted, failureReason)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (h *Handler) authenticate(c *gin.Context) (*publicAuthContext, bool) {
	req, _ := http.NewRequestWithContext(
		c.Request.Context(),
		http.MethodGet,
		strings.TrimRight(h.cfg.OpenAPIGateway.ControlPlaneBaseURL, "/")+"/internal/public-api/auth-context",
		nil,
	)
	req.Header.Set("x-openapi-internal-caller", h.cfg.OpenAPIGateway.InternalCaller)
	req.Header.Set("x-openapi-internal-token", h.cfg.OpenAPIGateway.InternalToken)
	if authz := c.GetHeader("Authorization"); authz != "" {
		req.Header.Set("Authorization", authz)
	} else if x := c.GetHeader("x-api-key"); x != "" {
		req.Header.Set("x-api-key", x)
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		c.JSON(503, gin.H{"ok": false, "error": "control_plane_unavailable"})
		return nil, false
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1*1024*1024))
	if resp.StatusCode != 200 {
		c.JSON(401, gin.H{"ok": false, "error": "invalid_api_key"})
		return nil, false
	}
	var parsed internalAuthResponse
	if err := json.Unmarshal(raw, &parsed); err != nil || !parsed.OK {
		c.JSON(401, gin.H{"ok": false, "error": "invalid_api_key"})
		return nil, false
	}
	return &publicAuthContext{
		TenantID:        parsed.Auth.TenantID,
		OwnerUserID:     parsed.Auth.OwnerUserID,
		KeyID:           parsed.Auth.KeyID,
		QPSLimit:        parsed.Auth.QPSLimit,
		ConcurrentLimit: parsed.Auth.ConcurrentLimit,
		ModelAllowlist:  parsed.Auth.ModelAllowlist,
	}, true
}

func (h *Handler) acquire(auth *publicAuthContext) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	lim := h.tenants[auth.TenantID]
	if lim == nil {
		lim = &tenantLimiter{
			concurrentLimit: maxInt(1, auth.ConcurrentLimit),
			qpsLimit:        maxInt(1, auth.QPSLimit),
			tokens:          float64(maxInt(1, auth.QPSLimit)),
			lastRef:         h.now(),
		}
		h.tenants[auth.TenantID] = lim
	}
	lim.concurrentLimit = maxInt(1, auth.ConcurrentLimit)
	lim.qpsLimit = maxInt(1, auth.QPSLimit)
	now := h.now()
	elapsed := now.Sub(lim.lastRef).Seconds()
	if elapsed > 0 {
		lim.tokens = minFloat(float64(lim.qpsLimit), lim.tokens+elapsed*float64(lim.qpsLimit))
		lim.lastRef = now
	}
	if lim.inFlight >= lim.concurrentLimit {
		return false
	}
	if lim.tokens < 1 {
		return false
	}
	lim.tokens -= 1
	lim.inFlight += 1
	return true
}

func (h *Handler) release(auth *publicAuthContext) {
	h.mu.Lock()
	defer h.mu.Unlock()
	lim := h.tenants[auth.TenantID]
	if lim == nil {
		return
	}
	if lim.inFlight > 0 {
		lim.inFlight -= 1
	}
}

type lease struct {
	TokenID   string
	AccountID string
	LeaseID   string
	Lane      string
	SnapshotID string
	DispatchEpoch string
	Provider string
}

func (h *Handler) requestLease(ctx context.Context, requestedModel string) (*lease, error) {
	reqID := "cliproxy-" + h.randHex8()

	// OpenAPI execution-plane leases are scoped by model *family* (e.g. "default"), not by the
	// requested model id (e.g. "gpt-5.4"). If we send the model id here, OpenAPI will deny the
	// lease request even though dispatchable tokens exist.
	modelFamilyScope := []string{"default"}

	payload := map[string]any{
		"meta": map[string]any{
			"contractName":    "cliproxyapi-execution-plane",
			"contractVersion": "v1alpha1",
			"schemaRevision":  h.cfg.OpenAPIGateway.SchemaRevision,
		},
		"requestId": reqID,
		"consumer": map[string]any{
			"executionInstanceId": h.cfg.OpenAPIGateway.ExecutionInstanceID,
			"executionNodeId":     h.cfg.OpenAPIGateway.ExecutionNodeID,
			"runtimeClass":        h.cfg.OpenAPIGateway.RuntimeClass,
			"region":              h.cfg.OpenAPIGateway.Region,
		},
		"lanes":             h.cfg.OpenAPIGateway.Lanes,
		"provider":          nil,
		"modelFamilyScope":  modelFamilyScope,
		"desiredLeaseCount": 1,
		"minLeaseTtlMs":     h.cfg.OpenAPIGateway.MinLeaseTtlMs,
		"requestedAt":       time.Now().UTC().Format(time.RFC3339Nano),
	}

	raw, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(h.cfg.OpenAPIGateway.ControlPlaneBaseURL, "/")+"/internal/execution/lease/request",
		bytes.NewReader(raw),
	)
	req.Header.Set("content-type", "application/json; charset=utf-8")
	req.Header.Set("x-openapi-internal-caller", h.cfg.OpenAPIGateway.InternalCaller)
	req.Header.Set("x-openapi-internal-token", h.cfg.OpenAPIGateway.InternalToken)
	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("lease_request_failed:%w", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("lease_request_status:%d", resp.StatusCode)
	}
	var parsed executionLeaseResponse
	if err := json.Unmarshal(b, &parsed); err != nil {
		return nil, fmt.Errorf("lease_request_parse_failed:%w", err)
	}
	if !parsed.OK || len(parsed.Snapshot.Leases) == 0 {
		return nil, fmt.Errorf("lease_denied:%s", parsed.DenyReason)
	}
	chosen := parsed.Snapshot.Leases[0]
	return &lease{
		TokenID:   chosen.TokenID,
		AccountID: chosen.AccountID,
		LeaseID:   chosen.LeaseID,
		Lane:      chosen.Lane,
		SnapshotID: parsed.Snapshot.SnapshotID,
		DispatchEpoch: parsed.Snapshot.DispatchEpoch,
		Provider: chosen.Provider,
	}, nil
}

func (h *Handler) releaseLease(ctx context.Context, lease *lease) {
	if lease == nil {
		return
	}
	reqID := "rel-" + h.randHex8()
	payload := map[string]any{
		"meta": map[string]any{
			"contractName":    "cliproxyapi-execution-plane",
			"contractVersion": "v1alpha1",
			"schemaRevision":  h.cfg.OpenAPIGateway.SchemaRevision,
		},
		"requestId":  reqID,
		"leaseId":    lease.LeaseID,
		"tokenId":    lease.TokenID,
		"accountId":  lease.AccountID,
		"releasedAt": time.Now().UTC().Format(time.RFC3339Nano),
	}
	raw, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(h.cfg.OpenAPIGateway.ControlPlaneBaseURL, "/")+"/internal/execution/lease/release",
		bytes.NewReader(raw),
	)
	req.Header.Set("content-type", "application/json; charset=utf-8")
	req.Header.Set("x-openapi-internal-caller", h.cfg.OpenAPIGateway.InternalCaller)
	req.Header.Set("x-openapi-internal-token", h.cfg.OpenAPIGateway.InternalToken)
	resp, err := h.httpClient.Do(req)
	if err == nil && resp != nil {
		_ = resp.Body.Close()
	}
}

func (h *Handler) authenticateInternal(c *gin.Context) bool {
	if strings.TrimSpace(c.GetHeader("x-openapi-internal-caller")) != strings.TrimSpace(h.cfg.OpenAPIGateway.InternalCaller) {
		c.JSON(403, gin.H{"ok": false, "error": "forbidden"})
		return false
	}
	if strings.TrimSpace(c.GetHeader("x-openapi-internal-token")) != strings.TrimSpace(h.cfg.OpenAPIGateway.InternalToken) {
		c.JSON(403, gin.H{"ok": false, "error": "forbidden"})
		return false
	}
	return true
}

func (h *Handler) callUpstream(ctx context.Context, accessToken string, accountID string, stream bool, body []byte) (status int, contentType string, peek string, err error) {
	upstreamURL := strings.TrimRight(h.cfg.OpenAPIGateway.UpstreamBaseURL, "/") + "/responses"
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("User-Agent", codexUserAgent)
	req.Header.Set("Originator", codexOriginator)
	req.Header.Set("Connection", "Keep-Alive")
	if strings.TrimSpace(accountID) != "" {
		req.Header.Set("Chatgpt-Account-Id", strings.TrimSpace(accountID))
	}
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
	resp, err := h.httpClientProxied.Do(req)
	if err != nil {
		return 0, "", "", err
	}
	defer resp.Body.Close()
	ct := resp.Header.Get("content-type")
	payload, _ := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	return resp.StatusCode, ct, strings.TrimSpace(string(payload[:minInt(len(payload), 200)])), nil
}

func (h *Handler) getCredential(ctx context.Context, tokenID string) (*credentialMaterial, error) {
	req, _ := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		strings.TrimRight(h.cfg.OpenAPIGateway.ControlPlaneBaseURL, "/")+"/internal/execution/credential-material/"+url.PathEscape(tokenID),
		nil,
	)
	req.Header.Set("x-openapi-internal-caller", h.cfg.OpenAPIGateway.InternalCaller)
	req.Header.Set("x-openapi-internal-token", h.cfg.OpenAPIGateway.InternalToken)
	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("credential_fetch_failed:%w", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("credential_fetch_status:%d", resp.StatusCode)
	}
	var parsed credentialMaterial
	if err := json.Unmarshal(b, &parsed); err != nil {
		return nil, fmt.Errorf("credential_fetch_parse_failed:%w", err)
	}
	if !parsed.OK || parsed.AccessToken == "" {
		return nil, fmt.Errorf("credential_missing:%s", parsed.Error)
	}
	return &parsed, nil
}

func (h *Handler) ingestUsage(
	auth *publicAuthContext,
	requestedModel string,
	requestID string,
	traceID string,
	tokenID *string,
	accountID *string,
	elapsed time.Duration,
	success bool,
	failureReason string,
	httpStatus int,
	errorCode string,
	errorMessage string,
	streamCompleted bool,
	usage map[string]int,
) {
	payload := map[string]any{
		"ownerUserId": auth.OwnerUserID,
		"tenantId":    auth.TenantID,
		"keyId":       auth.KeyID,
		"protocol":    "responses",
		// Preserve the user-requested model id for usage accounting (OpenAPI authority truth).
		"model":               requestedModel,
		"requestId":           requestID,
		"traceId":             traceID,
		"success":             success,
		"tokenId":             tokenID,
		"accountId":           accountID,
		"errorCode":           nullIfEmpty(errorCode),
		"errorMessage":        nullIfEmpty(errorMessage),
		"httpStatus":          httpStatus,
		"failureReason":       nullIfEmpty(failureReason),
		"streamCompleted":     streamCompleted,
		"queueWaitMs":         0,
		"cliproxyRoundtripMs": int(elapsed.Milliseconds()),
		"upstreamTotalMs":     int(elapsed.Milliseconds()),
		"transportPrimary":    nil,
		"transportFinal":      nil,
		"fallbackUsed":        false,
		"observationSource":   "server_sync",
		"createdAt":           time.Now().UTC().Format(time.RFC3339Nano),
	}
	if usage != nil {
		payload["inputTokens"] = usage["input_tokens"]
		payload["outputTokens"] = usage["output_tokens"]
		payload["totalTokens"] = usage["total_tokens"]
	}
	raw, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		strings.TrimRight(h.cfg.OpenAPIGateway.ControlPlaneBaseURL, "/")+"/internal/public-api/usage",
		bytes.NewReader(raw),
	)
	req.Header.Set("content-type", "application/json; charset=utf-8")
	req.Header.Set("x-openapi-internal-caller", h.cfg.OpenAPIGateway.InternalCaller)
	req.Header.Set("x-openapi-internal-token", h.cfg.OpenAPIGateway.InternalToken)
	resp, err := h.httpClient.Do(req)
	if err != nil {
		log.WithError(err).Warn("openapi_gateway_usage_ingest_failed")
		return
	}
	_ = resp.Body.Close()
}

func (h *Handler) reportFromUpstreamResult(
	auth *publicAuthContext,
	lease *lease,
	requestedModel string,
	requestID string,
	traceID string,
	startedAt time.Time,
	finishedAt time.Time,
	httpStatus int,
	usage map[string]int,
	success bool,
	streamCompleted bool,
	failureReason string,
) {
	if lease == nil {
		return
	}
	if success {
		in := 0
		out := 0
		if usage != nil {
			in = usage["input_tokens"]
			out = usage["output_tokens"]
		}
		total := in + out
		event := map[string]any{
			"eventType": "usage_report",
			"identity":  h.feedbackIdentityUsage(lease, requestedModel, requestID, traceID),
			"startedAt": startedAt.UTC().Format(time.RFC3339Nano),
			"finishedAt": finishedAt.UTC().Format(time.RFC3339Nano),
			"latencyMs": int(finishedAt.Sub(startedAt).Milliseconds()),
			"outcome":   "success",
			"usage": map[string]any{
				"inputTokens":     in,
				"outputTokens":    out,
				"cachedTokens":    nil,
				"reasoningTokens": nil,
				// total is derived server-side; included for convenience in payloadJson only.
				"_totalTokens": total,
			},
			"httpStatus":     httpStatus,
			"providerStatus": nil,
		}
		h.ingestExecutionFeedback(event)
		return
	}

	// Unauthorized signal gets strongest quarantine semantics server-side.
	if httpStatus == 401 {
		event := map[string]any{
			"eventType":   "signal",
			"signalType":  "unauthorized",
			"signal": map[string]any{
				"identity":    h.feedbackIdentityBase(lease, requestID, traceID),
				"observedAt":  finishedAt.UTC().Format(time.RFC3339Nano),
				"httpStatus":  httpStatus,
				"providerReasonCode": nullIfEmpty(failureReason),
				"rawEvidenceRef":     nil,
			},
		}
		h.ingestExecutionFeedback(event)
		return
	}

	class := h.failureClassFromStatus(httpStatus, streamCompleted, failureReason)
	h.reportExecutionFailure(auth, lease, requestedModel, requestID, traceID, startedAt, finishedAt, class, httpStatus, "", "", class != "policy_denied")
}

func (h *Handler) reportExecutionFailure(
	auth *publicAuthContext,
	lease *lease,
	requestedModel string,
	requestID string,
	traceID string,
	startedAt time.Time,
	finishedAt time.Time,
	failureClass string,
	httpStatus int,
	providerErrorCode string,
	providerErrorMessage string,
	retryable bool,
) {
	if lease == nil {
		return
	}
	event := map[string]any{
		"eventType": "failure_feedback",
		"identity":  h.feedbackIdentityCommon(lease, requestID, traceID),
		"occurredAt": finishedAt.UTC().Format(time.RFC3339Nano),
		"outcome":   "failed",
		"failureClass": failureClass,
		"httpStatus":   httpStatus,
		"providerErrorCode": nullIfEmpty(providerErrorCode),
		"providerErrorMessageExcerpt": nullIfEmpty(providerErrorMessage),
		"retryable": retryable,
		"executionAction": "dropped",
	}
	h.ingestExecutionFeedback(event)
}

func (h *Handler) feedbackIdentityUsage(lease *lease, requestedModel string, requestID string, traceID string) map[string]any {
	identity := h.feedbackIdentityCommon(lease, requestID, traceID)
	identity["model"] = requestedModel
	return identity
}

func (h *Handler) feedbackIdentityCommon(lease *lease, requestID string, traceID string) map[string]any {
	provider := strings.TrimSpace(lease.Provider)
	if provider == "" {
		provider = "cliproxyapi"
	}
	return map[string]any{
		"eventId":         "fb-" + h.randHex8(),
		"requestId":       requestID,
		"traceId":         traceID,
		"idempotencyKey":  requestID + ":" + lease.LeaseID,
		"snapshotId":      lease.SnapshotID,
		"dispatchEpoch":   lease.DispatchEpoch,
		"lane":            lease.Lane,
		"leaseId":         lease.LeaseID,
		"tokenId":         lease.TokenID,
		"accountId":       lease.AccountID,
		"provider":        provider,
		"attemptNo":       1,
		"routeKey":        "codex.responses",
	}
}

func (h *Handler) feedbackIdentityBase(lease *lease, requestID string, traceID string) map[string]any {
	// executionBaseSignalSchema expects attemptNo nullable. Keep it null for signals.
	provider := strings.TrimSpace(lease.Provider)
	if provider == "" {
		provider = "cliproxyapi"
	}
	return map[string]any{
		"eventId":        "fb-" + h.randHex8(),
		"requestId":      requestID,
		"traceId":        traceID,
		"idempotencyKey": requestID + ":" + lease.LeaseID,
		"snapshotId":     lease.SnapshotID,
		"dispatchEpoch":  lease.DispatchEpoch,
		"lane":           lease.Lane,
		"leaseId":        lease.LeaseID,
		"tokenId":        lease.TokenID,
		"accountId":      lease.AccountID,
		"provider":       provider,
		"attemptNo":      nil,
		"routeKey":       "codex.responses",
	}
}

func (h *Handler) failureClassFromStatus(status int, streamCompleted bool, failureReason string) string {
	if status == 429 {
		return "provider_429"
	}
	if status >= 500 {
		return "provider_5xx"
	}
	if status >= 400 {
		// Treat policy-ish failures (cloudflare/html) as policy_denied when explicitly marked.
		if strings.Contains(failureReason, "cloudflare") || strings.Contains(failureReason, "policy") {
			return "policy_denied"
		}
		return "provider_4xx_other"
	}
	if status >= 200 && status < 300 && !streamCompleted {
		return "stream_aborted"
	}
	return "network_transient"
}

func (h *Handler) ingestExecutionFeedback(event map[string]any) {
	if h.cfg == nil || !h.cfg.OpenAPIGateway.Enabled {
		return
	}
	payload := map[string]any{
		"meta": map[string]any{
			"contractName":    "cliproxyapi-execution-plane",
			"contractVersion": "v1alpha1",
			"schemaRevision":  h.cfg.OpenAPIGateway.SchemaRevision,
		},
		"consumer": map[string]any{
			"executionInstanceId": h.cfg.OpenAPIGateway.ExecutionInstanceID,
			"executionNodeId":     h.cfg.OpenAPIGateway.ExecutionNodeID,
			"runtimeClass":        h.cfg.OpenAPIGateway.RuntimeClass,
			"region":              h.cfg.OpenAPIGateway.Region,
		},
		"reportedAt": time.Now().UTC().Format(time.RFC3339Nano),
		"events":     []any{event},
	}
	raw, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		strings.TrimRight(h.cfg.OpenAPIGateway.ControlPlaneBaseURL, "/")+"/internal/execution/feedback",
		bytes.NewReader(raw),
	)
	req.Header.Set("content-type", "application/json; charset=utf-8")
	req.Header.Set("x-openapi-internal-caller", h.cfg.OpenAPIGateway.InternalCaller)
	req.Header.Set("x-openapi-internal-token", h.cfg.OpenAPIGateway.InternalToken)
	resp, err := h.httpClient.Do(req)
	if err != nil {
		log.WithError(err).Warn("openapi_gateway_feedback_ingest_failed")
		return
	}
	_ = resp.Body.Close()
}

func extractUsage(payload any) map[string]int {
	obj, ok := payload.(map[string]any)
	if !ok {
		return nil
	}
	u, ok := obj["usage"].(map[string]any)
	if !ok {
		return nil
	}
	in := toInt(u["input_tokens"])
	out := toInt(u["output_tokens"])
	total := toInt(u["total_tokens"])
	if total == 0 {
		total = in + out
	}
	return map[string]int{
		"input_tokens":  in,
		"output_tokens": out,
		"total_tokens":  total,
	}
}

func toInt(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	default:
		return 0
	}
}

func classifyFailure(status int, streamCompleted bool) string {
	if status >= 200 && status < 300 && streamCompleted {
		return ""
	}
	if status == 401 {
		return "unauthorized"
	}
	if status == 429 {
		return "server_or_cliproxy_429"
	}
	if status >= 500 {
		return "provider_5xx"
	}
	if status >= 400 {
		return "provider_4xx_other"
	}
	if status >= 200 && status < 300 && !streamCompleted {
		return "stream_incomplete"
	}
	return "unknown"
}

func contains(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func nullIfEmpty(s string) any {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}
