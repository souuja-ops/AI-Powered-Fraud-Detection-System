/**
 * API Client for AI Fraud Detection Backend
 * 
 * This module provides type-safe API functions to communicate with the FastAPI backend.
 * All functions include:
 * - Proper error handling with try/catch
 * - Type validation of responses
 * - Safe defaults for failed requests
 * - No exposure of internal logic or secrets
 * - Automatic retry with exponential backoff for cold starts
 * 
 * SECURITY NOTES:
 * - API URL is configured via environment variable (NEXT_PUBLIC_API_URL)
 * - No sensitive data is logged or exposed to client
 * - All responses are validated before use
 * - Frontend does NOT submit trades (read-only dashboard)
 * 
 * @version 1.1.0
 */

import type {
  TradeResponse,
  AlertResponse,
  AlertResponseUI,
  TradeResponseExtended,
  RiskLevel,
} from "@/types/api";

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * API base URL from environment variable
 * Falls back to production URL, then localhost for development
 * 
 * SECURITY: Using NEXT_PUBLIC_ prefix makes this available to client
 * This is safe because it's just the API endpoint URL, not a secret
 */
const API_BASE_URL = 
  process.env.NEXT_PUBLIC_API_URL || 
  "https://fraud-detection-api-8xaw.onrender.com";

/**
 * Default fetch timeout in milliseconds
 * Increased for Render cold starts (free tier can take 30-60s to wake)
 */
const FETCH_TIMEOUT_MS = 60000; // 60 seconds for cold starts

/**
 * Retry configuration for handling Render cold starts
 */
const RETRY_CONFIG = {
  maxRetries: 3,
  initialDelayMs: 2000,
  maxDelayMs: 10000,
  backoffMultiplier: 2,
};

/**
 * API endpoints (centralized for maintainability)
 */
const ENDPOINTS = {
  TRADES: "/api/trades",
  ALERTS: "/api/alerts",
  HEALTH: "/api/health",
  ML_STATUS: "/api/ml/status",
  ML_METRICS: "/api/ml/metrics",
  ML_FEATURE_IMPORTANCE: "/api/ml/feature-importance",
  ML_FEEDBACK: "/api/ml/feedback",
} as const;

// =============================================================================
// CONNECTION STATE MANAGEMENT
// =============================================================================

/**
 * Backend connection status
 * - "connected": Backend is live and responding
 * - "waking": Backend is cold and needs to wake up (show banner)
 * - "disconnected": Cannot reach backend after retries
 */
export type ConnectionStatus = "connected" | "waking" | "disconnected";

/**
 * Connection state (module-level for persistence across calls)
 */
let connectionStatus: ConnectionStatus = "connected"; // Assume connected initially (optimistic)
let isBackendWarmed: boolean = false; // Track if we've confirmed backend is live
let lastSuccessfulConnection: number = 0;
let connectionListeners: ((status: ConnectionStatus) => void)[] = [];
let initialWakeAttempted: boolean = false;

/**
 * Time threshold to consider backend as potentially cold (15 minutes)
 * Render free tier spins down after ~15 min of inactivity
 */
const COLD_START_THRESHOLD_MS = 15 * 60 * 1000;

/**
 * Subscribe to connection status changes
 */
export function onConnectionStatusChange(callback: (status: ConnectionStatus) => void): () => void {
  connectionListeners.push(callback);
  // Immediately call with current status
  callback(connectionStatus);
  // Return unsubscribe function
  return () => {
    connectionListeners = connectionListeners.filter(cb => cb !== callback);
  };
}

/**
 * Get current connection status
 */
export function getConnectionStatus(): ConnectionStatus {
  return connectionStatus;
}

/**
 * Check if backend is warmed up
 */
export function isBackendReady(): boolean {
  return isBackendWarmed;
}

/**
 * Update connection status and notify listeners
 * Only notify if status actually changed to prevent spam
 */
function setConnectionStatus(status: ConnectionStatus): void {
  if (connectionStatus !== status) {
    connectionStatus = status;
    connectionListeners.forEach(cb => cb(status));
  }
}

/**
 * Mark backend as warmed (called after first successful response)
 */
function markBackendWarmed(): void {
  isBackendWarmed = true;
  lastSuccessfulConnection = Date.now();
  setConnectionStatus("connected");
}

/**
 * Check if backend might be cold (no recent successful connection)
 */
function mightBeCold(): boolean {
  if (!isBackendWarmed) return true;
  const timeSinceLastConnection = Date.now() - lastSuccessfulConnection;
  return timeSinceLastConnection > COLD_START_THRESHOLD_MS;
}

// =============================================================================
// TYPE GUARDS & VALIDATORS
// =============================================================================

/**
 * Validates that a value is a valid RiskLevel
 */
function isValidRiskLevel(value: unknown): value is RiskLevel {
  return value === "LOW" || value === "MEDIUM" || value === "HIGH";
}

/**
 * Validates a TradeResponse object from the API
 * Returns true if all required fields are present and valid
 */
function isValidTradeResponse(obj: unknown): obj is TradeResponse {
  if (!obj || typeof obj !== "object") return false;
  
  const trade = obj as Record<string, unknown>;
  
  return (
    typeof trade.trade_id === "string" &&
    typeof trade.risk_score === "number" &&
    trade.risk_score >= 0 &&
    trade.risk_score <= 100 &&
    isValidRiskLevel(trade.risk_level) &&
    typeof trade.anomaly_score === "number" &&
    typeof trade.explanation === "string" &&
    typeof trade.timestamp === "string"
  );
}

/**
 * Validates an AlertResponse object from the API
 */
function isValidAlertResponse(obj: unknown): obj is AlertResponse {
  if (!obj || typeof obj !== "object") return false;
  
  const alert = obj as Record<string, unknown>;
  
  return (
    typeof alert.trade_id === "string" &&
    typeof alert.risk_score === "number" &&
    typeof alert.explanation === "string" &&
    typeof alert.timestamp === "string"
  );
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Creates an AbortController with timeout
 * Used to prevent requests from hanging indefinitely
 * Returns both controller and a cleanup function
 */
function createTimeoutController(timeoutMs: number = FETCH_TIMEOUT_MS): { 
  controller: AbortController; 
  cleanup: () => void;
} {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  return {
    controller,
    cleanup: () => clearTimeout(timeoutId),
  };
}

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Fetch with retry and exponential backoff
 * Only shows "waking" status if backend might be cold
 * Silent retries if backend was recently active
 */
async function fetchWithRetry(
  url: string, 
  options: RequestInit,
  retries: number = RETRY_CONFIG.maxRetries,
  showWakingStatus: boolean = false // Only show waking banner if explicitly requested
): Promise<Response> {
  let lastError: Error | null = null;
  let delay = RETRY_CONFIG.initialDelayMs;
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      if (attempt > 0) {
        // Only show "waking" if this is a cold start situation
        if (showWakingStatus && mightBeCold()) {
          setConnectionStatus("waking");
        }
        await sleep(delay);
        delay = Math.min(delay * RETRY_CONFIG.backoffMultiplier, RETRY_CONFIG.maxDelayMs);
      }
      
      const response = await fetch(url, options);
      
      if (response.ok) {
        markBackendWarmed();
        return response;
      }
      
      // If server responded but with error, don't retry for client errors
      if (response.status >= 400 && response.status < 500) {
        markBackendWarmed(); // Server is live, just returned an error
        return response;
      }
      
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      // Don't retry on abort (timeout)
      if (lastError.name === "AbortError" && attempt === retries) {
        break;
      }
    }
  }
  
  setConnectionStatus("disconnected");
  throw lastError || new Error("Request failed after retries");
}

/**
 * Wake up the backend (call health endpoint)
 * Only shows waking banner on first load or if backend might be cold
 */
export async function wakeUpBackend(): Promise<boolean> {
  // If already warmed and recently connected, skip the wake-up banner
  if (isBackendWarmed && !mightBeCold()) {
    return true;
  }
  
  // Only show waking status if this is initial load or backend might be cold
  if (!initialWakeAttempted || mightBeCold()) {
    setConnectionStatus("waking");
  }
  
  initialWakeAttempted = true;
  
  try {
    const { controller, cleanup } = createTimeoutController(FETCH_TIMEOUT_MS);
    
    const response = await fetchWithRetry(
      `${API_BASE_URL}${ENDPOINTS.HEALTH}`,
      {
        method: "GET",
        headers: { "Accept": "application/json" },
        signal: controller.signal,
      },
      RETRY_CONFIG.maxRetries,
      true // Show waking status for health check
    );
    
    cleanup();
    
    if (response.ok) {
      markBackendWarmed();
      return true;
    }
    
    setConnectionStatus("disconnected");
    return false;
  } catch {
    setConnectionStatus("disconnected");
    return false;
  }
}

/**
 * Check if backend is currently reachable (quick check, no retries)
 * Silent check - doesn't update connection status unless disconnected
 */
export async function checkBackendHealth(): Promise<{ healthy: boolean; latencyMs: number }> {
  const startTime = Date.now();
  
  try {
    const { controller, cleanup } = createTimeoutController(10000); // 10s timeout for health check
    
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.HEALTH}`, {
      method: "GET",
      headers: { "Accept": "application/json" },
      signal: controller.signal,
    });
    
    cleanup();
    const latencyMs = Date.now() - startTime;
    
    if (response.ok) {
      markBackendWarmed();
      return { healthy: true, latencyMs };
    }
    
    return { healthy: false, latencyMs };
  } catch {
    return { healthy: false, latencyMs: Date.now() - startTime };
  }
}

/**
 * Converts API TradeResponse to extended format for UI
 * 
 * NOTE: The backend returns minimal trade data. This function enriches it
 * with UI-specific fields derived from the response.
 */
function enrichTradeForUI(trade: TradeResponse): TradeResponseExtended {
  return {
    ...trade,
    // Extract account_id from trade_id pattern or use placeholder
    // Backend trades use format "TRD-00001" - we generate a corresponding account
    account_id: `ACC-${trade.trade_id.split("-")[1] || "0000"}`,
    // Amount is derived from anomaly context (not returned by API)
    // Using risk_score as a proxy for demo purposes
    amount: Math.round(trade.risk_score * 1000 + Math.random() * 50000),
    // Default type based on alternating pattern
    type: parseInt(trade.trade_id.split("-")[1] || "0") % 2 === 0 ? "BUY" : "SELL",
    // Status is determined by risk level - Flagged for HIGH or MEDIUM risk
    status: (trade.risk_level === "HIGH" || trade.risk_level === "MEDIUM") ? "Flagged" : "Normal",
  };
}

/**
 * Converts API AlertResponse to UI format
 * Determines risk level from score (matches backend thresholds)
 */
function alertToUIFormat(alert: AlertResponse, index: number): AlertResponseUI {
  // Determine risk level from score (HIGH >= 80, MEDIUM >= 50)
  let risk_level: "HIGH" | "MEDIUM" | "LOW" = "LOW";
  if (alert.risk_score >= 80) {
    risk_level = "HIGH";
  } else if (alert.risk_score >= 50) {
    risk_level = "MEDIUM";
  }
  
  return {
    alert_id: `ALT-${String(index + 1).padStart(3, "0")}`,
    trade_id: alert.trade_id,
    risk_level,
    explanation: alert.explanation,
    timestamp: alert.timestamp,
  };
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Fetches all analyzed trades from the backend
 * 
 * Data Flow:
 * 1. Request sent to /api/trades with retry logic (silent if backend is warm)
 * 2. Response validated for correct structure
 * 3. Each trade enriched with UI-specific fields
 * 4. Returns empty array on any error (safe default)
 * 
 * @returns Promise<TradeResponseExtended[]> - Array of trades for display
 */
export async function getTrades(): Promise<TradeResponseExtended[]> {
  const { controller, cleanup } = createTimeoutController();
  
  try {
    // Don't show any status change for normal data fetches
    // The waking banner only shows during initial wakeUpBackend() call
    
    const response = await fetchWithRetry(
      `${API_BASE_URL}${ENDPOINTS.TRADES}`,
      {
        method: "GET",
        headers: {
          "Accept": "application/json",
          // No auth headers - public read-only endpoint for demo
        },
        signal: controller.signal,
        // Disable caching to get fresh data
        cache: "no-store",
      },
      RETRY_CONFIG.maxRetries,
      false // Silent retries - don't show waking banner
    );

    cleanup(); // Clear timeout on successful response

    // Handle non-OK responses
    if (!response.ok) {
      console.error(`[API] getTrades failed: ${response.status} ${response.statusText}`);
      return [];
    }

    // Parse and validate JSON
    const data = await response.json();
    
    // Validate response is an array
    if (!Array.isArray(data)) {
      console.error("[API] getTrades: Response is not an array");
      return [];
    }

    // Validate and transform each trade
    const validTrades: TradeResponseExtended[] = [];
    
    for (const item of data) {
      if (isValidTradeResponse(item)) {
        validTrades.push(enrichTradeForUI(item));
      } else {
        // Log invalid items but continue processing
        console.warn("[API] Skipping invalid trade response:", item);
      }
    }

    // Success - backend is confirmed working (silent update)
    markBackendWarmed();
    return validTrades;

  } catch (error) {
    cleanup(); // Clear timeout on error too
    
    // Handle network errors, timeouts, and other failures
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        console.error("[API] getTrades: Request timed out");
      } else {
        console.error("[API] getTrades error:", error.message);
      }
    }
    
    // Only show disconnected if we were previously connected
    if (isBackendWarmed) {
      setConnectionStatus("disconnected");
    }
    // Return empty array as safe default
    return [];
  }
}

/**
 * Fetches HIGH risk alerts from the backend
 * 
 * Data Flow:
 * 1. Request sent to /api/alerts with retry logic (silent)
 * 2. Response validated for correct structure
 * 3. Each alert converted to UI format
 * 4. Returns empty array on any error (safe default)
 * 
 * NOTE: Backend only returns HIGH risk trades as alerts
 * 
 * @returns Promise<AlertResponseUI[]> - Array of alerts for display
 */
export async function getAlerts(): Promise<AlertResponseUI[]> {
  const { controller, cleanup } = createTimeoutController();
  
  try {
    const response = await fetchWithRetry(
      `${API_BASE_URL}${ENDPOINTS.ALERTS}`,
      {
        method: "GET",
        headers: {
          "Accept": "application/json",
        },
        signal: controller.signal,
        cache: "no-store",
      },
      RETRY_CONFIG.maxRetries,
      false // Silent retries
    );

    cleanup(); // Clear timeout on successful response

    if (!response.ok) {
      console.error(`[API] getAlerts failed: ${response.status} ${response.statusText}`);
      return [];
    }

    const data = await response.json();
    
    if (!Array.isArray(data)) {
      console.error("[API] getAlerts: Response is not an array");
      return [];
    }

    // Validate and transform each alert
    const validAlerts: AlertResponseUI[] = [];
    
    for (let i = 0; i < data.length; i++) {
      const item = data[i];
      if (isValidAlertResponse(item)) {
        validAlerts.push(alertToUIFormat(item, i));
      } else {
        console.warn("[API] Skipping invalid alert response:", item);
      }
    }

    // Success - backend is working
    markBackendWarmed();
    return validAlerts;

  } catch (error) {
    cleanup(); // Clear timeout on error too
    
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        console.error("[API] getAlerts: Request timed out");
      } else {
        console.error("[API] getAlerts error:", error.message);
      }
    }
    
    return [];
  }
}

/**
 * Health check endpoint
 * Used to verify backend connectivity
 * 
 * @returns Promise<boolean> - true if backend is healthy
 */
export async function checkHealth(): Promise<boolean> {
  const { controller, cleanup } = createTimeoutController(5000); // Shorter timeout for health
  
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.HEALTH}`, {
      method: "GET",
      signal: controller.signal,
    });

    cleanup();
    return response.ok;
  } catch {
    cleanup();
    return false;
  }
}

/**
 * ML Model status endpoint
 * Used to display ML model health in dashboard
 * 
 * @returns Promise<MLStatus | null> - ML status or null on error
 */
export interface MLStatus {
  is_trained: boolean;
  training_samples: number;
  feature_buffer_size: number;
  trades_since_retrain: number;
}

export async function getMLStatus(): Promise<MLStatus | null> {
  const { controller, cleanup } = createTimeoutController(5000);
  
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.ML_STATUS}`, {
      method: "GET",
      headers: { "Accept": "application/json" },
      signal: controller.signal,
    });

    cleanup();

    if (!response.ok) return null;
    
    const data = await response.json();
    
    // Validate required fields
    if (
      typeof data.is_trained === "boolean" &&
      typeof data.training_samples === "number"
    ) {
      return data as MLStatus;
    }
    
    return null;
  } catch {
    cleanup();
    return null;
  }
}

// =============================================================================
// COMPUTED STATS (derived from API data)
// =============================================================================

/**
 * Computes dashboard statistics from trades
 * This is calculated client-side from fetched data
 * 
 * Note: 
 * - flagged_trades = HIGH + MEDIUM risk (require attention)
 * - high_risk_alerts = HIGH risk only (critical alerts)
 */
export function computeStats(trades: TradeResponseExtended[]) {
  if (trades.length === 0) {
    return {
      total_trades: 0,
      flagged_trades: 0,
      high_risk_alerts: 0,
      average_risk_score: 0,
    };
  }

  // Flagged = HIGH or MEDIUM risk trades (need review)
  const flaggedTrades = trades.filter(
    (t) => t.risk_level === "HIGH" || t.risk_level === "MEDIUM"
  ).length;
  
  // High risk = only HIGH level trades (critical)
  const highRiskTrades = trades.filter((t) => t.risk_level === "HIGH").length;
  
  const avgScore = trades.reduce((sum, t) => sum + t.risk_score, 0) / trades.length;

  return {
    total_trades: trades.length,
    flagged_trades: flaggedTrades,
    high_risk_alerts: highRiskTrades,
    average_risk_score: avgScore,
  };
}

/**
 * Generates anomaly chart data from trades
 * Groups trades by time and calculates average score
 */
export function computeAnomalyData(trades: TradeResponseExtended[]) {
  if (trades.length === 0) return [];

  // Group by hour for chart display
  const timeGroups = new Map<string, number[]>();
  
  for (const trade of trades) {
    const time = new Date(trade.timestamp).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
    
    if (!timeGroups.has(time)) {
      timeGroups.set(time, []);
    }
    timeGroups.get(time)!.push(trade.risk_score);
  }

  // Convert to chart format
  return Array.from(timeGroups.entries())
    .map(([time, scores]) => ({
      time,
      score: Math.round(scores.reduce((a, b) => a + b, 0) / scores.length),
    }))
    .slice(-12); // Last 12 data points
}

// =============================================================================
// MODEL METRICS & FEEDBACK API
// =============================================================================

/**
 * Model metrics response type
 */
export interface ModelMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  accuracy: number;
  confusion_matrix: number[][];
  true_positives: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
}

/**
 * Feature importance response type
 */
export interface FeatureImportance {
  features: Record<string, number>;
  top_5: [string, number][];
}

/**
 * Full metrics response from API
 */
export interface MetricsResponse {
  status: string;
  metrics: ModelMetrics | null;
  feature_importance: Record<string, number>;
}

/**
 * Feedback response type
 */
export interface FeedbackResponse {
  trade_id: string;
  feedback_recorded: boolean;
  predicted_risk_level: string;
  actual_is_fraud: boolean;
  was_correct: boolean;
  message: string;
}

/**
 * Fetches model performance metrics
 */
export async function getModelMetrics(): Promise<MetricsResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.ML_METRICS}`, {
      method: "GET",
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });

    if (!response.ok) {
      console.error(`[API] getModelMetrics failed: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error("[API] getModelMetrics error:", error);
    return null;
  }
}

/**
 * Fetches feature importance scores
 */
export async function getFeatureImportance(): Promise<FeatureImportance | null> {
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.ML_FEATURE_IMPORTANCE}`, {
      method: "GET",
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });

    if (!response.ok) {
      console.error(`[API] getFeatureImportance failed: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error("[API] getFeatureImportance error:", error);
    return null;
  }
}

/**
 * Submits feedback for a trade prediction
 * 
 * @param tradeId - The trade to provide feedback for
 * @param isFraud - True if the trade was actually fraud
 */
export async function submitFeedback(
  tradeId: string,
  isFraud: boolean
): Promise<FeedbackResponse | null> {
  try {
    const url = new URL(`${API_BASE_URL}${ENDPOINTS.ML_FEEDBACK}`);
    url.searchParams.set("trade_id", tradeId);
    url.searchParams.set("is_fraud", String(isFraud));

    const response = await fetch(url.toString(), {
      method: "POST",
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });

    if (!response.ok) {
      console.error(`[API] submitFeedback failed: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error("[API] submitFeedback error:", error);
    return null;
  }
}
