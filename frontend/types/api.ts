/**
 * API Contract Types
 * 
 * This file defines the TypeScript interfaces for the backend API contract.
 * These types are LOCKED and should not be modified without backend coordination.
 * 
 * @version 1.0.0
 * @locked true
 */

// =============================================================================
// ENUMS & CONSTANTS
// =============================================================================

/**
 * Risk level classification for trades
 */
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

/**
 * Trade type classification
 */
export type TradeType = "BUY" | "SELL" | "TRANSFER";

// =============================================================================
// REQUEST TYPES
// =============================================================================

/**
 * Request payload for analyzing a trade
 */
export interface TradeRequest {
  /** Unique account identifier */
  account_id: string;
  /** Trade amount in USD */
  amount: number;
  /** Type of trade operation */
  type: TradeType;
  /** Optional metadata for additional context */
  metadata?: Record<string, unknown>;
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

/**
 * Response payload from trade analysis
 * 
 * @example
 * {
 *   trade_id: "TRD-001",
 *   risk_score: 85,
 *   risk_level: "HIGH",
 *   anomaly_score: 0.92,
 *   explanation: "Unusual transfer volume exceeds 5x daily average",
 *   timestamp: "2026-02-04T14:32:15.000Z"
 * }
 */
export interface TradeResponse {
  /** Unique trade identifier */
  trade_id: string;
  /** Risk score from 0 to 100 */
  risk_score: number;
  /** Categorized risk level */
  risk_level: RiskLevel;
  /** Anomaly detection score (normalized 0-1) */
  anomaly_score: number;
  /** Human-readable explanation of the risk assessment */
  explanation: string;
  /** ISO 8601 formatted timestamp */
  timestamp: string;
}

/**
 * Alert response for flagged activities
 * 
 * NOTE: Backend returns alerts in this format (matches TradeResponse subset)
 * This differs slightly from the original spec - trade_id is the identifier
 * 
 * @example
 * {
 *   trade_id: "TRD-003",
 *   risk_score: 85,
 *   explanation: "Cross-border transfer to flagged jurisdiction",
 *   timestamp: "2026-02-04T14:25:12.000Z"
 * }
 */
export interface AlertResponse {
  /** Trade identifier (used as alert ID) */
  trade_id: string;
  /** Risk score from 0 to 100 */
  risk_score: number;
  /** Human-readable explanation of why the alert was triggered */
  explanation: string;
  /** ISO 8601 formatted timestamp when alert was generated */
  timestamp: string;
}

/**
 * Legacy alert format for UI compatibility
 * Maps backend response to UI expected format
 */
export interface AlertResponseUI {
  /** Unique alert identifier (generated from trade_id) */
  alert_id: string;
  /** Associated trade identifier */
  trade_id: string;
  /** Risk level classification */
  risk_level: RiskLevel;
  /** Human-readable explanation of why the alert was triggered */
  explanation: string;
  /** ISO 8601 formatted timestamp when alert was generated */
  timestamp: string;
}

// =============================================================================
// EXTENDED TYPES (for UI compatibility)
// =============================================================================

/**
 * Extended trade response with additional UI fields
 * Used for display purposes in the dashboard
 */
export interface TradeResponseExtended extends TradeResponse {
  /** Account identifier for display */
  account_id: string;
  /** Trade amount in USD */
  amount: number;
  /** Type of trade */
  type: TradeType;
  /** Trade status for UI display */
  status: "Normal" | "Flagged";
}

/**
 * Anomaly data point for time-series charts
 */
export interface AnomalyDataPoint {
  /** Time label (e.g., "14:00") */
  time: string;
  /** Anomaly score value */
  score: number;
}

/**
 * Dashboard statistics summary
 */
export interface DashboardStats {
  /** Total number of trades processed */
  total_trades: number;
  /** Number of flagged trades */
  flagged_trades: number;
  /** Number of high-risk alerts */
  high_risk_alerts: number;
  /** Average risk score across all trades */
  average_risk_score: number;
}
