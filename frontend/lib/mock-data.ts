/**
 * Mock Data for AI Fraud Detection Dashboard
 * 
 * This file provides mock data that EXACTLY matches the API contract types.
 * Used for development and testing before backend integration.
 */

import type {
  AlertResponseUI,
  TradeResponseExtended,
  AnomalyDataPoint,
  DashboardStats,
} from "@/types/api";

// =============================================================================
// MOCK TRADE RESPONSES
// =============================================================================

export const mockTrades: TradeResponseExtended[] = [
  {
    trade_id: "TRD-001",
    account_id: "ACC-7821",
    trade_amount: 125000,
    trade_type: "BUY",
    amount: 125000,
    type: "BUY",
    timestamp: "2026-02-04T14:32:15.000Z",
    risk_score: 85,
    risk_level: "HIGH",
    anomaly_score: 0.85,
    explanation: "Rapid succession of trades from new account",
    status: "Flagged",
  },
  {
    trade_id: "TRD-002",
    account_id: "ACC-4523",
    trade_amount: 2500,
    trade_type: "SELL",
    amount: 2500,
    type: "SELL",
    timestamp: "2026-02-04T14:28:42.000Z",
    risk_score: 23,
    risk_level: "LOW",
    anomaly_score: 0.23,
    explanation: "No significant anomalies detected. Trade patterns are consistent with account history.",
    status: "Normal",
  },
  {
    trade_id: "TRD-003",
    account_id: "ACC-9012",
    trade_amount: 450000,
    trade_type: "TRANSFER",
    amount: 450000,
    type: "TRANSFER",
    timestamp: "2026-02-04T14:25:10.000Z",
    risk_score: 92,
    risk_level: "HIGH",
    anomaly_score: 0.92,
    explanation: "Unusual transfer volume exceeds 5x daily average",
    status: "Flagged",
  },
  {
    trade_id: "TRD-004",
    account_id: "ACC-3345",
    trade_amount: 1200,
    trade_type: "BUY",
    amount: 1200,
    type: "BUY",
    timestamp: "2026-02-04T14:22:33.000Z",
    risk_score: 15,
    risk_level: "LOW",
    anomaly_score: 0.15,
    explanation: "No significant anomalies detected. Trade patterns are consistent with account history.",
    status: "Normal",
  },
  {
    trade_id: "TRD-005",
    account_id: "ACC-6677",
    trade_amount: 35000,
    trade_type: "SELL",
    amount: 35000,
    type: "SELL",
    timestamp: "2026-02-04T14:18:55.000Z",
    risk_score: 55,
    risk_level: "MEDIUM",
    anomaly_score: 0.55,
    explanation: "Trade pattern deviates from account history",
    status: "Flagged",
  },
  {
    trade_id: "TRD-006",
    account_id: "ACC-1199",
    trade_amount: 89000,
    trade_type: "BUY",
    amount: 89000,
    type: "BUY",
    timestamp: "2026-02-04T14:15:20.000Z",
    risk_score: 82,
    risk_level: "HIGH",
    anomaly_score: 0.82,
    explanation: "Transaction pattern indicates potential layering activity",
    status: "Flagged",
  },
  {
    trade_id: "TRD-007",
    account_id: "ACC-8844",
    trade_amount: 800,
    trade_type: "SELL",
    amount: 800,
    type: "SELL",
    timestamp: "2026-02-04T14:12:08.000Z",
    risk_score: 12,
    risk_level: "LOW",
    anomaly_score: 0.12,
    explanation: "No significant anomalies detected. Trade patterns are consistent with account history.",
    status: "Normal",
  },
  {
    trade_id: "TRD-008",
    account_id: "ACC-2233",
    trade_amount: 275000,
    trade_type: "TRANSFER",
    amount: 275000,
    type: "TRANSFER",
    timestamp: "2026-02-04T14:08:45.000Z",
    risk_score: 88,
    risk_level: "HIGH",
    anomaly_score: 0.88,
    explanation: "Cross-border transfer to flagged jurisdiction",
    status: "Flagged",
  },
];

// =============================================================================
// MOCK ALERT RESPONSES
// =============================================================================

export const mockAlerts: AlertResponseUI[] = [
  {
    alert_id: "ALT-001",
    trade_id: "TRD-003",
    risk_level: "HIGH",
    explanation: "Unusual transfer volume exceeds 5x daily average",
    timestamp: "2026-02-04T14:25:12.000Z",
  },
  {
    alert_id: "ALT-002",
    trade_id: "TRD-001",
    risk_level: "HIGH",
    explanation: "Rapid succession of trades from new account",
    timestamp: "2026-02-04T14:32:18.000Z",
  },
  {
    alert_id: "ALT-003",
    trade_id: "TRD-008",
    risk_level: "HIGH",
    explanation: "Cross-border transfer to flagged jurisdiction",
    timestamp: "2026-02-04T14:08:48.000Z",
  },
  {
    alert_id: "ALT-004",
    trade_id: "TRD-006",
    risk_level: "MEDIUM",
    explanation: "Trade pattern deviates from account history",
    timestamp: "2026-02-04T14:15:25.000Z",
  },
];

// =============================================================================
// MOCK ANOMALY DATA
// =============================================================================

export const mockAnomalyData: AnomalyDataPoint[] = [
  { time: "00:00", score: 12 },
  { time: "01:00", score: 8 },
  { time: "02:00", score: 15 },
  { time: "03:00", score: 10 },
  { time: "04:00", score: 22 },
  { time: "05:00", score: 18 },
  { time: "06:00", score: 25 },
  { time: "07:00", score: 35 },
  { time: "08:00", score: 42 },
  { time: "09:00", score: 38 },
  { time: "10:00", score: 55 },
  { time: "11:00", score: 72 },
  { time: "12:00", score: 85 },
  { time: "13:00", score: 68 },
  { time: "14:00", score: 78 },
];

// =============================================================================
// MOCK DASHBOARD STATS
// =============================================================================

export const mockStats: DashboardStats = {
  total_trades: 2847,
  flagged_trades: 127,
  high_risk_alerts: 24,
  average_risk_score: 34.2,
};

// =============================================================================
// MOCK API FUNCTIONS (for development)
// =============================================================================

/**
 * Simulated delay to mimic network latency
 */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Mock implementation of fetchTrades
 */
export async function mockFetchTrades(): Promise<TradeResponseExtended[]> {
  await delay(300);
  return mockTrades;
}

/**
 * Mock implementation of fetchAlerts
 */
export async function mockFetchAlerts(): Promise<AlertResponseUI[]> {
  await delay(200);
  return mockAlerts;
}

/**
 * Mock implementation of fetchAnomalyData
 */
export async function mockFetchAnomalyData(): Promise<AnomalyDataPoint[]> {
  await delay(250);
  return mockAnomalyData;
}

/**
 * Mock implementation of fetchDashboardStats
 */
export async function mockFetchDashboardStats(): Promise<DashboardStats> {
  await delay(150);
  return mockStats;
}

// =============================================================================
// TYPE EXPORTS (for backward compatibility with existing components)
// =============================================================================

// Re-export types for components that import from mock-data
export type { TradeResponseExtended as Trade } from "@/types/api";
export type { AlertResponseUI as Alert } from "@/types/api";
export type { AnomalyDataPoint } from "@/types/api";
