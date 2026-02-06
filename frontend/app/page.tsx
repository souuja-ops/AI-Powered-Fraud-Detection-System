"use client"

/**
 * AI Fraud Detection Dashboard
 * 
 * Real-time dashboard for monitoring trade anomalies and fraud detection.
 * 
 * DATA FLOW:
 * 1. On mount, fetches initial data from backend
 * 2. Sets up polling interval (every 5 seconds) for live updates
 * 3. Validates and transforms API responses for display
 * 4. Falls back to mock data if backend is unavailable
 * 
 * SECURITY NOTES:
 * - Frontend is READ-ONLY (does not submit trades)
 * - No sensitive data is stored in localStorage/cookies
 * - API errors are handled gracefully without exposing internals
 * - All data is validated before rendering
 * 
 * @version 2.0.0 - Live data integration
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { 
  AlertTriangle, 
  ArrowUpRight, 
  Activity, 
  Shield, 
  TrendingUp, 
  WifiOff,
  RefreshCw,
  Brain
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts"

// New components for enhanced ML features
import { ModelPerformanceDashboard } from "@/components/model-performance"
import { AlertsPanel } from "@/components/alerts-panel"
import { TradesTable } from "@/components/trades-table"
import { ThreatMonitor } from "@/components/threat-monitor"
import { CompliancePanel } from "@/components/compliance-panel"

// API client functions
import { 
  getTrades, 
  getAlerts, 
  checkHealth,
  getMLStatus,
  computeStats,
  computeAnomalyData,
  submitFeedback,
  type MLStatus,
} from "@/lib/api"
import { toast } from "sonner"

// Types
import type { TradeResponseExtended, AlertResponseUI, DashboardStats, AnomalyDataPoint } from "@/types/api"

// Fallback mock data (used when backend is unavailable)
import { mockTrades, mockAlerts, mockAnomalyData, mockStats } from "@/lib/mock-data"

// =============================================================================
// CONFIGURATION
// =============================================================================

/** Polling interval for live data updates (milliseconds) */
const POLLING_INTERVAL_MS = 5000

/** Health check interval (milliseconds) */
const HEALTH_CHECK_INTERVAL_MS = 15000

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  })
}

// =============================================================================
// UI COMPONENTS
// =============================================================================

function ConnectionStatus({ isConnected, isLoading }: { isConnected: boolean; isLoading: boolean }) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
        <span className="text-muted-foreground">Connecting...</span>
      </div>
    )
  }
  
  if (isConnected) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-success"></span>
        </span>
        <span className="text-muted-foreground">Live</span>
      </div>
    )
  }
  
  return (
    <div className="flex items-center gap-2 text-sm">
      <WifiOff className="h-4 w-4 text-destructive" />
      <span className="text-destructive">Offline (Mock Data)</span>
    </div>
  )
}

function MLStatusBadge({ status }: { status: MLStatus | null }) {
  if (!status) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Brain className="h-3 w-3" />
        <span>ML: Unavailable</span>
      </div>
    )
  }
  
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <Brain className="h-3 w-3 text-success" />
      <span>ML: Active ({status.training_samples} samples)</span>
    </div>
  )
}

function StatCard({
  title,
  value,
  icon: Icon,
  trend,
  trendUp,
  isLoading,
}: {
  title: string
  value: string | number
  icon: React.ElementType
  trend?: string
  trendUp?: boolean
  isLoading?: boolean
}) {
  return (
    <Card className="bg-card border-border hover:border-accent/30 transition-colors group">
      <CardHeader className="flex flex-row items-center justify-between pb-1 sm:pb-2 space-y-0 p-3 sm:p-6">
        <CardTitle className="text-[10px] sm:text-xs font-medium text-muted-foreground uppercase tracking-wider">{title}</CardTitle>
        <div className="p-1 sm:p-1.5 rounded-md bg-secondary/50 group-hover:bg-accent/10 transition-colors">
          <Icon className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground group-hover:text-accent transition-colors" />
        </div>
      </CardHeader>
      <CardContent className="pt-0 p-3 sm:p-6 sm:pt-0">
        <div className={`text-xl sm:text-2xl md:text-3xl font-bold text-foreground transition-opacity tabular-nums ${isLoading ? 'opacity-50 animate-pulse' : ''}`}>
          {value}
        </div>
        {trend && (
          <p className={`text-[10px] sm:text-xs mt-1 sm:mt-1.5 flex items-center gap-1 ${trendUp ? "text-success" : "text-muted-foreground"}`}>
            <ArrowUpRight className={`h-2.5 w-2.5 sm:h-3 sm:w-3 ${!trendUp && "rotate-90"}`} />
            <span className="hidden sm:inline">{trend}</span>
            <span className="sm:hidden">{trendUp ? "Live" : "Mock"}</span>
          </p>
        )}
      </CardContent>
    </Card>
  )
}

// =============================================================================
// MAIN DASHBOARD COMPONENT
// =============================================================================

export default function FraudDetectionDashboard() {
  // Data state
  const [trades, setTrades] = useState<TradeResponseExtended[]>([])
  const [alerts, setAlerts] = useState<AlertResponseUI[]>([])
  const [stats, setStats] = useState<DashboardStats>(mockStats)
  const [anomalyData, setAnomalyData] = useState<AnomalyDataPoint[]>(mockAnomalyData)
  
  // Track actioned alerts (persists across polling refreshes)
  const [actionedAlertIds, setActionedAlertIds] = useState<Set<string>>(new Set())
  
  // Track trades marked as normal (false positive) - maps trade_id to true
  const [normalizedTradeIds, setNormalizedTradeIds] = useState<Set<string>>(new Set())
  
  // Connection state
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [mlStatus, setMLStatus] = useState<MLStatus | null>(null)
  
  // Refs for cleanup
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const healthIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Filter out actioned alerts for display
  const displayAlerts = useMemo(() => 
    alerts.filter(a => !actionedAlertIds.has(a.alert_id)),
    [alerts, actionedAlertIds]
  )
  
  // Apply normalized status to trades for display
  const displayTrades = useMemo(() => 
    trades.map(trade => {
      if (normalizedTradeIds.has(trade.trade_id)) {
        return {
          ...trade,
          status: "Normal" as const,
          risk_score: 0,
          risk_level: "LOW" as const,
          explanation: "Verified as normal by analyst"
        }
      }
      return trade
    }),
    [trades, normalizedTradeIds]
  )

  // Fetch data from backend
  const fetchData = useCallback(async () => {
    try {
      const [tradesData, alertsData] = await Promise.all([
        getTrades(),
        getAlerts(),
      ])

      const hasRealData = tradesData.length > 0

      if (hasRealData) {
        setTrades(tradesData)
        setAlerts(alertsData)
        setStats(computeStats(tradesData))
        setAnomalyData(computeAnomalyData(tradesData))
        setIsConnected(true)
      } else {
        setTrades(mockTrades)
        setAlerts(mockAlerts)
        setStats(mockStats)
        setAnomalyData(mockAnomalyData)
        setIsConnected(false)
      }

      setLastUpdate(new Date())
    } catch (error) {
      console.error("[Dashboard] Error fetching data:", error)
      setIsConnected(false)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Check backend health
  const checkBackendHealth = useCallback(async () => {
    const isHealthy = await checkHealth()
    setIsConnected(isHealthy)
    
    if (isHealthy) {
      const mlStatusData = await getMLStatus()
      setMLStatus(mlStatusData)
    } else {
      setMLStatus(null)
    }
  }, [])

  // Handle confirming fraud (true positive)
  const handleConfirmFraud = useCallback(async (alertId: string) => {
    // Find the alert to get the trade_id
    const alert = alerts.find(a => a.alert_id === alertId)
    if (!alert) {
      toast.error("Alert not found")
      return
    }

    toast.loading("Submitting feedback...", { id: "feedback" })
    
    try {
      const result = await submitFeedback(alert.trade_id, true)
      
      if (result?.feedback_recorded) {
        toast.success("Fraud confirmed! Model will learn from this feedback.", { 
          id: "feedback",
          description: `Alert ${alert.trade_id} marked as confirmed fraud.`
        })
        // Mark the alert as actioned (won't reappear on refresh)
        setActionedAlertIds(prev => new Set([...prev, alertId]))
      } else {
        toast.error("Failed to record feedback", { id: "feedback" })
      }
    } catch (error) {
      console.error("Error confirming fraud:", error)
      toast.error("Error submitting feedback. Please try again.", { id: "feedback" })
    }
  }, [alerts])

  // Handle dismissing alert (false positive) - marks trade as normal
  const handleDismissAlert = useCallback(async (alertId: string) => {
    const alert = alerts.find(a => a.alert_id === alertId)
    if (!alert) {
      toast.error("Alert not found")
      return
    }

    toast.loading("Submitting feedback...", { id: "feedback" })
    
    try {
      const result = await submitFeedback(alert.trade_id, false)
      
      if (result?.feedback_recorded) {
        toast.success("Marked as normal. Model will learn from this.", { 
          id: "feedback",
          description: `${alert.trade_id} marked as normal transaction.`
        })
        // Mark the alert as actioned (won't reappear on refresh)
        setActionedAlertIds(prev => new Set([...prev, alertId]))
        // Mark the trade as normalized (will show as "Normal" in trades table)
        setNormalizedTradeIds(prev => new Set([...prev, alert.trade_id]))
      } else {
        toast.error("Failed to record feedback", { id: "feedback" })
      }
    } catch (error) {
      console.error("Error dismissing alert:", error)
      toast.error("Error submitting feedback. Please try again.", { id: "feedback" })
    }
  }, [alerts])
  
  // Handle marking a trade as normal from the feedback dialog in Recent Trades
  const handleTradeMarkedNormal = useCallback((tradeId: string) => {
    // Mark the trade as normalized
    setNormalizedTradeIds(prev => new Set([...prev, tradeId]))
    
    // Also remove the corresponding alert (find alert by trade_id)
    const matchingAlert = alerts.find(a => a.trade_id === tradeId)
    if (matchingAlert) {
      setActionedAlertIds(prev => new Set([...prev, matchingAlert.alert_id]))
    }
  }, [alerts])

  // Setup polling on mount
  useEffect(() => {
    fetchData()
    checkBackendHealth()

    pollingIntervalRef.current = setInterval(fetchData, POLLING_INTERVAL_MS)
    healthIntervalRef.current = setInterval(checkBackendHealth, HEALTH_CHECK_INTERVAL_MS)

    return () => {
      if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)
      if (healthIntervalRef.current) clearInterval(healthIntervalRef.current)
    }
  }, [fetchData, checkBackendHealth])

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header - Responsive */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-3 sm:px-6 py-3 sm:py-4">
        <div className="flex items-center justify-between max-w-[1800px] mx-auto">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="p-1.5 sm:p-2 bg-accent/10 rounded-lg">
              <Shield className="h-5 w-5 sm:h-6 sm:w-6 text-accent" />
            </div>
            <div>
              <h1 className="text-sm sm:text-lg font-semibold text-foreground tracking-tight">AI Fraud Detector</h1>
              <p className="text-[10px] sm:text-xs text-muted-foreground hidden sm:block">Real-time anomaly monitoring</p>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <div className="hidden sm:flex flex-col items-end gap-0.5">
              <ConnectionStatus isConnected={isConnected} isLoading={isLoading} />
              <MLStatusBadge status={mlStatus} />
            </div>
            {/* Mobile connection indicator */}
            <div className="sm:hidden">
              {isConnected ? (
                <span className="flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-2.5 w-2.5 rounded-full bg-success opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-success"></span>
                </span>
              ) : (
                <WifiOff className="h-4 w-4 text-destructive" />
              )}
            </div>
            {lastUpdate && (
              <div className="text-[10px] sm:text-xs text-muted-foreground bg-secondary/50 px-1.5 sm:px-2.5 py-1 sm:py-1.5 rounded-md">
                <span className="hidden sm:inline">Updated: </span>{formatTime(lastUpdate.toISOString())}
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 p-3 sm:p-4 md:p-6 space-y-4 sm:space-y-6 max-w-[1800px] mx-auto w-full">
        {/* Stats Row */}
        <section aria-label="Statistics">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-3 md:gap-4">
            <StatCard
              title="Total Trades"
              value={stats.total_trades.toLocaleString()}
              icon={Activity}
              trend={isConnected ? "Live data" : "Mock data"}
              trendUp={isConnected}
              isLoading={isLoading}
            />
            <StatCard
              title="Flagged"
              value={stats.flagged_trades}
              icon={AlertTriangle}
              isLoading={isLoading}
            />
            <StatCard
              title="High Risk"
              value={stats.high_risk_alerts}
              icon={Shield}
              isLoading={isLoading}
            />
            <StatCard
              title="Avg Score"
              value={stats.average_risk_score.toFixed(1)}
              icon={TrendingUp}
              isLoading={isLoading}
            />
          </div>
        </section>

        {/* Chart, Threat Monitor, and Alerts Row */}
        <section aria-label="Analytics" className="grid grid-cols-1 lg:grid-cols-4 gap-3 sm:gap-4 md:gap-6">
          {/* Anomaly Chart */}
          <Card className="lg:col-span-2 bg-card border-border">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-foreground flex items-center gap-2">
                    <Activity className="h-4 w-4 text-accent" />
                    Anomaly Score Trend
                  </CardTitle>
                  <CardDescription className="mt-1">
                    Real-time risk score distribution {isConnected ? "(Live)" : "(Demo Data)"}
                  </CardDescription>
                </div>
                {isConnected && (
                  <Badge variant="outline" className="text-xs bg-success/10 border-success/30 text-success">
                    <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                    Live
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[280px] md:h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={anomalyData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(210, 100%, 55%)" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="hsl(210, 100%, 55%)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="time"
                      stroke="hsl(0, 0%, 40%)"
                      fontSize={11}
                      tickLine={false}
                      axisLine={false}
                      dy={10}
                    />
                    <YAxis
                      stroke="hsl(0, 0%, 40%)"
                      fontSize={11}
                      tickLine={false}
                      axisLine={false}
                      domain={[0, 100]}
                      tickFormatter={(value) => `${value}`}
                      dx={-5}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(0, 0%, 7%)",
                        border: "1px solid hsl(0, 0%, 20%)",
                        borderRadius: "8px",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
                      }}
                      labelStyle={{ color: "hsl(0, 0%, 95%)", fontWeight: 500 }}
                      itemStyle={{ color: "hsl(210, 100%, 55%)" }}
                      formatter={(value: number) => [`${value.toFixed(1)}`, 'Risk Score']}
                    />
                    <Area
                      type="monotone"
                      dataKey="score"
                      stroke="hsl(210, 100%, 55%)"
                      strokeWidth={2}
                      fill="url(#scoreGradient)"
                      animationDuration={300}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Threat Monitor */}
          <ThreatMonitor 
            trades={trades}
            alertCount={displayAlerts.length}
            isConnected={isConnected}
          />

          {/* Recent Alerts */}
          <AlertsPanel 
            alerts={displayAlerts}
            onConfirmFraud={handleConfirmFraud}
            onDismiss={handleDismissAlert}
          />
        </section>

        {/* Model Performance Section */}
        <section aria-label="Model Performance" className="grid grid-cols-1 lg:grid-cols-3 gap-3 sm:gap-4 md:gap-6">
          <ModelPerformanceDashboard />
          
          {/* Quick Stats Summary */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-2 sm:pb-3 p-3 sm:p-6">
              <CardTitle className="flex items-center gap-2 text-sm sm:text-base">
                <div className="p-1 sm:p-1.5 rounded-md bg-accent/10">
                  <Brain className="h-3 w-3 sm:h-4 sm:w-4 text-accent" />
                </div>
                <span className="hidden sm:inline">Ensemble Model Architecture</span>
                <span className="sm:hidden">ML Architecture</span>
              </CardTitle>
              <CardDescription className="text-xs sm:text-sm">
                Multi-model fraud detection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 sm:space-y-4 p-3 sm:p-6 pt-0 sm:pt-0">
              {/* Model Weights */}
              <div className="grid grid-cols-3 gap-1.5 sm:gap-3">
                <div className="p-2 sm:p-3 bg-accent/5 border border-accent/20 rounded-lg text-center group hover:bg-accent/10 transition-colors">
                  <div className="text-lg sm:text-2xl font-bold text-accent tabular-nums">30%</div>
                  <div className="text-[10px] sm:text-xs font-medium text-foreground mt-0.5 sm:mt-1">Isolation Forest</div>
                  <div className="text-[8px] sm:text-[10px] text-muted-foreground mt-0.5 hidden sm:block">Anomaly Detection</div>
                </div>
                <div className="p-2 sm:p-3 bg-success/5 border border-success/20 rounded-lg text-center group hover:bg-success/10 transition-colors">
                  <div className="text-lg sm:text-2xl font-bold text-success tabular-nums">50%</div>
                  <div className="text-[10px] sm:text-xs font-medium text-foreground mt-0.5 sm:mt-1">XGBoost</div>
                  <div className="text-[8px] sm:text-[10px] text-muted-foreground mt-0.5 hidden sm:block">Supervised ML</div>
                </div>
                <div className="p-2 sm:p-3 bg-warning/5 border border-warning/20 rounded-lg text-center group hover:bg-warning/10 transition-colors">
                  <div className="text-lg sm:text-2xl font-bold text-warning tabular-nums">20%</div>
                  <div className="text-[10px] sm:text-xs font-medium text-foreground mt-0.5 sm:mt-1">Heuristics</div>
                  <div className="text-[8px] sm:text-[10px] text-muted-foreground mt-0.5 hidden sm:block">Rule-based</div>
                </div>
              </div>
              
              {/* Fraud Patterns */}
              <div className="pt-2 sm:pt-3 border-t border-border space-y-1.5 sm:space-y-2">
                <h4 className="text-[10px] sm:text-xs font-medium text-muted-foreground uppercase tracking-wider">Fraud Patterns Detected</h4>
                <div className="grid grid-cols-2 gap-x-2 sm:gap-x-4 gap-y-1 sm:gap-y-1.5 text-[10px] sm:text-xs">
                  <div className="flex items-center gap-1.5 sm:gap-2 py-0.5 sm:py-1">
                    <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full bg-destructive flex-shrink-0" />
                    <span className="text-foreground truncate">Large Amount</span>
                  </div>
                  <div className="flex items-center gap-1.5 sm:gap-2 py-0.5 sm:py-1">
                    <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full bg-warning flex-shrink-0" />
                    <span className="text-foreground truncate">Velocity Attacks</span>
                  </div>
                  <div className="flex items-center gap-1.5 sm:gap-2 py-0.5 sm:py-1">
                    <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full bg-accent flex-shrink-0" />
                    <span className="text-foreground truncate">Off-Hours</span>
                  </div>
                  <div className="flex items-center gap-1.5 sm:gap-2 py-0.5 sm:py-1">
                    <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full bg-success flex-shrink-0" />
                    <span className="text-foreground truncate">Account Takeover</span>
                  </div>
                  <div className="flex items-center gap-1.5 sm:gap-2 py-0.5 sm:py-1">
                    <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                    <span className="text-foreground truncate">Structuring</span>
                  </div>
                  <div className="flex items-center gap-1.5 sm:gap-2 py-0.5 sm:py-1">
                    <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 rounded-full bg-cyan-500 flex-shrink-0" />
                    <span className="text-foreground truncate">New Account</span>
                  </div>
                </div>
              </div>
              
              {/* Training Info */}
              <div className="pt-2 sm:pt-3 border-t border-border">
                <div className="flex items-center justify-between text-[10px] sm:text-xs">
                  <div className="flex items-center gap-1 sm:gap-1.5 text-muted-foreground">
                    <Activity className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
                    <span className="hidden sm:inline">5,000 training samples</span>
                    <span className="sm:hidden">5K samples</span>
                  </div>
                  <Badge variant="outline" className="text-[8px] sm:text-[10px] px-1 sm:px-1.5 py-0 h-4 sm:h-auto">
                    10% Fraud
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Compliance Panel */}
          <CompliancePanel />
        </section>

        {/* Trades Table */}
        <section aria-label="Recent Trades">
          <TradesTable 
            trades={displayTrades}
            isConnected={isConnected}
            isLoading={isLoading}
            pollingInterval={POLLING_INTERVAL_MS}
            onTradeMarkedNormal={handleTradeMarkedNormal}
          />
        </section>
      </main>

      {/* Footer - Responsive */}
      <footer className="border-t border-border px-3 sm:px-6 py-2 sm:py-3 mt-auto bg-card/50">
        <div className="flex items-center justify-between text-[10px] sm:text-xs text-muted-foreground max-w-[1800px] mx-auto">
          <div className="flex items-center gap-1.5 sm:gap-2">
            <Shield className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
            <span className="hidden sm:inline">AI Fraud Detection MVP</span>
            <span className="sm:hidden">Fraud AI</span>
            <span className="hidden md:inline text-muted-foreground/50">•</span>
            <span className="hidden md:inline">Deriv AI Talent Sprint 2026</span>
          </div>
          <div className="flex items-center gap-1.5 sm:gap-2">
            {isConnected ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-success" />
                <span className="hidden sm:inline">Backend: Connected</span>
                <span className="sm:hidden">🟢</span>
                <span className="hidden sm:inline text-muted-foreground/50">|</span>
                <span className="hidden sm:inline">ML: {mlStatus?.is_trained ? "Active" : "Inactive"}</span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-destructive" />
                <span>Mock Mode</span>
              </>
            )}
          </div>
        </div>
      </footer>
    </div>
  )
}
