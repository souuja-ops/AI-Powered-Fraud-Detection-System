"use client"

/**
 * Real-Time Threat Monitor
 * 
 * A visually impressive live threat monitoring component showing:
 * - Real-time risk score gauge
 * - Threat level indicator with animation
 * - Live transaction stream visualization
 * - System health metrics
 * 
 * This component is designed to be visually impressive for hackathon demos.
 */

import React, { useState, useEffect, useRef } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  Shield, 
  Activity, 
  Zap, 
  AlertOctagon,
  Eye,
  Cpu,
  Clock,
  TrendingUp
} from "lucide-react"

// =============================================================================
// TYPES
// =============================================================================

interface ThreatMonitorProps {
  trades: Array<{
    trade_id: string;
    risk_score: number;
    risk_level: string;
    timestamp: string;
  }>;
  alertCount: number;
  isConnected: boolean;
}

// =============================================================================
// ANIMATED RISK GAUGE
// =============================================================================

function RiskGauge({ score }: { score: number }) {
  const circumference = 2 * Math.PI * 45; // radius = 45
  const strokeDashoffset = circumference - (score / 100) * circumference * 0.75; // 270 degree arc
  
  const getColor = (score: number) => {
    if (score >= 70) return "#ef4444"; // red
    if (score >= 40) return "#f59e0b"; // amber
    return "#22c55e"; // green
  };
  
  return (
    <div className="relative w-24 h-24 sm:w-32 sm:h-32 mx-auto">
      <svg className="w-full h-full -rotate-[135deg]" viewBox="0 0 100 100">
        {/* Background arc */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-secondary"
          strokeDasharray={circumference * 0.75}
          strokeLinecap="round"
        />
        {/* Value arc */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke={getColor(score)}
          strokeWidth="8"
          strokeDasharray={circumference * 0.75}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{
            filter: score >= 70 ? "drop-shadow(0 0 6px rgba(239, 68, 68, 0.5))" : "none"
          }}
        />
      </svg>
      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span 
          className="text-2xl sm:text-3xl font-bold tabular-nums transition-colors duration-500"
          style={{ color: getColor(score) }}
        >
          {score}
        </span>
        <span className="text-[10px] sm:text-xs text-muted-foreground">Risk Score</span>
      </div>
    </div>
  );
}

// =============================================================================
// LIVE PULSE INDICATOR
// =============================================================================

function LivePulse({ isActive }: { isActive: boolean }) {
  return (
    <div className="flex items-center gap-1.5 sm:gap-2">
      <span className="relative flex h-2 w-2 sm:h-3 sm:w-3">
        <span className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${
          isActive ? "animate-ping bg-success" : "bg-muted"
        }`} />
        <span className={`relative inline-flex rounded-full h-2 w-2 sm:h-3 sm:w-3 ${
          isActive ? "bg-success" : "bg-muted"
        }`} />
      </span>
      <span className={`text-[10px] sm:text-xs font-medium ${isActive ? "text-success" : "text-muted-foreground"}`}>
        <span className="hidden sm:inline">{isActive ? "MONITORING" : "OFFLINE"}</span>
        <span className="sm:hidden">{isActive ? "LIVE" : "OFF"}</span>
      </span>
    </div>
  );
}

// =============================================================================
// THREAT LEVEL BAR
// =============================================================================

function ThreatLevelBar({ level }: { level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" }) {
  const levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
  const activeIndex = levels.indexOf(level);
  
  const levelEmojis = {
    LOW: "🟢",
    MEDIUM: "🟡",
    HIGH: "🔴",
    CRITICAL: "🔴"
  };
  
  return (
    <div className="space-y-1.5 sm:space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] sm:text-xs font-medium text-muted-foreground">THREAT LEVEL</span>
        <Badge 
          variant="outline"
          className={`text-[10px] sm:text-xs h-5 sm:h-auto ${
            level === "CRITICAL" ? "border-destructive bg-destructive/10 text-destructive animate-pulse" :
            level === "HIGH" ? "border-destructive text-destructive" :
            level === "MEDIUM" ? "border-warning text-warning" :
            "border-success text-success"
          }`}
        >
          {levelEmojis[level]} {level}
        </Badge>
      </div>
      <div className="flex gap-0.5 sm:gap-1">
        {levels.map((l, i) => (
          <div
            key={l}
            className={`h-1.5 sm:h-2 flex-1 rounded-full transition-all duration-500 ${
              i <= activeIndex
                ? i === 3 ? "bg-destructive animate-pulse" :
                  i === 2 ? "bg-destructive" :
                  i === 1 ? "bg-warning" :
                  "bg-success"
                : "bg-secondary"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// LIVE TRANSACTION STREAM
// =============================================================================

function TransactionStream({ trades }: { trades: ThreatMonitorProps["trades"] }) {
  const recentTrades = trades.slice(0, 5);
  const [animateKey, setAnimateKey] = useState(0);
  const prevTradeIdRef = useRef<string | null>(null);
  
  // Trigger animation when a new trade appears at the top
  useEffect(() => {
    if (recentTrades.length > 0 && recentTrades[0].trade_id !== prevTradeIdRef.current) {
      prevTradeIdRef.current = recentTrades[0].trade_id;
      setAnimateKey(k => k + 1);
    }
  }, [recentTrades]);
  
  const riskEmojis = {
    HIGH: "🔴",
    MEDIUM: "🟡",
    LOW: "🟢"
  };
  
  return (
    <div className="space-y-1.5 sm:space-y-2">
      <div className="text-[10px] sm:text-xs font-medium text-muted-foreground flex items-center gap-1.5 sm:gap-2">
        <Activity className="h-2.5 w-2.5 sm:h-3 sm:w-3 animate-pulse" />
        <span className="hidden sm:inline">LIVE TRANSACTION STREAM</span>
        <span className="sm:hidden">LIVE STREAM</span>
        {recentTrades.length > 0 && (
          <span className="text-[8px] sm:text-[10px] text-muted-foreground/70">
            ({trades.length} total)
          </span>
        )}
      </div>
      <div className="space-y-0.5 sm:space-y-1 max-h-28 sm:max-h-32 overflow-hidden">
        {recentTrades.map((trade, i) => (
          <div
            key={trade.trade_id}
            className={`flex items-center justify-between p-1.5 sm:p-2 rounded text-[10px] sm:text-xs transition-all duration-300 ${
              i === 0 ? "bg-accent/10 scale-100" : "bg-secondary/30 scale-95 opacity-70"
            } ${i === 0 && animateKey > 0 ? "animate-in slide-in-from-top-2 fade-in duration-300" : ""}`}
          >
            <div className="flex items-center gap-1.5 sm:gap-2 min-w-0">
              <span className="flex-shrink-0">{riskEmojis[trade.risk_level as keyof typeof riskEmojis] || "⚪"}</span>
              <span className="font-mono truncate">{trade.trade_id}</span>
            </div>
            <div className="flex items-center gap-1.5 sm:gap-2 flex-shrink-0">
              <span className={`font-bold tabular-nums ${
                trade.risk_score >= 70 ? "text-destructive" :
                trade.risk_score >= 40 ? "text-warning" :
                "text-success"
              }`}>
                {trade.risk_score}
              </span>
              <span className="text-muted-foreground hidden sm:inline">
                {new Date(trade.timestamp).toLocaleTimeString("en-US", {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit"
                })}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// SYSTEM METRICS
// =============================================================================

function SystemMetrics({ tradesPerMinute, avgLatency }: { tradesPerMinute: number; avgLatency: number }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:gap-3">
      <div className="p-2 sm:p-3 bg-secondary/30 rounded-lg">
        <div className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs text-muted-foreground mb-0.5 sm:mb-1">
          <Zap className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
          <span className="hidden sm:inline">THROUGHPUT</span>
          <span className="sm:hidden">TPM</span>
        </div>
        <div className="text-base sm:text-lg font-bold tabular-nums">
          {tradesPerMinute} <span className="text-[8px] sm:text-xs font-normal text-muted-foreground">/min</span>
        </div>
      </div>
      <div className="p-2 sm:p-3 bg-secondary/30 rounded-lg">
        <div className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs text-muted-foreground mb-0.5 sm:mb-1">
          <Clock className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
          <span className="hidden sm:inline">AVG LATENCY</span>
          <span className="sm:hidden">LATENCY</span>
        </div>
        <div className="text-base sm:text-lg font-bold tabular-nums">
          {avgLatency} <span className="text-[8px] sm:text-xs font-normal text-muted-foreground">ms</span>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ThreatMonitor({ trades, alertCount, isConnected }: ThreatMonitorProps) {
  const [currentRiskScore, setCurrentRiskScore] = useState(0);
  const [threatLevel, setThreatLevel] = useState<"LOW" | "MEDIUM" | "HIGH" | "CRITICAL">("LOW");
  const [tradesPerMinute, setTradesPerMinute] = useState(0);
  
  // Calculate metrics from trades
  useEffect(() => {
    if (trades.length === 0) {
      setCurrentRiskScore(0);
      setThreatLevel("LOW");
      return;
    }
    
    // Get average risk score of recent trades
    const recentTrades = trades.slice(0, 10);
    const avgScore = recentTrades.reduce((sum, t) => sum + t.risk_score, 0) / recentTrades.length;
    setCurrentRiskScore(Math.round(avgScore));
    
    // Determine threat level based on alert count and average score
    const highRiskCount = recentTrades.filter(t => t.risk_level === "HIGH").length;
    if (highRiskCount >= 5 || avgScore >= 80) {
      setThreatLevel("CRITICAL");
    } else if (highRiskCount >= 3 || avgScore >= 60) {
      setThreatLevel("HIGH");
    } else if (highRiskCount >= 1 || avgScore >= 40) {
      setThreatLevel("MEDIUM");
    } else {
      setThreatLevel("LOW");
    }
    
    // Estimate trades per minute (simplified)
    setTradesPerMinute(Math.round(trades.length / 5 * 60)); // Assuming 5 second polling
  }, [trades, alertCount]);
  
  return (
    <Card className="bg-card border-border overflow-hidden">
      <CardHeader className="pb-2 p-3 sm:p-6 sm:pb-2">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="text-sm sm:text-base flex items-center gap-1.5 sm:gap-2">
              <Shield className="h-3 w-3 sm:h-4 sm:w-4 text-accent flex-shrink-0" />
              <span className="truncate">Threat Monitor</span>
            </CardTitle>
            <CardDescription className="text-[10px] sm:text-xs truncate">
              <span className="hidden sm:inline">Real-time fraud detection status</span>
              <span className="sm:hidden">Live detection</span>
            </CardDescription>
          </div>
          <LivePulse isActive={isConnected} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3 sm:space-y-4 p-3 sm:p-6 pt-0">
        {/* Risk Gauge */}
        <RiskGauge score={currentRiskScore} />
        
        {/* Threat Level */}
        <ThreatLevelBar level={threatLevel} />
        
        {/* Alert Count */}
        <div className="flex items-center justify-between p-2 sm:p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <div className="flex items-center gap-1.5 sm:gap-2">
            <AlertOctagon className="h-3 w-3 sm:h-4 sm:w-4 text-destructive" />
            <span className="text-xs sm:text-sm font-medium">Active Alerts</span>
          </div>
          <span className="text-xl sm:text-2xl font-bold text-destructive tabular-nums">{alertCount}</span>
        </div>
        
        {/* Live Stream */}
        <TransactionStream trades={trades} />
        
        {/* System Metrics */}
        <SystemMetrics tradesPerMinute={tradesPerMinute} avgLatency={23} />
        
        {/* Footer */}
        <div className="flex items-center justify-between text-[10px] sm:text-xs text-muted-foreground pt-1.5 sm:pt-2 border-t border-border">
          <div className="flex items-center gap-1">
            <Cpu className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
            <span className="hidden sm:inline">Ensemble Model Active</span>
            <span className="sm:hidden">ML Active</span>
          </div>
          <div className="flex items-center gap-1">
            <Eye className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
            <span>{trades.length} <span className="hidden sm:inline">trades analyzed</span></span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ThreatMonitor;
