"use client"

/**
 * Risk Score Breakdown Component
 * 
 * Visualizes how different signals contributed to the final risk score.
 * Critical for explainable AI - users can see WHY a trade was flagged.
 * 
 * Shows:
 * - Individual signal contributions
 * - Model agreement (how many models flagged)
 * - Visual breakdown with progress bars
 */

import React from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  PieChart,
  BarChart3,
  Brain,
  AlertTriangle,
  Clock,
  TrendingUp,
  DollarSign,
  Zap,
  User,
  Activity,
} from "lucide-react"

// =============================================================================
// TYPES
// =============================================================================

interface SignalBreakdown {
  name: string;
  score: number;
  maxScore: number;
  triggered: boolean;
  description: string;
  icon: React.ReactNode;
}

interface RiskBreakdownProps {
  tradeId: string;
  riskScore: number;
  riskLevel: string;
  explanation: string;
  amount: number;
  // These would come from detailed API response in production
  isolationForestScore?: number;
  xgboostScore?: number;
  heuristicScore?: number;
}

// =============================================================================
// SIGNAL BREAKDOWN HELPER
// =============================================================================

function parseExplanationToSignals(
  explanation: string,
  riskScore: number,
  amount: number
): SignalBreakdown[] {
  const signals: SignalBreakdown[] = [];
  const explanationLower = explanation.toLowerCase();
  
  // Amount Signal
  const amountTriggered = explanationLower.includes("amount") || 
                          explanationLower.includes("large") ||
                          amount > 50000;
  signals.push({
    name: "Amount Analysis",
    score: amountTriggered ? Math.min(30, Math.floor(amount / 10000) * 5) : 0,
    maxScore: 30,
    triggered: amountTriggered,
    description: amount > 100000 
      ? `Large amount: $${amount.toLocaleString()}`
      : amount > 50000
      ? `Elevated amount: $${amount.toLocaleString()}`
      : "Amount within normal range",
    icon: <DollarSign className="h-4 w-4" />,
  });
  
  // Velocity Signal
  const velocityTriggered = explanationLower.includes("velocity") || 
                            explanationLower.includes("rapid") ||
                            explanationLower.includes("burst");
  signals.push({
    name: "Trading Velocity",
    score: velocityTriggered ? 25 : 0,
    maxScore: 30,
    triggered: velocityTriggered,
    description: velocityTriggered 
      ? "Multiple trades in short window detected"
      : "Normal trading frequency",
    icon: <Zap className="h-4 w-4" />,
  });
  
  // Time Signal
  const timeTriggered = explanationLower.includes("hour") || 
                        explanationLower.includes("time") ||
                        explanationLower.includes("unusual");
  signals.push({
    name: "Time Analysis",
    score: timeTriggered ? 15 : 0,
    maxScore: 20,
    triggered: timeTriggered,
    description: timeTriggered 
      ? "Trading at unusual hours"
      : "Normal trading hours",
    icon: <Clock className="h-4 w-4" />,
  });
  
  // Behavior Signal
  const behaviorTriggered = explanationLower.includes("behavior") || 
                            explanationLower.includes("spike") ||
                            explanationLower.includes("deviation");
  signals.push({
    name: "Behavior Analysis",
    score: behaviorTriggered ? 20 : 0,
    maxScore: 25,
    triggered: behaviorTriggered,
    description: behaviorTriggered 
      ? "Deviation from account's typical pattern"
      : "Consistent with account history",
    icon: <User className="h-4 w-4" />,
  });
  
  // ML Signal
  const mlTriggered = explanationLower.includes("ml") || 
                      explanationLower.includes("model") ||
                      explanationLower.includes("anomaly");
  signals.push({
    name: "ML Anomaly Detection",
    score: mlTriggered ? Math.min(25, riskScore - signals.reduce((a, s) => a + s.score, 0)) : 0,
    maxScore: 25,
    triggered: mlTriggered,
    description: mlTriggered 
      ? "ML model detected anomalous pattern"
      : "No ML anomaly detected",
    icon: <Brain className="h-4 w-4" />,
  });
  
  return signals;
}

// =============================================================================
// SIGNAL BAR COMPONENT
// =============================================================================

function SignalBar({ signal }: { signal: SignalBreakdown }) {
  const percentage = (signal.score / signal.maxScore) * 100;
  
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={signal.triggered ? "text-warning" : "text-muted-foreground"}>
            {signal.icon}
          </div>
          <span className={`text-sm ${signal.triggered ? "font-medium" : "text-muted-foreground"}`}>
            {signal.name}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            {signal.score}/{signal.maxScore}
          </span>
          {signal.triggered && (
            <Badge variant="outline" className="text-xs py-0 px-1">
              Active
            </Badge>
          )}
        </div>
      </div>
      <Progress 
        value={percentage} 
        className={`h-2 ${signal.triggered ? "" : "opacity-50"}`}
      />
      <p className="text-xs text-muted-foreground">{signal.description}</p>
    </div>
  );
}

// =============================================================================
// MODEL AGREEMENT INDICATOR
// =============================================================================

function ModelAgreement({
  isolationForest,
  xgboost,
  heuristic,
}: {
  isolationForest?: number;
  xgboost?: number;
  heuristic?: number;
}) {
  // Default values if not provided
  const ifScore = isolationForest ?? 0.3;
  const xgScore = xgboost ?? 0.4;
  const heurScore = heuristic ?? 0.3;
  
  const models = [
    { name: "Isolation Forest", score: ifScore, flagged: ifScore > 0.5 },
    { name: "XGBoost", score: xgScore, flagged: xgScore > 0.5 },
    { name: "Heuristics", score: heurScore, flagged: heurScore > 0.5 },
  ];
  
  const flaggedCount = models.filter(m => m.flagged).length;
  
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Model Agreement</span>
        <Badge
          variant="outline"
          className={
            flaggedCount === 3
              ? "border-destructive text-destructive"
              : flaggedCount >= 2
              ? "border-warning text-warning"
              : "border-muted-foreground"
          }
        >
          {flaggedCount}/3 models flagged
        </Badge>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {models.map((model) => (
          <div
            key={model.name}
            className={`p-2 rounded-lg border text-center ${
              model.flagged
                ? "border-warning/50 bg-warning/10"
                : "border-border bg-secondary/30"
            }`}
          >
            <div className="text-xs text-muted-foreground mb-1">{model.name}</div>
            <div className={`text-lg font-bold ${model.flagged ? "text-warning" : "text-muted-foreground"}`}>
              {Math.round(model.score * 100)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function RiskBreakdownDialog({
  tradeId,
  riskScore,
  riskLevel,
  explanation,
  amount,
  isolationForestScore,
  xgboostScore,
  heuristicScore,
}: RiskBreakdownProps) {
  const signals = parseExplanationToSignals(explanation, riskScore, amount);
  const triggeredCount = signals.filter(s => s.triggered).length;
  
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
          <BarChart3 className="h-3 w-3 mr-1" />
          Breakdown
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg bg-card border-border max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <PieChart className="h-5 w-5 text-accent" />
            Risk Score Breakdown
          </DialogTitle>
          <DialogDescription>
            Understanding why {tradeId} received a score of {riskScore}/100
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Overall Score */}
          <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
            <div>
              <div className="text-sm text-muted-foreground">Final Risk Score</div>
              <div className="text-3xl font-bold">{riskScore}/100</div>
            </div>
            <Badge
              className={`text-lg px-4 py-2 ${
                riskLevel === "HIGH"
                  ? "bg-destructive/20 text-destructive border-destructive/30"
                  : riskLevel === "MEDIUM"
                  ? "bg-warning/20 text-warning border-warning/30"
                  : "bg-success/20 text-success border-success/30"
              }`}
            >
              {riskLevel}
            </Badge>
          </div>

          {/* Model Agreement */}
          <ModelAgreement
            isolationForest={isolationForestScore}
            xgboost={xgboostScore}
            heuristic={heuristicScore}
          />

          {/* Signal Breakdown */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Signal Analysis</span>
              <span className="text-xs text-muted-foreground">
                {triggeredCount} of {signals.length} signals active
              </span>
            </div>
            
            <div className="space-y-4">
              {signals.map((signal) => (
                <SignalBar key={signal.name} signal={signal} />
              ))}
            </div>
          </div>

          {/* AI Explanation */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Activity className="h-4 w-4 text-accent" />
              AI Explanation
            </div>
            <p className="text-sm text-muted-foreground bg-secondary/30 p-3 rounded-lg">
              {explanation}
            </p>
          </div>

          {/* Legend */}
          <div className="text-xs text-muted-foreground border-t border-border pt-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-warning" />
                <span>Active signal</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-muted-foreground/30" />
                <span>Inactive signal</span>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default RiskBreakdownDialog;
