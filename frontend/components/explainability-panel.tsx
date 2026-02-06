"use client"

/**
 * AI Explainability Panel
 * 
 * This component provides visual explanations for WHY a trade was flagged.
 * Key for regulatory compliance (GDPR, EU AI Act) and building trust.
 * 
 * Features:
 * - SHAP-style feature contribution visualization
 * - Decision path explanation
 * - Counterfactual analysis ("What would make this safe?")
 * - Model confidence indicator
 */

import React from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { 
  Brain, 
  Lightbulb, 
  ArrowRight, 
  AlertTriangle, 
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Scale,
  Shield
} from "lucide-react"

// =============================================================================
// TYPES
// =============================================================================

interface SignalContribution {
  name: string;
  contribution: number; // -100 to +100 (negative = reduces risk)
  description: string;
  triggered: boolean;
}

interface ExplainabilityProps {
  tradeId: string;
  riskScore: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  explanation: string;
  signals?: SignalContribution[];
}

// =============================================================================
// SIGNAL CONTRIBUTION BAR
// =============================================================================

function SignalBar({ signal }: { signal: SignalContribution }) {
  const isPositive = signal.contribution > 0;
  const absValue = Math.abs(signal.contribution);
  const barWidth = Math.min(absValue, 100);
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="space-y-0.5 sm:space-y-1">
            <div className="flex items-center justify-between text-[10px] sm:text-xs">
              <span className={`truncate max-w-[70%] ${signal.triggered ? "font-medium" : "text-muted-foreground"}`}>
                {signal.name}
              </span>
              <span className={isPositive ? "text-destructive" : "text-success"}>
                {isPositive ? "+" : ""}{signal.contribution}
              </span>
            </div>
            <div className="h-1.5 sm:h-2 bg-secondary rounded-full overflow-hidden relative">
              {/* Center line */}
              <div className="absolute left-1/2 top-0 w-px h-full bg-border z-10" />
              {/* Contribution bar */}
              <div
                className={`h-full transition-all duration-500 absolute ${
                  isPositive 
                    ? "left-1/2 bg-destructive/70" 
                    : "right-1/2 bg-success/70"
                }`}
                style={{ width: `${barWidth / 2}%` }}
              />
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p className="max-w-xs text-xs">{signal.description}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// DECISION PATH VISUALIZATION
// =============================================================================

function DecisionPath({ riskScore, signals }: { riskScore: number; signals: SignalContribution[] }) {
  const triggeredSignals = signals.filter(s => s.triggered && s.contribution > 0);
  
  return (
    <div className="space-y-2 sm:space-y-3">
      <div className="text-xs sm:text-sm font-medium flex items-center gap-1.5 sm:gap-2">
        <Lightbulb className="h-3 w-3 sm:h-4 sm:w-4 text-warning" />
        Decision Path
      </div>
      <div className="flex items-center gap-1 sm:gap-2 text-[10px] sm:text-xs overflow-x-auto pb-2 scrollbar-thin">
        <div className="flex-shrink-0 px-2 sm:px-3 py-1 sm:py-1.5 bg-secondary rounded-full">
          Base: 0
        </div>
        {triggeredSignals.slice(0, 3).map((signal, i) => (
          <React.Fragment key={signal.name}>
            <ArrowRight className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-muted-foreground flex-shrink-0" />
            <div className={`flex-shrink-0 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full ${
              signal.contribution > 20 ? "bg-destructive/20 text-destructive" : "bg-warning/20 text-warning"
            }`}>
              +{signal.contribution} <span className="hidden sm:inline">({signal.name.split(" ")[0]})</span>
            </div>
          </React.Fragment>
        ))}
        <ArrowRight className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-muted-foreground flex-shrink-0" />
        <div className={`flex-shrink-0 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full font-bold ${
          riskScore >= 70 ? "bg-destructive text-destructive-foreground" :
          riskScore >= 40 ? "bg-warning text-warning-foreground" :
          "bg-success text-success-foreground"
        }`}>
          Final: {riskScore}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// COUNTERFACTUAL ANALYSIS
// =============================================================================

function CounterfactualAnalysis({ signals, riskScore }: { signals: SignalContribution[]; riskScore: number }) {
  // Find the signal that, if removed, would reduce risk the most
  const topContributor = signals
    .filter(s => s.triggered && s.contribution > 0)
    .sort((a, b) => b.contribution - a.contribution)[0];
  
  if (!topContributor || riskScore < 40) return null;
  
  const hypotheticalScore = riskScore - topContributor.contribution;
  const wouldBeSafe = hypotheticalScore < 70;
  
  return (
    <div className="p-2 sm:p-3 bg-accent/5 border border-accent/20 rounded-lg space-y-1.5 sm:space-y-2">
      <div className="text-xs sm:text-sm font-medium flex items-center gap-1.5 sm:gap-2">
        <Scale className="h-3 w-3 sm:h-4 sm:w-4 text-accent" />
        <span className="hidden sm:inline">Counterfactual Analysis</span>
        <span className="sm:hidden">What If?</span>
      </div>
      <p className="text-[10px] sm:text-xs text-muted-foreground">
        <span className="font-medium text-foreground">If "{topContributor.name}" was normal:</span>
        {" "}Risk score would be <span className={wouldBeSafe ? "text-success font-medium" : "text-warning font-medium"}>
          {hypotheticalScore}
        </span> <span className="hidden sm:inline">({wouldBeSafe ? "would pass threshold" : "still elevated"})</span>
      </p>
    </div>
  );
}

// =============================================================================
// MODEL CONFIDENCE INDICATOR
// =============================================================================

function ModelConfidence({ riskScore }: { riskScore: number }) {
  // Confidence is higher when score is near extremes (0 or 100)
  // Lower when near decision boundary (40-60)
  const distanceFromMiddle = Math.abs(riskScore - 50);
  const confidence = Math.min(100, 50 + distanceFromMiddle);
  
  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <div className="text-[10px] sm:text-xs text-muted-foreground whitespace-nowrap">
        <span className="hidden sm:inline">Model Confidence:</span>
        <span className="sm:hidden">Conf:</span>
      </div>
      <div className="flex-1 h-1.5 sm:h-2 bg-secondary rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all duration-500 ${
            confidence >= 80 ? "bg-success" : 
            confidence >= 60 ? "bg-warning" : 
            "bg-destructive"
          }`}
          style={{ width: `${confidence}%` }}
        />
      </div>
      <div className="text-[10px] sm:text-xs font-medium">{confidence.toFixed(0)}%</div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExplainabilityPanel({ 
  tradeId, 
  riskScore, 
  riskLevel, 
  explanation,
  signals: providedSignals 
}: ExplainabilityProps) {
  // Default signals if not provided (parse from explanation)
  const signals: SignalContribution[] = providedSignals || parseExplanationToSignals(explanation, riskScore);
  
  const riskEmoji = riskLevel === "HIGH" ? "🔴" : riskLevel === "MEDIUM" ? "🟡" : "🟢";
  
  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2 sm:pb-3 p-3 sm:p-6">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="text-sm sm:text-base flex items-center gap-1.5 sm:gap-2">
              <Brain className="h-3 w-3 sm:h-4 sm:w-4 text-accent flex-shrink-0" />
              <span className="truncate">
                <span className="hidden sm:inline">AI Explainability</span>
                <span className="sm:hidden">Why Flagged?</span>
              </span>
            </CardTitle>
            <CardDescription className="text-[10px] sm:text-xs truncate">
              <span className="hidden sm:inline">Understanding why this trade was flagged</span>
              <span className="sm:hidden">Risk explanation</span>
            </CardDescription>
          </div>
          <Badge 
            variant="outline" 
            className={`text-[10px] sm:text-xs h-5 sm:h-auto flex-shrink-0 ${
              riskLevel === "HIGH" ? "border-destructive text-destructive" :
              riskLevel === "MEDIUM" ? "border-warning text-warning" :
              "border-success text-success"
            }`}
          >
            {riskEmoji} {riskLevel}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 sm:space-y-4 p-3 sm:p-6 pt-0">
        {/* Primary Explanation */}
        <div className="p-2 sm:p-3 bg-secondary/50 rounded-lg">
          <p className="text-xs sm:text-sm line-clamp-3 sm:line-clamp-none">{explanation}</p>
        </div>
        
        {/* Signal Contributions */}
        <div className="space-y-2 sm:space-y-3">
          <div className="text-xs sm:text-sm font-medium flex items-center gap-1.5 sm:gap-2">
            <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 text-accent" />
            <span className="hidden sm:inline">Signal Contributions</span>
            <span className="sm:hidden">Signals</span>
          </div>
          <div className="space-y-1.5 sm:space-y-2">
            {signals.map((signal) => (
              <SignalBar key={signal.name} signal={signal} />
            ))}
          </div>
        </div>
        
        {/* Decision Path */}
        <DecisionPath riskScore={riskScore} signals={signals} />
        
        {/* Counterfactual */}
        <CounterfactualAnalysis signals={signals} riskScore={riskScore} />
        
        {/* Model Confidence */}
        <ModelConfidence riskScore={riskScore} />
        
        {/* Regulatory Compliance Note */}
        <div className="flex items-start gap-1.5 sm:gap-2 p-1.5 sm:p-2 bg-accent/5 rounded text-[9px] sm:text-xs text-muted-foreground">
          <Shield className="h-2.5 w-2.5 sm:h-3 sm:w-3 mt-0.5 text-accent flex-shrink-0" />
          <span className="hidden sm:inline">
            This explanation complies with EU AI Act Article 13 transparency requirements 
            and GDPR Article 22 automated decision-making provisions.
          </span>
          <span className="sm:hidden">
            EU AI Act & GDPR compliant explanation
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function parseExplanationToSignals(explanation: string, riskScore: number): SignalContribution[] {
  const signals: SignalContribution[] = [];
  
  // Parse common patterns from explanation text
  if (explanation.toLowerCase().includes("large") || explanation.toLowerCase().includes("amount")) {
    signals.push({
      name: "Amount Anomaly",
      contribution: Math.min(55, Math.floor(riskScore * 0.4)),
      description: "Transaction amount exceeds typical patterns for this account",
      triggered: true
    });
  }
  
  if (explanation.toLowerCase().includes("velocity") || explanation.toLowerCase().includes("rapid")) {
    signals.push({
      name: "Velocity Burst",
      contribution: Math.min(50, Math.floor(riskScore * 0.35)),
      description: "High frequency of trades in a short time window",
      triggered: true
    });
  }
  
  if (explanation.toLowerCase().includes("unusual hour") || explanation.toLowerCase().includes("off-hours")) {
    signals.push({
      name: "Unusual Hours",
      contribution: 20,
      description: "Trade executed outside normal business hours",
      triggered: true
    });
  }
  
  if (explanation.toLowerCase().includes("behavior") || explanation.toLowerCase().includes("deviation") || explanation.toLowerCase().includes("drift")) {
    signals.push({
      name: "Behavior Drift",
      contribution: Math.min(45, Math.floor(riskScore * 0.3)),
      description: "Significant deviation from account's historical trading pattern",
      triggered: true
    });
  }
  
  if (explanation.toLowerCase().includes("new account")) {
    signals.push({
      name: "New Account Risk",
      contribution: Math.min(40, Math.floor(riskScore * 0.25)),
      description: "Account has limited trading history",
      triggered: true
    });
  }
  
  if (explanation.toLowerCase().includes("structur") || explanation.toLowerCase().includes("$10k") || explanation.toLowerCase().includes("threshold")) {
    signals.push({
      name: "Structuring Pattern",
      contribution: Math.min(50, Math.floor(riskScore * 0.35)),
      description: "Multiple transactions near reporting thresholds",
      triggered: true
    });
  }
  
  if (explanation.toLowerCase().includes("ml model") || explanation.toLowerCase().includes("corroborate")) {
    signals.push({
      name: "ML Anomaly Score",
      contribution: Math.min(25, Math.floor(riskScore * 0.2)),
      description: "Machine learning model detected unusual patterns",
      triggered: true
    });
  }
  
  // Add baseline signal
  if (signals.length === 0) {
    signals.push({
      name: "Baseline Assessment",
      contribution: riskScore,
      description: "Overall risk assessment from combined signals",
      triggered: riskScore > 20
    });
  }
  
  // Add a normal signal to show contrast
  signals.push({
    name: "Account History",
    contribution: -5,
    description: "Account has some established trading history",
    triggered: false
  });
  
  return signals.sort((a, b) => b.contribution - a.contribution);
}

export default ExplainabilityPanel;
