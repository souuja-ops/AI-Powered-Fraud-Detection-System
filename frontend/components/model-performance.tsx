"use client"

/**
 * Model Performance Dashboard Component
 * 
 * Displays comprehensive ML model metrics including:
 * - Precision, Recall, F1-Score, ROC-AUC
 * - Confusion Matrix visualization
 * - Feature Importance chart
 * 
 * This component demonstrates ML maturity and explainability.
 */

import React, { useState, useEffect } from "react"
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
import { Brain, Target, TrendingUp, AlertTriangle, CheckCircle, XCircle } from "lucide-react"
import { getModelMetrics, type MetricsResponse, type ModelMetrics } from "@/lib/api"

// =============================================================================
// CONFUSION MATRIX COMPONENT
// =============================================================================

interface ConfusionMatrixProps {
  matrix: number[][];
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

function ConfusionMatrix({ matrix, tp, tn, fp, fn }: ConfusionMatrixProps) {
  const total = tp + tn + fp + fn;
  
  return (
    <div className="space-y-2 sm:space-y-3">
      <div className="text-xs sm:text-sm font-medium text-muted-foreground">Confusion Matrix</div>
      <div className="grid grid-cols-3 gap-0.5 sm:gap-1 text-[10px] sm:text-xs">
        {/* Header row */}
        <div className=""></div>
        <div className="text-center text-muted-foreground p-1 sm:p-2 truncate">Pred: Normal</div>
        <div className="text-center text-muted-foreground p-1 sm:p-2 truncate">Pred: Fraud</div>
        
        {/* Actual Normal row */}
        <div className="text-right text-muted-foreground p-1 sm:p-2 truncate">Actual: Normal</div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="bg-success/20 border border-success/30 rounded p-1.5 sm:p-3 text-center">
                <div className="text-sm sm:text-lg font-bold text-success">{tn}</div>
                <div className="text-[8px] sm:text-xs text-muted-foreground">TN</div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>True Negatives: Correctly identified as normal</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="bg-destructive/20 border border-destructive/30 rounded p-1.5 sm:p-3 text-center">
                <div className="text-sm sm:text-lg font-bold text-destructive">{fp}</div>
                <div className="text-[8px] sm:text-xs text-muted-foreground">FP</div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>False Positives: Incorrectly flagged as fraud</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        
        {/* Actual Fraud row */}
        <div className="text-right text-muted-foreground p-1 sm:p-2 truncate">Actual: Fraud</div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="bg-warning/20 border border-warning/30 rounded p-1.5 sm:p-3 text-center">
                <div className="text-sm sm:text-lg font-bold text-warning">{fn}</div>
                <div className="text-[8px] sm:text-xs text-muted-foreground">FN</div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>False Negatives: Missed fraud (dangerous!)</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="bg-success/20 border border-success/30 rounded p-1.5 sm:p-3 text-center">
                <div className="text-sm sm:text-lg font-bold text-success">{tp}</div>
                <div className="text-[8px] sm:text-xs text-muted-foreground">TP</div>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>True Positives: Correctly identified as fraud</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <div className="text-[10px] sm:text-xs text-center text-muted-foreground">
        Total samples: {total}
      </div>
    </div>
  )
}

// =============================================================================
// METRIC CARD COMPONENT
// =============================================================================

interface MetricCardProps {
  title: string;
  value: number;
  description: string;
  icon: React.ReactNode;
  color: "success" | "warning" | "destructive" | "default";
}

function MetricCard({ title, value, description, icon, color }: MetricCardProps) {
  const colorClasses = {
    success: "text-success",
    warning: "text-warning",
    destructive: "text-destructive",
    default: "text-foreground",
  };
  
  const percentage = Math.round(value * 100);
  
  return (
    <div className="space-y-1 sm:space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1 sm:gap-2">
          <div className="[&>svg]:h-3 [&>svg]:w-3 sm:[&>svg]:h-4 sm:[&>svg]:w-4">{icon}</div>
          <span className="text-[10px] sm:text-sm font-medium truncate">{title}</span>
        </div>
        <span className={`text-sm sm:text-lg font-bold ${colorClasses[color]}`}>
          {percentage}%
        </span>
      </div>
      <Progress value={percentage} className="h-1.5 sm:h-2" />
      <p className="text-[9px] sm:text-xs text-muted-foreground line-clamp-2">{description}</p>
    </div>
  )
}

// =============================================================================
// FEATURE IMPORTANCE COMPONENT
// =============================================================================

interface FeatureImportanceProps {
  features: Record<string, number>;
}

function FeatureImportanceChart({ features }: FeatureImportanceProps) {
  // Sort features by importance and take top 10
  const sortedFeatures = Object.entries(features)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10);
  
  const maxValue = sortedFeatures[0]?.[1] || 1;
  
  // Format feature names for display
  const formatFeatureName = (name: string) => {
    return name
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .replace("Zscore", "Z-Score")
      .replace("Avg", "Avg.")
      .replace("Std", "Std.");
  };
  
  return (
    <div className="space-y-2 sm:space-y-3">
      <div className="text-xs sm:text-sm font-medium text-muted-foreground">
        <span className="hidden sm:inline">Top 10 Features by Importance</span>
        <span className="sm:hidden">Top Features</span>
      </div>
      <div className="space-y-1.5 sm:space-y-2">
        {sortedFeatures.map(([name, value], index) => {
          const percentage = (value / maxValue) * 100;
          const isTopFeature = index < 3;
          
          return (
            <div key={name} className="space-y-0.5 sm:space-y-1">
              <div className="flex items-center justify-between text-[10px] sm:text-xs">
                <span className={`truncate max-w-[60%] ${isTopFeature ? "font-medium" : "text-muted-foreground"}`}>
                  {formatFeatureName(name)}
                </span>
                <span className="text-muted-foreground">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 sm:h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    isTopFeature ? "bg-accent" : "bg-muted-foreground/30"
                  }`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  )
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ModelPerformanceDashboard() {
  const [metricsData, setMetricsData] = useState<MetricsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMetrics() {
      try {
        setIsLoading(true);
        const data = await getModelMetrics();
        if (data) {
          setMetricsData(data);
          setError(null);
        } else {
          setError("Failed to load metrics");
        }
      } catch (err) {
        setError("Error fetching metrics");
      } finally {
        setIsLoading(false);
      }
    }

    fetchMetrics();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <Card className="bg-card border-border">
        <CardContent className="p-4 sm:p-6">
          <div className="flex items-center justify-center h-32 sm:h-40">
            <div className="animate-pulse text-muted-foreground text-sm">
              Loading model metrics...
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !metricsData || !metricsData.metrics) {
    return (
      <Card className="bg-card border-border">
        <CardContent className="p-4 sm:p-6">
          <div className="flex flex-col items-center justify-center h-32 sm:h-40 text-muted-foreground">
            <AlertTriangle className="h-6 w-6 sm:h-8 sm:w-8 mb-2" />
            <p className="text-sm">{error || "Model not trained yet"}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { metrics, feature_importance } = metricsData;

  return (
    <Card className="bg-card border-border">
      <CardHeader className="p-3 sm:p-6 pb-2 sm:pb-4">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="flex items-center gap-1.5 sm:gap-2 text-sm sm:text-base">
              <Brain className="h-4 w-4 sm:h-5 sm:w-5 text-accent flex-shrink-0" />
              <span className="truncate">Model Performance</span>
            </CardTitle>
            <CardDescription className="text-[10px] sm:text-sm truncate">
              <span className="hidden sm:inline">Ensemble model metrics (Isolation Forest + XGBoost)</span>
              <span className="sm:hidden">Ensemble metrics</span>
            </CardDescription>
          </div>
          <Badge variant="outline" className="text-[10px] sm:text-xs flex-shrink-0 h-5 sm:h-auto">
            <CheckCircle className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-1 text-success" />
            Trained
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 sm:space-y-6 p-3 sm:p-6 pt-0">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-2 sm:gap-4">
          <MetricCard
            title="Precision"
            value={metrics.precision}
            description="Of predicted fraud, how many were correct"
            icon={<Target className="h-4 w-4 text-accent" />}
            color={metrics.precision >= 0.8 ? "success" : metrics.precision >= 0.6 ? "warning" : "destructive"}
          />
          <MetricCard
            title="Recall"
            value={metrics.recall}
            description="Of actual fraud, how many we caught"
            icon={<AlertTriangle className="h-4 w-4 text-warning" />}
            color={metrics.recall >= 0.8 ? "success" : metrics.recall >= 0.6 ? "warning" : "destructive"}
          />
          <MetricCard
            title="F1 Score"
            value={metrics.f1_score}
            description="Harmonic mean of precision and recall"
            icon={<TrendingUp className="h-4 w-4 text-success" />}
            color={metrics.f1_score >= 0.8 ? "success" : metrics.f1_score >= 0.6 ? "warning" : "destructive"}
          />
          <MetricCard
            title="ROC-AUC"
            value={metrics.roc_auc}
            description="Area under ROC curve (model quality)"
            icon={<Brain className="h-4 w-4 text-accent" />}
            color={metrics.roc_auc >= 0.9 ? "success" : metrics.roc_auc >= 0.7 ? "warning" : "destructive"}
          />
        </div>

        {/* Confusion Matrix */}
        <ConfusionMatrix
          matrix={metrics.confusion_matrix}
          tp={metrics.true_positives}
          tn={metrics.true_negatives}
          fp={metrics.false_positives}
          fn={metrics.false_negatives}
        />

        {/* Feature Importance */}
        {feature_importance && Object.keys(feature_importance).length > 0 && (
          <FeatureImportanceChart features={feature_importance} />
        )}

        {/* Model Info */}
        <div className="pt-2 sm:pt-4 border-t border-border">
          <div className="flex items-center justify-between text-[10px] sm:text-xs text-muted-foreground">
            <span>Accuracy: {(metrics.accuracy * 100).toFixed(1)}%</span>
            <span className="hidden sm:inline">Test samples: {metrics.true_positives + metrics.true_negatives + metrics.false_positives + metrics.false_negatives}</span>
            <span className="sm:hidden">Samples: {metrics.true_positives + metrics.true_negatives + metrics.false_positives + metrics.false_negatives}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default ModelPerformanceDashboard;
