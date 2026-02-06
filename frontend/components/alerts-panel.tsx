"use client"

/**
 * Alerts Panel Component
 * 
 * Enhanced alerts display with:
 * - Full alerts list in expandable modal
 * - Time-based filtering
 * - Risk level filtering
 * - Quick actions (Confirm/Dismiss)
 * - Real-time updates
 * - AI Explainability panel for selected alerts
 * - Fully responsive design for all devices
 */

import React, { useState, useMemo } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  AlertTriangle,
  Shield,
  Clock,
  Filter,
  ExternalLink,
  CheckCircle,
  XCircle,
  Bell,
  ChevronRight,
  ChevronLeft,
} from "lucide-react"
import { ExplainabilityPanel } from "@/components/explainability-panel"
import type { AlertResponseUI, RiskLevel } from "@/types/api"

// =============================================================================
// TYPES
// =============================================================================

interface AlertsPanelProps {
  alerts: AlertResponseUI[];
  onConfirmFraud?: (alertId: string) => void;
  onDismiss?: (alertId: string) => void;
}

type TimeFilter = "5min" | "15min" | "1hour" | "all";
type RiskFilter = "all" | "HIGH" | "MEDIUM" | "LOW";

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function formatDateTime(timestamp: string) {
  return new Date(timestamp).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function getTimeAgo(timestamp: string): string {
  const now = new Date();
  const time = new Date(timestamp);
  const diffMs = now.getTime() - time.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${Math.floor(diffHours / 24)}d ago`;
}

function filterByTime(alerts: AlertResponseUI[], filter: TimeFilter): AlertResponseUI[] {
  if (filter === "all") return alerts;
  
  const now = Date.now();
  let cutoffMs: number;
  
  switch (filter) {
    case "5min":
      cutoffMs = now - 5 * 60 * 1000;
      break;
    case "15min":
      cutoffMs = now - 15 * 60 * 1000;
      break;
    case "1hour":
      cutoffMs = now - 60 * 60 * 1000;
      break;
    default:
      return alerts;
  }
  
  return alerts.filter(alert => {
    const alertTime = new Date(alert.timestamp).getTime();
    return alertTime >= cutoffMs;
  });
}

function filterByRisk(alerts: AlertResponseUI[], filter: RiskFilter): AlertResponseUI[] {
  if (filter === "all") return alerts;
  return alerts.filter(alert => alert.risk_level === filter);
}

// =============================================================================
// RISK BADGE COMPONENT - Consistent across all views
// =============================================================================

function RiskBadge({ level, compact = false }: { level: RiskLevel; compact?: boolean }) {
  const config = {
    HIGH: { 
      className: "bg-destructive/20 text-destructive border border-destructive/30", 
      icon: "🔴",
      label: "HIGH" 
    },
    MEDIUM: { 
      className: "bg-warning/20 text-warning border border-warning/30", 
      icon: "🟡",
      label: "MEDIUM" 
    },
    LOW: { 
      className: "bg-success/20 text-success border border-success/30", 
      icon: "🟢",
      label: "LOW" 
    },
  };
  
  const { className, icon, label } = config[level] || config.MEDIUM;
  
  if (compact) {
    return (
      <Badge className={`${className} font-medium text-xs px-1.5 py-0.5`}>
        <span className="mr-0.5 text-xs">{icon}</span>
        {label}
      </Badge>
    );
  }
  
  return (
    <Badge className={`${className} font-medium`}>
      <span className="mr-1">{icon}</span>
      {label}
    </Badge>
  );
}

// =============================================================================
// SINGLE ALERT CARD COMPONENT
// =============================================================================

interface AlertCardProps {
  alert: AlertResponseUI;
  compact?: boolean;
  onConfirm?: () => void;
  onDismiss?: () => void;
}

function AlertCard({ alert, compact = false, onConfirm, onDismiss }: AlertCardProps) {
  const [actionTaken, setActionTaken] = useState<"confirmed" | "dismissed" | null>(null);
  
  const handleConfirm = () => {
    setActionTaken("confirmed");
    onConfirm?.();
  };
  
  const handleDismiss = () => {
    setActionTaken("dismissed");
    onDismiss?.();
  };
  
  if (actionTaken) {
    return (
      <div className={`flex items-center gap-2 sm:gap-3 p-2 sm:p-3 rounded-lg border ${
        actionTaken === "confirmed" 
          ? "bg-destructive/10 border-destructive/30" 
          : "bg-muted/30 border-muted"
      }`}>
        {actionTaken === "confirmed" ? (
          <CheckCircle className="h-3 w-3 sm:h-4 sm:w-4 text-destructive flex-shrink-0" />
        ) : (
          <XCircle className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground flex-shrink-0" />
        )}
        <span className="text-xs sm:text-sm text-muted-foreground">
          {actionTaken === "confirmed" ? "Confirmed as fraud" : "Dismissed"}
        </span>
      </div>
    );
  }
  
  return (
    <div className={`flex items-start gap-2 sm:gap-3 p-2 sm:p-3 rounded-lg bg-secondary/30 border border-border transition-all hover:bg-secondary/50 ${
      alert.risk_level === "HIGH" ? "border-l-4 border-l-destructive" : ""
    }`}>
      <RiskBadge level={alert.risk_level} compact={compact} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs sm:text-sm font-medium text-foreground truncate">
            {alert.trade_id}
          </p>
          <span className="text-[10px] sm:text-xs text-muted-foreground whitespace-nowrap">
            {getTimeAgo(alert.timestamp)}
          </span>
        </div>
        <p className={`text-[10px] sm:text-xs text-muted-foreground mt-1 ${compact ? "line-clamp-1" : "line-clamp-2"}`}>
          {alert.explanation}
        </p>
        
        {!compact && (
          <div className="flex items-center gap-1 sm:gap-2 mt-2">
            <Button
              variant="outline"
              size="sm"
              className="h-6 sm:h-7 text-[10px] sm:text-xs border-destructive/50 hover:bg-destructive/10 text-destructive px-2"
              onClick={handleConfirm}
            >
              <AlertTriangle className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-1" />
              Confirm
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-6 sm:h-7 text-[10px] sm:text-xs px-2"
              onClick={handleDismiss}
            >
              <XCircle className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-1" />
              Dismiss
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// FILTER BAR COMPONENT - Responsive
// =============================================================================

interface FilterBarProps {
  timeFilter: TimeFilter;
  riskFilter: RiskFilter;
  onTimeFilterChange: (value: TimeFilter) => void;
  onRiskFilterChange: (value: RiskFilter) => void;
  totalCount: number;
  filteredCount: number;
}

function FilterBar({
  timeFilter,
  riskFilter,
  onTimeFilterChange,
  onRiskFilterChange,
  totalCount,
  filteredCount,
}: FilterBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 pb-3 border-b border-border">
      <Filter className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
      
      <Select value={timeFilter} onValueChange={(v) => onTimeFilterChange(v as TimeFilter)}>
        <SelectTrigger className="h-7 sm:h-8 w-[90px] sm:w-[110px] text-[10px] sm:text-xs">
          <Clock className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-1" />
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="5min">Last 5 min</SelectItem>
          <SelectItem value="15min">Last 15 min</SelectItem>
          <SelectItem value="1hour">Last 1 hour</SelectItem>
          <SelectItem value="all">All time</SelectItem>
        </SelectContent>
      </Select>
      
      <Select value={riskFilter} onValueChange={(v) => onRiskFilterChange(v as RiskFilter)}>
        <SelectTrigger className="h-7 sm:h-8 w-[80px] sm:w-[100px] text-[10px] sm:text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Risk</SelectItem>
          <SelectItem value="HIGH">🔴 High</SelectItem>
          <SelectItem value="MEDIUM">🟡 Medium</SelectItem>
          <SelectItem value="LOW">🟢 Low</SelectItem>
        </SelectContent>
      </Select>
      
      <span className="text-[10px] sm:text-xs text-muted-foreground ml-auto">
        {filteredCount}/{totalCount}
      </span>
    </div>
  );
}

// =============================================================================
// ALL ALERTS MODAL - Fully Responsive
// =============================================================================

interface AllAlertsModalProps {
  alerts: AlertResponseUI[];
  onConfirmFraud?: (alertId: string) => void;
  onDismiss?: (alertId: string) => void;
}

function AllAlertsModal({ alerts, onConfirmFraud, onDismiss }: AllAlertsModalProps) {
  const [timeFilter, setTimeFilter] = useState<TimeFilter>("all");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [selectedAlert, setSelectedAlert] = useState<AlertResponseUI | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  
  const filteredAlerts = useMemo(() => {
    let result = alerts;
    result = filterByTime(result, timeFilter);
    result = filterByRisk(result, riskFilter);
    return result.sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [alerts, timeFilter, riskFilter]);
  
  const stats = useMemo(() => ({
    high: filteredAlerts.filter(a => a.risk_level === "HIGH").length,
    medium: filteredAlerts.filter(a => a.risk_level === "MEDIUM").length,
    low: filteredAlerts.filter(a => a.risk_level === "LOW").length,
  }), [filteredAlerts]);

  const handleSelectAlert = (alert: AlertResponseUI) => {
    setSelectedAlert(alert);
    setShowDetails(true);
  };

  const handleBackToList = () => {
    setShowDetails(false);
  };
  
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="w-full mt-3">
          <ExternalLink className="h-3 w-3 mr-2" />
          View All Alerts ({alerts.length})
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-5xl w-[95vw] h-[90vh] sm:h-[85vh] bg-card border-border p-0 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex-shrink-0 p-3 sm:p-5 pb-3 sm:pb-4 border-b border-border bg-card">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-base sm:text-lg">
              <Bell className="h-4 w-4 sm:h-5 sm:w-5 text-destructive" />
              Alert Investigation Center
            </DialogTitle>
            <DialogDescription className="text-xs sm:text-sm">
              Review and investigate flagged transactions.
            </DialogDescription>
          </DialogHeader>
          
          {/* Stats Bar - Responsive */}
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-3 mt-3 sm:mt-4">
            <div className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 bg-destructive/10 rounded-md border border-destructive/20">
              <span className="text-xs sm:text-sm">🔴</span>
              <span className="text-[10px] sm:text-sm font-medium text-destructive">{stats.high} High</span>
            </div>
            <div className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 bg-warning/10 rounded-md border border-warning/20">
              <span className="text-xs sm:text-sm">🟡</span>
              <span className="text-[10px] sm:text-sm font-medium text-warning">{stats.medium} Med</span>
            </div>
            <div className="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 bg-success/10 rounded-md border border-success/20">
              <span className="text-xs sm:text-sm">🟢</span>
              <span className="text-[10px] sm:text-sm font-medium text-success">{stats.low} Low</span>
            </div>
            <div className="hidden sm:block ml-auto text-xs text-muted-foreground bg-secondary/50 px-2 py-1 rounded">
              {filteredAlerts.length} of {alerts.length} alerts
            </div>
          </div>
        </div>
        
        {/* Filter Bar */}
        <div className="flex-shrink-0 px-3 sm:px-5 py-2 sm:py-3 border-b border-border bg-muted/30">
          <FilterBar
            timeFilter={timeFilter}
            riskFilter={riskFilter}
            onTimeFilterChange={setTimeFilter}
            onRiskFilterChange={setRiskFilter}
            totalCount={alerts.length}
            filteredCount={filteredAlerts.length}
          />
        </div>
        
        {/* Main Content - Responsive Split Panel */}
        <div className="flex-1 flex flex-col md:flex-row min-h-0 overflow-hidden">
          {/* Left: Alerts List */}
          <div className={`md:w-2/5 border-b md:border-b-0 md:border-r border-border flex flex-col min-h-0 ${
            showDetails ? 'hidden md:flex' : 'flex'
          }`}>
            <div className="px-3 sm:px-4 py-2 border-b border-border bg-muted/20 flex-shrink-0">
              <span className="text-[10px] sm:text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Alerts Queue ({filteredAlerts.length})
              </span>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-2 sm:p-3 space-y-2">
                {filteredAlerts.length === 0 ? (
                  <div className="text-center py-12 sm:py-16 text-muted-foreground">
                    <Shield className="h-10 w-10 sm:h-12 sm:w-12 mx-auto mb-3 opacity-40" />
                    <p className="text-sm font-medium">No alerts found</p>
                    <p className="text-xs mt-1 text-muted-foreground/70">
                      Try adjusting your filters
                    </p>
                  </div>
                ) : (
                  filteredAlerts.map((alert, index) => (
                    <div
                      key={alert.alert_id}
                      onClick={() => handleSelectAlert(alert)}
                      className={`p-2 sm:p-3 rounded-lg border cursor-pointer transition-all duration-150 ${
                        selectedAlert?.alert_id === alert.alert_id
                          ? "border-primary bg-primary/5 shadow-sm"
                          : "border-border hover:border-primary/40 hover:bg-muted/50"
                      }`}
                    >
                      <div className="flex items-start gap-2 sm:gap-3">
                        <div className="flex-shrink-0 mt-0.5">
                          <RiskBadge level={alert.risk_level} compact />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-xs sm:text-sm font-medium text-foreground truncate">
                              {alert.trade_id}
                            </p>
                            <span className="text-[9px] sm:text-[10px] text-muted-foreground whitespace-nowrap">
                              #{index + 1}
                            </span>
                          </div>
                          <p className="text-[10px] sm:text-xs text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
                            {alert.explanation}
                          </p>
                          <p className="text-[10px] sm:text-[11px] text-muted-foreground/60 mt-1.5">
                            {getTimeAgo(alert.timestamp)}
                          </p>
                        </div>
                        <ChevronRight className={`h-3 w-3 sm:h-4 sm:w-4 flex-shrink-0 transition-colors ${
                          selectedAlert?.alert_id === alert.alert_id 
                            ? "text-primary" 
                            : "text-muted-foreground/40"
                        }`} />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </div>
          
          {/* Right: Detail Panel */}
          <div className={`md:w-3/5 flex flex-col min-h-0 bg-muted/10 ${
            showDetails ? 'flex' : 'hidden md:flex'
          }`}>
            {/* Mobile Back Button */}
            <div className="md:hidden px-3 py-2 border-b border-border bg-muted/20 flex-shrink-0">
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={handleBackToList}
                className="h-7 text-xs"
              >
                <ChevronLeft className="h-3 w-3 mr-1" />
                Back to list
              </Button>
            </div>
            
            <div className="hidden md:block px-5 py-2 border-b border-border bg-muted/20 flex-shrink-0">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Alert Details
              </span>
            </div>
            
            <ScrollArea className="flex-1">
              {selectedAlert ? (
                <div className="p-3 sm:p-5">
                  {/* Alert Header */}
                  <div className="flex items-start gap-3 sm:gap-4 pb-3 sm:pb-4 border-b border-border">
                    <RiskBadge level={selectedAlert.risk_level} />
                    <div className="flex-1">
                      <h3 className="text-sm sm:text-base font-semibold text-foreground">
                        {selectedAlert.trade_id}
                      </h3>
                      <p className="text-[10px] sm:text-xs text-muted-foreground mt-0.5">
                        Alert ID: <span className="font-mono">{selectedAlert.alert_id}</span>
                      </p>
                    </div>
                  </div>
                  
                  {/* Detection Details */}
                  <div className="py-3 sm:py-4 space-y-3 sm:space-y-4">
                    <div>
                      <label className="text-[10px] sm:text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                        <AlertTriangle className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
                        Detection Reason
                      </label>
                      <div className="mt-2 p-3 sm:p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                        <p className="text-xs sm:text-sm text-foreground leading-relaxed">
                          {selectedAlert.explanation}
                        </p>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                      <div>
                        <label className="text-[10px] sm:text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                          <Clock className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
                          Timestamp
                        </label>
                        <p className="mt-2 text-xs sm:text-sm text-foreground font-mono bg-muted/50 px-2 sm:px-3 py-1.5 sm:py-2 rounded">
                          {formatDateTime(selectedAlert.timestamp)}
                        </p>
                      </div>
                      
                      <div>
                        <label className="text-[10px] sm:text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                          Risk Level
                        </label>
                        <div className="mt-2 p-2 sm:p-3 bg-muted/50 rounded">
                          <div className="flex items-center justify-between mb-2">
                            <RiskBadge level={selectedAlert.risk_level} compact />
                            <span className="text-[10px] sm:text-xs text-muted-foreground">
                              {selectedAlert.risk_level === "HIGH" ? "100%" : selectedAlert.risk_level === "MEDIUM" ? "66%" : "33%"}
                            </span>
                          </div>
                          <div className="w-full h-1.5 sm:h-2 bg-secondary rounded-full overflow-hidden">
                            <div 
                              className={`h-full transition-all duration-300 ${
                                selectedAlert.risk_level === "HIGH" 
                                  ? "bg-destructive w-full" 
                                  : selectedAlert.risk_level === "MEDIUM"
                                  ? "bg-warning w-2/3"
                                  : "bg-success w-1/3"
                              }`}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* AI Explainability Section */}
                  <div className="pt-3 sm:pt-4 mt-2 border-t border-border">
                    <ExplainabilityPanel
                      tradeId={selectedAlert.trade_id}
                      riskScore={selectedAlert.risk_level === "HIGH" ? 85 : selectedAlert.risk_level === "MEDIUM" ? 55 : 25}
                      riskLevel={selectedAlert.risk_level as "LOW" | "MEDIUM" | "HIGH"}
                      explanation={selectedAlert.explanation}
                    />
                  </div>
                  
                  {/* Action Buttons */}
                  <div className="pt-3 sm:pt-4 mt-2 border-t border-border">
                    <p className="text-[10px] sm:text-xs text-muted-foreground mb-2 sm:mb-3 flex items-center gap-1">
                      <Shield className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
                      Take action on this alert:
                    </p>
                    <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
                      <Button
                        variant="destructive"
                        size="default"
                        className="flex-1 h-9 sm:h-10 text-xs sm:text-sm"
                        onClick={() => {
                          onConfirmFraud?.(selectedAlert.alert_id)
                          setSelectedAlert(null)
                          setShowDetails(false)
                        }}
                      >
                        <AlertTriangle className="h-3 w-3 sm:h-4 sm:w-4 mr-2" />
                        Confirm Fraud
                      </Button>
                      <Button
                        variant="outline"
                        size="default"
                        className="flex-1 h-9 sm:h-10 text-xs sm:text-sm"
                        onClick={() => {
                          onDismiss?.(selectedAlert.alert_id)
                          setSelectedAlert(null)
                          setShowDetails(false)
                        }}
                      >
                        <XCircle className="h-3 w-3 sm:h-4 sm:w-4 mr-2" />
                        False Positive
                      </Button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full min-h-[300px] sm:min-h-[400px] flex items-center justify-center text-muted-foreground">
                  <div className="text-center p-6 sm:p-8">
                    <div className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-3 sm:mb-4 rounded-full bg-muted/50 flex items-center justify-center">
                      <AlertTriangle className="h-6 w-6 sm:h-8 sm:w-8 opacity-40" />
                    </div>
                    <p className="text-sm font-medium">No alert selected</p>
                    <p className="text-xs mt-1 text-muted-foreground/70">
                      Click on an alert from the list to view details
                    </p>
                  </div>
                </div>
              )}
            </ScrollArea>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// MAIN ALERTS PANEL COMPONENT
// =============================================================================

export function AlertsPanel({ alerts, onConfirmFraud, onDismiss }: AlertsPanelProps) {
  const [timeFilter, setTimeFilter] = useState<TimeFilter>("all");
  
  const filteredAlerts = useMemo(() => {
    let result = filterByTime(alerts, timeFilter);
    return result.sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [alerts, timeFilter]);
  
  const highRiskCount = alerts.filter(a => a.risk_level === "HIGH").length;
  
  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-foreground flex items-center gap-2 text-sm sm:text-base">
            <AlertTriangle className="h-4 w-4 sm:h-5 sm:w-5 text-destructive" />
            <span className="hidden sm:inline">Recent Alerts</span>
            <span className="sm:hidden">Alerts</span>
            {alerts.length > 0 && (
              <Badge variant="destructive" className="ml-1 sm:ml-2 text-[10px] sm:text-xs">
                {alerts.length}
              </Badge>
            )}
          </CardTitle>
          {highRiskCount > 0 && (
            <Badge className="bg-destructive/20 text-destructive border-destructive/30 animate-pulse text-[10px] sm:text-xs">
              🔴 {highRiskCount} HIGH
            </Badge>
          )}
        </div>
        <CardDescription className="text-xs sm:text-sm">Trades requiring attention</CardDescription>
        
        {/* Quick Time Filter */}
        <div className="flex items-center gap-1 sm:gap-2 pt-2">
          <Clock className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-muted-foreground" />
          <div className="flex gap-0.5 sm:gap-1">
            {(["5min", "15min", "1hour", "all"] as TimeFilter[]).map((filter) => (
              <Button
                key={filter}
                variant={timeFilter === filter ? "secondary" : "ghost"}
                size="sm"
                className="h-5 sm:h-6 px-1.5 sm:px-2 text-[10px] sm:text-xs"
                onClick={() => setTimeFilter(filter)}
              >
                {filter === "all" ? "All" : filter === "5min" ? "5m" : filter === "15min" ? "15m" : "1h"}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-2 sm:space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-4 sm:py-6 text-muted-foreground">
            <Shield className="h-6 w-6 sm:h-8 sm:w-8 mx-auto mb-2 opacity-50" />
            <p className="text-xs sm:text-sm">
              {timeFilter === "all" ? "No alerts yet" : "No alerts in this time period"}
            </p>
          </div>
        ) : (
          <>
            {/* Show first 5 alerts */}
            <div className="space-y-2 max-h-[200px] sm:max-h-[280px] overflow-y-auto">
              {filteredAlerts.slice(0, 5).map((alert) => (
                <AlertCard
                  key={alert.alert_id}
                  alert={alert}
                  compact
                  onConfirm={() => onConfirmFraud?.(alert.alert_id)}
                  onDismiss={() => onDismiss?.(alert.alert_id)}
                />
              ))}
            </div>
            
            {/* Show "more" indicator */}
            {filteredAlerts.length > 5 && (
              <div className="text-center text-[10px] sm:text-xs text-muted-foreground pt-2">
                +{filteredAlerts.length - 5} more alerts
              </div>
            )}
          </>
        )}
        
        {/* View All Button */}
        <AllAlertsModal
          alerts={alerts}
          onConfirmFraud={onConfirmFraud}
          onDismiss={onDismiss}
        />
      </CardContent>
    </Card>
  );
}

export default AlertsPanel;
