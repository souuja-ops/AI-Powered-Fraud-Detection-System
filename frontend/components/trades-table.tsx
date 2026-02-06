"use client"

/**
 * Enhanced Trades Table Component
 * 
 * Features:
 * - Time-based filtering (5min, 15min, 1hour, all)
 * - Risk level filtering (All, HIGH, MEDIUM, LOW)
 * - Status filtering (All, Flagged, Normal)
 * - Sortable columns
 * - Pagination / Load more
 * - New trade highlighting
 * - Quick actions
 */

import React, { useState, useMemo, useEffect } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Activity,
  Clock,
  Filter,
  Search,
  ChevronDown,
  ArrowUpDown,
  Eye,
  RefreshCw,
} from "lucide-react"
import type { TradeResponseExtended, RiskLevel } from "@/types/api"
import { RiskBreakdownDialog } from "@/components/risk-breakdown"
import { FeedbackButton } from "@/components/feedback-dialog"

// =============================================================================
// TYPES
// =============================================================================

interface TradesTableProps {
  trades: TradeResponseExtended[];
  isConnected: boolean;
  isLoading: boolean;
  pollingInterval: number;
  onTradeMarkedNormal?: (tradeId: string) => void;
}

type TimeFilter = "5min" | "15min" | "1hour" | "all";
type RiskFilter = "all" | "HIGH" | "MEDIUM" | "LOW";
type StatusFilter = "all" | "flagged" | "normal";
type SortField = "time" | "amount" | "risk";
type SortDirection = "asc" | "desc";

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function formatCurrency(amount: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
}

function formatTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function getRiskLevel(score: number): RiskLevel {
  if (score >= 70) return "HIGH";
  if (score >= 40) return "MEDIUM";
  return "LOW";
}

function filterByTime(trades: TradeResponseExtended[], filter: TimeFilter): TradeResponseExtended[] {
  if (filter === "all") return trades;
  
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
      return trades;
  }
  
  return trades.filter(trade => {
    const tradeTime = new Date(trade.timestamp).getTime();
    return tradeTime >= cutoffMs;
  });
}

function isNewTrade(timestamp: string, thresholdSeconds: number = 10): boolean {
  const tradeTime = new Date(timestamp).getTime();
  const now = new Date().getTime();
  return (now - tradeTime) < thresholdSeconds * 1000;
}

// =============================================================================
// RISK BADGE COMPONENT - Consistent with emoji indicators
// =============================================================================

function RiskBadge({ score }: { score: number }) {
  const level = getRiskLevel(score);
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
  
  const { className, icon, label } = config[level];
  
  return (
    <Badge className={`${className} font-medium text-[10px] sm:text-xs`}>
      <span className="mr-0.5">{icon}</span>
      {label}
    </Badge>
  );
}

// =============================================================================
// STATUS BADGE COMPONENT
// =============================================================================

function StatusBadge({ status, riskScore }: { status: string; riskScore: number }) {
  // Derive status from risk score if status is not properly set
  const isFlagged = status === "Flagged" || riskScore >= 70;
  
  if (isFlagged) {
    return (
      <Badge className="bg-destructive/20 text-destructive border border-destructive/30 text-[10px] sm:text-xs">
        🚨 Flagged
      </Badge>
    );
  }
  
  if (riskScore >= 40) {
    return (
      <Badge className="bg-warning/20 text-warning border border-warning/30 text-[10px] sm:text-xs">
        ⚠️ Review
      </Badge>
    );
  }
  
  return (
    <Badge className="bg-success/20 text-success border border-success/30 text-[10px] sm:text-xs">
      ✓ Normal
    </Badge>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function TradesTable({
  trades,
  isConnected,
  isLoading,
  pollingInterval,
  onTradeMarkedNormal,
}: TradesTableProps) {
  // Filter states
  const [timeFilter, setTimeFilter] = useState<TimeFilter>("all");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");
  
  // Sort state
  const [sortField, setSortField] = useState<SortField>("time");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  
  // Pagination
  const [displayCount, setDisplayCount] = useState(20);
  
  // Track new trades for highlighting
  const [newTradeIds, setNewTradeIds] = useState<Set<string>>(new Set());
  
  // Update new trade highlighting
  useEffect(() => {
    const recentIds = new Set(
      trades
        .filter(t => isNewTrade(t.timestamp, 10))
        .map(t => t.trade_id)
    );
    setNewTradeIds(recentIds);
    
    // Clear highlighting after 10 seconds
    const timer = setTimeout(() => {
      setNewTradeIds(new Set());
    }, 10000);
    
    return () => clearTimeout(timer);
  }, [trades]);
  
  // Apply filters and sorting
  const filteredTrades = useMemo(() => {
    let result = [...trades];
    
    // Time filter
    result = filterByTime(result, timeFilter);
    
    // Risk filter
    if (riskFilter !== "all") {
      result = result.filter(t => getRiskLevel(t.risk_score) === riskFilter);
    }
    
    // Status filter
    if (statusFilter !== "all") {
      result = result.filter(t => {
        const isFlagged = t.status === "Flagged" || t.risk_score >= 70;
        return statusFilter === "flagged" ? isFlagged : !isFlagged;
      });
    }
    
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(t =>
        t.trade_id.toLowerCase().includes(query) ||
        t.account_id.toLowerCase().includes(query)
      );
    }
    
    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case "time":
          comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
          break;
        case "amount":
          comparison = a.amount - b.amount;
          break;
        case "risk":
          comparison = a.risk_score - b.risk_score;
          break;
      }
      return sortDirection === "asc" ? comparison : -comparison;
    });
    
    return result;
  }, [trades, timeFilter, riskFilter, statusFilter, searchQuery, sortField, sortDirection]);
  
  // Stats
  const stats = useMemo(() => ({
    total: filteredTrades.length,
    high: filteredTrades.filter(t => t.risk_score >= 70).length,
    medium: filteredTrades.filter(t => t.risk_score >= 40 && t.risk_score < 70).length,
    low: filteredTrades.filter(t => t.risk_score < 40).length,
  }), [filteredTrades]);
  
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };
  
  const handleLoadMore = () => {
    setDisplayCount(c => c + 20);
  };
  
  const displayedTrades = filteredTrades.slice(0, displayCount);
  const hasMore = filteredTrades.length > displayCount;
  
  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-3 sm:pb-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <CardTitle className="text-foreground flex items-center gap-2 text-sm sm:text-base">
              <Activity className="h-4 w-4 sm:h-5 sm:w-5 text-accent" />
              <span className="hidden sm:inline">Recent Trades</span>
              <span className="sm:hidden">Trades</span>
            </CardTitle>
            <CardDescription className="text-xs sm:text-sm">
              {isConnected 
                ? `Live feed - ${trades.length} trades`
                : "Demo mode"
              }
            </CardDescription>
          </div>
          {isConnected && (
            <div className="flex items-center gap-2">
              <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />
              <span className="text-[10px] sm:text-xs text-muted-foreground">
                Every {pollingInterval / 1000}s
              </span>
            </div>
          )}
        </div>
        
        {/* Filter Bar - Responsive */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3 pt-3 sm:pt-4 border-t border-border mt-3 sm:mt-4">
          <Filter className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
          
          {/* Time Filter */}
          <Select value={timeFilter} onValueChange={(v) => setTimeFilter(v as TimeFilter)}>
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
          
          {/* Risk Filter */}
          <Select value={riskFilter} onValueChange={(v) => setRiskFilter(v as RiskFilter)}>
            <SelectTrigger className="h-7 sm:h-8 w-[80px] sm:w-[100px] text-[10px] sm:text-xs">
              <SelectValue placeholder="Risk" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Risk</SelectItem>
              <SelectItem value="HIGH">🔴 High</SelectItem>
              <SelectItem value="MEDIUM">🟡 Medium</SelectItem>
              <SelectItem value="LOW">🟢 Low</SelectItem>
            </SelectContent>
          </Select>
          
          {/* Status Filter */}
          <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as StatusFilter)}>
            <SelectTrigger className="h-7 sm:h-8 w-[80px] sm:w-[100px] text-[10px] sm:text-xs">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="flagged">🚨 Flagged</SelectItem>
              <SelectItem value="normal">✓ Normal</SelectItem>
            </SelectContent>
          </Select>
          
          {/* Search - Hidden on smallest screens */}
          <div className="relative hidden sm:block flex-1 min-w-[120px] max-w-[200px]">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
            <Input
              placeholder="Search ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-7 sm:h-8 pl-7 text-[10px] sm:text-xs"
            />
          </div>
          
          {/* Stats Summary - Responsive */}
          <div className="flex items-center gap-1 sm:gap-2 ml-auto text-[10px] sm:text-xs">
            <span className="text-muted-foreground hidden sm:inline">{stats.total} trades:</span>
            <Badge variant="outline" className="text-destructive border-destructive/30 text-[10px] sm:text-xs px-1 sm:px-2">
              🔴 {stats.high}
            </Badge>
            <Badge variant="outline" className="text-warning border-warning/30 text-[10px] sm:text-xs px-1 sm:px-2">
              🟡 {stats.medium}
            </Badge>
            <Badge variant="outline" className="text-success border-success/30 text-[10px] sm:text-xs px-1 sm:px-2">
              🟢 {stats.low}
            </Badge>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0 sm:p-6 sm:pt-0">
        <ScrollArea className="h-[400px] sm:h-[500px] w-full">
        {/* Mobile Card View */}
        <div className="block sm:hidden p-3 space-y-2">
          {displayedTrades.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              {isLoading ? "Loading..." : "No trades match filters"}
            </div>
          ) : (
            displayedTrades.map((trade) => {
              const isNew = newTradeIds.has(trade.trade_id);
              const riskLevel = getRiskLevel(trade.risk_score);
              
              return (
                <div 
                  key={trade.trade_id}
                  className={`p-3 rounded-lg border border-border ${
                    isNew ? "bg-accent/10" : riskLevel === "HIGH" ? "bg-destructive/5" : "bg-card"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        {isNew && <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />}
                        <span className="font-medium text-sm truncate">{trade.trade_id}</span>
                      </div>
                      <p className="text-[10px] text-muted-foreground mt-0.5">{trade.account_id}</p>
                    </div>
                    <RiskBadge score={trade.risk_score} />
                  </div>
                  <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/50">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-[10px]">{trade.type}</Badge>
                      <span className="text-xs font-medium">{formatCurrency(trade.amount)}</span>
                    </div>
                    <span className="text-[10px] text-muted-foreground">{formatTime(trade.timestamp)}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <RiskBreakdownDialog 
                      tradeId={trade.trade_id}
                      riskScore={trade.risk_score}
                      riskLevel={trade.risk_level}
                      explanation={trade.explanation}
                      amount={trade.amount}
                    />
                    {trade.risk_score >= 70 && (
                      <FeedbackButton 
                        tradeId={trade.trade_id} 
                        predictedRiskLevel={trade.risk_level}
                        onFeedbackSubmitted={(response) => {
                          if (!response.actual_is_fraud) {
                            onTradeMarkedNormal?.(trade.trade_id);
                          }
                        }}
                      />
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
        
        {/* Desktop Table View */}
        <Table className="hidden sm:table">
          <TableHeader>
            <TableRow className="border-border hover:bg-transparent">
              <TableHead className="text-muted-foreground text-xs">Trade ID</TableHead>
              <TableHead className="text-muted-foreground text-xs">Account</TableHead>
              <TableHead className="text-muted-foreground text-xs">Type</TableHead>
              <TableHead 
                className="text-muted-foreground text-xs cursor-pointer hover:text-foreground"
                onClick={() => handleSort("amount")}
              >
                <div className="flex items-center gap-1">
                  Amount
                  <ArrowUpDown className="h-3 w-3" />
                </div>
              </TableHead>
              <TableHead 
                className="text-muted-foreground text-xs cursor-pointer hover:text-foreground"
                onClick={() => handleSort("risk")}
              >
                <div className="flex items-center gap-1">
                  Risk
                  <ArrowUpDown className="h-3 w-3" />
                </div>
              </TableHead>
              <TableHead className="text-muted-foreground text-xs">Status</TableHead>
              <TableHead 
                className="text-muted-foreground text-xs cursor-pointer hover:text-foreground"
                onClick={() => handleSort("time")}
              >
                <div className="flex items-center gap-1">
                  Time
                  <ArrowUpDown className="h-3 w-3" />
                </div>
              </TableHead>
              <TableHead className="text-muted-foreground text-xs text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayedTrades.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                  {isLoading ? "Loading trades..." : "No trades match your filters"}
                </TableCell>
              </TableRow>
            ) : (
              displayedTrades.map((trade) => {
                const isNew = newTradeIds.has(trade.trade_id);
                const riskLevel = getRiskLevel(trade.risk_score);
                
                return (
                  <TableRow 
                    key={trade.trade_id} 
                    className={`border-border transition-all ${
                      isNew 
                        ? "bg-accent/10 animate-pulse" 
                        : riskLevel === "HIGH" 
                        ? "bg-destructive/5" 
                        : ""
                    }`}
                  >
                    <TableCell className="font-medium text-foreground text-xs sm:text-sm">
                      <div className="flex items-center gap-2">
                        {isNew && (
                          <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                        )}
                        {trade.trade_id}
                      </div>
                    </TableCell>
                    <TableCell className="text-muted-foreground text-xs sm:text-sm">{trade.account_id}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="text-[10px] sm:text-xs">
                        {trade.type}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-foreground font-medium">
                      {formatCurrency(trade.amount)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-2 bg-secondary rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${
                              trade.risk_score >= 70
                                ? "bg-destructive"
                                : trade.risk_score >= 40
                                ? "bg-warning"
                                : "bg-success"
                            }`}
                            style={{ width: `${trade.risk_score}%` }}
                          />
                        </div>
                        <span className="text-sm text-muted-foreground w-8">
                          {trade.risk_score}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={trade.status} riskScore={trade.risk_score} />
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatTime(trade.timestamp)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1">
                        <RiskBreakdownDialog
                          tradeId={trade.trade_id}
                          riskScore={trade.risk_score}
                          riskLevel={riskLevel}
                          explanation={trade.explanation}
                          amount={trade.amount}
                        />
                        <FeedbackButton
                          tradeId={trade.trade_id}
                          predictedRiskLevel={riskLevel}
                          onFeedbackSubmitted={(response) => {
                            // If marked as not fraud (normal), update the trade display
                            if (response && !response.actual_is_fraud) {
                              onTradeMarkedNormal?.(trade.trade_id)
                            }
                          }}
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
        </ScrollArea>
        
        {/* Load More */}
        {hasMore && (
          <div className="flex justify-center pt-4">
            <Button
              variant="outline"
              size="sm"
              onClick={handleLoadMore}
              className="text-xs"
            >
              <ChevronDown className="h-3 w-3 mr-2" />
              Load More ({filteredTrades.length - displayCount} remaining)
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default TradesTable;
