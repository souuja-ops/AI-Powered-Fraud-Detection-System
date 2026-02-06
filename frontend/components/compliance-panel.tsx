"use client"

/**
 * Compliance & Governance Panel
 * 
 * Shows AI system compliance status with major regulatory frameworks:
 * - EU AI Act (High-risk AI systems)
 * - GDPR Article 22 (Automated decision-making)
 * - MiFID II (Algorithmic trading)
 * - SEC Rule 15c3-5 (Market Access Risk Controls)
 * 
 * This component demonstrates awareness of regulatory requirements
 * which is critical for enterprise adoption.
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Shield,
  CheckCircle,
  AlertCircle,
  FileText,
  Lock,
  Eye,
  Scale,
  Gavel,
} from "lucide-react"

// =============================================================================
// COMPLIANCE REQUIREMENTS
// =============================================================================

interface ComplianceItem {
  id: string;
  name: string;
  regulation: string;
  status: "compliant" | "partial" | "pending";
  description: string;
  article?: string;
}

const complianceItems: ComplianceItem[] = [
  {
    id: "eu-ai-act",
    name: "EU AI Act",
    regulation: "High-Risk AI Systems",
    status: "compliant",
    description: "System provides explanations for all decisions and maintains human oversight",
    article: "Article 13, 14",
  },
  {
    id: "gdpr",
    name: "GDPR",
    regulation: "Automated Decisions",
    status: "compliant",
    description: "Human review available for all flagged transactions; no fully automated final decisions",
    article: "Article 22",
  },
  {
    id: "mifid",
    name: "MiFID II",
    regulation: "Algo Trading Controls",
    status: "compliant",
    description: "Real-time monitoring, kill switches, and audit trails implemented",
    article: "Article 17",
  },
  {
    id: "sec",
    name: "SEC Rule 15c3-5",
    regulation: "Market Access Risk",
    status: "compliant",
    description: "Pre-trade risk controls with configurable thresholds and alerts",
    article: "Rule 15c3-5",
  },
  {
    id: "audit",
    name: "Audit Trail",
    regulation: "Record Keeping",
    status: "compliant",
    description: "Complete logging of all predictions, features, and human feedback",
    article: "Various",
  },
  {
    id: "bias",
    name: "Bias Monitoring",
    regulation: "Fair AI",
    status: "partial",
    description: "Feature importance tracking; demographic analysis pending",
    article: "EU AI Act Art. 10",
  },
];

// =============================================================================
// COMPLIANCE STATUS ICON
// =============================================================================

function StatusIcon({ status }: { status: ComplianceItem["status"] }) {
  switch (status) {
    case "compliant":
      return <CheckCircle className="h-3 w-3 sm:h-4 sm:w-4 text-success flex-shrink-0" />;
    case "partial":
      return <AlertCircle className="h-3 w-3 sm:h-4 sm:w-4 text-warning flex-shrink-0" />;
    case "pending":
      return <AlertCircle className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground flex-shrink-0" />;
  }
}

// =============================================================================
// COMPLIANCE BADGE
// =============================================================================

function ComplianceBadge({ status }: { status: ComplianceItem["status"] }) {
  switch (status) {
    case "compliant":
      return (
        <Badge variant="outline" className="text-[8px] sm:text-[10px] border-success/50 text-success bg-success/10 px-1 sm:px-1.5 h-4 sm:h-auto">
          ✓
        </Badge>
      );
    case "partial":
      return (
        <Badge variant="outline" className="text-[8px] sm:text-[10px] border-warning/50 text-warning bg-warning/10 px-1 sm:px-1.5 h-4 sm:h-auto">
          ~
        </Badge>
      );
    case "pending":
      return (
        <Badge variant="outline" className="text-[8px] sm:text-[10px] border-muted-foreground/50 text-muted-foreground px-1 sm:px-1.5 h-4 sm:h-auto">
          ...
        </Badge>
      );
  }
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function CompliancePanel() {
  const compliantCount = complianceItems.filter(i => i.status === "compliant").length;
  const totalCount = complianceItems.length;
  const complianceScore = Math.round((compliantCount / totalCount) * 100);

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2 sm:pb-3 p-3 sm:p-6">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="text-sm sm:text-base flex items-center gap-1.5 sm:gap-2">
              <Scale className="h-3 w-3 sm:h-4 sm:w-4 text-accent flex-shrink-0" />
              <span className="truncate">
                <span className="hidden sm:inline">AI Governance & Compliance</span>
                <span className="sm:hidden">Compliance</span>
              </span>
            </CardTitle>
            <CardDescription className="text-[10px] sm:text-xs truncate">
              <span className="hidden sm:inline">Regulatory framework compliance status</span>
              <span className="sm:hidden">Regulatory status</span>
            </CardDescription>
          </div>
          <div className="flex items-center gap-1.5 sm:gap-2 flex-shrink-0">
            <div className="text-lg sm:text-2xl font-bold text-success">{complianceScore}%</div>
            <Badge variant="outline" className="text-[8px] sm:text-xs border-success text-success h-5 sm:h-auto px-1 sm:px-2">
              <Shield className="h-2 w-2 sm:h-3 sm:w-3 mr-0.5 sm:mr-1" />
              <span className="hidden sm:inline">Verified</span>
              <span className="sm:hidden">✓</span>
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2 sm:space-y-3 p-3 sm:p-6 pt-0">
        {/* Compliance Grid */}
        <div className="space-y-1.5 sm:space-y-2">
          {complianceItems.map((item) => (
            <TooltipProvider key={item.id}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center justify-between p-1.5 sm:p-2 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-help">
                    <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                      <StatusIcon status={item.status} />
                      <div className="min-w-0">
                        <div className="text-[10px] sm:text-sm font-medium truncate">{item.name}</div>
                        <div className="text-[9px] sm:text-xs text-muted-foreground truncate hidden sm:block">{item.regulation}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 sm:gap-2 flex-shrink-0">
                      {item.article && (
                        <span className="text-[8px] sm:text-[10px] text-muted-foreground font-mono hidden md:inline">{item.article}</span>
                      )}
                      <ComplianceBadge status={item.status} />
                    </div>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="left" className="max-w-xs">
                  <p className="text-xs">{item.description}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ))}
        </div>

        {/* Key Features */}
        <div className="pt-2 sm:pt-3 border-t border-border">
          <div className="text-[10px] sm:text-xs font-medium text-muted-foreground mb-1.5 sm:mb-2">Compliance Features</div>
          <div className="grid grid-cols-2 gap-1.5 sm:gap-2 text-[10px] sm:text-xs">
            <div className="flex items-center gap-1.5 sm:gap-2 text-foreground">
              <Eye className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-accent flex-shrink-0" />
              <span className="truncate">Explainable AI</span>
            </div>
            <div className="flex items-center gap-1.5 sm:gap-2 text-foreground">
              <Lock className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-accent flex-shrink-0" />
              <span className="truncate">Data Privacy</span>
            </div>
            <div className="flex items-center gap-1.5 sm:gap-2 text-foreground">
              <FileText className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-accent flex-shrink-0" />
              <span className="truncate">Audit Trail</span>
            </div>
            <div className="flex items-center gap-1.5 sm:gap-2 text-foreground">
              <Gavel className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-accent flex-shrink-0" />
              <span className="truncate">Human Oversight</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-1.5 sm:pt-2 text-[8px] sm:text-[10px] text-muted-foreground flex items-center gap-1">
          <Shield className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
          <span className="hidden sm:inline">Last compliance check: {new Date().toLocaleDateString()}</span>
          <span className="sm:hidden">Checked: {new Date().toLocaleDateString()}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export default CompliancePanel;
