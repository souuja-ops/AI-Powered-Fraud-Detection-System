"use client"

/**
 * Feedback Dialog Component
 * 
 * Allows users to provide feedback on model predictions.
 * This enables the active learning loop:
 * 1. Model makes prediction
 * 2. Human reviews and provides ground truth
 * 3. Model learns from feedback
 * 
 * Critical for production ML systems.
 */

import React, { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  ThumbsUp,
  ThumbsDown,
  Loader2,
  MessageSquare,
} from "lucide-react"
import { submitFeedback, type FeedbackResponse } from "@/lib/api"
import { toast } from "sonner"

// =============================================================================
// TYPES
// =============================================================================

interface FeedbackButtonProps {
  tradeId: string;
  predictedRiskLevel: string;
  onFeedbackSubmitted?: (response: FeedbackResponse) => void;
}

// =============================================================================
// FEEDBACK BUTTON COMPONENT
// =============================================================================

export function FeedbackButton({
  tradeId,
  predictedRiskLevel,
  onFeedbackSubmitted,
}: FeedbackButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedbackResult, setFeedbackResult] = useState<FeedbackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFeedback = async (isFraud: boolean) => {
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await submitFeedback(tradeId, isFraud);
      
      if (response) {
        setFeedbackResult(response);
        onFeedbackSubmitted?.(response);
        
        // Show toast notification
        if (isFraud) {
          toast.success("Marked as fraud", {
            description: `${tradeId} confirmed as fraudulent. Model will learn from this.`
          });
        } else {
          toast.success("Marked as normal", {
            description: `${tradeId} marked as normal transaction. Model will learn from this.`
          });
        }
      } else {
        setError("Failed to submit feedback. Please try again.");
        toast.error("Failed to submit feedback");
      }
    } catch (err) {
      setError("An error occurred. Please try again.");
      toast.error("Error submitting feedback");
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetDialog = () => {
    setFeedbackResult(null);
    setError(null);
    setIsOpen(false);
  };

  const wasPredictedAsFraud = predictedRiskLevel === "HIGH";

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs"
          title="Provide feedback"
        >
          <MessageSquare className="h-3 w-3 mr-1" />
          Feedback
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md bg-card border-border">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-accent" />
            Provide Feedback
          </DialogTitle>
          <DialogDescription>
            Help improve the model by confirming or correcting this prediction.
          </DialogDescription>
        </DialogHeader>

        {!feedbackResult ? (
          <div className="space-y-4 py-4">
            {/* Current Prediction */}
            <div className="bg-secondary/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Trade ID</span>
                <span className="font-mono text-sm">{tradeId}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Model Prediction</span>
                <Badge
                  className={
                    wasPredictedAsFraud
                      ? "bg-destructive/20 text-destructive border-destructive/30"
                      : "bg-success/20 text-success border-success/30"
                  }
                >
                  {wasPredictedAsFraud ? "Fraud Detected" : "Normal"}
                </Badge>
              </div>
            </div>

            {/* Question */}
            <div className="text-center">
              <p className="text-sm font-medium mb-4">
                Was this transaction actually fraudulent?
              </p>
              
              <div className="flex gap-3 justify-center">
                <Button
                  variant="outline"
                  className="flex-1 border-destructive/50 hover:bg-destructive/10 hover:border-destructive"
                  onClick={() => handleFeedback(true)}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 mr-2 text-destructive" />
                  )}
                  Yes, Fraud
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 border-success/50 hover:bg-success/10 hover:border-success"
                  onClick={() => handleFeedback(false)}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <CheckCircle className="h-4 w-4 mr-2 text-success" />
                  )}
                  No, Normal
                </Button>
              </div>

              {error && (
                <p className="text-xs text-destructive mt-3">{error}</p>
              )}
            </div>
          </div>
        ) : (
          /* Feedback Result */
          <div className="space-y-4 py-4">
            <div className="flex flex-col items-center text-center">
              {feedbackResult.was_correct ? (
                <>
                  <div className="h-12 w-12 rounded-full bg-success/20 flex items-center justify-center mb-3">
                    <ThumbsUp className="h-6 w-6 text-success" />
                  </div>
                  <h3 className="font-medium text-success">Model was correct!</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    The prediction matched your feedback.
                  </p>
                </>
              ) : (
                <>
                  <div className="h-12 w-12 rounded-full bg-warning/20 flex items-center justify-center mb-3">
                    <ThumbsDown className="h-6 w-6 text-warning" />
                  </div>
                  <h3 className="font-medium text-warning">Feedback recorded</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    The model will learn from this correction.
                  </p>
                </>
              )}
            </div>

            {/* Details */}
            <div className="bg-secondary/30 rounded-lg p-3 text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Predicted:</span>
                <span>{feedbackResult.predicted_risk_level}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Actual:</span>
                <span>{feedbackResult.actual_is_fraud ? "Fraud" : "Normal"}</span>
              </div>
            </div>

            <p className="text-xs text-center text-muted-foreground">
              {feedbackResult.message}
            </p>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={resetDialog} className="w-full">
            {feedbackResult ? "Close" : "Cancel"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// INLINE FEEDBACK BUTTONS (for table rows)
// =============================================================================

interface InlineFeedbackProps {
  tradeId: string;
  predictedRiskLevel: string;
  compact?: boolean;
}

export function InlineFeedbackButtons({
  tradeId,
  predictedRiskLevel,
  compact = false,
}: InlineFeedbackProps) {
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");
  const [result, setResult] = useState<"correct" | "incorrect" | null>(null);

  const handleQuickFeedback = async (isFraud: boolean) => {
    setStatus("submitting");
    
    try {
      const response = await submitFeedback(tradeId, isFraud);
      if (response) {
        setStatus("success");
        setResult(response.was_correct ? "correct" : "incorrect");
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  };

  if (status === "success") {
    return (
      <div className="flex items-center gap-1">
        {result === "correct" ? (
          <CheckCircle className="h-4 w-4 text-success" />
        ) : (
          <XCircle className="h-4 w-4 text-warning" />
        )}
        <span className="text-xs text-muted-foreground">
          {result === "correct" ? "Correct" : "Noted"}
        </span>
      </div>
    );
  }

  if (status === "submitting") {
    return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
  }

  return (
    <div className="flex items-center gap-1">
      <Button
        variant="ghost"
        size="sm"
        className="h-6 w-6 p-0"
        onClick={() => handleQuickFeedback(true)}
        title="Confirm as fraud"
      >
        <AlertTriangle className="h-3 w-3 text-destructive/70 hover:text-destructive" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className="h-6 w-6 p-0"
        onClick={() => handleQuickFeedback(false)}
        title="Mark as normal"
      >
        <CheckCircle className="h-3 w-3 text-success/70 hover:text-success" />
      </Button>
    </div>
  );
}

export default FeedbackButton;
