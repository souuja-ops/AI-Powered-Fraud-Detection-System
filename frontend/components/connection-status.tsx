"use client";

import { useEffect, useState } from "react";
import { Wifi, WifiOff, Loader2, AlertCircle } from "lucide-react";
import { 
  onConnectionStatusChange, 
  wakeUpBackend, 
  type ConnectionStatus 
} from "@/lib/api";

/**
 * Connection Status Indicator
 * 
 * Shows the current connection status to the backend.
 * Handles Render cold starts gracefully with user feedback.
 */
export function ConnectionStatus() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [isWaking, setIsWaking] = useState(false);

  useEffect(() => {
    // Subscribe to connection status changes
    const unsubscribe = onConnectionStatusChange(setStatus);
    
    // Wake up backend on mount
    setIsWaking(true);
    wakeUpBackend().finally(() => setIsWaking(false));
    
    return unsubscribe;
  }, []);

  const getStatusConfig = () => {
    switch (status) {
      case "connected":
        return {
          icon: Wifi,
          text: "Connected",
          color: "text-green-500",
          bgColor: "bg-green-500/10",
          borderColor: "border-green-500/20",
        };
      case "connecting":
        return {
          icon: Loader2,
          text: "Connecting...",
          color: "text-yellow-500",
          bgColor: "bg-yellow-500/10",
          borderColor: "border-yellow-500/20",
          animate: true,
        };
      case "waking":
        return {
          icon: Loader2,
          text: "Waking server...",
          color: "text-blue-500",
          bgColor: "bg-blue-500/10",
          borderColor: "border-blue-500/20",
          animate: true,
        };
      case "disconnected":
        return {
          icon: WifiOff,
          text: "Disconnected",
          color: "text-red-500",
          bgColor: "bg-red-500/10",
          borderColor: "border-red-500/20",
        };
      default:
        return {
          icon: AlertCircle,
          text: "Unknown",
          color: "text-gray-500",
          bgColor: "bg-gray-500/10",
          borderColor: "border-gray-500/20",
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div
      className={`
        inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium
        ${config.bgColor} ${config.color} border ${config.borderColor}
        transition-all duration-300
      `}
    >
      <Icon 
        className={`h-3.5 w-3.5 ${config.animate ? "animate-spin" : ""}`} 
      />
      <span>{config.text}</span>
      {status === "waking" && (
        <span className="text-[10px] opacity-70">(~30s)</span>
      )}
    </div>
  );
}

/**
 * Connection Banner
 * 
 * Full-width banner shown when backend is waking up or disconnected.
 * Provides more context to the user.
 */
export function ConnectionBanner() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [showBanner, setShowBanner] = useState(true);

  useEffect(() => {
    const unsubscribe = onConnectionStatusChange((newStatus) => {
      setStatus(newStatus);
      // Auto-hide banner after connection
      if (newStatus === "connected") {
        setTimeout(() => setShowBanner(false), 2000);
      } else {
        setShowBanner(true);
      }
    });
    
    return unsubscribe;
  }, []);

  if (!showBanner || status === "connected") {
    return null;
  }

  const getMessage = () => {
    switch (status) {
      case "waking":
        return {
          title: "Waking up the server...",
          description: "Free tier servers sleep after inactivity. This takes about 30 seconds.",
          color: "bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400",
        };
      case "connecting":
        return {
          title: "Connecting to backend...",
          description: "Establishing connection to the fraud detection API.",
          color: "bg-yellow-500/10 border-yellow-500/30 text-yellow-600 dark:text-yellow-400",
        };
      case "disconnected":
        return {
          title: "Connection lost",
          description: "Unable to reach the backend. Data may be stale. Retrying...",
          color: "bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400",
        };
      default:
        return null;
    }
  };

  const message = getMessage();
  if (!message) return null;

  return (
    <div className={`w-full px-4 py-3 border-b ${message.color}`}>
      <div className="max-w-7xl mx-auto flex items-center gap-3">
        <Loader2 className="h-4 w-4 animate-spin flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm">{message.title}</p>
          <p className="text-xs opacity-80">{message.description}</p>
        </div>
      </div>
    </div>
  );
}
