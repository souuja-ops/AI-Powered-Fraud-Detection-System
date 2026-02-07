"use client";

import { useEffect, useState } from "react";
import { Wifi, WifiOff, Loader2 } from "lucide-react";
import { 
  onConnectionStatusChange, 
  wakeUpBackend, 
  type ConnectionStatus 
} from "@/lib/api";

/**
 * Connection Status Indicator (Badge)
 * 
 * Shows the current connection status to the backend as a small badge.
 * Only shows "waking" during cold starts, otherwise connected/disconnected.
 */
export function ConnectionStatus() {
  const [status, setStatus] = useState<ConnectionStatus>("connected");

  useEffect(() => {
    // Subscribe to connection status changes
    const unsubscribe = onConnectionStatusChange(setStatus);
    
    // Wake up backend on mount (handles cold start detection internally)
    wakeUpBackend();
    
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
          icon: Wifi,
          text: "Connected",
          color: "text-green-500",
          bgColor: "bg-green-500/10",
          borderColor: "border-green-500/20",
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
 * Full-width banner shown ONLY when backend is waking up from cold start.
 * Does NOT show during normal data fetches.
 */
export function ConnectionBanner() {
  const [status, setStatus] = useState<ConnectionStatus>("connected");
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    const unsubscribe = onConnectionStatusChange((newStatus) => {
      setStatus(newStatus);
      // Only show banner for waking or disconnected
      if (newStatus === "waking" || newStatus === "disconnected") {
        setShowBanner(true);
      } else if (newStatus === "connected") {
        // Hide banner after a short delay when connected
        setTimeout(() => setShowBanner(false), 1500);
      }
    });
    
    return unsubscribe;
  }, []);

  // Don't render anything if banner shouldn't show
  if (!showBanner) {
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
      case "disconnected":
        return {
          title: "Connection lost",
          description: "Unable to reach the backend. Data may be stale. Retrying...",
          color: "bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400",
        };
      case "connected":
        return {
          title: "Connected!",
          description: "Backend is ready.",
          color: "bg-green-500/10 border-green-500/30 text-green-600 dark:text-green-400",
        };
      default:
        return null;
    }
  };

  const message = getMessage();
  if (!message) return null;

  return (
    <div className={`w-full px-4 py-3 border-b ${message.color} transition-all duration-300`}>
      <div className="max-w-7xl mx-auto flex items-center gap-3">
        {status !== "connected" ? (
          <Loader2 className="h-4 w-4 animate-spin flex-shrink-0" />
        ) : (
          <Wifi className="h-4 w-4 flex-shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm">{message.title}</p>
          <p className="text-xs opacity-80">{message.description}</p>
        </div>
      </div>
    </div>
  );
}
