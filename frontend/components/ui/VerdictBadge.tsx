import React from "react";
import { VerdictType } from "@/lib/types";
import { getVerdictTheme, cn } from "@/lib/utils";
import { CheckCircle2, XCircle, HelpCircle } from "lucide-react";

interface VerdictBadgeProps {
  verdict: VerdictType | string;
  size?: "sm" | "md" | "lg";
  showIcon?: boolean;
  className?: string;
}

export function VerdictBadge({
  verdict,
  size = "md",
  showIcon = true,
  className,
}: VerdictBadgeProps) {
  const theme = getVerdictTheme(verdict);
  const norm = (verdict || "").toLowerCase();

  const sizeClasses = {
    sm: "px-2.5 py-0.5 text-[11px] gap-1",
    md: "px-3.5 py-1.5 text-xs gap-1.5 font-bold",
    lg: "px-5 py-2.5 text-sm md:text-base gap-2 font-extrabold tracking-wide",
  };

  const Icon = norm === "true" ? CheckCircle2 : norm === "false" ? XCircle : HelpCircle;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border shadow-sm transition-transform select-none",
        theme.bg,
        sizeClasses[size],
        className
      )}
    >
      {showIcon && <Icon className={size === "lg" ? "w-5 h-5" : "w-3.5 h-3.5"} />}
      <span>{theme.label}</span>
    </span>
  );
}

