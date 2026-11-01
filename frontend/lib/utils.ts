import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { VerdictType } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number | undefined | null, decimals = 1): string {
  if (value === undefined || value === null || isNaN(value)) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(value: number | undefined | null): string {
  if (value === undefined || value === null || isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatMs(value: number | undefined | null): string {
  if (value === undefined || value === null || isNaN(value)) return "—";
  return `${value.toFixed(1)} ms`;
}

export function getVerdictTheme(verdict: VerdictType | string) {
  const norm = (verdict || "").toLowerCase();
  if (norm === "true") {
    return {
      label: "True",
      bg: "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-400 dark:border-emerald-800/60",
      pill: "bg-[#239B79] text-white",
      dot: "bg-[#239B79]",
      border: "border-[#239B79]",
    };
  }
  if (norm === "false") {
    return {
      label: "False",
      bg: "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/40 dark:text-rose-400 dark:border-rose-800/60",
      pill: "bg-[#D95465] text-white",
      dot: "bg-[#D95465]",
      border: "border-[#D95465]",
    };
  }
  return {
    label: "Not Enough Info (NEI)",
    bg: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-800/60",
    pill: "bg-[#D9A441] text-white",
    dot: "bg-[#D9A441]",
    border: "border-[#D9A441]",
  };
}

