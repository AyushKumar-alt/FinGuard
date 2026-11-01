import React from "react";
import { cn } from "@/lib/utils";

interface CircularProgressProps {
  percentage: number;
  label: string;
  category: string;
  note?: string;
  color?: "emerald" | "purple" | "amber" | "rose";
  className?: string;
}

export function CircularProgress({
  percentage,
  label,
  category,
  note,
  color = "emerald",
  className,
}: CircularProgressProps) {
  const radius = 32;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (Math.min(100, Math.max(0, percentage)) / 100) * circumference;

  const colorStyles = {
    emerald: {
      stroke: "#239B79",
      text: "text-[#239B79]",
      badge: "bg-emerald-50 text-[#239B79] dark:bg-emerald-950/40 dark:text-emerald-400",
    },
    purple: {
      stroke: "#7065E8",
      text: "text-[#7065E8]",
      badge: "bg-[#EEECFC] text-[#7065E8] dark:bg-[#2B2564] dark:text-[#A59DFF]",
    },
    amber: {
      stroke: "#D9A441",
      text: "text-[#D9A441]",
      badge: "bg-amber-50 text-[#D9A441] dark:bg-amber-950/40 dark:text-amber-400",
    },
    rose: {
      stroke: "#D95465",
      text: "text-[#D95465]",
      badge: "bg-rose-50 text-[#D95465] dark:bg-rose-950/40 dark:text-rose-400",
    },
  };

  const style = colorStyles[color];

  return (
    <div
      className={cn(
        "bg-white dark:bg-[#18162E] rounded-3xl p-5 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card flex items-center justify-between gap-4",
        className
      )}
    >
      <div className="flex-1">
        <span className="text-[10px] font-bold uppercase tracking-wider text-[#85839A] dark:text-[#A09EB7]">
          {category}
        </span>
        <h4 className="text-sm font-bold text-[#171638] dark:text-white mt-0.5">
          {label}
        </h4>
        {note && (
          <p className="text-[11px] text-[#85839A] dark:text-[#8E8BA6] mt-1">
            {note}
          </p>
        )}
      </div>

      <div className="relative w-20 h-20 shrink-0 flex items-center justify-center">
        <svg className="w-full h-full -rotate-90 transform" viewBox="0 0 80 80">
          <circle
            cx="40"
            cy="40"
            r={radius}
            className="text-[#E9E7F2] dark:text-[#2D294F]"
            strokeWidth="7"
            stroke="currentColor"
            fill="transparent"
          />
          <circle
            cx="40"
            cy="40"
            r={radius}
            stroke={style.stroke}
            strokeWidth="7"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={cn("text-xs font-extrabold tracking-tight", style.text)}>
            {Math.round(percentage)}%
          </span>
        </div>
      </div>
    </div>
  );
}

