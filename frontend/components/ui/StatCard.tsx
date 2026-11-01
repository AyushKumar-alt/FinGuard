import React from "react";
import { cn } from "@/lib/utils";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ElementType;
  badge?: string;
  badgeColor?: "purple" | "emerald" | "amber" | "rose" | "slate";
  className?: string;
  onClick?: () => void;
}

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
  badgeColor = "purple",
  className,
  onClick,
}: StatCardProps) {
  const badgeClasses = {
    purple: "bg-[#EEECFC] text-[#7065E8] dark:bg-[#2B2564] dark:text-[#A59DFF]",
    emerald: "bg-emerald-50 text-[#239B79] dark:bg-emerald-950/40 dark:text-emerald-400",
    amber: "bg-amber-50 text-[#D9A441] dark:bg-amber-950/40 dark:text-amber-400",
    rose: "bg-rose-50 text-[#D95465] dark:bg-rose-950/40 dark:text-rose-400",
    slate: "bg-[#F4F3FA] text-[#85839A] dark:bg-[#28244D] dark:text-[#A09EB7]",
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        "bg-white dark:bg-[#18162E] rounded-3xl p-5 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card hover:shadow-card-hover transition-all duration-200 flex flex-col justify-between",
        onClick && "cursor-pointer hover:border-primary/50",
        className
      )}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs font-bold text-[#85839A] dark:text-[#A09EB7] uppercase tracking-wider">
          {title}
        </span>
        {badge && (
          <span className={cn("text-[10px] font-bold px-2 py-0.5 rounded-full", badgeClasses[badgeColor])}>
            {badge}
          </span>
        )}
      </div>

      <div className="flex items-baseline justify-between gap-3 mt-1">
        <div className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
          {value}
        </div>
        {Icon && (
          <div className="w-10 h-10 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-center text-[#7065E8] dark:text-[#A59DFF] shrink-0">
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      {subtitle && (
        <p className="text-[11px] text-[#85839A] dark:text-[#8E8BA6] mt-2 font-medium">
          {subtitle}
        </p>
      )}
    </div>
  );
}

