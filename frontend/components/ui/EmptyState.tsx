import React from "react";
import { LucideIcon, SearchX } from "lucide-react";

interface EmptyStateProps {
  title: string;
  description: string;
  icon?: LucideIcon;
  actionText?: string;
  onAction?: () => void;
}

export function EmptyState({
  title,
  description,
  icon: Icon = SearchX,
  actionText,
  onAction,
}: EmptyStateProps) {
  return (
    <div className="w-full py-14 px-4 bg-white dark:bg-[#18162E] rounded-3xl border border-[#E9E7F2] dark:border-[#2D294F] flex flex-col items-center justify-center text-center">
      <div className="w-14 h-14 rounded-2xl bg-[#F4F3FA] dark:bg-[#28244D] flex items-center justify-center text-[#85839A] dark:text-[#A09EB7] mb-3">
        <Icon className="w-6 h-6" />
      </div>
      <h3 className="text-base font-bold text-[#171638] dark:text-white">
        {title}
      </h3>
      <p className="text-xs text-[#85839A] dark:text-[#8E8BA6] mt-1 max-w-md">
        {description}
      </p>
      {actionText && onAction && (
        <button
          onClick={onAction}
          className="mt-4 px-4 py-2 rounded-full bg-[#171638] dark:bg-white text-white dark:text-[#171638] text-xs font-bold shadow-sm hover:scale-105 transition-transform"
        >
          {actionText}
        </button>
      )}
    </div>
  );
}

