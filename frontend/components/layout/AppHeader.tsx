"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "./ThemeProvider";
import {
  Search,
  Sun,
  Moon,
  Bell,
  Settings,
  Download,
  Plus,
  ShieldCheck,
  CheckCircle2,
} from "lucide-react";

const TOP_TABS = [
  { href: "/", label: "Dashboard" },
  { href: "/verify", label: "Verify Claim" },
  { href: "/retrieval-lab", label: "Retrieval Lab" },
  { href: "/gold-benchmark", label: "Gold Benchmark" },
];

export function AppHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const { theme, toggleTheme } = useTheme();
  const [searchQuery, setSearchQuery] = useState("");
  const [showNotifications, setShowNotifications] = useState(false);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/evidence?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  return (
    <header className="w-full pt-4 pb-2 px-4 md:px-8 flex flex-col md:flex-row items-center justify-between gap-4 select-none">
      {/* Left: Top Tab Navigation */}
      <div className="flex items-center gap-1 md:gap-2 self-start md:self-auto overflow-x-auto max-w-full">
        {TOP_TABS.map((tab) => {
          const isActive = tab.href === "/" ? pathname === "/" : pathname.startsWith(tab.href);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`relative px-3.5 py-1.5 text-sm font-semibold rounded-full transition-all duration-150 whitespace-nowrap ${
                isActive
                  ? "text-[#171638] dark:text-white bg-white/70 dark:bg-white/10 shadow-sm"
                  : "text-[#85839A] dark:text-[#A09EB7] hover:text-[#171638] dark:hover:text-white"
              }`}
            >
              {tab.label}
              {isActive && (
                <span className="absolute bottom-0 left-3 right-3 h-[2px] bg-[#7065E8] rounded-full" />
              )}
            </Link>
          );
        })}
      </div>

      {/* Center: Search pill */}
      <form
        onSubmit={handleSearchSubmit}
        className="relative w-full md:w-80 lg:w-96 flex items-center"
      >
        <Search className="w-4 h-4 text-[#85839A] absolute left-3.5 pointer-events-none" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search claims, evidence, or tokens..."
          className="w-full h-10 pl-10 pr-12 text-xs md:text-sm bg-white dark:bg-[#18162E] text-[#171638] dark:text-white rounded-full border border-[#E9E7F2] dark:border-[#2D294F] shadow-sm focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary transition-all placeholder:text-[#85839A]"
        />
        <kbd className="hidden lg:inline-flex absolute right-3 px-1.5 py-0.5 text-[10px] font-mono text-[#85839A] bg-[#F4F3FA] dark:bg-[#28244D] rounded-md border border-[#E9E7F2] dark:border-[#383366]">
          ↵
        </kbd>
      </form>

      {/* Right Controls: Theme Pill, Bell, Settings, Export, Primary CTA */}
      <div className="flex items-center gap-2.5 self-end md:self-auto">
        {/* Light / Dark Mode Pill Toggle */}
        <button
          onClick={toggleTheme}
          className="h-9 px-2.5 rounded-full bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center gap-1.5 shadow-sm text-xs font-medium text-[#171638] dark:text-white hover:border-primary/50 transition-colors"
          title="Toggle light/dark mode"
        >
          {theme === "light" ? (
            <>
              <Sun className="w-4 h-4 text-amber-500 fill-amber-500" />
              <span className="text-[11px] font-bold">Light</span>
            </>
          ) : (
            <>
              <Moon className="w-4 h-4 text-primary fill-primary" />
              <span className="text-[11px] font-bold">Dark</span>
            </>
          )}
        </button>

        {/* Notifications Icon with popover */}
        <div className="relative">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className="w-9 h-9 rounded-full bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-center text-[#85839A] hover:text-[#171638] dark:hover:text-white shadow-sm transition-colors relative"
            title="System notifications"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-primary" />
          </button>

          {showNotifications && (
            <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] rounded-2xl shadow-xl p-4 z-50 animate-in fade-in slide-in-from-top-2 duration-150">
              <div className="flex items-center justify-between pb-2 border-b border-[#E9E7F2] dark:border-[#2D294F]">
                <h4 className="text-xs font-bold text-[#171638] dark:text-white uppercase tracking-wider">
                  Notifications
                </h4>
                <span className="text-[10px] text-primary font-semibold">Live System</span>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div className="p-2.5 rounded-xl bg-[#F8F7FD] dark:bg-[#211E3E] flex items-start gap-2.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-[#171638] dark:text-white">Gold Benchmark v2 Frozen</p>
                    <p className="text-[11px] text-[#85839A] mt-0.5">4,374 non-relevant pairs adjudicated automatically (97.6% human labor reduction).</p>
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-[#F8F7FD] dark:bg-[#211E3E] flex items-start gap-2.5">
                  <ShieldCheck className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-[#171638] dark:text-white">Neural Cross-Encoder Active</p>
                    <p className="text-[11px] text-[#85839A] mt-0.5">ms-marco-MiniLM-L-6-v2 joint reranker achieved 0.8713 MRR.</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Settings button */}
        <Link
          href="/settings"
          className="w-9 h-9 rounded-full bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-center text-[#85839A] hover:text-[#171638] dark:hover:text-white shadow-sm transition-colors"
          title="FinGuard settings"
        >
          <Settings className="w-4 h-4" />
        </Link>

        {/* Export Data button */}
        <Link
          href="/reports"
          className="hidden sm:inline-flex items-center gap-1.5 h-9 px-3.5 rounded-full bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] text-xs font-semibold text-[#171638] dark:text-white shadow-sm hover:border-primary/50 transition-colors"
        >
          <Download className="w-3.5 h-3.5 text-primary" />
          <span>Export data</span>
          <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-[#F4F3FA] dark:bg-[#28244D] text-[#85839A] font-mono">
            .json
          </span>
        </Link>

        {/* Primary Action Button */}
        <Link
          href="/verify"
          className="inline-flex items-center gap-1.5 h-9 px-4 rounded-full bg-[#171638] hover:bg-[#2A2855] text-white text-xs font-bold shadow-md transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          <Plus className="w-4 h-4" />
          <span>Verify Claim</span>
        </Link>
      </div>
    </header>
  );
}

