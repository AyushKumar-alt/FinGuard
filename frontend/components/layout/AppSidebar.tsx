"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutGrid,
  ShieldCheck,
  Clock,
  Search,
  Cpu,
  BarChart3,
  Database,
  UserCheck,
  FileText,
  Settings,
  Menu,
  X,
  Sparkles,
} from "lucide-react";

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
  badge?: string;
  category?: "core" | "research";
}

const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Overview", icon: LayoutGrid, category: "core" },
  { href: "/verify", label: "Verify Claim", icon: ShieldCheck, badge: "Live", category: "core" },
  { href: "/investigations", label: "Investigations", icon: Clock, category: "core" },
  { href: "/evidence", label: "Evidence Explorer", icon: Search, category: "core" },
  { href: "/retrieval-lab", label: "Retrieval Lab", icon: Cpu, badge: "8 Configs", category: "research" },
  { href: "/evaluation", label: "Verification & Eval", icon: BarChart3, category: "research" },
  { href: "/gold-benchmark", label: "Gold Benchmark", icon: Database, badge: "v2.0", category: "research" },
  { href: "/review-queue", label: "Human Review Queue", icon: UserCheck, badge: "107", category: "research" },
  { href: "/reports", label: "Reports & Audits", icon: FileText, category: "research" },
  { href: "/settings", label: "Settings & Status", icon: Settings, category: "research" },
];

export function AppSidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Mobile Menu Toggle Button */}
      <div className="lg:hidden fixed top-4 left-4 z-50">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-2.5 rounded-2xl bg-white dark:bg-[#18162E] text-finguard-text dark:text-white shadow-md border border-finguard-border dark:border-[#2D294F] flex items-center justify-center"
          aria-label="Toggle navigation menu"
        >
          {mobileOpen ? <X className="w-5 h-5 text-primary" /> : <Menu className="w-5 h-5 text-primary" />}
        </button>
      </div>

      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-navy/40 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar Rail - Inspired directly by the SaaS dashboard screenshot */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-40 transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        } w-20 md:w-[72px] lg:w-20 my-0 lg:my-3 lg:ml-3 h-full lg:h-[calc(100vh-24px)] rounded-none lg:rounded-[28px] bg-[#5448C8] dark:bg-[#322A82] shadow-2xl flex flex-col items-center py-6 select-none`}
      >
        {/* Brand Logo / Icon */}
        <Link
          href="/"
          onClick={() => setMobileOpen(false)}
          className="mb-6 flex flex-col items-center group"
          title="FinGuard: Financial Fact-Checking"
        >
          <div className="w-12 h-12 rounded-2xl bg-white/15 backdrop-blur-md border border-white/20 flex items-center justify-center text-white shadow-sm group-hover:scale-105 transition-transform duration-200">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <span className="text-[10px] font-bold text-white/90 uppercase tracking-widest mt-1.5 opacity-80">
            FinGuard
          </span>
        </Link>

        {/* Navigation Rail Items */}
        <nav className="flex-1 w-full px-2 space-y-1.5 overflow-y-auto overflow-x-hidden no-scrollbar flex flex-col items-center">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

            return (
              <div key={item.href} className="relative group w-full flex justify-center">
                <Link
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={`relative w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-200 ${
                    isActive
                      ? "bg-white text-[#5448C8] shadow-md scale-100 font-semibold"
                      : "text-white/70 hover:text-white hover:bg-white/10"
                  }`}
                  aria-label={item.label}
                >
                  <Icon className={`w-5 h-5 ${isActive ? "text-[#5448C8]" : "text-white"}`} />

                  {/* Badge pill dot for queue or live */}
                  {item.badge && !isActive && (
                    <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-emerald-400 ring-2 ring-[#5448C8]" />
                  )}
                </Link>

                {/* Floating Tooltip Label */}
                <div className="hidden lg:group-hover:flex absolute left-full ml-3.5 top-1/2 -translate-y-1/2 z-50 pointer-events-none items-center">
                  <div className="bg-[#171638] text-white text-xs font-medium py-1.5 px-3 rounded-xl whitespace-nowrap shadow-xl border border-white/10 flex items-center gap-1.5">
                    <span>{item.label}</span>
                    {item.badge && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-primary text-white font-bold">
                        {item.badge}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </nav>

        {/* Bottom Status Indicator / CTA */}
        <div className="mt-auto pt-3 flex flex-col items-center w-full px-2">
          <Link
            href="/verify"
            onClick={() => setMobileOpen(false)}
            title="Start New Claim Verification"
            className="w-12 h-12 rounded-2xl bg-emerald-400/20 hover:bg-emerald-400/30 border border-emerald-300/40 text-emerald-300 flex items-center justify-center transition-all duration-200 hover:scale-105"
          >
            <Sparkles className="w-5 h-5" />
          </Link>
        </div>
      </aside>
    </>
  );
}

