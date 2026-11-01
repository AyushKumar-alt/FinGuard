import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/layout/ThemeProvider";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { AppHeader } from "@/components/layout/AppHeader";

export const metadata: Metadata = {
  title: "FinGuard | Financial Claim Verification & Retrieval Intelligence",
  description:
    "Evidence-grounded financial claim verification, multi-strategy retrieval benchmarking, and gold evidence benchmark governance platform evaluated on FinFact.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-[#F4F3FA] dark:bg-[#0E0D1F] text-[#171638] dark:text-[#F4F3FA] antialiased">
        <ThemeProvider>
          <div className="min-h-screen flex flex-row">
            {/* Left Nav Rail (Inspired by screenshot) */}
            <AppSidebar />

            {/* Main Content Area */}
            <div className="flex-1 min-w-0 flex flex-col pl-0 lg:pl-[104px] pr-0 lg:pr-3 py-0 lg:py-3 transition-all duration-300">
              <div className="flex-1 bg-[#F4F3FA] dark:bg-[#0E0D1F] rounded-none lg:rounded-[36px] flex flex-col min-h-screen lg:min-h-[calc(100vh-24px)] overflow-hidden">
                <AppHeader />
                <main className="flex-1 px-4 md:px-8 pb-10 pt-2 max-w-[1600px] w-full mx-auto">
                  {children}
                </main>
              </div>
            </div>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}

