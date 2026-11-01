import Link from "next/link";
import { FileQuestion, ArrowLeft } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center text-center px-4">
      <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-[#7065E8]/10 text-[#7065E8] mb-6">
        <FileQuestion className="h-10 w-10" />
      </div>
      <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl mb-2">
        Page Not Found
      </h1>
      <p className="max-w-md text-muted-foreground text-sm mb-8">
        The research module or evidence report you are looking for does not exist or has been moved.
      </p>
      <Link
        href="/"
        className="inline-flex items-center gap-2 rounded-xl bg-[#7065E8] px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-[#5e53d9] transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Return to Overview Dashboard
      </Link>
    </div>
  );
}

