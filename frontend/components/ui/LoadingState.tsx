import React from "react";
import { Loader2 } from "lucide-react";

export function LoadingState({ message = "Retrieving and verifying evidence..." }: { message?: string }) {
  return (
    <div className="w-full py-16 flex flex-col items-center justify-center text-center">
      <div className="w-14 h-14 rounded-2xl bg-[#EEECFC] dark:bg-[#2B2564] flex items-center justify-center text-primary mb-4 animate-pulse">
        <Loader2 className="w-7 h-7 animate-spin" />
      </div>
      <h3 className="text-base font-bold text-[#171638] dark:text-white">
        {message}
      </h3>
      <p className="text-xs text-[#85839A] dark:text-[#8E8BA6] mt-1 max-w-sm">
        Parsing financial metrics, scanning 12,941 candidate evidence sentences, and computing cross-encoder joint relevance.
      </p>
    </div>
  );
}

