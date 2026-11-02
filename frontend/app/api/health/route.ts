import { NextResponse } from "next/server";
import { getSystemStatus } from "@/lib/data-service";

const PYTHON_API_URL = (process.env.PYTHON_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

export async function GET() {
  try {
    let pythonHealthy = false;
    let pythonData: any = null;
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/health`, {
        cache: "no-store",
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        pythonHealthy = true;
        pythonData = await res.json();
      }
    } catch (_) {}

    const localStatus = getSystemStatus();
    return NextResponse.json({
      status: pythonHealthy ? "healthy" : "online",
      healthy: true,
      service: "FinGuard Phase 2 Verification Platform",
      backend_bridge: pythonHealthy ? "connected" : "standby",
      details: pythonData || localStatus,
    });
  } catch (error: any) {
    return NextResponse.json({ status: "error", error: error.message }, { status: 500 });
  }
}

