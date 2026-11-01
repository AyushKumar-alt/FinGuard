import { NextResponse } from "next/server";
import { getRetrievalMetrics } from "@/lib/data-service";

export async function GET() {
  try {
    const metrics = getRetrievalMetrics();
    return NextResponse.json(metrics);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

