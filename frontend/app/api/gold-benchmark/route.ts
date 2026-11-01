import { NextResponse } from "next/server";
import { getGoldBenchmarkOverview } from "@/lib/data-service";

export async function GET() {
  try {
    const data = getGoldBenchmarkOverview();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

