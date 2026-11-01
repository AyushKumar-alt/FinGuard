import { NextResponse } from "next/server";
import { getSystemStatus } from "@/lib/data-service";

export async function GET() {
  try {
    const status = getSystemStatus();
    return NextResponse.json(status);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

