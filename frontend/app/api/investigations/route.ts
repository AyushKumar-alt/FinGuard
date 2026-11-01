import { NextResponse } from "next/server";
import { getSampleInvestigations } from "@/lib/data-service";

export async function GET() {
  try {
    const list = getSampleInvestigations();
    return NextResponse.json(list);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

