import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const LOG_FILE = path.join(REPO_ROOT, "outputs", "human_reviews_logged.jsonl");

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { claim_id, evidence_id, assigned_grade, reviewer_notes } = body;

    if (!claim_id || !evidence_id || assigned_grade === undefined) {
      return NextResponse.json({ error: "Missing required review fields" }, { status: 400 });
    }

    const record = {
      timestamp: new Date().toISOString(),
      claim_id,
      evidence_id,
      assigned_grade,
      reviewer_notes: reviewer_notes || "",
    };

    // Append to local audit log file
    fs.appendFileSync(LOG_FILE, JSON.stringify(record) + "\n", "utf-8");

    return NextResponse.json({
      success: true,
      message: `Judgment recorded successfully for ${claim_id} - ${evidence_id}`,
      record,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

