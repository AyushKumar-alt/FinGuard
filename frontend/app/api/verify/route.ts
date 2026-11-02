import { NextRequest, NextResponse } from "next/server";

const PYTHON_API_URL = (process.env.PYTHON_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  const claim = typeof body === "object" && body !== null && "claim" in body
    ? (body as { claim?: unknown }).claim
    : undefined;
  if (typeof claim !== "string" || !claim.trim()) {
    return NextResponse.json({ error: "Enter a non-empty financial claim." }, { status: 400 });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180_000);
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/phase2/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: controller.signal,
    });
    const responseText = await response.text();
    let data: unknown;
    try {
      data = JSON.parse(responseText);
    } catch {
      return NextResponse.json(
        { error: "The Python verification service returned an invalid response." },
        { status: 502 }
      );
    }
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "AbortError";
    return NextResponse.json(
      {
        error: timedOut
          ? "Verification timed out while waiting for the Python inference service."
          : "Verification service unavailable. Start the Python API and try again.",
      },
      { status: timedOut ? 504 : 503 }
    );
  } finally {
    clearTimeout(timeout);
  }
}
