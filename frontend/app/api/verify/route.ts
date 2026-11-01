import { NextRequest, NextResponse } from "next/server";
import { execFile } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";

const REPO_ROOT = path.resolve(process.cwd(), "..");

function runPythonVerify(payload: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const tempFile = path.join(os.tmpdir(), `finguard_verify_${Date.now()}_${Math.random().toString(36).substring(7)}.json`);
    try {
      fs.writeFileSync(tempFile, JSON.stringify(payload), "utf-8");
    } catch (err) {
      return reject(err);
    }

    const pythonCmd = process.platform === "win32" ? "python" : "python3";
    const args = ["-m", "src.api_bridge", "verify", "--file", tempFile];

    execFile(
      pythonCmd,
      args,
      {
        cwd: REPO_ROOT,
        timeout: 45000,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          HF_HUB_OFFLINE: "1",
          TRANSFORMERS_OFFLINE: "1",
        },
      },
      (error, stdout, stderr) => {
        // Clean up temp file
        try {
          if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
        } catch (_) {}

        if (error) {
          console.error("Python bridge execution error:", stderr || error.message);
          return reject(new Error(stderr || error.message));
        }

        try {
          // Find first '{' and last '}' in stdout
          const firstBrace = stdout.indexOf("{");
          const lastBrace = stdout.lastIndexOf("}");
          if (firstBrace !== -1 && lastBrace !== -1) {
            const jsonText = stdout.substring(firstBrace, lastBrace + 1);
            const parsed = JSON.parse(jsonText);
            return resolve(parsed);
          }
          reject(new Error("No JSON found in Python output"));
        } catch (parseErr) {
          reject(parseErr);
        }
      }
    );
  });
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const claim = (body.claim || "").trim();

    if (!claim) {
      return NextResponse.json(
        {
          claim: "",
          verdict: "NEI",
          verdict_display: "Not Enough Information",
          confidence: 0.0,
          is_conflicting: false,
          is_uncertain: true,
          uncertainty_reason: "Claim text was empty.",
          explanation: "Please enter a non-empty financial claim to verify.",
          supporting_evidence: [],
          contradicting_evidence: [],
          neutral_evidence: [],
        },
        { status: 400 }
      );
    }

    const payload = {
      claim,
      retrieval_mode: body.retrieval_mode || "bm25",
      use_query_expansion: body.use_query_expansion ?? true,
      use_reranker: body.use_reranker ?? true,
      use_nli: body.use_nli ?? true,
      top_k_retrieve: body.top_k_retrieve || 15,
      top_k_rerank: body.top_k_rerank || 4,
    };

    // 1. Try connecting to local Python HTTP API daemon if active
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const apiRes = await fetch("http://127.0.0.1:8000/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (apiRes.ok) {
        const data = await apiRes.json();
        return NextResponse.json(data);
      }
    } catch (_) {
      // Local daemon not running or timed out, fall back to child process
    }

    // 2. Invoke Python bridge CLI
    try {
      const startTime = Date.now();
      const result = await runPythonVerify(payload);
      result.latency_ms = Date.now() - startTime;
      return NextResponse.json(result);
    } catch (cliErr: any) {
      console.warn("Python bridge CLI direct execution error:", cliErr.message);
      return NextResponse.json(
        {
          error: "Verification service unavailable",
          details: cliErr.message,
          recovery: "Start the Python API daemon or verify Python dependencies and local model availability.",
        },
        { status: 503 }
      );
    }
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
