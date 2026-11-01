import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const REPO_ROOT = path.resolve(process.cwd(), "..");
let cachedCorpus: any[] | null = null;

function loadCorpus(maxItems = 1000): any[] {
  if (cachedCorpus) return cachedCorpus;
  const jsonPath = path.join(REPO_ROOT, "finfact.json");
  if (!fs.existsSync(jsonPath)) return [];

  try {
    const raw = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
    const items: any[] = [];
    let docId = 0;

    for (let c = 0; c < raw.length && items.length < maxItems; c++) {
      const claimObj = raw[c];
      const evList = claimObj.evidence || [];
      for (const ev of evList) {
        if (!ev.sentence || ev.sentence.split(/\s+/).length < 4) continue;
        items.push({
          doc_id: docId,
          canonical_id: `ev_${String(docId).padStart(5, "0")}`,
          text: ev.sentence,
          source_urls: ev.hrefs || [],
          parent_claim_id: c,
          claim_text: claimObj.claim || "",
          label: claimObj.label || "",
          word_count: ev.sentence.split(/\s+/).length,
        });
        docId++;
        if (items.length >= maxItems) break;
      }
    }
    cachedCorpus = items;
    return items;
  } catch (e) {
    console.error("Error loading finfact.json corpus", e);
    return [];
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const q = (searchParams.get("q") || "").toLowerCase().trim();
    const limit = Math.min(parseInt(searchParams.get("limit") || "20", 10), 100);

    const corpus = loadCorpus(2000);
    if (!q) {
      return NextResponse.json(corpus.slice(0, limit));
    }

    const filtered = corpus.filter(
      (item) =>
        item.text.toLowerCase().includes(q) ||
        item.claim_text.toLowerCase().includes(q) ||
        item.canonical_id.toLowerCase().includes(q)
    );

    return NextResponse.json(filtered.slice(0, limit));
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

