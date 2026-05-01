import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { RunClient } from "./RunClient";

export async function generateStaticParams(): Promise<Array<{ runId: string }>> {
  const indexPath = join(process.cwd(), "public", "runs", "index.json");
  const raw = await readFile(indexPath, "utf-8");
  const idx = JSON.parse(raw) as { runs?: Array<{ runId: string }> };
  const runs = idx.runs ?? [];
  return runs.map((r) => ({ runId: r.runId }));
}

export default async function RunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  return <RunClient runId={runId} />;
}

