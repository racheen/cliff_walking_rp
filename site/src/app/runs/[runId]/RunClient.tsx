"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { MetricChart } from "@/components/MetricCharts";
import { QWedgeHeatmap } from "@/components/QWedgeHeatmap";
import type { QValuesSnapshots, RunIndex, RunIndexEntry, RunSummary } from "@/lib/runs";
import { fetchRunIndex, fetchSummary } from "@/lib/runs";

export function RunClient({ runId }: { runId: string }) {
  const [entry, setEntry] = useState<RunIndexEntry | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [qvalues, setQvalues] = useState<QValuesSnapshots | null>(null);
  const [error, setError] = useState<string | null>(null);

  const media = useMemo(() => summary?.media ?? {}, [summary]);
  const metrics = useMemo(() => summary?.metrics ?? {}, [summary]);

  useEffect(() => {
    let alive = true;
    fetchRunIndex()
      .then((v: RunIndex) => {
        if (!alive) return;
        const e = v.runs.find((r) => r.runId === runId) ?? null;
        setEntry(e);
        if (!e) return;
        return fetchSummary(e.summaryPath).then(async (s) => {
          if (!alive) return;
          setSummary(s);
          const qpath = s.media?.qvalues_snapshots_json;
          if (qpath) {
            try {
              const res = await fetch(`/runs/${e.runId}/${qpath}`, { cache: "no-store" });
              if (!res.ok) return;
              const q = (await res.json()) as QValuesSnapshots;
              if (!alive) return;
              setQvalues(q);
            } catch {
              // Optional artifact; ignore if missing
            }
          }
        });
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [runId]);

  if (error) {
    return (
      <section className="card">
        <h1>Failed to load run</h1>
        <p className="small">{error}</p>
        <p className="small">
          <Link href="/runs">Back to runs</Link>
        </p>
      </section>
    );
  }

  if (!entry) {
    return (
      <section className="card">
        <h1>Run not found</h1>
        <p className="small">
          Could not find <code>{runId}</code> in <code>/runs/index.json</code>.
        </p>
        <p className="small">
          <Link href="/runs">Back to runs</Link>
        </p>
      </section>
    );
  }

  if (!summary) {
    return (
      <section className="card">
        <h1>{runId}</h1>
        <p className="small">Loading…</p>
        <p className="small">
          <Link href="/runs">Back to runs</Link>
        </p>
      </section>
    );
  }

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <section className="card">
        <div className="prose">
          <h1>{runId}</h1>
        </div>
        <p className="small">
          <span className="pill pill--accent">{summary.variant}</span>
        </p>
        <p className="small">
          Grid: {summary.xsize} × {summary.ysize}
        </p>
        <p className="small">
          <Link href="/runs">Back to runs</Link>
        </p>
      </section>

      <section className="card">
        <div className="prose">
          <h2>Learning curves</h2>
        </div>
        <div className="grid2">
          <MetricChart
            title="Episode returns"
            subtitle="Interactive chart (from summary.json)"
            values={metrics.episode_returns}
          />
          <MetricChart
            title="Steps per episode"
            subtitle="Interactive chart (from summary.json)"
            values={metrics.steps_per_episode}
          />
        </div>

        {(media.returns_png || media.steps_png) && (
          <div style={{ marginTop: 16 }} className="small">
            PNGs are still available in the bundle for reference.
          </div>
        )}
      </section>

      <section className="card">
        <div className="prose">
          <h2>Q-value landscape</h2>
        </div>
        {qvalues ? (
          <QWedgeHeatmap data={qvalues} />
        ) : (
          <div className="empty">No per-action Q-value JSON found for this run yet.</div>
        )}
      </section>

      <section className="card">
        <div className="prose">
          <h2>Hyperparameters</h2>
        </div>
        <pre className="small" style={{ whiteSpace: "pre-wrap" }}>
          {JSON.stringify(summary.hyperparameters ?? {}, null, 2)}
        </pre>
      </section>
    </div>
  );
}
