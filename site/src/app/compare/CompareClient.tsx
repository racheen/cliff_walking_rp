"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { MetricChart } from "@/components/MetricCharts";
import type { RunIndex, RunIndexEntry, RunSummary } from "@/lib/runs";
import { fetchRunIndex, fetchSummary } from "@/lib/runs";

function findEntry(idx: RunIndex, runId: string): RunIndexEntry | null {
  return idx.runs.find((r) => r.runId === runId) ?? null;
}

export function CompareClient({
  initialA,
  initialB
}: {
  initialA: string | null;
  initialB: string | null;
}) {
  const [idx, setIdx] = useState<RunIndex | null>(null);
  const [a, setA] = useState<string>(initialA ?? "");
  const [b, setB] = useState<string>(initialB ?? "");
  const [sumA, setSumA] = useState<RunSummary | null>(null);
  const [sumB, setSumB] = useState<RunSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetchRunIndex()
      .then((v) => {
        if (!alive) return;
        setIdx(v);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!idx) return;
    let alive = true;
    setError(null);
    setSumA(null);
    setSumB(null);

    const ea = a ? findEntry(idx, a) : null;
    const eb = b ? findEntry(idx, b) : null;

    Promise.all([
      ea ? fetchSummary(ea.summaryPath) : Promise.resolve(null),
      eb ? fetchSummary(eb.summaryPath) : Promise.resolve(null)
    ])
      .then(([sa, sb]) => {
        if (!alive) return;
        setSumA(sa);
        setSumB(sb);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      });

    return () => {
      alive = false;
    };
  }, [idx, a, b]);

  const options = useMemo(() => idx?.runs ?? [], [idx]);
  const ready = !!(sumA && sumB);

  return (
    <div className="grid">
      <section className="card">
        <div className="prose">
          <h1>Compare runs</h1>
        </div>
        <p className="small">
          Pick two runs to compare learning curves side by side. This renders directly from{" "}
          <code>summary.json</code> (no matplotlib images required).
        </p>

        {error ? (
          <p className="small" style={{ marginTop: 12 }}>
            <strong>Load error:</strong> {error}
          </p>
        ) : null}

        <div className="grid2" style={{ marginTop: 10 }}>
          <div className="card">
            <div className="cardTitle">Run A</div>
            <div className="small" style={{ marginTop: 8 }}>
              <select
                value={a}
                onChange={(e) => setA(e.target.value)}
                style={{ width: "100%", padding: 10, borderRadius: 12 }}
              >
                <option value="">Select a run…</option>
                {options.map((r) => (
                  <option key={r.runId} value={r.runId}>
                    {r.runId} ({r.variant})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="card">
            <div className="cardTitle">Run B</div>
            <div className="small" style={{ marginTop: 8 }}>
              <select
                value={b}
                onChange={(e) => setB(e.target.value)}
                style={{ width: "100%", padding: 10, borderRadius: 12 }}
              >
                <option value="">Select a run…</option>
                {options.map((r) => (
                  <option key={r.runId} value={r.runId}>
                    {r.runId} ({r.variant})
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {ready ? (
          <div className="kpi">
            <Link className="pill" href={`/runs/${encodeURIComponent(sumA.run_id)}`}>
              Open A
            </Link>
            <Link className="pill" href={`/runs/${encodeURIComponent(sumB.run_id)}`}>
              Open B
            </Link>
          </div>
        ) : (
          <div className="empty" style={{ marginTop: 14 }}>
            Select two runs to render the comparison.
          </div>
        )}
      </section>

      {ready ? (
        <section className="card">
          <div className="prose">
            <h2>Learning curves</h2>
          </div>

          <div className="grid2">
            <div>
              <div className="pill pill--accent" style={{ marginBottom: 10 }}>
                A: {sumA.variant} • {sumA.run_id}
              </div>
              <MetricChart title="Episode returns" values={sumA.metrics?.episode_returns} />
              <div style={{ height: 12 }} />
              <MetricChart title="Steps per episode" values={sumA.metrics?.steps_per_episode} />
            </div>

            <div>
              <div
                className="pill"
                style={{
                  marginBottom: 10,
                  borderColor: "rgba(45,212,191,0.35)",
                  background: "rgba(45,212,191,0.10)",
                  color: "rgba(232,236,255,0.92)"
                }}
              >
                B: {sumB.variant} • {sumB.run_id}
              </div>
              <MetricChart title="Episode returns" values={sumB.metrics?.episode_returns} />
              <div style={{ height: 12 }} />
              <MetricChart title="Steps per episode" values={sumB.metrics?.steps_per_episode} />
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}

