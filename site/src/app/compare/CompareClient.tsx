"use client";

import { useEffect, useMemo, useState } from "react";

import { MetricChart } from "@/components/MetricCharts";
import { QWedgeHeatmap } from "@/components/QWedgeHeatmap";
import type { QValuesSnapshots, RunIndex, RunIndexEntry, RunSummary } from "@/lib/runs";
import { fetchRunIndex, fetchSummary } from "@/lib/runs";

function findEntry(idx: RunIndex, runId: string): RunIndexEntry | null {
  return idx.runs.find((r) => r.runId === runId) ?? null;
}

function latestVariant(idx: RunIndex, variant: RunIndexEntry["variant"]): RunIndexEntry | null {
  return idx.runs.find((r) => r.variant === variant) ?? null;
}

function formatVariant(variant: RunSummary["variant"]): string {
  return variant === "positive_only" ? "Positive-only" : "Traditional";
}

function valueText(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (value == null) return "not set";
  return JSON.stringify(value);
}

function sharedHyperparameters(sumA: RunSummary, sumB: RunSummary) {
  const a = sumA.hyperparameters ?? {};
  const b = sumB.hyperparameters ?? {};
  const keys = ["episodes", "seed", "alpha", "gamma", "epsilon", "epsilon_decay", "render"];

  return keys.map((key) => ({
    key,
    label:
      key === "epsilon_decay"
        ? "epsilon decay"
        : key === "alpha"
          ? "learning rate"
          : key,
    value: valueText(a[key]),
    matched: JSON.stringify(a[key]) === JSON.stringify(b[key])
  }));
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
  const [qA, setQA] = useState<QValuesSnapshots | null>(null);
  const [qB, setQB] = useState<QValuesSnapshots | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetchRunIndex()
      .then((v) => {
        if (!alive) return;
        setIdx(v);
        if (!initialA) setA(latestVariant(v, "positive_only")?.runId ?? "");
        if (!initialB) setB(latestVariant(v, "traditional")?.runId ?? "");
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, [initialA, initialB]);

  useEffect(() => {
    if (!idx) return;
    let alive = true;
    setError(null);
    setSumA(null);
    setSumB(null);
    setQA(null);
    setQB(null);

    const ea = a ? findEntry(idx, a) : null;
    const eb = b ? findEntry(idx, b) : null;

    async function loadSummaryAndQvalues(entry: RunIndexEntry | null) {
      if (!entry) return { summary: null, qvalues: null };
      const summary = await fetchSummary(entry.summaryPath);
      const qpath = summary.media?.qvalues_snapshots_json;
      if (!qpath) return { summary, qvalues: null };
      const res = await fetch(`/runs/${entry.runId}/${qpath}`, { cache: "no-store" });
      if (!res.ok) return { summary, qvalues: null };
      return { summary, qvalues: (await res.json()) as QValuesSnapshots };
    }

    Promise.all([loadSummaryAndQvalues(ea), loadSummaryAndQvalues(eb)])
      .then(([ra, rb]) => {
        if (!alive) return;
        setSumA(ra.summary);
        setSumB(rb.summary);
        setQA(ra.qvalues);
        setQB(rb.qvalues);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      });

    return () => {
      alive = false;
    };
  }, [idx, a, b]);

  const availableVariants = useMemo(() => {
    const runs = idx?.runs ?? [];
    return {
      positiveOnly: runs.some((r) => r.variant === "positive_only"),
      traditional: runs.some((r) => r.variant === "traditional")
    };
  }, [idx]);
  const ready = !!(sumA && sumB);
  const setup = ready ? sharedHyperparameters(sumA, sumB) : [];
  const setupMatched = setup.every((item) => item.matched);

  return (
    <div className="flow">
      <section className="section introBand">
        <div className="prose">
          <h1>Positive-only vs traditional</h1>
        </div>
        <p className="small">
          This viewer compares the current pair of exported runs: one positive-only agent and one
          traditional agent trained with the same setup.
        </p>

        {error ? (
          <p className="small" style={{ marginTop: 12 }}>
            <strong>Load error:</strong> {error}
          </p>
        ) : null}

        {ready ? (
          <div className="comparisonHeader">
            <div className="runBadge runBadge--positive">
              <span>{formatVariant(sumA.variant)}</span>
              <strong>{sumA.run_id}</strong>
            </div>
            <div className="runBadge runBadge--traditional">
              <span>{formatVariant(sumB.variant)}</span>
              <strong>{sumB.run_id}</strong>
            </div>
          </div>
        ) : (
          <div className="empty" style={{ marginTop: 14 }}>
            {availableVariants.positiveOnly || availableVariants.traditional
              ? "Add both a positive-only and a traditional run to render the comparison."
              : "No exported runs found yet."}
          </div>
        )}
      </section>

      {ready ? (
        <section className="section setupBand">
          <div className="sectionTitleRow">
            <div className="prose">
              <h2>Shared setup</h2>
            </div>
            <span className={`statusPill ${setupMatched ? "statusPill--ok" : "statusPill--warn"}`}>
              {setupMatched ? "matched hyperparameters" : "check hyperparameters"}
            </span>
          </div>

          <div className="setupGrid">
            <div className="setupItem">
              <span>environment</span>
              <strong>
                {sumA.xsize} x {sumA.ysize} cliff grid
              </strong>
            </div>
            {setup.map((item) => (
              <div className="setupItem" key={item.key}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                {!item.matched ? <em>differs between runs</em> : null}
              </div>
            ))}
          </div>

          <div className="variantNotes">
            <div>
              <span>Positive-only reward landscape</span>
              <strong>Progress is rewarded; cliff contact resets without a negative penalty.</strong>
            </div>
            <div>
              <span>Traditional reward landscape</span>
              <strong>Every step is penalized, and the cliff carries the large aversive penalty.</strong>
            </div>
          </div>
        </section>
      ) : null}

      {ready ? (
        <section className="section">
          <div className="prose">
            <h2>Learning curves</h2>
          </div>

          <div className="grid2">
            <div>
              <div className="seriesLabel seriesLabel--positive">
                {formatVariant(sumA.variant)}
              </div>
              <MetricChart title="Episode returns" values={sumA.metrics?.episode_returns} />
              <div style={{ height: 12 }} />
              <MetricChart title="Steps per episode" values={sumA.metrics?.steps_per_episode} />
            </div>

            <div>
              <div className="seriesLabel seriesLabel--traditional">
                {formatVariant(sumB.variant)}
              </div>
              <MetricChart title="Episode returns" values={sumB.metrics?.episode_returns} />
              <div style={{ height: 12 }} />
              <MetricChart title="Steps per episode" values={sumB.metrics?.steps_per_episode} />
            </div>
          </div>
        </section>
      ) : null}

      {ready ? (
        <section className="section">
          <div className="prose">
            <h2>Q-value landscapes</h2>
          </div>

          <div className="grid2">
            <div>
              <div className="seriesLabel seriesLabel--positive">{formatVariant(sumA.variant)}</div>
              {qA ? (
                <QWedgeHeatmap data={qA} title="Positive-only mental state" variant="positive_only" />
              ) : (
                <div className="empty">No per-action Q-value JSON found for positive-only.</div>
              )}
            </div>

            <div>
              <div className="seriesLabel seriesLabel--traditional">{formatVariant(sumB.variant)}</div>
              {qB ? (
                <QWedgeHeatmap data={qB} title="Traditional mental state" variant="traditional" />
              ) : (
                <div className="empty">No per-action Q-value JSON found for traditional.</div>
              )}
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
