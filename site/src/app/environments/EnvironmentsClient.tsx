"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { MetricChart } from "@/components/MetricCharts";
import { QWedgeHeatmap } from "@/components/QWedgeHeatmap";
import type { QValuesSnapshots, RunIndex, RunIndexEntry, RunSummary } from "@/lib/runs";
import { fetchRunIndex, fetchSummary } from "@/lib/runs";

type LoadedRun = {
  entry: RunIndexEntry;
  summary: RunSummary;
  qvalues: QValuesSnapshots | null;
};

type EnvironmentGroup = {
  key: string;
  label: string;
  width: number;
  height: number;
  layout: string;
  traps: [number, number][];
  runs: LoadedRun[];
};

function formatVariant(variant: RunIndexEntry["variant"]): string {
  return variant === "positive_only" ? "Positive-only" : "Traditional";
}

function defaultCliffs(width: number, height: number): [number, number][] {
  return Array.from({ length: Math.max(0, width - 2) }, (_, i) => [i + 1, height - 1]);
}

function average(values: number[] | undefined, count = 25): string {
  if (!values?.length) return "n/a";
  const slice = values.slice(Math.max(0, values.length - count));
  const n = slice.reduce((sum, value) => sum + value, 0) / slice.length;
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function layoutName(summary: RunSummary): string {
  const raw = summary.hyperparameters?.cliff_layout;
  return typeof raw === "string" ? raw : "bottom";
}

function groupKey(summary: RunSummary): string {
  return `${summary.xsize}x${summary.ysize}:${layoutName(summary)}`;
}

function groupRuns(runs: LoadedRun[]): EnvironmentGroup[] {
  const groups = new Map<string, EnvironmentGroup>();

  for (const run of runs) {
    const { summary } = run;
    const key = groupKey(summary);
    const layout = layoutName(summary);
    const traps = summary.cliff_positions?.length
      ? summary.cliff_positions
      : defaultCliffs(summary.xsize, summary.ysize);

    const group =
      groups.get(key) ??
      ({
        key,
        label: `${summary.xsize} x ${summary.ysize} ${layout} traps`,
        width: summary.xsize,
        height: summary.ysize,
        layout,
        traps,
        runs: []
      } satisfies EnvironmentGroup);

    group.runs.push(run);
    groups.set(key, group);
  }

  return [...groups.values()].sort((a, b) => a.label.localeCompare(b.label));
}

function GridPreview({ group }: { group: EnvironmentGroup }) {
  const traps = new Set(group.traps.map(([x, y]) => `${x},${y}`));

  return (
    <div className="layoutPreview">
      <div className="cardHeader">
        <div>
          <div className="cardTitle">{group.label}</div>
          <div className="cardSubtitle">
            {group.runs.length} run{group.runs.length === 1 ? "" : "s"} exported
          </div>
        </div>
      </div>
      <div className="miniGrid" style={{ gridTemplateColumns: `repeat(${group.width}, 1fr)` }}>
        {Array.from({ length: group.width * group.height }, (_, i) => {
          const x = i % group.width;
          const y = Math.floor(i / group.width);
          const kind =
            x === 0 && y === group.height - 1
              ? "start"
              : x === group.width - 1 && y === group.height - 1
                ? "goal"
                : traps.has(`${x},${y}`)
                  ? "cliff"
                  : "open";
          return <span className={`miniCell miniCell--${kind}`} key={`${x}-${y}`} />;
        })}
      </div>
    </div>
  );
}

function latestRunForVariant(group: EnvironmentGroup | null, variant: RunIndexEntry["variant"]) {
  return [...(group?.runs ?? [])]
    .filter((run) => run.summary.variant === variant)
    .sort((a, b) => b.summary.run_id.localeCompare(a.summary.run_id))[0] ?? null;
}

export function EnvironmentsClient() {
  const [runs, setRuns] = useState<LoadedRun[]>([]);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;

    async function load() {
      const idx = await fetchRunIndex();
      const loaded = await Promise.all(
        idx.runs.map(async (entry) => {
          const summary = await fetchSummary(entry.summaryPath);
          const qpath = summary.media?.qvalues_snapshots_json;
          let qvalues: QValuesSnapshots | null = null;
          if (qpath) {
            const res = await fetch(`/runs/${entry.runId}/${qpath}`, { cache: "no-store" });
            if (res.ok) qvalues = (await res.json()) as QValuesSnapshots;
          }
          return { entry, summary, qvalues };
        })
      );

      if (!alive) return;
      setRuns(loaded);
    }

    load().catch((e: unknown) => {
      if (!alive) return;
      setError(e instanceof Error ? e.message : String(e));
    });

    return () => {
      alive = false;
    };
  }, []);

  const groups = useMemo(() => groupRuns(runs), [runs]);
  const selectedGroup = groups.find((group) => group.key === selectedKey) ?? groups[0] ?? null;
  const positiveRun = latestRunForVariant(selectedGroup, "positive_only");
  const traditionalRun = latestRunForVariant(selectedGroup, "traditional");
  const comparisonReady = !!(positiveRun && traditionalRun);

  useEffect(() => {
    if (!selectedGroup) return;
    setSelectedKey((current) => current || selectedGroup.key);
  }, [selectedGroup]);

  return (
    <div className="flow">
      <section className="section introBand">
        <div className="prose">
          <h1>Training environments</h1>
          <p>
            Export runs with different grid sizes and trap layouts, then come here to inspect the
            returns, steps, and learned Q-values for each environment.
          </p>
        </div>
        <div className="kpi">
          <span className="pill pill--accent">{groups.length} environment set{groups.length === 1 ? "" : "s"}</span>
          <span className="pill">{runs.length} exported run{runs.length === 1 ? "" : "s"}</span>
        </div>
        {error ? <p className="small">Load error: {error}</p> : null}
      </section>

      {groups.length > 0 ? (
        <>
          <section className="section setupBand">
            <div className="sectionTitleRow">
              <div className="prose">
                <h2>Choose an environment</h2>
              </div>
              <select
                className="selectControl"
                value={selectedGroup?.key ?? ""}
                onChange={(event) => {
                  setSelectedKey(event.target.value);
                }}
                aria-label="Environment"
              >
                {groups.map((group) => (
                  <option key={group.key} value={group.key}>
                    {group.label}
                  </option>
                ))}
              </select>
            </div>

            {selectedGroup ? <GridPreview group={selectedGroup} /> : null}
          </section>

          {selectedGroup ? (
            <section className="section">
              <div className="sectionTitleRow">
                <div className="prose">
                  <h2>Positive-only vs traditional</h2>
                </div>
                <span className={`statusPill ${comparisonReady ? "statusPill--ok" : "statusPill--warn"}`}>
                  {comparisonReady ? "both variants available" : "export both variants"}
                </span>
              </div>

              {comparisonReady ? (
                <>
                  <div className="comparisonHeader">
                    <div className="runBadge runBadge--positive">
                      <span>{formatVariant(positiveRun.summary.variant)}</span>
                      <strong>{positiveRun.summary.run_id}</strong>
                    </div>
                    <div className="runBadge runBadge--traditional">
                      <span>{formatVariant(traditionalRun.summary.variant)}</span>
                      <strong>{traditionalRun.summary.run_id}</strong>
                    </div>
                  </div>

                  <div className="grid2" style={{ marginTop: 16 }}>
                    <div>
                      <div className="seriesLabel seriesLabel--positive">
                        {formatVariant(positiveRun.summary.variant)}
                      </div>
                      <MetricChart title="Episode returns" values={positiveRun.summary.metrics?.episode_returns} />
                      <div style={{ height: 12 }} />
                      <MetricChart title="Steps per episode" values={positiveRun.summary.metrics?.steps_per_episode} />
                      <div className="reflectionStats">
                        <span>Recent return: {average(positiveRun.summary.metrics?.episode_returns)}</span>
                        <span>Recent steps: {average(positiveRun.summary.metrics?.steps_per_episode)}</span>
                      </div>
                    </div>

                    <div>
                      <div className="seriesLabel seriesLabel--traditional">
                        {formatVariant(traditionalRun.summary.variant)}
                      </div>
                      <MetricChart title="Episode returns" values={traditionalRun.summary.metrics?.episode_returns} />
                      <div style={{ height: 12 }} />
                      <MetricChart title="Steps per episode" values={traditionalRun.summary.metrics?.steps_per_episode} />
                      <div className="reflectionStats">
                        <span>Recent return: {average(traditionalRun.summary.metrics?.episode_returns)}</span>
                        <span>Recent steps: {average(traditionalRun.summary.metrics?.steps_per_episode)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="grid2" style={{ marginTop: 16 }}>
                    <div>
                      <div className="seriesLabel seriesLabel--positive">
                        {formatVariant(positiveRun.summary.variant)}
                      </div>
                      {positiveRun.qvalues ? (
                        <QWedgeHeatmap
                          data={positiveRun.qvalues}
                          title="Positive-only Q-values"
                          variant="positive_only"
                        />
                      ) : (
                        <div className="empty">No Q-value JSON found for positive-only.</div>
                      )}
                    </div>
                    <div>
                      <div className="seriesLabel seriesLabel--traditional">
                        {formatVariant(traditionalRun.summary.variant)}
                      </div>
                      {traditionalRun.qvalues ? (
                        <QWedgeHeatmap
                          data={traditionalRun.qvalues}
                          title="Traditional Q-values"
                          variant="traditional"
                        />
                      ) : (
                        <div className="empty">No Q-value JSON found for traditional.</div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div className="empty">
                  This environment needs one positive-only run and one traditional run. Use{" "}
                  <code>train-site</code> for this grid to export both.
                </div>
              )}
            </section>
          ) : null}
        </>
      ) : (
        <section className="section">
          <div className="empty">
            No exported runs found yet. Generate a run with the trainer, then refresh this page.
          </div>
        </section>
      )}

      <section className="section">
        <div className="prose">
          <h2>Generate another set</h2>
          <p>
            Example: <code>PYTHONPATH=src:. python3 -m cliff_walking_rp train-site --episodes 200 --width 12 --height 6 --trap-layout scattered</code>
          </p>
          <p>
            Supported layouts: <code>scattered</code>, <code>maze</code>, <code>islands</code>,{" "}
            <code>mixed</code>, <code>bottom</code>, <code>gap</code>, <code>middle</code>, and{" "}
            <code>double</code>.
          </p>
          <p>
            <Link href="/">Back to reflection</Link>
          </p>
        </div>
      </section>
    </div>
  );
}
