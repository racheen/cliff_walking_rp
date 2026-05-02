"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { MetricChart } from "@/components/MetricCharts";
import { QWedgeHeatmap } from "@/components/QWedgeHeatmap";
import type { QValuesSnapshots, RunIndex, RunIndexEntry, RunSummary } from "@/lib/runs";
import { fetchRunIndex, fetchSummary } from "@/lib/runs";

type Variant = RunIndexEntry["variant"];
type LayoutPreview = {
  name: string;
  size: [number, number];
  traps: [number, number][];
};

function formatVariant(variant: Variant): string {
  return variant === "positive_only" ? "Positive-only" : "Traditional";
}

function preferDefaultGrid(
  runs: { entry: RunIndexEntry; summary: RunSummary; qvalues: QValuesSnapshots | null }[],
  variant: Variant
) {
  const matches = runs.filter((run) => run.entry.variant === variant);
  return matches.find((run) => run.summary.xsize === 12 && run.summary.ysize === 4) ?? matches[0] ?? null;
}

function average(values: number[] | undefined, count = 25): string {
  if (!values?.length) return "n/a";
  const slice = values.slice(Math.max(0, values.length - count));
  const n = slice.reduce((sum, v) => sum + v, 0) / slice.length;
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function layoutPresets(): LayoutPreview[] {
  return [
    { name: "Scattered mines", size: [12, 6], traps: [[4, 2], [8, 2], [6, 3], [3, 4], [9, 4]] },
    { name: "Maze traps", size: [14, 7], traps: [[4, 1], [4, 2], [4, 4], [4, 5], [9, 0], [9, 2], [9, 3], [9, 4]] },
    {
      name: "Trap islands",
      size: [14, 6],
      traps: [[4, 3], [3, 3], [5, 3], [4, 2], [4, 4], [9, 3], [8, 3], [10, 3], [9, 2], [9, 4]]
    },
    { name: "Classic bottom trap line", size: [12, 4], traps: Array.from({ length: 10 }, (_, i) => [i + 1, 3]) }
  ];
}

function GridPreview({ layout }: { layout: LayoutPreview }) {
  const [width, height] = layout.size;
  const traps = new Set(layout.traps.map(([x, y]) => `${x},${y}`));

  return (
    <div className="layoutPreview">
      <div className="cardHeader">
        <div>
          <div className="cardTitle">{layout.name}</div>
          <div className="cardSubtitle">
            {width} x {height}, {layout.traps.length} trap cells
          </div>
        </div>
      </div>
      <div className="miniGrid" style={{ gridTemplateColumns: `repeat(${width}, 1fr)` }}>
        {Array.from({ length: width * height }, (_, i) => {
          const x = i % width;
          const y = Math.floor(i / width);
          const kind =
            x === 0 && y === height - 1
              ? "start"
              : x === width - 1 && y === height - 1
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

export function ReflectionClient() {
  const [positive, setPositive] = useState<RunSummary | null>(null);
  const [traditional, setTraditional] = useState<RunSummary | null>(null);
  const [positiveQ, setPositiveQ] = useState<QValuesSnapshots | null>(null);
  const [traditionalQ, setTraditionalQ] = useState<QValuesSnapshots | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;

    async function load() {
      const idx = await fetchRunIndex();

      const loaded = await Promise.all(
        idx.runs.map(async (entry) => {
          const summary = await fetchSummary(entry.summaryPath);
          const qpath = summary.media?.qvalues_snapshots_json;
          if (!qpath) return { entry, summary, qvalues: null };
          const res = await fetch(`/runs/${entry.runId}/${qpath}`, { cache: "no-store" });
          return {
            entry,
            summary,
            qvalues: res.ok ? ((await res.json()) as QValuesSnapshots) : null
          };
        })
      );

      const positiveRun = preferDefaultGrid(loaded, "positive_only");
      const traditionalRun = preferDefaultGrid(loaded, "traditional");

      if (!alive) return;
      setPositive(positiveRun?.summary ?? null);
      setPositiveQ(positiveRun?.qvalues ?? null);
      setTraditional(traditionalRun?.summary ?? null);
      setTraditionalQ(traditionalRun?.qvalues ?? null);
    }

    load().catch((e: unknown) => {
      if (!alive) return;
      setError(e instanceof Error ? e.message : String(e));
    });

    return () => {
      alive = false;
    };
  }, []);

  const ready = !!(positive && traditional);
  const presets = useMemo(layoutPresets, []);

  return (
    <div className="flow">
      <section className="section introBand">
        <div className="prose">
          <h1>Reflection: learning without punishment</h1>
          <p>
            This project asks whether an agent can learn useful behavior when traps stop being a
            punishment signal. The answer is more interesting than a simple yes: the
            agent still learns, but the shape of the learning changes.
          </p>
        </div>
        <div className="kpi">
          <Link className="pill pill--accent" href="/environments">
            Training environments
          </Link>
          {positive ? <span className="pill">{formatVariant(positive.variant)}: {positive.run_id}</span> : null}
          {traditional ? <span className="pill">{formatVariant(traditional.variant)}: {traditional.run_id}</span> : null}
          {ready ? (
            <span className="pill">
              Reflection grid: {positive.xsize} x {positive.ysize}
            </span>
          ) : null}
        </div>
        {error ? <p className="small">Load error: {error}</p> : null}
      </section>

      <section className="section">
        <div className="prose">
          <h2>The graphs are the reflection</h2>
          <p>
            In the traditional run, touching a trap produces large negative returns. In the
            positive-only run, the agent is rewarded for progress and reset after unsafe choices, so
            the graph reads as steadier practice instead of repeated punishment spikes.
          </p>
        </div>

        {ready ? (
          <div className="grid2">
            <div>
              <div className="seriesLabel seriesLabel--positive">{formatVariant(positive.variant)}</div>
              <MetricChart title="Episode returns" values={positive.metrics?.episode_returns} />
              <div className="reflectionStats">
                <span>Recent average return: {average(positive.metrics?.episode_returns)}</span>
                <span>Recent average steps: {average(positive.metrics?.steps_per_episode)}</span>
              </div>
            </div>
            <div>
              <div className="seriesLabel seriesLabel--traditional">{formatVariant(traditional.variant)}</div>
              <MetricChart title="Episode returns" values={traditional.metrics?.episode_returns} />
              <div className="reflectionStats">
                <span>Recent average return: {average(traditional.metrics?.episode_returns)}</span>
                <span>Recent average steps: {average(traditional.metrics?.steps_per_episode)}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="empty">Export one positive-only run and one traditional run to show the graphs here.</div>
        )}
      </section>

      {ready ? (
        <section className="section">
          <div className="prose">
            <h2>Behavior gets cleaner in different ways</h2>
            <p>
              Steps per episode show how quickly each agent settles into a route. Positive-only reward
              shaping makes progress legible without making each trap emotionally loud; the
              traditional setup teaches avoidance by making the bad state dominate the return.
            </p>
          </div>

          <div className="grid2">
            <MetricChart title="Positive-only steps per episode" values={positive.metrics?.steps_per_episode} />
            <MetricChart title="Traditional steps per episode" values={traditional.metrics?.steps_per_episode} />
          </div>
        </section>
      ) : null}

      {ready ? (
        <section className="section">
          <div className="prose">
            <h2>What the agents believe</h2>
            <p>
              The Q-value grids show the learned action values inside each state. Yellow and teal cells
              mark rewarded paths; pink cells mark penalty pressure. The contrast makes the training
              philosophy visible instead of hiding it in a table.
            </p>
          </div>

          <div className="grid2">
            <div>
              <div className="seriesLabel seriesLabel--positive">{formatVariant(positive.variant)}</div>
              {positiveQ ? (
                <QWedgeHeatmap data={positiveQ} title="Positive-only Q-values" variant="positive_only" />
              ) : (
                <div className="empty">No Q-value graph found for positive-only.</div>
              )}
            </div>
            <div>
              <div className="seriesLabel seriesLabel--traditional">{formatVariant(traditional.variant)}</div>
              {traditionalQ ? (
                <QWedgeHeatmap data={traditionalQ} title="Traditional Q-values" variant="traditional" />
              ) : (
                <div className="empty">No Q-value graph found for traditional.</div>
              )}
            </div>
          </div>
        </section>
      ) : null}

      <section className="section setupBand">
        <div className="prose">
          <h2>More grids, more traps</h2>
          <p>
            The trainer now supports different board sizes and trap layouts, so the hazards can sit
            anywhere in the grid instead of only along the bottom. Run the exporter with{" "}
            <code>--width</code>, <code>--height</code>, and <code>--trap-layout</code> to generate
            new graph sets.
          </p>
        </div>
        <div className="layoutGrid">
          {presets.map((preset) => (
            <GridPreview layout={preset} key={preset.name} />
          ))}
        </div>
      </section>
    </div>
  );
}
