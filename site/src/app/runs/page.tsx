"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import type { RunIndex } from "@/lib/runs";
import { fetchRunIndex } from "@/lib/runs";

export default function RunsPage() {
  const [idx, setIdx] = useState<RunIndex>({
    schemaVersion: "1",
    generatedAt: new Date().toISOString(),
    runs: []
  });
  const [error, setError] = useState<string | null>(null);
  const [pickA, setPickA] = useState<string>("");
  const [pickB, setPickB] = useState<string>("");

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

  return (
    <div className="grid">
      <section className="card">
        <div className="prose">
          <h1>Runs</h1>
        </div>
        <p className="small">
          This page lists exported run bundles under <code>site/public/runs</code>. Generate bundles
          via the Python CLI, then sync them into the site for deployment.
        </p>
        <div className="kpi">
          <div className="pill">
            <span>Runs available</span>
            <strong>{idx.runs.length}</strong>
          </div>
          <div className="pill pill--accent">
            Tip: pick two runs → <Link href="/compare">compare</Link>
          </div>
        </div>
        {error ? (
          <p className="small" style={{ marginTop: 12 }}>
            <strong>Index load error:</strong> {error}
          </p>
        ) : null}
      </section>

      <section className="card">
        {idx.runs.length === 0 ? (
          <p className="small">
            No runs found. Add bundles under <code>site/public/runs</code> and ensure{" "}
            <code>index.json</code> exists.
          </p>
        ) : (
          <div className="grid">
            {idx.runs.map((r) => {
              const a = pickA === r.runId;
              const b = pickB === r.runId;
              return (
                <div key={r.runId} className="card card--hover">
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                    <div>
                      <div style={{ fontWeight: 750 }}>{r.runId}</div>
                      <div className="small">
                        <span className={`pill ${r.variant === "positive_only" ? "pill--accent" : ""}`}>
                          {r.variant}
                        </span>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <Link className="pill" href={`/runs/${encodeURIComponent(r.runId)}`}>
                        open
                      </Link>
                    </div>
                  </div>

                  <div className="kpi">
                    <label className="pill" style={{ cursor: "pointer", userSelect: "none" }}>
                      <input
                        type="radio"
                        name="compareA"
                        checked={a}
                        onChange={() => setPickA(r.runId)}
                        style={{ accentColor: "var(--accent)" }}
                      />
                      <span>A</span>
                    </label>
                    <label className="pill" style={{ cursor: "pointer", userSelect: "none" }}>
                      <input
                        type="radio"
                        name="compareB"
                        checked={b}
                        onChange={() => setPickB(r.runId)}
                        style={{ accentColor: "var(--accent2)" }}
                      />
                      <span>B</span>
                    </label>
                    <Link
                      className="pill pill--accent"
                      href={
                        pickA && pickB
                          ? `/compare?a=${encodeURIComponent(pickA)}&b=${encodeURIComponent(pickB)}`
                          : "/compare"
                      }
                      style={{ marginLeft: "auto" }}
                    >
                      Compare selected
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

