"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { QValuesSnapshots } from "@/lib/runs";

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function mixColor(a: [number, number, number], b: [number, number, number], t: number) {
  const u = clamp01(t);
  const r = Math.round(a[0] + (b[0] - a[0]) * u);
  const g = Math.round(a[1] + (b[1] - a[1]) * u);
  const bl = Math.round(a[2] + (b[2] - a[2]) * u);
  return `rgb(${r},${g},${bl})`;
}

function colorForValue(v: number, stressMin: number, rewardMax: number) {
  if (!Number.isFinite(v)) {
    return "rgba(255, 255, 255, 0.77)";
  }

  const neutral: [number, number, number] = [70, 110, 179];
  const stress: [number, number, number] = [230, 73, 94];
  const learned: [number, number, number] = [69, 171, 103];
  const reward: [number, number, number] = [28, 136, 76];

  if (v < 0 && stressMin < 0) {
    return mixColor(neutral, stress, Math.abs(v / stressMin));
  }

  if (v > 0 && rewardMax > 0) {
    const t = clamp01(v / rewardMax);
    return t < 0.6
      ? mixColor(neutral, learned, t / 0.6)
      : mixColor(learned, reward, (t - 0.6) / 0.4);
  }

  return `rgb(${neutral[0]},${neutral[1]},${neutral[2]})`;
}

function fmt(v: number) {
  return Number.isFinite(v) ? v.toFixed(2) : "?";
}

export function QWedgeHeatmap({
  data,
  initialFrame = 0,
  title = "Q-table mental state",
  variant = "positive_only"
}: {
  data: QValuesSnapshots;
  initialFrame?: number;
  title?: string;
  variant?: "positive_only" | "traditional";
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [frame, setFrame] = useState<number>(Math.max(0, Math.min(initialFrame, data.qvalues.length - 1)));
  const [playing, setPlaying] = useState(false);
  const [canvasWidth, setCanvasWidth] = useState(720);

  const stressMin = Math.min(0, data.min);
  const rewardMax = Math.max(0, data.max);
  const showRewardMeter = variant === "positive_only" && rewardMax > 0;
  const xsize = data.xsize;
  const ysize = data.ysize;

  const label = useMemo(() => {
    const ep = data.frame_episodes?.[frame];
    return Number.isFinite(ep) ? `Episode ${ep}` : `Frame ${frame + 1}`;
  }, [data.frame_episodes, frame]);

  const frameStats = useMemo(() => {
    const values = data.qvalues[frame]?.flat(2) ?? [];
    if (values.length === 0) return { stress: 0, learned: 0, neutral: 0 };
    const stress = values.filter((v) => v < -0.0001).length;
    const learned = values.filter((v) => v > 0.0001).length;
    return {
      stress: Math.round((stress / values.length) * 100),
      learned: Math.round((learned / values.length) * 100),
      neutral: Math.round(((values.length - stress - learned) / values.length) * 100)
    };
  }, [data.qvalues, frame]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    const updateWidth = () => setCanvasWidth(Math.max(280, Math.floor(el.clientWidth)));
    updateWidth();

    const observer = new ResizeObserver(updateWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!playing || data.qvalues.length <= 1) return;
    const id = window.setInterval(() => {
      setFrame((f) => (f + 1) % data.qvalues.length);
    }, 220);
    return () => window.clearInterval(id);
  }, [data.qvalues.length, playing]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const pad = 8;
    const cell = Math.floor(Math.min(58, Math.max(20, (canvasWidth - pad * 2) / xsize)));
    const w = pad * 2 + xsize * cell;
    const h = pad * 2 + ysize * cell;

    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    c.width = Math.floor(w * dpr);
    c.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, w, h);

    const grid = data.qvalues[frame];
    if (!grid) return;

    // Q-table indices follow the environment action order: [right, up, left, down].
    for (let y = 0; y < ysize; y++) {
      for (let x = 0; x < xsize; x++) {
        const q = grid[y][x];
        const x0 = pad + x * cell;
        const y0 = pad + y * cell;
        const cx = x0 + cell / 2;
        const cy = y0 + cell / 2;

        // Up triangle
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0 + cell, y0);
        ctx.lineTo(cx, cy);
        ctx.closePath();
        ctx.fillStyle = colorForValue(q[1], stressMin, rewardMax);
        ctx.fill();

        // Right triangle
        ctx.beginPath();
        ctx.moveTo(x0 + cell, y0);
        ctx.lineTo(x0 + cell, y0 + cell);
        ctx.lineTo(cx, cy);
        ctx.closePath();
        ctx.fillStyle = colorForValue(q[0], stressMin, rewardMax);
        ctx.fill();

        // Down triangle
        ctx.beginPath();
        ctx.moveTo(x0, y0 + cell);
        ctx.lineTo(x0 + cell, y0 + cell);
        ctx.lineTo(cx, cy);
        ctx.closePath();
        ctx.fillStyle = colorForValue(q[3], stressMin, rewardMax);
        ctx.fill();

        // Left triangle
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0, y0 + cell);
        ctx.lineTo(cx, cy);
        ctx.closePath();
        ctx.fillStyle = colorForValue(q[2], stressMin, rewardMax);
        ctx.fill();

        // Cell outline
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, cell - 1, cell - 1);

        // Cross lines
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0 + cell, y0 + cell);
        ctx.moveTo(x0 + cell, y0);
        ctx.lineTo(x0, y0 + cell);
        ctx.strokeStyle = "rgba(18,35,58,0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, [canvasWidth, data, frame, rewardMax, stressMin, xsize, ysize]);

  return (
    <div className="qHeatmap">
      <div className="cardHeader">
        <div>
          <div className="cardTitle">{title}</div>
          <div className="cardSubtitle">
            {variant === "positive_only"
              ? "Positive Q-values show learned reward; neutral actions stay dark."
              : "Traditional Q-values mostly show penalty pressure; neutral actions stay dark."}
          </div>
        </div>
        <div className="episodeBadge">
          <span>Episode</span>
          <strong>{label.replace("Episode ", "")}</strong>
        </div>
      </div>

      {data.qvalues.length > 1 ? (
        <div className="qHeatmapControls">
          <button
            type="button"
            className="iconButton"
            onClick={() => setPlaying((v) => !v)}
            aria-label={playing ? "Pause training playback" : "Play training playback"}
            title={playing ? "Pause" : "Play"}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <input
            type="range"
            min={0}
            max={data.qvalues.length - 1}
            value={frame}
            onChange={(e) => {
              setPlaying(false);
              setFrame(Number(e.target.value));
            }}
            aria-label="Training frame"
          />
        </div>
      ) : null}

      <div className="qHeatmapCanvasWrap" ref={wrapRef}>
        <canvas ref={canvasRef} />
      </div>

      <div className={`qHeatmapLegend ${showRewardMeter ? "" : "qHeatmapLegend--stressOnly"}`}>
        <span>Stress {fmt(stressMin)}</span>
        <span>Neutral 0.00</span>
        {showRewardMeter ? <span>Reward {fmt(rewardMax)}</span> : null}
      </div>

      <div className="qHeatmapStats">
        <span>Stress-valued actions: {frameStats.stress}%</span>
        <span>Neutral actions: {frameStats.neutral}%</span>
        {showRewardMeter ? <span>Reward-valued actions: {frameStats.learned}%</span> : null}
      </div>
    </div>
  );
}
