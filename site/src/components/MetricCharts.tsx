"use client";

import React, { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type Point = { x: number; y: number };

function toSeries(values: number[] | undefined): Point[] {
  if (!values || values.length === 0) return [];
  return values.map((y, i) => ({ x: i + 1, y }));
}

export function MetricChart({
  title,
  subtitle,
  values,
  height = 220
}: {
  title: string;
  subtitle?: string;
  values: number[] | undefined;
  height?: number;
}) {
  const data = useMemo(() => toSeries(values), [values]);

  return (
    <div className="card card--hover">
      <div className="cardHeader">
        <div>
          <div className="cardTitle">{title}</div>
          {subtitle ? <div className="cardSubtitle">{subtitle}</div> : null}
        </div>
      </div>

      {data.length === 0 ? (
        <div className="empty">No data available.</div>
      ) : (
        <div style={{ width: "100%", height }}>
          <ResponsiveContainer>
            <AreaChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="fillAccent" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.35} />
                  <stop offset="90%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
              <XAxis
                dataKey="x"
                tick={{ fill: "rgba(232,236,255,0.75)", fontSize: 12 }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tick={{ fill: "rgba(232,236,255,0.75)", fontSize: 12 }}
                tickLine={false}
                axisLine={false}
                width={36}
              />
              <Tooltip
                contentStyle={{
                  background: "rgba(10,14,28,0.92)",
                  border: "1px solid rgba(255,255,255,0.14)",
                  borderRadius: 12,
                  color: "var(--text)"
                }}
                labelStyle={{ color: "rgba(232,236,255,0.85)" }}
              />
              <Area
                type="monotone"
                dataKey="y"
                stroke="var(--accent)"
                strokeWidth={2}
                fill="url(#fillAccent)"
                dot={false}
                activeDot={{ r: 3 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

