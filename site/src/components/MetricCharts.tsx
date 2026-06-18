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
              <CartesianGrid stroke="rgba(67,109,82,0.10)" strokeDasharray="3 3" />
              <XAxis
                dataKey="x"
                tick={{ fill: "rgba(29,53,38,0.58)", fontSize: 12 }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tick={{ fill: "rgba(29,53,38,0.58)", fontSize: 12 }}
                tickLine={false}
                axisLine={false}
                width={36}
              />
              <Tooltip
                contentStyle={{
                  background: "rgba(255,255,255,0.97)",
                  border: "1px solid rgba(67,109,82,0.18)",
                  borderRadius: 12,
                  color: "var(--text)",
                  boxShadow: "0 18px 40px rgba(35,72,49,0.12)"
                }}
                labelStyle={{ color: "rgba(29,53,38,0.78)" }}
              />
              <Area
                type="monotone"
                dataKey="y"
                stroke="var(--accent)"
                strokeWidth={2.5}
                fill="url(#fillAccent)"
                dot={false}
                activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
