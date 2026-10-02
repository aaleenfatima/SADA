"use client";

import clsx from "clsx";
import { RotateCcw } from "lucide-react";
import { useReducedMotion } from "motion/react";
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Point {
  phase: number;
  flux: number;
}

export function FoldedCurve({
  curve,
  period,
  durationDays,
  onRerun,
}: {
  curve?: number[];
  period: number;
  durationDays?: number | null;
  onRerun: () => void;
}) {
  const reduced = useReducedMotion();
  const [zoom, setZoom] = useState(false);

  const half = period / 2;
  const canZoom = typeof durationDays === "number" && durationDays > 0;
  const window = canZoom ? Math.min(half, durationDays * 3) : half;
  const zoomed = zoom && canZoom;

  // Bins are spread evenly over one orbit, centred on the transit epoch.
  const data = useMemo<Point[]>(() => {
    if (!curve?.length) return [];
    const all = curve.map((flux, i) => ({
      phase: -half + ((i + 0.5) * period) / curve.length,
      flux,
    }));
    return zoomed ? all.filter((d) => Math.abs(d.phase) <= window) : all;
  }, [curve, half, period, window, zoomed]);

  if (!curve?.length) {
    return (
      <div className="grid min-h-80 place-items-center p-6 text-center">
        <div className="max-w-sm">
          <p className="font-display text-lg font-semibold text-paper">No folded curve in this result</p>
          <p className="mt-1 text-sm text-dim">
            This result was cached before the API started storing the curve. Run the analysis again
            to get it.
          </p>
          <button
            type="button"
            onClick={onRerun}
            className="mt-4 inline-flex items-center gap-2 border border-rule-hi px-3 py-2 text-sm text-paper transition-colors hover:border-signal hover:bg-deck-hi"
          >
            <RotateCcw className="size-4" aria-hidden />
            Run again without cache
          </button>
        </div>
      </div>
    );
  }

  const bound = zoomed ? window : half;

  return (
    <div className="p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-paper">Light curve folded at the candidate period</h3>
        <div role="group" aria-label="Chart range" className="flex border border-rule text-sm">
          {[
            { label: "Full orbit", value: false },
            { label: "Zoom on transit", value: true },
          ].map((o) => (
            <button
              key={o.label}
              type="button"
              aria-pressed={zoomed === o.value}
              disabled={o.value && !canZoom}
              onClick={() => setZoom(o.value)}
              className={clsx(
                "px-3 py-1 transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                zoomed === o.value ? "bg-nasa text-white" : "text-dim hover:text-paper",
              )}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      <div
        role="img"
        aria-label={`Folded light curve, period ${period.toFixed(4)} days, ${curve.length} bins`}
        className="h-80 w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
            <CartesianGrid stroke="var(--color-rule)" strokeDasharray="2 5" />
            <XAxis
              dataKey="phase"
              type="number"
              domain={[-bound, bound]}
              allowDataOverflow
              ticks={[-bound, (-2 * bound) / 3, -bound / 3, 0, bound / 3, (2 * bound) / 3, bound]}
              tickFormatter={(v: number) => v.toFixed(zoomed ? 2 : 1)}
              stroke="var(--color-rule-hi)"
              tick={{ fill: "var(--color-dim)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              label={{
                value: "Days from transit centre",
                position: "insideBottom",
                offset: -10,
                fill: "var(--color-dim)",
                fontSize: 12,
              }}
            />
            <YAxis
              width={46}
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => v.toFixed(1)}
              stroke="var(--color-rule-hi)"
              tick={{ fill: "var(--color-dim)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              label={{
                value: "Flux (z-score)",
                angle: -90,
                position: "insideLeft",
                offset: 8,
                fill: "var(--color-dim)",
                fontSize: 12,
                style: { textAnchor: "middle" },
              }}
            />
            {canZoom && (
              <ReferenceArea
                x1={-durationDays / 2}
                x2={durationDays / 2}
                fill="var(--color-stop)"
                fillOpacity={0.1}
                stroke="var(--color-stop)"
                strokeOpacity={0.45}
                strokeDasharray="3 3"
              />
            )}
            <ReferenceLine x={0} stroke="var(--color-rule-hi)" />
            <Tooltip
              cursor={{ stroke: "var(--color-rule-hi)" }}
              content={({ active, payload }) => {
                const p = payload?.[0]?.payload as Point | undefined;
                if (!active || !p) return null;
                return (
                  <div className="border border-rule-hi bg-space px-2.5 py-1.5 font-mono text-xs text-paper">
                    <div>
                      {p.phase >= 0 ? "+" : ""}
                      {p.phase.toFixed(3)} d
                    </div>
                    <div className="text-dim">z = {p.flux.toFixed(3)}</div>
                  </div>
                );
              }}
            />
            <Line
              type="linear"
              dataKey="flux"
              stroke="var(--color-signal)"
              strokeWidth={1.75}
              dot={zoomed ? { r: 2.5, fill: "var(--color-signal)", strokeWidth: 0 } : false}
              activeDot={{ r: 4, fill: "var(--color-paper)", strokeWidth: 0 }}
              isAnimationActive={!reduced}
              animationDuration={700}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 max-w-prose text-sm leading-relaxed text-dim">
        This is the normalized 200-bin curve the CNN scored, so flux is in z-score units, not
        physical depth.
        {canZoom && (
          <>
            {" "}
            The red band marks the transit window ({(durationDays * 24).toFixed(1)} h).
          </>
        )}
      </p>
    </div>
  );
}
