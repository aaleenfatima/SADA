"use client";

import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { formatElapsed } from "@/lib/api";
import type { AnalysisState } from "@/lib/hooks";
import { Lamp, Panel, type LampState } from "./ui";

const NODES = ["Fetch", "Search", "Classify", "Report"] as const;

// The agent logs "done" when classification ends, before the report is written.
const STAGE_NODE: Record<string, number> = {
  fetch: 0,
  search: 1,
  classify: 2,
  refine: 2,
  done: 2,
  report: 3,
};

function nodeState(i: number, reached: number, phase: AnalysisState["phase"]): LampState {
  if (phase === "done") return "ok";
  if (phase === "error") return i < reached ? "ok" : i === reached ? "off" : "idle";
  if (i < reached) return "ok";
  if (i === reached) return "warn";
  return "idle";
}

export function MissionLog({ state }: { state: AnalysisState }) {
  const { phase, job, startedAt, finishedAt, cached, starId } = state;
  const busy = phase === "submitting" || phase === "running";

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [busy]);

  const progress = job?.progress ?? [];
  const reached = progress.reduce((m, p) => Math.max(m, STAGE_NODE[p.stage] ?? -1), -1);
  const active = Math.max(reached, 0);

  const scrollRef = useRef<HTMLOListElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [progress.length]);

  const elapsed =
    startedAt === null ? null : ((finishedAt ?? now) - startedAt) / 1000;

  return (
    <Panel
      title="Analysis progress"
      aside={
        <span className="font-mono text-sm tabular-nums text-dim" aria-label="Elapsed time">
          T+{elapsed === null ? "--:--" : formatElapsed(elapsed)}
        </span>
      }
    >
      <ol className="grid grid-cols-4 gap-1 border-b border-rule px-4 py-3" aria-label="Pipeline stages">
        {NODES.map((label, i) => {
          const effective: LampState =
            phase === "idle" || cached
              ? "idle"
              : phase === "submitting"
                ? i === 0
                  ? "warn"
                  : "idle"
                : nodeState(i, active, phase);
          return (
            <li key={label} className="min-w-0">
              <div
                className={clsx(
                  "h-0.5",
                  effective === "ok" && "bg-go",
                  effective === "warn" && "bg-hold",
                  effective === "off" && "bg-stop",
                  effective === "idle" && "bg-rule",
                )}
              />
              <div className="mt-2 flex items-center gap-1.5 text-sm">
                <Lamp state={effective} pulse={busy && effective === "warn"} />
                <span className={effective === "idle" ? "text-dim" : "text-paper"}>{label}</span>
              </div>
            </li>
          );
        })}
      </ol>

      <ol
        ref={scrollRef}
        className="thin-scroll max-h-64 min-h-24 space-y-1.5 overflow-y-auto px-4 py-3 font-mono text-xs leading-relaxed"
        aria-live="polite"
      >
        {phase === "idle" && <li className="text-dim">Waiting for a target.</li>}
        {phase === "submitting" && <li className="text-dim">Sending request to the API.</li>}
        {cached && starId !== null && (
          <li className="text-paper">
            KIC {starId} was served from the result cache. No new analysis ran.
          </li>
        )}
        {busy && job?.status === "queued" && <li className="text-dim">Job queued, waiting to start.</li>}
        {progress.map((p, i) => (
          <li key={i} className="grid grid-cols-[3.25rem_4.5rem_1fr] gap-2">
            <span className="tabular-nums text-dim">
              {job ? formatElapsed(p.at - job.created_at) : "--:--"}
            </span>
            <span className="text-signal">{p.stage}</span>
            <span className="min-w-0 break-words text-paper">{p.message}</span>
          </li>
        ))}
      </ol>
    </Panel>
  );
}
