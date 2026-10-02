"use client";

import clsx from "clsx";
import { motion } from "motion/react";
import type { AnalysisResult, Decision } from "@/lib/api";
import type { AnalysisState } from "@/lib/hooks";
import { DECISION, DecisionBadge, ErrorNotice, Lamp, Panel } from "./ui";

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="max-w-prose p-6 text-[15px] leading-relaxed text-dim">{children}</p>;
}

function count(result: AnalysisResult, d: Decision) {
  return result.candidates.filter((c) => c.decision === d).length;
}

function Meta({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div title={title}>
      <dt className="text-sm text-dim">{label}</dt>
      <dd className="text-[15px] text-paper">{value}</dd>
    </div>
  );
}

const TH = "px-3 py-2 text-left text-sm font-medium text-dim";
const TD = "px-3 py-2.5 font-mono text-sm tabular-nums text-paper";

export function ResultsPanel({
  state,
  selected,
  onSelect,
}: {
  state: AnalysisState;
  selected: number;
  onSelect: (i: number) => void;
}) {
  const { phase, result, starId, error, errorDetail, cached } = state;
  const busy = phase === "submitting" || phase === "running";

  let body: React.ReactNode;

  if (phase === "error") {
    body = <ErrorNotice title="The analysis failed" message={error ?? "Unknown error."} detail={errorDetail} />;
  } else if (result && result.status === "failed") {
    body = (
      <ErrorNotice
        title={`KIC ${result.star_id} could not be analyzed`}
        message={result.error ?? "The pipeline reported a failure without a message."}
      />
    );
  } else if (result) {
    body = (
      <>
        <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4 border-b border-rule px-4 py-4">
          <div>
            <p className="font-display text-4xl font-bold leading-none text-paper">KIC {result.star_id}</p>
            <p className="mt-1.5 text-sm text-dim">
              {result.candidates.length} candidate {result.candidates.length === 1 ? "signal" : "signals"}
              {cached ? ", loaded from cache" : ""}
            </p>
          </div>
          <dl className="grid grid-cols-2 gap-x-8 gap-y-2 sm:grid-cols-4">
            <Meta
              label="Periods from"
              value={result.ephemeris_source === "dr25_catalog" ? "DR25 catalog" : "Live BLS search"}
            />
            <Meta
              label="Random Forest"
              value={result.rf_available ? "Applied" : "Not applied"}
              title={result.rf_unavailable_reason ?? undefined}
            />
            <Meta
              label="Run time"
              value={typeof result.elapsed_seconds === "number" ? `${result.elapsed_seconds.toFixed(1)} s` : "n/a"}
            />
            <div>
              <dt className="text-sm text-dim">Outcome</dt>
              <dd className="flex flex-wrap gap-x-3 text-[15px] text-paper">
                {(["accept", "escalate", "reject"] as const).map((d) => (
                  <span key={d} className="inline-flex items-center gap-1.5">
                    <Lamp state={DECISION[d].lamp} />
                    <span className="font-mono tabular-nums">{count(result, d)}</span>
                    <span className="text-dim">{DECISION[d].label.toLowerCase()}</span>
                  </span>
                ))}
              </dd>
            </div>
          </dl>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse">
            <thead>
              <tr className="border-b border-rule">
                <th className={clsx(TH, "pl-4")}>Candidate</th>
                <th className={TH}>Period (d)</th>
                <th className={TH}>Duration (h)</th>
                <th className={TH}>CNN</th>
                <th className={TH}>Random Forest</th>
                <th className={TH}>CNN band</th>
                <th className={TH}>Decision</th>
              </tr>
            </thead>
            <tbody>
              {result.candidates.map((c, i) => {
                const isSel = i === selected;
                return (
                  <tr
                    key={i}
                    onClick={() => onSelect(i)}
                    className={clsx(
                      "cursor-pointer border-b border-rule/60 transition-colors last:border-b-0",
                      isSel ? "bg-deck-hi" : "hover:bg-deck-hi/60",
                    )}
                  >
                    <td className="relative py-2.5 pl-4 pr-3">
                      {isSel && (
                        <motion.span
                          layoutId="candidate-marker"
                          className="absolute inset-y-0 left-0 w-[3px] bg-signal"
                          transition={{ type: "spring", stiffness: 500, damping: 40 }}
                        />
                      )}
                      <button
                        type="button"
                        aria-pressed={isSel}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelect(i);
                        }}
                        className="font-display text-lg font-semibold text-paper"
                      >
                        {i + 1}
                      </button>
                    </td>
                    <td className={TD}>{c.period.toFixed(4)}</td>
                    <td className={TD}>
                      {typeof c.duration_days === "number" ? (c.duration_days * 24).toFixed(1) : "n/a"}
                    </td>
                    <td className={TD}>{c.cnn_probability === null ? "n/a" : c.cnn_probability.toFixed(3)}</td>
                    <td className={TD}>{c.rf_probability === null ? "n/a" : c.rf_probability.toFixed(3)}</td>
                    <td className={clsx(TD, "font-sans text-[15px]")}>{c.confidence_band ?? "n/a"}</td>
                    <td className="px-3 py-2.5">
                      <DecisionBadge decision={c.decision} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </>
    );
  } else if (busy) {
    body = (
      <Empty>
        Analyzing KIC {starId}. Candidate signals appear here when the job finishes. The progress panel
        shows each stage as it completes.
      </Empty>
    );
  } else {
    body = (
      <Empty>
        No star loaded. Enter a Kepler ID, or pick a featured star, to see its candidate signals and
        how each one was scored.
      </Empty>
    );
  }

  return <Panel title="Candidate signals">{body}</Panel>;
}
