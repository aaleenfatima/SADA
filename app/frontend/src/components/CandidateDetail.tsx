"use client";

import { motion, useReducedMotion } from "motion/react";
import type { AnalysisResult, Candidate, OddEven } from "@/lib/api";
import { FoldedCurve } from "./FoldedCurve";
import { DecisionBadge, Lamp, Panel } from "./ui";

function ModelGauge({
  label,
  value,
  bands,
  missing,
}: {
  label: string;
  value: number | null;
  bands?: boolean;
  missing?: string;
}) {
  const reduced = useReducedMotion();
  const pct = value === null ? 0 : Math.min(1, Math.max(0, value)) * 100;

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm text-dim">{label}</span>
        <span className="font-mono text-lg tabular-nums text-paper">
          {value === null ? "n/a" : value.toFixed(3)}
        </span>
      </div>
      {value === null ? (
        <p className="text-sm leading-relaxed text-dim">{missing}</p>
      ) : (
        <>
          <div className="relative h-2.5 bg-rule">
            {bands && (
              <>
                <span className="absolute inset-y-0 left-[40%] w-[40%] bg-rule-hi" />
                <span className="absolute inset-y-0 left-[80%] right-0 bg-nasa" />
              </>
            )}
            <span className="absolute -bottom-1 -top-1 left-1/2 w-px bg-dim/70" />
            <motion.span
              className="absolute -bottom-1.5 -top-1.5 w-0.5 -translate-x-1/2 bg-paper"
              initial={{ left: "0%" }}
              animate={{ left: `${pct}%` }}
              transition={reduced ? { duration: 0 } : { type: "spring", stiffness: 90, damping: 18 }}
            />
          </div>
          {bands && (
            <div className="relative mt-2 h-4 font-mono text-[10px] text-dim">
              <span className="absolute left-0">low</span>
              <span className="absolute left-[40%]">ambiguous</span>
              <span className="absolute left-[80%]">high</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function OddEvenBlock({ oe }: { oe: OddEven | null }) {
  if (!oe) return null;

  if (oe.odd_depth === null || oe.even_depth === null) {
    return (
      <p className="text-sm leading-relaxed text-dim">
        {oe.note ?? "Not enough odd and even transits to compare."}
      </p>
    );
  }

  const max = Math.max(oe.odd_depth, oe.even_depth, 1e-12);
  const rows = [
    { label: "Odd transits", depth: oe.odd_depth },
    { label: "Even transits", depth: oe.even_depth },
  ];

  return (
    <div>
      <div className="space-y-2">
        {rows.map((r) => (
          <div key={r.label}>
            <div className="mb-1 flex items-baseline justify-between text-sm">
              <span className="text-dim">{r.label}</span>
              <span className="font-mono tabular-nums text-paper">{(r.depth * 1e6).toFixed(1)} ppm</span>
            </div>
            <div className="h-1.5 bg-rule">
              <div
                className="h-full bg-signal"
                style={{ width: `${(Math.max(0, r.depth) / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 flex items-start gap-2 text-sm leading-relaxed">
        <span className="mt-1.5">
          <Lamp state={oe.is_suspicious ? "warn" : "ok"} />
        </span>
        <span className="text-paper">
          {oe.sigma === null ? "Difference not measurable" : `Depths differ by ${oe.sigma.toFixed(2)} σ`}.{" "}
          <span className="text-dim">
            {oe.is_suspicious
              ? "Above the 3 σ limit, which is how eclipsing binaries often look."
              : "Below the 3 σ limit used to flag eclipsing binaries."}
          </span>
        </span>
      </p>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-sm text-dim">{label}</dt>
      <dd className="font-mono text-[15px] tabular-nums text-paper">{value}</dd>
    </div>
  );
}

export function CandidateDetail({
  candidate: c,
  index,
  result,
  onRerun,
}: {
  candidate: Candidate;
  index: number;
  result: AnalysisResult;
  onRerun: () => void;
}) {
  const title = `Candidate ${index + 1} at ${c.period.toFixed(4)} days`;

  if (c.status !== "ok") {
    return (
      <Panel title={title} aside={<DecisionBadge decision={c.decision} />}>
        <div className="p-4">
          <p className="font-display text-lg font-semibold text-paper">This period could not be scored</p>
          <p className="mt-1 max-w-prose text-[15px] leading-relaxed text-paper/90">{c.reason}</p>
        </div>
      </Panel>
    );
  }

  const rfMissing =
    result.rf_unavailable_reason ?? "The Random Forest was not applied to this candidate.";

  return (
    <Panel title={title} aside={<DecisionBadge decision={c.decision} />}>
      <div className="grid lg:grid-cols-[minmax(0,1fr)_330px]">
        <div className="min-w-0 border-rule lg:border-r">
          <FoldedCurve
            curve={c.folded_curve}
            period={c.period}
            durationDays={c.duration_days}
            onRerun={onRerun}
          />
          <dl className="grid grid-cols-3 gap-4 border-t border-rule px-4 py-3">
            <Fact label="Period" value={`${c.period.toFixed(4)} d`} />
            <Fact label="Epoch" value={c.epoch.toFixed(3)} />
            <Fact
              label="Duration"
              value={typeof c.duration_days === "number" ? `${(c.duration_days * 24).toFixed(1)} h` : "n/a"}
            />
          </dl>
        </div>

        <div className="space-y-6 border-t border-rule p-4 lg:border-t-0">
          <section aria-labelledby={`models-${index}`} key={`models-${index}`} className="space-y-5">
            <h3 id={`models-${index}`} className="text-sm font-semibold text-paper">
              Model scores
            </h3>
            <ModelGauge label="CNN probability" value={c.cnn_probability} bands />
            <ModelGauge label="Random Forest probability" value={c.rf_probability} missing={rfMissing} />
            <p className="text-sm leading-relaxed text-dim">
              The tick at 0.5 is the cut-off used to decide whether the two models agree.
            </p>
            {c.models_agree !== null && c.models_agree !== undefined && (
              <p className="flex items-center gap-2 text-sm text-paper">
                <Lamp state={c.models_agree ? "ok" : "warn"} />
                {c.models_agree ? "The models agree" : "The models disagree"}
              </p>
            )}
          </section>

          <section aria-labelledby={`oe-${index}`}>
            <h3 id={`oe-${index}`} className="mb-3 text-sm font-semibold text-paper">
              Odd/even transit depth
            </h3>
            <OddEvenBlock oe={c.odd_even} />
          </section>

          {(c.refined || c.possible_alias_of) && (
            <section className="space-y-1.5 text-sm leading-relaxed text-paper">
              <h3 className="mb-1 font-semibold">Flags</h3>
              {c.refined && <p>The period was refined with a finer BLS grid.</p>}
              {c.possible_alias_of && (
                <p>
                  Possible alias: this period is a {c.possible_alias_of.ratio} ratio of the stronger
                  candidate at {c.possible_alias_of.period.toFixed(4)} d.
                </p>
              )}
            </section>
          )}
        </div>
      </div>

      <div className="grid gap-6 border-t border-rule p-4 lg:grid-cols-2">
        <section>
          <h3 className="mb-1.5 text-sm font-semibold text-paper">Why this decision</h3>
          <p className="max-w-prose text-[15px] leading-relaxed text-paper/90">{c.reason}</p>
          <p className="mt-2 text-sm text-dim">Set by fixed rules in the pipeline.</p>
        </section>
        <section>
          <h3 className="mb-1.5 text-sm font-semibold text-paper">Vetting summary</h3>
          {c.report ? (
            <>
              <p className="max-w-prose text-[15px] leading-relaxed text-paper/90">{c.report.text}</p>
              <p className="mt-2 text-sm text-dim">
                {c.report.source === "llm"
                  ? "Written by a language model from the measured values. Check it against the readouts."
                  : "Built from a template. No language model was used."}
              </p>
            </>
          ) : (
            <p className="text-sm text-dim">No summary was generated for this candidate.</p>
          )}
        </section>
      </div>
    </Panel>
  );
}
