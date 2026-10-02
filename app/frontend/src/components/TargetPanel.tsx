"use client";

import clsx from "clsx";
import { Check, LoaderCircle } from "lucide-react";
import { useState, type FormEvent } from "react";
import type { AnalysisResult, RunOptions } from "@/lib/api";
import { Panel } from "./ui";

function Toggle({
  checked,
  disabled,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint: string;
}) {
  return (
    <label
      className={clsx(
        "flex items-start gap-2.5 text-sm",
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer",
      )}
    >
      <input
        type="checkbox"
        className="peer sr-only"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="mt-0.5 grid size-4 shrink-0 place-items-center border border-rule-hi bg-space text-signal peer-checked:border-signal peer-checked:bg-signal/15 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-signal">
        {checked && <Check className="size-3" strokeWidth={3} aria-hidden />}
      </span>
      <span>
        <span className="block text-paper">{label}</span>
        <span className="block text-dim">{hint}</span>
      </span>
    </label>
  );
}

export function TargetPanel({
  busy,
  activeStar,
  featured,
  onRun,
}: {
  busy: boolean;
  activeStar: number | null;
  featured: AnalysisResult[] | null;
  onRun: (starId: number, opts: RunOptions) => void;
}) {
  const [value, setValue] = useState("");
  const [live, setLive] = useState(false);
  const [fresh, setFresh] = useState(false);

  const trimmed = value.trim();
  const parsed = /^\d{1,9}$/.test(trimmed) ? Number(trimmed) : null;
  const invalid = trimmed !== "" && parsed === null;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (parsed !== null && !busy) onRun(parsed, { forceLive: live, useCache: !fresh });
  };

  return (
    <Panel title="Target">
      <form onSubmit={submit} className="space-y-4 p-4">
        <div>
          <label htmlFor="kic" className="mb-1.5 block text-sm text-dim">
            Kepler Input Catalog ID
          </label>
          <div className="flex">
            <span className="grid place-items-center border border-r-0 border-rule bg-space px-3 font-mono text-sm text-dim">
              KIC
            </span>
            <input
              id="kic"
              inputMode="numeric"
              autoComplete="off"
              spellCheck={false}
              placeholder="11442793"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              aria-invalid={invalid}
              aria-describedby={invalid ? "kic-hint" : undefined}
              className="min-w-0 flex-1 border border-rule bg-space px-3 py-2 font-mono text-base tabular-nums text-paper placeholder:text-faint focus-visible:border-signal focus-visible:outline-none"
            />
          </div>
          {invalid && (
            <p id="kic-hint" className="mt-1.5 text-sm text-hold">
              Kepler IDs are whole numbers, for example 11442793.
            </p>
          )}
        </div>

        <div className="space-y-3">
          <Toggle
            checked={live}
            onChange={setLive}
            label="Search for periods live"
            hint="Skip the DR25 catalog and run BLS on the light curve. Slower, and the Random Forest is not applied."
          />
          <Toggle
            checked={fresh || live}
            disabled={live}
            onChange={setFresh}
            label="Ignore cached result"
            hint="Run the full analysis again even if this star was analyzed before."
          />
        </div>

        <button
          type="submit"
          disabled={parsed === null || busy}
          className="flex w-full items-center justify-center gap-2 bg-nasa px-4 py-2.5 font-display text-lg font-semibold tracking-wide text-white transition-colors hover:bg-nasa-hi disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-nasa"
        >
          {busy && <LoaderCircle className="size-4 animate-spin" aria-hidden />}
          {busy ? "Analyzing" : "Analyze star"}
        </button>
      </form>

      {featured !== null && (
        <div className="border-t border-rule p-4">
          <h3 className="mb-2 text-sm font-semibold text-paper">Featured stars</h3>
          {featured.length === 0 ? (
            <p className="text-sm text-dim">
              None cached yet. Run <code className="font-mono text-paper">python -m scripts.precache</code>{" "}
              in the backend to add some.
            </p>
          ) : (
            <ul className="grid gap-1.5">
              {featured.map((s) => (
                <li key={s.star_id}>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => {
                      setValue(String(s.star_id));
                      onRun(s.star_id, { forceLive: false, useCache: true });
                    }}
                    className={clsx(
                      "flex w-full items-center justify-between border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                      activeStar === s.star_id
                        ? "border-signal/60 bg-deck-hi"
                        : "border-rule hover:border-rule-hi hover:bg-deck-hi",
                    )}
                  >
                    <span className="font-mono text-sm tabular-nums text-paper">KIC {s.star_id}</span>
                    <span className="text-sm text-dim">
                      {s.candidates.length} {s.candidates.length === 1 ? "candidate" : "candidates"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Panel>
  );
}
