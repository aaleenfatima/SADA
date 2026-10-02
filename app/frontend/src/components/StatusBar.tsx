"use client";

import { useEffect, useState } from "react";
import type { Health } from "@/lib/api";
import { Lamp, type LampState } from "./ui";

function Chip({ state, label, value }: { state: LampState; label: string; value: string }) {
  return (
    <li className="flex items-center gap-2 text-sm">
      <Lamp state={state} />
      <span className="text-dim">{label}</span>
      <span className="font-mono text-paper">{value}</span>
    </li>
  );
}

export function StatusBar({
  health,
  reachable,
}: {
  health: Health | null;
  reachable: boolean | null;
}) {
  // Clock renders only after mount so server and client HTML agree.
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const utc = now ? now.toISOString().slice(11, 19) : "--:--:--";

  const api: LampState = reachable === null ? "idle" : reachable ? "ok" : "off";
  const cnn: LampState = !health ? "idle" : health.models.cnn_loaded ? "ok" : "off";
  const rf: LampState = !health ? "idle" : health.models.rf_loaded ? "ok" : "warn";

  return (
    <header className="border-b border-rule bg-deck">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-x-8 gap-y-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <svg viewBox="0 0 32 32" className="size-8" aria-hidden>
            <circle cx="16" cy="16" r="11" fill="var(--color-nasa)" />
            <circle cx="16" cy="16" r="11" fill="none" stroke="var(--color-signal)" strokeWidth="1" />
            <circle cx="19.5" cy="16" r="4.6" fill="var(--color-space)" />
          </svg>
          <div className="leading-none">
            <p className="font-display text-[28px] font-bold tracking-wide text-paper">SADA</p>
            <p className="mt-1 text-[13px] text-dim">Stellar Anomaly Detection Architecture</p>
          </div>
        </div>

        <ul className="flex flex-wrap items-center gap-x-6 gap-y-2" aria-label="System status">
          <Chip
            state={api}
            label="API"
            value={reachable === null ? "connecting" : reachable ? "online" : "offline"}
          />
          <Chip
            state={cnn}
            label="CNN"
            value={!health ? "unknown" : health.models.cnn_loaded ? "loaded" : "missing"}
          />
          <Chip
            state={rf}
            label="Random Forest"
            value={!health ? "unknown" : health.models.rf_loaded ? "loaded" : "missing"}
          />
          <Chip
            state={!health ? "idle" : health.catalog_entries > 0 ? "ok" : "warn"}
            label="DR25 catalog"
            value={!health ? "unknown" : `${health.catalog_entries.toLocaleString("en-US")} stars`}
          />
          <li className="font-mono text-sm tabular-nums text-paper" suppressHydrationWarning>
            {utc} <span className="text-dim">UTC</span>
          </li>
        </ul>
      </div>
    </header>
  );
}
