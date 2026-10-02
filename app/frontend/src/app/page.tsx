"use client";

import { useCallback, useEffect, useState } from "react";
import { API_URL } from "@/lib/api";
import { useAnalysis, useFeatured, useHealth } from "@/lib/hooks";
import { CandidateDetail } from "@/components/CandidateDetail";
import { MissionLog } from "@/components/MissionLog";
import { ResultsPanel } from "@/components/ResultsPanel";
import { StatusBar } from "@/components/StatusBar";
import { TargetPanel } from "@/components/TargetPanel";

export default function Page() {
  const { health, reachable } = useHealth();
  const featured = useFeatured(reachable);
  const { state, run } = useAnalysis();
  const [selected, setSelected] = useState(0);

  useEffect(() => setSelected(0), [state.result]);

  const busy = state.phase === "submitting" || state.phase === "running";
  const result = state.result?.status === "complete" ? state.result : null;
  const candidate = result?.candidates[selected];

  // Re-run the same star down the same path, skipping the cache.
  const rerun = useCallback(() => {
    if (state.starId === null) return;
    run(state.starId, {
      forceLive: state.result?.ephemeris_source === "live_bls_search",
      useCache: false,
    });
  }, [state.starId, state.result, run]);

  return (
    <div className="flex min-h-dvh flex-col">
      <StatusBar health={health} reachable={reachable} />

      {reachable === false && (
        <div role="alert" className="border-b border-stop/50 bg-stop/10 px-4 py-2.5 text-sm text-paper">
          <div className="mx-auto max-w-[1500px]">
            Cannot reach the SADA API at <span className="font-mono">{API_URL}</span>. Start the backend
            with <span className="font-mono">uvicorn app.main:app --reload</span> from{" "}
            <span className="font-mono">app/backend</span>. This page reconnects on its own.
          </div>
        </div>
      )}

      <main className="mx-auto grid w-full max-w-[1500px] flex-1 content-start gap-4 p-4 lg:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="flex min-w-0 flex-col gap-4">
          <TargetPanel
            busy={busy}
            activeStar={state.starId}
            featured={featured}
            onRun={run}
          />
          <MissionLog state={state} />
        </aside>

        <div className="flex min-w-0 flex-col gap-4">
          <ResultsPanel state={state} selected={selected} onSelect={setSelected} />
          {result && candidate && (
            <CandidateDetail
              key={`${result.star_id}-${selected}`}
              candidate={candidate}
              index={selected}
              result={result}
              onRerun={rerun}
            />
          )}
        </div>
      </main>
    </div>
  );
}
