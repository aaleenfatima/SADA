"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  analyze,
  describeError,
  getFeatured,
  getHealth,
  getJob,
  type AnalysisResult,
  type Health,
  type Job,
  type RunOptions,
} from "./api";

/** Polls /healthz so the status bar reflects the real API and model state. */
export function useHealth(intervalMs = 15_000) {
  const [health, setHealth] = useState<Health | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const h = await getHealth();
        if (!alive) return;
        setHealth(h);
        setReachable(true);
      } catch {
        if (alive) setReachable(false);
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { health, reachable };
}

/** Loads precomputed demo stars once the API is reachable. null = not loaded yet. */
export function useFeatured(reachable: boolean | null) {
  const [stars, setStars] = useState<AnalysisResult[] | null>(null);

  useEffect(() => {
    if (reachable !== true || stars !== null) return;
    let alive = true;
    getFeatured()
      .then((s) => alive && setStars(s))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [reachable, stars]);

  return stars;
}

export type Phase = "idle" | "submitting" | "running" | "done" | "error";

export interface AnalysisState {
  phase: Phase;
  starId: number | null;
  job: Job | null;
  result: AnalysisResult | null;
  cached: boolean;
  error: string | null;
  errorDetail: string | null;
  startedAt: number | null;
  finishedAt: number | null;
}

const IDLE: AnalysisState = {
  phase: "idle",
  starId: null,
  job: null,
  result: null,
  cached: false,
  error: null,
  errorDetail: null,
  startedAt: null,
  finishedAt: null,
};

const POLL_MS = 1500;

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });
}

/**
 * Submits a star and follows the job to completion.
 * /analyze answers either with a cached result (no job) or a job id to poll.
 * Only job.status decides when to stop: the agent logs a "done" stage before
 * the report-writing stage, so stage names are not a reliable end signal.
 */
export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>(IDLE);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = useCallback(async (starId: number, opts: RunOptions) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setState({ ...IDLE, phase: "submitting", starId, startedAt: Date.now() });

    try {
      const res = await analyze(starId, opts, ac.signal);
      if (ac.signal.aborted) return;

      if (res.cached && res.result) {
        setState((s) => ({
          ...s,
          phase: "done",
          result: res.result,
          cached: true,
          finishedAt: Date.now(),
        }));
        return;
      }
      if (!res.job_id) {
        throw new Error("The API returned neither a result nor a job id.");
      }

      while (!ac.signal.aborted) {
        const job = await getJob(res.job_id, ac.signal);
        if (ac.signal.aborted) return;

        if (job.status === "complete") {
          setState((s) => ({
            ...s,
            phase: "done",
            job,
            result: job.result,
            finishedAt: Date.now(),
          }));
          return;
        }
        if (job.status === "failed") {
          setState((s) => ({
            ...s,
            phase: "error",
            job,
            error: job.error?.split("\n")[0] ?? "The analysis job failed.",
            errorDetail: job.error,
            finishedAt: Date.now(),
          }));
          return;
        }
        setState((s) => ({ ...s, phase: "running", job }));
        await sleep(POLL_MS, ac.signal);
      }
    } catch (e) {
      if (ac.signal.aborted) return;
      setState((s) => ({
        ...s,
        phase: "error",
        error: describeError(e),
        finishedAt: Date.now(),
      }));
    }
  }, []);

  return { state, run };
}
