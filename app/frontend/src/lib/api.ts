export const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

/* ---- Types: mirror the FastAPI responses in app/backend ------------------ */

export type Decision = "accept" | "reject" | "escalate";

export interface OddEven {
  odd_depth: number | null;
  even_depth: number | null;
  sigma: number | null;
  is_suspicious: boolean;
  note: string | null;
}

export interface Report {
  text: string;
  source: "llm" | "template" | string;
}

export interface Alias {
  ratio: string;
  period: number;
}

export interface Candidate {
  period: number;
  epoch: number;
  duration_days?: number | null;
  status: "ok" | "preprocessing_failed" | string;
  decision: Decision;
  reason: string;
  cnn_probability: number | null;
  rf_probability: number | null;
  confidence_band: "high" | "ambiguous" | "low" | null;
  odd_even: OddEven | null;
  refined: boolean;
  possible_alias_of?: Alias | null;
  models_agree?: boolean | null;
  report?: Report;
  /** 200-bin z-scored folded curve; absent on results cached before it was added. */
  folded_curve?: number[];
}

export interface AnalysisResult {
  star_id: number;
  status: "complete" | "failed";
  error?: string;
  ephemeris_source?: "dr25_catalog" | "live_bls_search";
  rf_available?: boolean;
  rf_unavailable_reason?: string | null;
  elapsed_seconds?: number;
  candidates: Candidate[];
}

export interface ProgressEntry {
  stage: string;
  message: string;
  at: number;
}

export interface Job {
  job_id: string;
  star_id: number;
  status: "queued" | "running" | "complete" | "failed";
  stage: string | null;
  progress: ProgressEntry[];
  result: AnalysisResult | null;
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface Health {
  ready: boolean;
  models: { cnn_loaded: boolean; rf_loaded: boolean; [key: string]: unknown };
  catalog_entries: number;
  note: string | null;
}

export interface AnalyzeResponse {
  cached: boolean;
  job_id: string | null;
  result: AnalysisResult | null;
}

export interface RunOptions {
  forceLive: boolean;
  useCache: boolean;
}

/* ---- Requests ------------------------------------------------------------ */

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* body was not JSON; keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const getHealth = () => request<Health>("/healthz");

export const getFeatured = () =>
  request<{ stars: AnalysisResult[] }>("/featured").then((r) => r.stars);

export const getJob = (jobId: string, signal?: AbortSignal) =>
  request<Job>(`/jobs/${jobId}`, { signal });

export const analyze = (starId: number, opts: RunOptions, signal?: AbortSignal) =>
  request<AnalyzeResponse>("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      star_id: starId,
      force_live_search: opts.forceLive,
      use_cache: opts.useCache,
    }),
    signal,
  });

export function describeError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof TypeError) {
    return `Cannot reach the SADA API at ${API_URL}. Check that the backend is running.`;
  }
  return e instanceof Error ? e.message : "Something went wrong.";
}

/* ---- Formatting ---------------------------------------------------------- */

/** Seconds -> "mm:ss" for mission-elapsed-time style readouts. */
export function formatElapsed(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}
