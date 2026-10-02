import clsx from "clsx";
import { TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import type { Decision } from "@/lib/api";

export function Panel({
  title,
  aside,
  children,
  className,
}: {
  title: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={clsx("border border-rule bg-deck", className)}>
      <header className="flex min-h-11 items-center justify-between gap-3 border-b border-rule px-4 py-2">
        <h2 className="font-display text-xl font-semibold leading-none text-paper">
          {title}
        </h2>
        {aside}
      </header>
      {children}
    </section>
  );
}

export type LampState = "ok" | "warn" | "off" | "idle";

const LAMP: Record<LampState, string> = {
  ok: "bg-go shadow-[0_0_8px_0_var(--color-go)]",
  warn: "bg-hold shadow-[0_0_8px_0_var(--color-hold)]",
  off: "bg-stop shadow-[0_0_8px_0_var(--color-stop)]",
  idle: "bg-faint",
};

export function Lamp({ state, pulse }: { state: LampState; pulse?: boolean }) {
  return (
    <span
      aria-hidden
      className={clsx(
        "inline-block size-2 shrink-0",
        LAMP[state],
        pulse && "animate-pulse motion-reduce:animate-none",
      )}
    />
  );
}

export const DECISION: Record<
  Decision,
  { label: string; text: string; box: string; lamp: LampState }
> = {
  accept: {
    label: "ACCEPT",
    text: "text-go",
    box: "border-go/40 bg-go/10 text-go",
    lamp: "ok",
  },
  escalate: {
    label: "REVIEW",
    text: "text-hold",
    box: "border-hold/40 bg-hold/10 text-hold",
    lamp: "warn",
  },
  reject: {
    label: "REJECT",
    text: "text-stop",
    box: "border-stop/40 bg-stop/10 text-stop",
    lamp: "off",
  },
};

export function DecisionBadge({ decision }: { decision: Decision }) {
  const d = DECISION[decision] ?? DECISION.escalate;
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 border px-2 py-0.5 font-mono text-[11px] font-medium tracking-wider",
        d.box,
      )}
    >
      <Lamp state={d.lamp} />
      {d.label}
    </span>
  );
}

export function ErrorNotice({
  title,
  message,
  detail,
}: {
  title: string;
  message: string;
  detail?: string | null;
}) {
  return (
    <div role="alert" className="m-4 border border-stop/50 bg-stop/10 p-4">
      <div className="flex items-start gap-3">
        <TriangleAlert className="mt-0.5 size-5 shrink-0 text-stop" aria-hidden />
        <div className="min-w-0">
          <p className="font-display text-lg font-semibold leading-tight text-paper">
            {title}
          </p>
          <p className="mt-1 text-[15px] text-paper/90">{message}</p>
          {detail && (
            <details className="mt-3 text-sm text-dim">
              <summary className="cursor-pointer select-none hover:text-paper">
                Server details
              </summary>
              <pre className="thin-scroll mt-2 max-h-56 overflow-auto bg-space p-3 font-mono text-xs leading-relaxed text-dim">
                {detail}
              </pre>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
