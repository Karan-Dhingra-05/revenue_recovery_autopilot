/**
 * Phase 0 landing page.
 *
 * Minimal, professional status screen that:
 *   - Shows the project name and tagline
 *   - Live-polls GET /api/health and displays service status
 *
 * No dashboard data yet — that is Phase 1+.
 */

"use client";

import { useEffect, useState } from "react";
import { fetchHealth, type HealthStatus } from "@/lib/api";

// ─── Status indicator ─────────────────────────────────────────────────────────

function StatusBadge({ value }: { value: string }) {
  const ok = value === "ok";
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
        ok
          ? "bg-emerald-950 text-emerald-400 ring-1 ring-emerald-800"
          : "bg-red-950 text-red-400 ring-1 ring-red-800",
      ].join(" ")}
    >
      <span
        className={[
          "h-1.5 w-1.5 rounded-full",
          ok ? "bg-emerald-400" : "bg-red-400",
        ].join(" ")}
      />
      {ok ? "ok" : value}
    </span>
  );
}

// ─── Status row ──────────────────────────────────────────────────────────────

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-gray-800 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <StatusBadge value={value} />
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Home() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((data) => {
        setHealth(data);
        setFetchError(null);
      })
      .catch((err: Error) => {
        setFetchError(err.message);
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="flex min-h-full flex-col items-center justify-center p-6">
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="w-full max-w-sm space-y-8">
        <div className="space-y-1 text-center">
          <p className="text-xs font-semibold uppercase tracking-widest text-indigo-400">
            Razorpay AI Buildathon · Track 3
          </p>
          <h1 className="text-2xl font-bold text-white">
            Revenue Recovery Autopilot
          </h1>
          <p className="text-sm text-gray-500">
            Recover more revenue from failed payments, automatically — but safely.
          </p>
        </div>

        {/* ── Service Status ────────────────────────────────────────────── */}
        <div className="rounded-xl border border-gray-800 bg-gray-900 px-5 py-4">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Service Status
          </h2>

          {loading && (
            <p className="py-2 text-sm text-gray-600">Checking services…</p>
          )}

          {!loading && fetchError && (
            <div className="rounded-lg bg-red-950 px-4 py-3 text-sm text-red-400 ring-1 ring-red-800">
              Backend unreachable — {fetchError}
            </div>
          )}

          {!loading && !fetchError && health && (
            <div>
              <StatusRow label="API" value={health.status} />
              <StatusRow label="Database (PostgreSQL)" value={health.db} />
              <StatusRow label="Cache (Redis)" value={health.redis} />
            </div>
          )}
        </div>

        {/* ── Phase badge ───────────────────────────────────────────────── */}
        <p className="text-center text-xs text-gray-700">
          Phase 0 · Infrastructure Scaffold · Dashboard coming in Phase 6
        </p>
      </div>
    </main>
  );
}
