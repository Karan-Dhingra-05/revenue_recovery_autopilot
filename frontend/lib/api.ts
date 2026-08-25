/**
 * Thin API client for the Revenue Recovery Autopilot backend.
 *
 * All backend calls go through this module so the base URL is configured
 * in exactly one place (NEXT_PUBLIC_API_URL).
 */

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export interface HealthStatus {
  /** "ok" | "degraded" */
  status: string;
  /** "ok" | "error: <message>" */
  db: string;
  /** "ok" | "error: <message>" */
  redis: string;
}

/**
 * Fetch the health status of the backend and its dependencies.
 * Throws if the HTTP request itself fails (network error, non-2xx).
 */
export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/api/health`, {
    // Don't cache health checks — we always want a live response.
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Health check returned ${res.status}`);
  }

  return res.json() as Promise<HealthStatus>;
}
