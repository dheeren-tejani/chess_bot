import type { GamePayload } from '../types';

/* Same-origin everywhere (Vite dev proxy / Netlify edge function in prod).
   The real backend URL exists ONLY in server-side env vars. */

/* Engine searches legitimately take seconds, and a waking container holds
   requests far longer. The old 1.5s health timeout aborted answers that
   curl — which has no timeout — saw succeed. That was the "healthz not
   properly going and coming" flicker. */
const API_TIMEOUT_MS = 15000;
const HEALTH_TIMEOUT_MS = 8000;

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) { super(message); this.status = status; }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs: number = API_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(path, {
      ...init,
      signal: ctrl.signal,
      headers: { 'Content-Type': 'application/json', ...(init.headers || {}) },
    });
    if (!res.ok) throw new ApiError('HTTP ' + res.status, res.status);
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  getMove(fen: string, moves: string[], depth = 3): Promise<{ best_move: string; evaluation: number }> {
    return request('/api/move', { method: 'POST', body: JSON.stringify({ fen, moves, depth }) });
  },
  saveGame(payload: GamePayload): Promise<{ replay_code: string }> {
    return request('/api/games', { method: 'POST', body: JSON.stringify(payload) });
  },
  loadGame(code: string): Promise<GamePayload> {
    return request('/api/games/' + encodeURIComponent(code), { method: 'GET' });
  },
  /** Liveness probe — generous default timeout for waking containers. */
  health(timeoutMs: number = HEALTH_TIMEOUT_MS): Promise<{ ok: boolean }> {
    return request('/healthz', { method: 'GET' }, timeoutMs);
  },
};

/** Poll /healthz until the backend reports ready. Attempts use a 12s
    timeout. NOTE: this backend serves 200 + {"ok": false} while the
    model loads, so readiness is judged from the body, not the status. */
export async function waitForHealth(totalMs: number, intervalMs = 2000): Promise<boolean> {
  const deadline = Date.now() + totalMs;
  while (Date.now() < deadline) {
    try {
      const h: any = await api.health(12000);
      if (!h || h.ok !== false) return true;
    } catch { /* not answering (yet) — each failed poll still nudges the wake */ }
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}