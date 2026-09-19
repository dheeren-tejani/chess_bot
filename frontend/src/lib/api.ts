import type { GamePayload } from '../types';

/* Same-origin everywhere:
   - dev:  Vite's server.proxy forwards /api + /healthz to localhost:8000
   - prod: the Netlify edge function forwards them to the deployed backend
   The real backend URL exists ONLY in server-side env vars.
   WARNING: never put it in a VITE_-prefixed variable — Vite inlines those
   into the shipped JS bundle, which defeats the whole point. */
const API_BASE: string = (globalThis as any).__GAMBIT_API__ ?? '';
const API_TIMEOUT_MS = 4000;

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) { super(message); this.status = status; }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs: number = API_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(API_BASE + path, {
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
  health(): Promise<{ ok: boolean }> {
    return request('/healthz', { method: 'GET' }, 1500);
  },
};

/** Poll /healthz until the backend reports itself ready — used to ride
    out a serverless cold start (each failed poll still nudges the wake).
    NOTE: this backend serves 200 + {"ok": false} while the model loads,
    so readiness is judged from the body, not the status code. */
export async function waitForHealth(totalMs: number, intervalMs = 2000): Promise<boolean> {
  const deadline = Date.now() + totalMs;
  while (Date.now() < deadline) {
    try {
      const h: any = await api.health();
      if (!h || h.ok !== false) return true;   // ok:true, or unknown schema
    } catch { /* not answering yet — keep polling */ }
    await new Promise(r => setTimeout(r, intervalMs));
  }
  return false;
}