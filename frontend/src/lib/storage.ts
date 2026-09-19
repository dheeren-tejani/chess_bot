import type { GamePayload } from '../types';

const LS_PREFIX = 'gambit:replay:';
const CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

export function makeCode(): string {
  const b = new Uint8Array(8);
  crypto.getRandomValues(b);
  return Array.from(b, x => CODE_CHARS[x % CODE_CHARS.length]).join('');
}

export function saveLocalGame(code: string, payload: GamePayload): void {
  try { localStorage.setItem(LS_PREFIX + code, JSON.stringify(payload)); } catch { /* private mode */ }
}

export function loadLocalGame(code: string): GamePayload | null {
  try { const s = localStorage.getItem(LS_PREFIX + code); return s ? JSON.parse(s) : null; } catch { return null; }
}

export interface LocalGameSummary { code: string; result: string; plies: number; }

export function listLocalGames(): LocalGameSummary[] {
  try {
    return Object.keys(localStorage)
      .filter(k => k.startsWith(LS_PREFIX))
      .map(k => {
        const p = JSON.parse(localStorage.getItem(k) || '{}');
        return { code: k.slice(LS_PREFIX.length), result: p.result || '—', plies: (p.moves || []).length };
      })
      .filter(g => g.plies > 0)
      .slice(0, 6);
  } catch { return []; }
}