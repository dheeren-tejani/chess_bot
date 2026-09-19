import { create } from 'zustand';
import { Chess } from 'chess.js';
import type { VMove, GamePayload, GameOverInfo } from '../types';
import { api, waitForHealth } from '../lib/api';
import { engineBestMove } from '../lib/engine';
import { SFX } from '../lib/sfx';
import { toast } from '../lib/toasts';
import { makeCode, saveLocalGame, loadLocalGame } from '../lib/storage';
import { wasDrag } from '../lib/input';

const live = new Chess();     // authoritative game (live mode)
const viewer = new Chess();   // replay position tracker (replay mode)
let viewerPly = 0;            // ply currently loaded into `viewer`
let reqSeq = 0;               // token to discard stale bot responses
const wideScreen = () => typeof window !== 'undefined' && window.innerWidth >= 1024;

const clampCp = (v: number) => Math.max(-1500, Math.min(1500, Math.round(v) || 0));

function kingSquare(ch: any, color: 'w' | 'b'): string | null {
  const rows = ch.board();
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
    const pc = rows[i][j];
    if (pc && pc.type === 'k' && pc.color === color) return pc.square;
  }
  return null;
}

/** Accepts UCI ("e7e8q") or SAN ("Nf3"); falls back to a legal default. */
function matchMove(ch: any, bm?: string): VMove | null {
  const legal = ch.moves({ verbose: true }) as unknown as VMove[];
  if (!bm) return legal[0] ?? null;
  const s = bm.trim();
  const uci = s.toLowerCase().match(/^([a-h][1-8])([a-h][1-8])([qrbn])?$/);
  if (uci) {
    const m = legal.find(x => x.from === uci[1] && x.to === uci[2] && (x.promotion || '') === (uci[3] || ''));
    if (m) return m;
  }
  const san = legal.find(x => x.san === s);
  return san ?? legal[0] ?? null;
}

function evalAt(evals: Record<number, number>, ply: number): number {
  for (let i = ply; i >= 0; i--) if (evals[i] !== undefined) return evals[i];
  return 0;
}
function evalSeries(moves: VMove[], evals: Record<number, number>): number[] {
  const out: number[] = [];
  let last = 0;
  for (let i = 0; i <= moves.length; i++) { if (evals[i] !== undefined) last = evals[i]; out.push(last); }
  return out;
}

interface GameState {
  screen: 'home' | 'game';
  mode: 'live' | 'replay';
  viewMode: '3d' | '2d';
  playerColor: 'w' | 'b';
  orientation: 'w' | 'b';     // which side the board is viewed from
  panelOpen: boolean;
  gameKey: number;
  moves: VMove[];
  viewPly: number;
  animDur: number;
  fen: string;
  turn: 'w' | 'b';
  inCheck: boolean;
  checkSquare: string | null;
  lastMove: { from: string; to: string } | null;
  selected: string | null;
  legalTargets: VMove[];
  hovered: string | null;
  promotion: { from: string; to: string } | null;
  thinking: boolean;
  gameOver: GameOverInfo | null;
  showGameOver: boolean;
  showResignConfirm: boolean;
  savingCode: boolean;
  replayCode: string | null;
  replayResult: string | null;
  evalsByPly: Record<number, number>;
  currentEval: number;
  playing: boolean;
  speed: 1 | 2 | 4;
  muted: boolean;
  camCmd: { id: number; type: 'reset' | 'flip' } | null;
  apiOnline: boolean | null;
  onSquareClick: (sq: string) => void;
  resolvePromotion: (p: string) => void;
  clearPromotion: () => void;
  clearSelection: () => void;
  startGame: (color: 'w' | 'b') => void;
  resign: () => void;
  goHome: () => void;
  loadReplay: (code: string) => Promise<void>;
  setViewPly: (p: number) => void;
  setPlaying: (v: boolean) => void;
  setSpeed: (v: 1 | 2 | 4) => void;
  toggleMute: () => void;
  toggleView: () => void;
  togglePanel: () => void;
  setResignConfirm: (v: boolean) => void;
  resetCamera: () => void;
  flipBoard: () => void;
  dismissGameOver: () => void;
}

const resetFields = () => ({
  moves: [] as VMove[], viewPly: 0, animDur: 0.4,
  fen: live.fen(), turn: 'w' as const, inCheck: false, checkSquare: null, lastMove: null,
  selected: null, legalTargets: [] as VMove[], hovered: null, promotion: null,
  thinking: false, gameOver: null, showGameOver: false, showResignConfirm: false,
  savingCode: false, replayCode: null, replayResult: null,
  evalsByPly: {} as Record<number, number>, currentEval: 0, playing: false,
});

export const useGame = create<GameState>((set, get) => ({
  screen: 'home',
  mode: 'live',
  viewMode: '3d',
  playerColor: 'w',
  orientation: 'w',
  panelOpen: wideScreen(),
  gameKey: 0,
  ...resetFields(),
  speed: 1,
  muted: false,
  camCmd: null,
  apiOnline: null,

  onSquareClick: (sq) => {
    if (wasDrag()) return;
    const s = get();
    if (s.mode !== 'live' || s.thinking || s.gameOver || s.promotion || s.turn !== s.playerColor) {
      set({ selected: null, legalTargets: [] });
      return;
    }
    if (s.selected) {
      const cand = s.legalTargets.filter(m => m.to === sq);
      if (cand.length > 0) {
        if (cand[0].promotion) { set({ promotion: { from: s.selected, to: sq } }); SFX.play('ui'); return; }
        commitMove(s.selected, sq, undefined);
        return;
      }
    }
    const pc = (live as any).get(sq);
      if (pc && pc.color === live.turn()) {
      set({ selected: sq, legalTargets: live.moves({ square: sq as any, verbose: true }) as unknown as VMove[], hovered: null });
      SFX.play('select');
    } else {
      set({ selected: null, legalTargets: [] });
    }
  },
  resolvePromotion: (p) => {
    const pr = get().promotion;
    if (pr) commitMove(pr.from, pr.to, p);
  },
  clearPromotion: () => set({ promotion: null }),
  clearSelection: () => set({ selected: null, legalTargets: [], hovered: null }),

  startGame: (color) => {
    reqSeq++;
    live.reset();
    set({
      screen: 'game', mode: 'live', playerColor: color, orientation: color,
      gameKey: get().gameKey + 1, ...resetFields(), panelOpen: wideScreen(),
      camCmd: { id: (get().camCmd?.id ?? 0) + 1, type: 'reset' },
    });
    if (color === 'b') void botTurn(); // engine (White) opens the game
  },

  resign: () => {
    const s = get();
    if (s.mode !== 'live' || s.gameOver || s.screen !== 'game') return;
    const winner = s.playerColor === 'w' ? 'b' : 'w';
    void finishGame(winner === 'w' ? '1-0' : '0-1', 'resignation', winner);
  },

  goHome: () => {
    reqSeq++;
    live.reset();
    set({
      screen: 'home', mode: 'live', playerColor: 'w', orientation: 'w',
      gameKey: get().gameKey + 1, ...resetFields(), panelOpen: wideScreen(),
    });
  },

  loadReplay: async (code) => {
    const c = code.trim().toUpperCase();
    let payload: GamePayload | null = null;
    try { payload = await api.loadGame(c); } catch { payload = loadLocalGame(c); }
    if (!payload || !Array.isArray(payload.moves) || payload.moves.length === 0) throw new Error('not found');
    viewer.reset();
    viewerPly = 0;
    const vlist: VMove[] = [];
    for (const san of payload.moves) {
      try { const v = viewer.move(String(san)); if (v) vlist.push(v as unknown as VMove); else break; } catch { break; }
    }
    if (vlist.length === 0) throw new Error('corrupt');
    const evals: Record<number, number> = {};
    if (Array.isArray(payload.evaluations)) {
      // evaluations[i] = engine opinion of the position after ply i
      payload.evaluations.forEach((e, i) => { if (typeof e === 'number' && isFinite(e)) evals[i] = clampCp(e); });
    }
    reqSeq++;
    set({
      screen: 'game', mode: 'replay',
      orientation: payload.player_color === 'black' ? 'b' : 'w',
      playerColor: 'w',
      gameKey: get().gameKey + 1, ...resetFields(),
      moves: vlist, animDur: 0.5, panelOpen: wideScreen(),
      replayCode: c, replayResult: payload.result || null,
      evalsByPly: evals, currentEval: evals[1] ?? 0,
      camCmd: { id: (get().camCmd?.id ?? 0) + 1, type: 'reset' },
    });
  },

  setViewPly: (p) => {
    const s = get();
    if (s.mode !== 'replay') return;
    const target = Math.max(0, Math.min(s.moves.length, p | 0));
    if (target === s.viewPly) return;
    try {
      if (target > viewerPly) { for (let i = viewerPly; i < target; i++) viewer.move(s.moves[i].san); }
      else { for (let i = viewerPly; i > target; i--) viewer.undo(); }
      viewerPly = target;
    } catch { /* never let position sync crash the viewer */ }
    const vm = target > 0 ? s.moves[target - 1] : null;
    const chk = viewer.inCheck();
    set({
      viewPly: target,
      animDur: Math.abs(target - s.viewPly) === 1 ? 0.5 : 0.16,
      lastMove: vm ? { from: vm.from, to: vm.to } : null,
      turn: viewer.turn(),
      inCheck: chk,
      checkSquare: chk ? kingSquare(viewer, viewer.turn()) : null,
      currentEval: evalAt(s.evalsByPly, target),
    });
  },
  setPlaying: (v) => set({ playing: v }),
  setSpeed: (v) => set({ speed: v }),
  toggleMute: () => { const m = !get().muted; SFX.muted = m; set({ muted: m }); },
  toggleView: () => {
    const next = get().viewMode === '3d' ? '2d' : '3d';
    SFX.play('ui');
    set({ viewMode: next, hovered: null, selected: null, legalTargets: [] });
  },
  togglePanel: () => set({ panelOpen: !get().panelOpen }),
  setResignConfirm: (v) => set({ showResignConfirm: v }),
  resetCamera: () => set({ camCmd: { id: (get().camCmd?.id ?? 0) + 1, type: 'reset' } }),
  // replay-only in the UI — flips viewing side to inspect the opponent's angle
  flipBoard: () => set({
    orientation: get().orientation === 'w' ? 'b' : 'w',
    camCmd: { id: (get().camCmd?.id ?? 0) + 1, type: 'flip' },
  }),
  dismissGameOver: () => set({ showGameOver: false }),
}));

function commitMove(from: string, to: string, promotion?: string) {
  let v: VMove | null = null;
  try {
    v = live.move(promotion ? { from, to, promotion } : { from, to }) as unknown as VMove;
  } catch { v = null; }
  if (!v) { useGame.setState({ promotion: null }); return; }
  pushMove(v, true);
}

function pushMove(v: VMove, isPlayer: boolean) {
  const s = useGame.getState();
  if (s.gameOver) return; // already decided (e.g. resign during engine think)
  const moves = [...s.moves, v];
  const chk = live.inCheck();
  useGame.setState({
    moves, viewPly: moves.length, animDur: 0.5,
    fen: live.fen(), turn: live.turn(),
    inCheck: chk, checkSquare: chk ? kingSquare(live, live.turn()) : null,
    lastMove: { from: v.from, to: v.to },
    selected: null, legalTargets: [], hovered: null, promotion: null,
  });
  SFX.play(v.captured ? 'capture' : 'move');
  if (chk) setTimeout(() => SFX.play('check'), 150);
  if (live.isGameOver()) {
    const o = detectOutcome();
    if (o) void finishGame(o.result, o.reason, o.winner);
    return;
  }
  if (isPlayer) void botTurn();
}

function detectOutcome(): GameOverInfo | null {
  if (!live.isGameOver()) return null;
  if (live.isCheckmate()) {
    const winner = live.turn() === 'w' ? 'b' : 'w';
    return { result: winner === 'w' ? '1-0' : '0-1', reason: 'checkmate', winner };
  }
  if (live.isStalemate()) return { result: '1/2-1/2', reason: 'stalemate', winner: null };
  if (live.isInsufficientMaterial()) return { result: '1/2-1/2', reason: 'insufficient material', winner: null };
  if (live.isThreefoldRepetition()) return { result: '1/2-1/2', reason: 'threefold repetition', winner: null };
  if (live.isDraw()) return { result: '1/2-1/2', reason: 'fifty-move rule', winner: null };
  return { result: '1/2-1/2', reason: 'draw', winner: null };
}

async function botTurn() {
  const s0 = useGame.getState();
  useGame.setState({ thinking: true });
  const token = ++reqSeq;
  const fen = live.fen();
  const sans = useGame.getState().moves.map(m => m.san);
  let res: { best_move: string; evaluation: number } | null = null;
  let fromApi = false;
  try {
    res = await api.getMove(fen, sans, 3);
    fromApi = true;
    if (useGame.getState().apiOnline !== true) useGame.setState({ apiOnline: true });
  } catch {
    // A serverless backend may just still be waking up. Wait it out and
    // retry once before falling back to the local engine.
    //   - was online, just blipped  → short wait (8 s)
    //   - never connected (cold)    → long wait in prod, short in dev
    //   - confirmed down            → no wait, fail over instantly
    const state = useGame.getState().apiOnline;
    const budget = state === false ? 0
      : state === true ? 8000
      : (import.meta.env.DEV ? 3000 : 30000);
    if (budget > 0) {
      const woke = await waitForHealth(budget);
      if (woke) {
        try {
          res = await api.getMove(fen, sans, 3);
          fromApi = true;
          useGame.setState({ apiOnline: true });
        } catch { /* fall through */ }
      }
    }
    if (!fromApi) {
      if (useGame.getState().apiOnline !== false) {
        useGame.setState({ apiOnline: false });
        toast('API offline — built-in engine is playing');
      }
      res = await engineBestMove(fen);
    }
  }
  const st = useGame.getState();
  if (token !== reqSeq || st.gameKey !== s0.gameKey || st.mode !== 'live' || st.screen !== 'game' || st.gameOver) {
    useGame.setState({ thinking: false });
    return;
  }
  const m = matchMove(live, res?.best_move);
  let v: VMove | null = null;
  try { if (m) v = live.move(m.san) as unknown as VMove; } catch { v = null; }
  useGame.setState({ thinking: false });
  // Eval is recorded ONLY when it came from the backend — the local
  // fallback engine's numbers are never shown.
  if (fromApi && res && typeof res.evaluation === 'number' && isFinite(res.evaluation)) {
    // The engine evaluated the position AFTER st.moves.length plies (the
    // position it was asked to move in) — anchor the curve point there,
    // not at the ply of the move it then played.
    useGame.setState({
      evalsByPly: { ...st.evalsByPly, [st.moves.length]: clampCp(res.evaluation) },
      currentEval: clampCp(res.evaluation),
    });
  }
  if (v) pushMove(v, false);
}

async function finishGame(result: string, reason: string, winner: 'w' | 'b' | null) {
  useGame.setState({ thinking: false, showResignConfirm: false });
  useGame.setState({ gameOver: { result, reason, winner }, showGameOver: true, savingCode: true });
  SFX.play('end');
  const s = useGame.getState();
  const payload: GamePayload = {
    moves: s.moves.map(m => m.san),
    result,
    evaluations: evalSeries(s.moves, s.evalsByPly),
    player_color: s.playerColor === 'w' ? 'white' : 'black',
  };
  let code = '';
  try {
    const r = await api.saveGame(payload);
    code = String(r.replay_code || '').toUpperCase();
    useGame.setState({ apiOnline: true });
  } catch {
    useGame.setState({ apiOnline: false });
  }
  if (!/^[A-Z0-9]{8,12}$/.test(code)) { code = makeCode(); saveLocalGame(code, payload); }
  useGame.setState({ replayCode: code, savingCode: false });
}

export function setHoverIfTarget(sq: string | null) {
  const s = useGame.getState();
  if (sq === null) { if (s.hovered) useGame.setState({ hovered: null }); return; }
  if (s.hovered !== sq && s.legalTargets.some(m => m.to === sq)) useGame.setState({ hovered: sq });
}

/** convenience facade for HUD components */
export function useChessGame() {
  const moves = useGame(s => s.moves);
  const viewPly = useGame(s => s.viewPly);
  return {
    plyCount: moves.length,
    viewPly,
    moveNumber: Math.floor(viewPly / 2) + 1,
    turn: useGame(s => s.turn),
    inCheck: useGame(s => s.inCheck),
    gameOver: useGame(s => s.gameOver),
    isLive: useGame(s => s.mode === 'live'),
  };
}