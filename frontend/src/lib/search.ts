import { Chess } from 'chess.js';

/** Alpha-beta negamax + quiescence + PSTs with a hard time budget.
    Pure logic, no DOM access — shared by the engine worker and the
    synchronous fallback path. */

const VAL: Record<string, number> = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
const PST: Record<string, number[]> = {
  p: [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10, 5,5,10,25,25,10,5,5,
      0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5, 5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
  n: [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30,
      -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30,
      -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
  b: [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10,
      -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10,
      -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
  r: [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
      -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
  q: [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10,
      -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10,
      -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
  k: [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
      -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
      -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
      20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20],
};
let nodes = 0;
let deadline = 0;

function evalBoard(ch: any): number {
  let s = 0, wb = 0, bb = 0;
  const rows = ch.board();
  for (let i = 0; i < 8; i++) {
    const row = rows[i];
    for (let j = 0; j < 8; j++) {
      const pc = row[j];
      if (!pc) continue;
      const idx = pc.color === 'w' ? i * 8 + j : (7 - i) * 8 + j;
      const v = VAL[pc.type] + PST[pc.type][idx];
      s += pc.color === 'w' ? v : -v;
      if (pc.type === 'b') { if (pc.color === 'w') wb++; else bb++; }
    }
  }
  if (wb >= 2) s += 30;
  if (bb >= 2) s -= 30;
  return s;
}
const side = (ch: any) => (ch.turn() === 'w' ? 1 : -1) * evalBoard(ch);

function order(moves: any[]): any[] {
  return moves
    .map((m: any) => ({ m, s: (m.captured ? 10 * VAL[m.captured] - VAL[m.piece] : 0) + (m.promotion ? 800 : 0) }))
    .sort((a: any, b: any) => b.s - a.s)
    .map((x: any) => x.m);
}

function quiesce(ch: any, alpha: number, beta: number): number {
  nodes++;
  const stand = side(ch);
  if (stand >= beta) return beta;
  if (stand > alpha) alpha = stand;
  const caps = order(ch.moves({ verbose: true }).filter((m: any) => m.captured || m.promotion));
  for (let i = 0; i < caps.length; i++) {
    ch.move(caps[i]);
    const sc = -quiesce(ch, -beta, -alpha);
    ch.undo();
    if (sc >= beta) return beta;
    if (sc > alpha) alpha = sc;
    if ((nodes & 511) === 0 && Date.now() > deadline) return alpha;
  }
  return alpha;
}

function search(ch: any, depth: number, alpha: number, beta: number): number {
  nodes++;
  if (depth === 0) return quiesce(ch, alpha, beta);
  const moves = ch.moves({ verbose: true });
  if (moves.length === 0) return ch.inCheck() ? -99000 - depth * 10 : 0;
  const list = order(moves);
  for (let i = 0; i < list.length; i++) {
    ch.move(list[i]);
    const sc = -search(ch, depth - 1, -beta, -alpha);
    ch.undo();
    if (sc >= beta) return beta;
    if (sc > alpha) alpha = sc;
    if ((nodes & 511) === 0 && Date.now() > deadline) return alpha;
  }
  return alpha;
}

export function bestMove(fen: string, maxDepth: number, timeMs: number): { uci: string; evaluation: number; nodes: number } {
  const ch = new Chess(fen);
  deadline = Date.now() + timeMs;
  nodes = 0;
  let rootMoves: any[] = order(ch.moves({ verbose: true }));
  if (!rootMoves.length) return { uci: '', evaluation: 0, nodes: 0 };
  let scored: any[] = [];
  for (let d = 1; d <= maxDepth; d++) {
    let alpha = -1000000;
    scored = [];
    for (let i = 0; i < rootMoves.length; i++) {
      const m = rootMoves[i];
      ch.move(m);
      const sc = -search(ch, d - 1, -1000000, -alpha);
      ch.undo();
      scored.push({ m, sc });
      if (sc > alpha) alpha = sc;
    }
    scored.sort((a: any, b: any) => b.sc - a.sc);
    rootMoves = scored.map((x: any) => x.m);
    if (Date.now() > deadline && d > 1) break;
  }
  const top = scored.length ? scored[0].sc : 0;
  const pool = scored.filter((x: any) => x.sc >= top - 12); // variety among near-equals
  const pick = pool[Math.floor(Math.random() * pool.length)].m;
  const evalWhite = (ch.turn() === 'w' ? 1 : -1) * top;
  return { uci: pick.from + pick.to + (pick.promotion || ''), evaluation: Math.max(-1500, Math.min(1500, evalWhite)), nodes };
}