import { bestMove } from './search';

/* Vite bundles this file as a real module worker (see engine.ts), so the
   engine runs off the main thread — the 3D render loop never stalls. */

const ctx: any = self;

ctx.onmessage = (e: MessageEvent) => {
  const { id, fen, depth, timeMs } = e.data;
  try {
    const r = bestMove(fen, depth || 3, timeMs || 1400);
    ctx.postMessage({ id, ok: true, uci: r.uci, evaluation: r.evaluation });
  } catch (err) {
    ctx.postMessage({ id, ok: false, error: String(err) });
  }
};