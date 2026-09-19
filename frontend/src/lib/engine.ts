let engineWorker: Worker | null = null;
let engineWorkerFailed = false;
let engineReqId = 0;
const enginePending = new Map<number, { resolve: (v: any) => void; timer: number }>();

function getEngineWorker(): Worker | null {
  if (engineWorkerFailed) return null;
  if (engineWorker) return engineWorker;
  try {
    engineWorker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
    engineWorker.onmessage = (e: MessageEvent) => {
      const entry = enginePending.get(e.data?.id);
      if (!entry) return;
      enginePending.delete(e.data.id);
      clearTimeout(entry.timer);
      entry.resolve(e.data);
    };
    engineWorker.onerror = () => { engineWorkerFailed = true; };
  } catch { engineWorkerFailed = true; }
  return engineWorker;
}

/** Best move via the engine worker; falls back to a shallower
    synchronous search if module workers are unavailable. */
export async function engineBestMove(fen: string): Promise<{ best_move: string; evaluation: number }> {
  const w = getEngineWorker();
  if (w) {
    const id = ++engineReqId;
    const data: any = await new Promise((resolve) => {
      const timer = window.setTimeout(() => { enginePending.delete(id); resolve(null); }, 6000);
      enginePending.set(id, { resolve, timer });
      w.postMessage({ id, fen, depth: 3, timeMs: 1400 });
    });
    if (data && data.ok) return { best_move: data.uci, evaluation: data.evaluation };
  }
  const { bestMove } = await import('./search');
  const r = bestMove(fen, 2, 700);
  return { best_move: r.uci, evaluation: r.evaluation };
}