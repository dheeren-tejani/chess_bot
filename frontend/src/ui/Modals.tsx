import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Crown, Copy, Check, Loader2, Flag } from 'lucide-react';
import { useGame } from '../state/useChessGame';
import { useToasts, toast } from '../lib/toasts';

export function PromotionModal() {
  const promotion = useGame(s => s.promotion);
  const resolve = useGame(s => s.resolvePromotion);
  const clear = useGame(s => s.clearPromotion);
  const NAMES: Record<string, string> = { q: 'QUEEN', r: 'ROOK', b: 'BISHOP', n: 'KNIGHT' };
  return (
    <AnimatePresence>
      {promotion && (
        <motion.div className="absolute inset-0 z-30 grid place-items-center bg-black/40 backdrop-blur-[2px]"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <motion.div
            initial={{ scale: 0.94, y: 10, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.96, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 400, damping: 28 }}
            className="rounded-2xl border border-[#27272a] bg-[#0d0d11]/95 p-6"
          >
            <div className="text-center font-mono text-[11px] tracking-[0.25em] text-zinc-500">PROMOTE PAWN</div>
            <div className="mt-4 flex gap-2">
              {(['q', 'r', 'b', 'n'] as const).map(t => (
                <button key={t} onClick={() => resolve(t)}
                  className="group flex w-[72px] flex-col items-center gap-1.5 rounded-xl border border-[#27272a] py-4 transition hover:border-accent/60 hover:bg-zinc-800/50">
                  <span className="font-display text-2xl text-zinc-200 group-hover:text-accent">{t.toUpperCase()}</span>
                  <span className="font-mono text-[10px] text-zinc-600">{NAMES[t]}</span>
                </button>
              ))}
            </div>
            <button onClick={clear} className="mt-4 w-full text-center font-mono text-[11px] text-zinc-600 transition hover:text-zinc-300">CANCEL</button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function ResignModal() {
  const show = useGame(s => s.showResignConfirm);
  const playerColor = useGame(s => s.playerColor);
  const { resign, setResignConfirm } = useGame.getState();
  const loss = playerColor === 'w' ? '0-1' : '1-0';
  return (
    <AnimatePresence>
      {show && (
        <motion.div className="absolute inset-0 z-30 grid place-items-center bg-black/45 backdrop-blur-[2px]"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          onClick={() => setResignConfirm(false)}>
          <motion.div
            onClick={e => e.stopPropagation()}
            initial={{ scale: 0.94, y: 10, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.96, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 400, damping: 28 }}
            className="w-[360px] max-w-[92vw] rounded-2xl border border-[#27272a] bg-[#0d0d11]/95 p-6 text-center"
          >
            <Flag size={22} strokeWidth={1.6} className="mx-auto text-accent" />
            <div className="mt-3 font-mono text-[11px] tracking-[0.25em] text-zinc-500">RESIGN</div>
            <p className="mt-3 font-display text-sm leading-6 text-zinc-300">
              Resign as {playerColor === 'w' ? 'White' : 'Black'}? The game ends {loss} and is saved with a replay code.
            </p>
            <div className="mt-5 flex gap-2">
              <button onClick={() => setResignConfirm(false)}
                className="flex-1 rounded-xl border border-[#27272a] py-2.5 font-display text-sm text-zinc-300 transition hover:border-zinc-600 hover:text-zinc-100 active:scale-[0.98]">
                Keep playing
              </button>
              <button onClick={resign}
                className="flex-1 rounded-xl border border-red-900/60 bg-red-950/40 py-2.5 font-display text-sm text-red-300 transition hover:bg-red-950/70 active:scale-[0.98]">
                Resign
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function GameOverModal() {
  const info = useGame(s => s.gameOver);
  const show = useGame(s => s.showGameOver);
  const saving = useGame(s => s.savingCode);
  const code = useGame(s => s.replayCode);
  const movesLen = useGame(s => s.moves.length);
  const { dismissGameOver, goHome, loadReplay, setPlaying } = useGame.getState();
  const [copied, setCopied] = useState(false);
  if (!info) return null;
  const title = info.reason === 'checkmate' ? 'Checkmate'
    : info.reason === 'resignation' ? 'Resignation'
    : info.reason === 'stalemate' ? 'Stalemate' : 'Draw';
  const sub = (info.winner ? (info.winner === 'w' ? 'White' : 'Black') + ' wins · ' : info.reason + ' · ')
    + Math.ceil(movesLen / 2) + ' moves';
  const copy = async () => {
    if (!code) return;
    try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1600); }
    catch { toast('Clipboard unavailable'); }
  };
  const watch = async () => {
    if (!code) return;
    try { await loadReplay(code); setPlaying(true); } catch { toast('Could not open replay'); }
  };
  return (
    <AnimatePresence>
      {show && (
        <motion.div className="absolute inset-0 z-40 grid place-items-center bg-black/55 backdrop-blur-[3px]"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <motion.div
            onClick={e => e.stopPropagation()}
            initial={{ scale: 0.94, y: 14, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.96, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 320, damping: 26 }}
            className="w-[400px] max-w-[92vw] rounded-2xl border border-[#27272a] bg-[#0d0d11]/95 p-7 text-center shadow-2xl"
          >
            {info.winner
              ? <Crown size={26} strokeWidth={1.6} className="mx-auto text-accent" />
              : <div className="mx-auto h-6 w-6 rounded-full border border-zinc-700" />}
            <h2 className="mt-3 font-display text-xl tracking-wide text-zinc-100">{title}</h2>
            <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">{sub}</p>
            <div className="mt-6 rounded-xl border border-[#27272a] bg-[#09090b] px-4 py-4">
              <div className="font-mono text-[9px] tracking-[0.3em] text-zinc-600">REPLAY CODE</div>
              {saving || !code ? (
                <div className="mt-2 flex justify-center"><Loader2 size={22} className="spin text-zinc-500" /></div>
              ) : (
                <div className="mt-2 font-mono text-[26px] tracking-[0.38em] text-zinc-100">{code}</div>
              )}
              <button onClick={copy} disabled={!code}
                className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-[#27272a] px-3 py-1.5 font-mono text-[11px] text-zinc-300 transition hover:border-zinc-600 hover:text-zinc-100 disabled:opacity-40">
                {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? 'COPIED' : 'COPY CODE'}
              </button>
            </div>
            <div className="mt-5 flex gap-2">
              <button onClick={watch} className="flex-1 rounded-xl bg-zinc-100 py-2.5 font-display text-sm text-zinc-950 transition hover:bg-white active:scale-[0.98]">Watch replay</button>
              <button onClick={goHome} className="flex-1 rounded-xl border border-[#27272a] py-2.5 font-display text-sm text-zinc-300 transition hover:border-zinc-600 hover:text-zinc-100 active:scale-[0.98]">Back to home</button>
            </div>
            <button onClick={dismissGameOver} className="mt-3 font-mono text-[10px] tracking-[0.15em] text-zinc-600 transition hover:text-zinc-400">
              KEEP EXPLORING THE POSITION
            </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function Toasts() {
  const list = useToasts(s => s.list);
  return (
    <div className="pointer-events-none absolute bottom-20 left-1/2 z-50 flex -translate-x-1/2 flex-col items-center gap-1.5 lg:bottom-6">
      <AnimatePresence>
        {list.map(t => (
          <motion.div key={t.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="rounded-lg border border-[#27272a] border-l-2 border-l-accent bg-[#0d0d11]/95 px-3.5 py-2 font-mono text-[11px] text-zinc-300 backdrop-blur">
            {t.msg}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}