import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, X } from 'lucide-react';
import { useGame } from '../state/useChessGame';
import { PieceGlyph } from '../lib/glyphs';

type RowState = 'pending' | 'active' | 'done' | 'error';

function StageRow({ label, value, state }: { label: string; value: string; state: RowState }) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2.5">
        <span className="grid h-4 w-4 place-items-center">
          {state === 'done' && <Check size={13} strokeWidth={2.4} className="text-accent" />}
          {state === 'error' && <X size={13} strokeWidth={2.4} className="text-amber-500" />}
          {state === 'active' && <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />}
          {state === 'pending' && <span className="h-2 w-2 rounded-full border border-zinc-700" />}
        </span>
        <span className={`font-mono text-[11px] tracking-[0.18em] ${
          state === 'pending' ? 'text-zinc-600' : 'text-zinc-300'}`}>{label}</span>
      </div>
      <span className={`font-mono text-[10px] tracking-[0.15em] ${
        state === 'active' ? 'text-accent'
        : state === 'error' ? 'text-amber-500'
        : state === 'pending' ? 'text-zinc-700' : 'text-zinc-500'}`}>{value}</span>
    </div>
  );
}

export function LoadingScreen() {
  const screen = useGame(s => s.screen);
  const stage = useGame(s => s.loadingStage);
  const color = useGame(s => s.loadingColor);
  const goHome = useGame.getState().goHome;
  const playLocally = useGame.getState().playLocally;
  const [t0, setT0] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (screen === 'loading') { setT0(performance.now()); setElapsed(0); }
  }, [screen]);

  // timer freezes once we've given up (offline)
  useEffect(() => {
    if (screen !== 'loading' || stage === 'offline' || t0 === 0) return;
    const iv = window.setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    return () => window.clearInterval(iv);
  }, [screen, stage, t0]);

  const offline = stage === 'offline';
  const isBlack = color === 'b';
  const engineState: RowState = offline ? 'error' : stage === 'waking' ? 'active' : 'done';
  const engineValue = offline ? 'OFFLINE' : stage === 'waking' ? 'WAKING' : 'READY';
  const openState: RowState = stage === 'opening' ? 'active' : 'pending';
  const openValue = stage === 'opening' ? 'THINKING' : 'PENDING';

  return (
    <AnimatePresence>
      {screen === 'loading' && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}
          className="absolute inset-0 z-30 grid place-items-center overflow-y-auto bg-[#09090b]/90 backdrop-blur-md"
        >
          <motion.div
            initial={{ scale: 0.96, y: 12, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.97, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 320, damping: 26 }}
            className="w-[400px] max-w-[92vw] rounded-2xl border border-[#27272a] bg-[#0d0d11]/95 p-7 shadow-2xl"
          >
            <div className="text-center">
              <div className="font-display text-lg font-semibold tracking-[0.42em] text-zinc-100">GAMBIT</div>
              <div className="mt-1 font-mono text-[10px] tracking-[0.28em] text-zinc-600">
                PREPARING MATCH — YOU PLAY {isBlack ? 'BLACK' : 'WHITE'}
              </div>
            </div>

            <motion.div
              animate={{ y: [0, -7, 0] }}
              transition={{ repeat: Infinity, duration: 2.6, ease: 'easeInOut' }}
              className="mx-auto mt-6 h-28 w-28"
            >
              <PieceGlyph type="n" color={color} />
            </motion.div>

            <div className="mt-6 space-y-3 border-t border-[#27272a] pt-5">
              <StageRow label="BOARD" value="READY" state="done" />
              <StageRow label="ENGINE" value={engineValue} state={engineState} />
              {isBlack && <StageRow label="ENGINE'S OPENING" value={openValue} state={openState} />}
            </div>

            {/* indeterminate sweep while anything is in flight */}
            <div className="mt-5 h-[3px] overflow-hidden rounded-full bg-[#18181b]">
              <motion.div
                className="h-full w-1/4 rounded-full bg-accent"
                animate={offline ? { x: 0, opacity: 0 } : { x: ['-100%', '400%'] }}
                transition={offline ? { duration: 0.3 } : { repeat: Infinity, duration: 1.4, ease: 'easeInOut' }}
              />
            </div>

            <div className="mt-3 flex items-center justify-between font-mono text-[10px] text-zinc-600">
              <span className="tabular-nums">{elapsed.toFixed(1)}s</span>
              {!offline && <span>WAKING THE ENGINE</span>}
            </div>

            {offline ? (
              <>
                <p className="mt-4 text-center font-display text-[13px] leading-5 text-zinc-400">
                  The engine server didn't respond in time — it may still be booting.
                  You can retry from home, or play now with the built-in local engine.
                </p>
                <div className="mt-5 flex gap-2">
                  <button onClick={playLocally}
                    className="flex-1 rounded-xl bg-zinc-100 py-2.5 font-display text-sm text-zinc-950 transition hover:bg-white active:scale-[0.98]">
                    Play locally
                  </button>
                  <button onClick={goHome}
                    className="flex-1 rounded-xl border border-[#27272a] py-2.5 font-display text-sm text-zinc-300 transition hover:border-zinc-600 hover:text-zinc-100 active:scale-[0.98]">
                    Back to home
                  </button>
                </div>
              </>
            ) : (
              <>
                <AnimatePresence>
                  {elapsed > 3 && (
                    <motion.p
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="mt-4 text-center font-mono text-[10px] leading-4 text-zinc-600"
                    >
                      SERVERLESS ENGINES SLEEP BETWEEN GAMES —<br />THE FIRST WAKE-UP CAN TAKE A MINUTE
                    </motion.p>
                  )}
                </AnimatePresence>
                <button onClick={goHome}
                  className="mt-5 w-full text-center font-mono text-[10px] tracking-[0.2em] text-zinc-600 transition hover:text-zinc-300">
                  CANCEL
                </button>
              </>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}