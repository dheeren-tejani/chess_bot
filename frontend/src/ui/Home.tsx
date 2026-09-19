import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { useGame } from '../state/useChessGame';
import { api } from '../lib/api';
import { listLocalGames } from '../lib/storage';
import type { LocalGameSummary } from '../lib/storage';
import { PieceGlyph } from '../lib/glyphs';
import { toast } from '../lib/toasts';
import { BUILD_ID } from '../lib/build';

export function Home() {
  const screen = useGame(s => s.screen);
  const startGame = useGame.getState().startGame;
  const loadReplay = useGame.getState().loadReplay;
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [code, setCode] = useState('');
  const [err, setErr] = useState<null | 'format' | 'notfound'>(null);
  const [busy, setBusy] = useState(false);
  const [locals, setLocals] = useState<LocalGameSummary[]>([]);

  useEffect(() => {
    if (screen !== 'home') return;
    setLocals(listLocalGames());
    let cancelled = false;
    const ping = async () => {
      try { await api.health(); if (!cancelled) setStatus('online'); }
      catch { if (!cancelled) setStatus('offline'); }
    };
    void ping();
    const iv = window.setInterval(ping, 5000);
    return () => { cancelled = true; window.clearInterval(iv); };
  }, [screen]);

  const submit = async (raw?: string) => {
    const c = (raw ?? code).toUpperCase();
    if (!/^[A-Z0-9]{8}$/.test(c)) { setErr('format'); return; }
    setBusy(true); setErr(null);
    try { await loadReplay(c); setCode(''); toast('Replay ' + c + ' loaded'); }
    catch { setErr('notfound'); }
    finally { setBusy(false); }
  };

  return (
    <AnimatePresence>
      {screen === 'home' && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}
          className="absolute inset-0 z-30 grid place-items-center overflow-y-auto bg-[#09090b]/85 backdrop-blur-md"
        >
          <motion.div
            initial={{ scale: 0.96, y: 12, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.97, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 320, damping: 26 }}
            className="my-auto w-[460px] max-w-[94vw] rounded-2xl border border-[#27272a] bg-[#0d0d11]/90 p-7 shadow-2xl"
          >
            <div className="text-center">
              <div className="font-display text-xl font-semibold tracking-[0.42em] text-zinc-100">GAMBIT</div>
              <div className="mt-1.5 font-mono text-[10px] tracking-[0.3em] text-zinc-600">3D CHESS · CHOOSE YOUR SIDE</div>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-3">
              {(['w', 'b'] as const).map(c => (
                <button
                  key={c}
                  onClick={() => startGame(c)}
                  className="group flex flex-col items-center gap-2.5 rounded-xl border border-[#27272a] bg-[#09090b] px-4 pb-4 pt-6 transition hover:border-accent/50 hover:bg-zinc-800/40 active:scale-[0.98]"
                >
                  <div className="h-24 w-24 transition-transform duration-200 group-hover:scale-105">
                    <PieceGlyph type="q" color={c} />
                  </div>
                  <div className="font-display text-sm tracking-[0.18em] text-zinc-100">PLAY {c === 'w' ? 'WHITE' : 'BLACK'}</div>
                  <div className="font-mono text-[9px] tracking-[0.12em] text-zinc-600">{c === 'w' ? 'YOU MOVE FIRST' : 'ENGINE OPENS'}</div>
                </button>
              ))}
            </div>

            <div className="mt-5 border-t border-[#27272a] pt-5">
              <div className="font-mono text-[10px] tracking-[0.25em] text-zinc-600">WATCH A REPLAY</div>
              <div className="mt-2.5 flex gap-2">
                <input
                  value={code} spellCheck={false}
                  onChange={e => { setCode(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8)); setErr(null); }}
                  onKeyDown={e => { if (e.key === 'Enter') submit(); }}
                  placeholder="8K3N7P2A"
                  className={`min-w-0 flex-1 rounded-xl border bg-[#09090b] px-3 py-2.5 text-center font-mono text-base tracking-[0.35em] outline-none transition placeholder:text-zinc-800 ${
                    err ? 'border-red-900/70' : 'border-[#27272a] focus:border-zinc-600'}`}
                />
                <button
                  onClick={() => submit()} disabled={busy}
                  className="flex items-center gap-2 rounded-xl bg-zinc-100 px-4 font-display text-sm text-zinc-950 transition hover:bg-white disabled:opacity-50"
                >
                  {busy ? <Loader2 size={14} className="spin" /> : null} LOAD
                </button>
              </div>
              {err && (
                <div className="mt-2 text-center font-mono text-[11px] text-red-400">
                  {err === 'format' ? 'Codes are 8 alphanumeric characters.' : 'Replay not found — check the code.'}
                </div>
              )}
            </div>

            {locals.length > 0 && (
              <div className="mt-4">
                <div className="font-mono text-[10px] tracking-[0.25em] text-zinc-600">SAVED ON THIS DEVICE</div>
                <div className="mt-2 grid grid-cols-2 gap-1.5">
                  {locals.map(g => (
                    <button key={g.code} onClick={() => submit(g.code)}
                      className="flex items-center justify-between rounded-lg border border-transparent px-2.5 py-2 transition hover:border-[#27272a] hover:bg-zinc-800/50">
                      <span className="font-mono text-xs tracking-[0.18em] text-zinc-300">{g.code}</span>
                      <span className="font-mono text-[9px] text-zinc-600">{g.result}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-5 flex items-center justify-between border-t border-[#27272a] pt-4 font-mono text-[10px]">
              <div className="flex items-center gap-2">
                <span className={`inline-block h-2 w-2 rounded-full ${
                  status === 'online' ? 'bg-emerald-500'
                  : status === 'offline' ? 'bg-amber-500'
                  : 'animate-pulse bg-zinc-600'}`} />
                <span className="text-zinc-500">
                  {status === 'online' ? 'ENGINE ONLINE — API' : status === 'offline' ? 'ENGINE LOCAL — BUILT-IN' : 'CONNECTING'}
                </span>
              </div>
              <span className="text-zinc-700">{BUILD_ID}V — 2D/3D</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}