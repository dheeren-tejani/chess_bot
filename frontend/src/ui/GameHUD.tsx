import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Volume2, VolumeX, Crosshair, RefreshCw, Box, LayoutGrid, Flag,
  Home as HomeIcon, PanelRightClose, PanelRightOpen, LineChart, X,
} from 'lucide-react';
import { useGame, useChessGame } from '../state/useChessGame';
import { ReplayControls } from './ReplayViewer';

function HudButton({ icon: Icon, label, onClick }: { icon: any; label: string; onClick: () => void }) {
  return (
    <div className="group relative">
      <button
        onClick={onClick} aria-label={label}
        className="grid h-8 w-8 place-items-center rounded-lg border border-[#27272a] bg-[#0c0c0f]/80 text-zinc-400 backdrop-blur-md transition hover:border-zinc-600 hover:bg-zinc-800/70 hover:text-zinc-100 active:scale-95 sm:h-9 sm:w-9"
      >
        <Icon size={15} strokeWidth={1.8} />
      </button>
      <div className="pointer-events-none absolute right-0 top-full z-20 mt-1.5 whitespace-nowrap rounded-md border border-[#27272a] bg-[#0d0d11] px-2 py-1 font-mono text-[10px] text-zinc-400 opacity-0 transition group-hover:opacity-100">
        {label}
      </div>
    </div>
  );
}

/** Live: 2D/3D · Reset · Resign · Mute.  Replay adds Flip + Exit-to-home. */
export function ActionButtons() {
  const muted = useGame(s => s.muted);
  const viewMode = useGame(s => s.viewMode);
  const mode = useGame(s => s.mode);
  const screen = useGame(s => s.screen);
  const { resetCamera, flipBoard, toggleMute, toggleView, goHome, setResignConfirm } = useGame.getState();
  if (screen !== 'game') return null;
  return (
    <div className="absolute right-3 top-3 z-20 flex gap-1.5 sm:right-4 sm:top-4">
      <HudButton icon={viewMode === '3d' ? LayoutGrid : Box} label={viewMode === '3d' ? '2D board (V)' : '3D board (V)'} onClick={toggleView} />
      <HudButton icon={Crosshair} label="Reset view" onClick={resetCamera} />
      {mode === 'live' ? (
        <HudButton icon={Flag} label="Resign" onClick={() => setResignConfirm(true)} />
      ) : (
        <>
          <HudButton icon={RefreshCw} label="Flip board" onClick={flipBoard} />
          <HudButton icon={HomeIcon} label="Exit to home" onClick={goHome} />
        </>
      )}
      <HudButton icon={muted ? VolumeX : Volume2} label={muted ? 'Sound on' : 'Mute'} onClick={toggleMute} />
    </div>
  );
}

export function TopBar() {
  const g = useChessGame();
  const thinking = useGame(s => s.thinking);
  return (
    /* width reserves exactly the action-button row on mobile
       (Tailwind: underscores become spaces inside calc()) */
    <div className="pointer-events-none absolute left-3 top-3 z-10 flex w-[calc(100%_-_196px)] flex-wrap items-center gap-1.5 sm:left-4 sm:top-4 sm:w-auto sm:gap-2">
      <div className="pointer-events-auto flex items-center gap-2.5 rounded-full border border-[#27272a] bg-[#0c0c0f]/80 px-3 py-1.5 font-mono text-[10px] backdrop-blur-md sm:gap-3 sm:px-4 sm:py-2 sm:text-[11px]">
        <span className="text-zinc-600">MOVE</span>
        <span className="tabular-nums text-zinc-100">{g.moveNumber}</span>
        <span className="hidden items-center gap-2.5 sm:flex">
          <span className="text-zinc-800">|</span>
          <span className="text-zinc-600">PLY</span>
          <span className="tabular-nums text-zinc-100">{g.viewPly}</span>
        </span>
        <span className="text-zinc-800">|</span>
        <span className={`inline-block h-2 w-2 rounded-[2px] ${g.turn === 'w' ? 'bg-[#e9e5da]' : 'border border-zinc-700 bg-[#2a2a33]'}`} />
        <span className="text-zinc-200">{g.turn === 'w' ? 'WHITE' : 'BLACK'}</span>
      </div>
      {g.inCheck && (
        <div className="pointer-events-auto animate-pulse rounded-full border border-red-900/60 bg-red-950/60 px-3 py-1.5 font-mono text-[10px] tracking-[0.2em] text-[#ff8080] backdrop-blur-md">
          CHECK
        </div>
      )}
      {thinking && (
        <div className="pointer-events-auto flex items-center gap-1.5 rounded-full border border-[#27272a] bg-[#0c0c0f]/80 px-3 py-1.5 font-mono text-[10px] tracking-[0.15em] text-zinc-500 backdrop-blur-md">
          ENGINE<span className="dots"><span>.</span><span>.</span><span>.</span></span>
        </div>
      )}
    </div>
  );
}

/* ── eval curve ── Catmull-Rom → bezier smoothing */
function smoothPath(pts: [number, number][]): string {
  if (pts.length < 2) return '';
  let d = `M${pts[0][0].toFixed(2)},${pts[0][1].toFixed(2)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] ?? pts[i + 1];
    const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += `C${c1x.toFixed(2)},${c1y.toFixed(2)} ${c2x.toFixed(2)},${c2y.toFixed(2)} ${p2[0].toFixed(2)},${p2[1].toFixed(2)}`;
  }
  return d;
}

const CX0 = 6, CX1 = 258, CY0 = 10, CY1 = 56, CMID = (CY0 + CY1) / 2;

function EvalChart() {
  const moves = useGame(s => s.moves);
  const evalsByPly = useGame(s => s.evalsByPly);
  const viewPly = useGame(s => s.viewPly);
  const mode = useGame(s => s.mode);
  const thinking = useGame(s => s.thinking);
  const setViewPly = useGame(s => s.setViewPly);
  const setPlaying = useGame(s => s.setPlaying);
  const [hover, setHover] = useState<number | null>(null);

  const N = moves.length;
  const val = (i: number): number => {
    for (let k = Math.min(i, N); k >= 0; k--) if (evalsByPly[k] !== undefined) return evalsByPly[k];
    return 0;
  };
  const xOf = (i: number) => (N > 0 ? CX0 + (i / N) * (CX1 - CX0) : (CX0 + CX1) / 2);
  const yOf = (v: number) => CMID - (Math.max(-1500, Math.min(1500, v)) / 1500) * (CMID - CY0);

  const pts = useMemo(() => {
    const arr: [number, number][] = [];
    for (let i = 0; i <= N; i++) arr.push([xOf(i), yOf(val(i))]);
    return arr;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [N, evalsByPly]);

  const linePath = useMemo(() => smoothPath(pts), [pts]);
  const areaPath = linePath ? `${linePath} L ${pts[pts.length - 1][0].toFixed(2)},${CY1 + 8} L ${pts[0][0].toFixed(2)},${CY1 + 8} Z` : '';
  const playhead = mode === 'replay' ? viewPly : N;
  const cp = hover !== null ? val(hover) : useGame.getState().currentEval;
  const mate = Math.abs(cp) >= 1500;
  const label = mate ? (cp > 0 ? '+MATE' : '−MATE') : (cp > 0 ? '+' : cp < 0 ? '−' : '') + (Math.abs(cp) / 100).toFixed(1);
  const prob = cp === 0 ? 0.5 : 1 / (1 + Math.pow(10, -cp / 400));

  const idxFromEvt = (e: React.MouseEvent<SVGRectElement>): number => {
    const r = e.currentTarget.getBoundingClientRect();
    return Math.max(0, Math.min(N, Math.round(((e.clientX - r.left) / r.width) * N)));
  };

  return (
    <div className="border-b border-[#27272a] px-4 py-3">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] tracking-[0.25em] text-zinc-600">
          EVAL{thinking && <span className="dots ml-1"><span>.</span><span>.</span><span>.</span></span>}
        </span>
        <span className="font-mono text-xs text-zinc-200">
          {label}{hover !== null && <span className="text-zinc-600"> · PLY {hover}</span>}
        </span>
      </div>
      <svg viewBox="0 0 264 76" className="mt-1.5 w-full">
        <defs>
          <linearGradient id="evgrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#e0a758" stopOpacity="0.26" />
            <stop offset="100%" stopColor="#e0a758" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <line x1={CX0} x2={CX1} y1={CMID} y2={CMID} stroke="#27272a" strokeDasharray="3 3" />
        {areaPath && <path d={areaPath} fill="url(#evgrad)" />}
        {linePath && <path d={linePath} fill="none" stroke="#e0a758" strokeWidth="1.6" strokeLinecap="round" />}
        {hover !== null && (
          <line x1={xOf(hover)} x2={xOf(hover)} y1={CY0 - 2} y2={CY1 + 6} stroke="#52525b" strokeWidth="1" />
        )}
        <line x1={xOf(playhead)} x2={xOf(playhead)} y1={CY0 - 2} y2={CY1 + 6} stroke="#3f3f46" strokeWidth="1" />
        <circle cx={xOf(playhead)} cy={yOf(val(playhead))} r="3.2" fill="#e0a758" stroke="#0c0c0f" strokeWidth="1" />
        {hover !== null && (
          <circle cx={xOf(hover)} cy={yOf(val(hover))} r="2.4" fill="#f2c179" />
        )}
        <rect
          x={CX0 - 4} y={0} width={CX1 - CX0 + 8} height={76} fill="transparent"
          className={mode === 'replay' ? 'cursor-pointer' : ''}
          onMouseMove={e => setHover(idxFromEvt(e))}
          onMouseLeave={() => setHover(null)}
          onClick={e => { if (mode === 'replay') { setPlaying(false); setViewPly(idxFromEvt(e)); } }}
        />
      </svg>
      <div className="mt-1 flex justify-between font-mono text-[9px] text-zinc-600">
        <span>{Math.round(prob * 100)}% WHITE</span>
        <span>{Math.round((1 - prob) * 100)}% BLACK</span>
      </div>
    </div>
  );
}

function MoveToken({ san, active, clickable, onClick }: { san: string; active: boolean; clickable: boolean; onClick: () => void }) {
  return (
    <button
      data-active={active ? '1' : undefined}
      disabled={!clickable}
      onClick={onClick}
      className={`flex-1 rounded-md px-2 py-1.5 text-left font-mono text-[12.5px] transition ${
        active ? 'bg-zinc-800/60 text-accent'
        : clickable ? 'text-zinc-300 hover:bg-zinc-800/40'
        : 'cursor-default text-zinc-500'}`}
    >
      {san}
    </button>
  );
}

function MoveLog() {
  const moves = useGame(s => s.moves);
  const viewPly = useGame(s => s.viewPly);
  const mode = useGame(s => s.mode);
  const setViewPly = useGame(s => s.setViewPly);
  const setPlaying = useGame(s => s.setPlaying);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const el = box.querySelector('[data-active="1"]') as HTMLElement | null;
    if (!el) return;
    // manual scroll ONLY on this container — scrollIntoView() is allowed to
    // scroll overflow:hidden ancestors (the fixed root), which dragged the
    // entire UI sideways on every move.
    const delta =
      el.getBoundingClientRect().top - box.getBoundingClientRect().top -
      (box.clientHeight - el.offsetHeight) / 2;
    box.scrollTop = Math.max(0, box.scrollTop + delta);
  }, [viewPly, moves.length]);

  if (moves.length === 0) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center px-6 text-center font-mono text-[11px] leading-5 text-zinc-600">
        {mode === 'replay' ? 'Replay loaded — use the transport below.' : 'No moves yet — make your move.'}
      </div>
    );
  }
  const rows = Math.ceil(moves.length / 2);
  return (
    <div ref={boxRef} className="min-h-0 flex-1 overflow-y-auto py-1">
      {Array.from({ length: rows }, (_, k) => {
        const wPly = 2 * k + 1, bPly = 2 * k + 2;
        const wm = moves[2 * k], bm = moves[2 * k + 1];
        const jump = (ply: number) => { setPlaying(false); setViewPly(ply); };
        return (
          <div key={k} className="flex items-center gap-1 px-2">
            <span className="w-8 shrink-0 pr-1 text-right font-mono text-[11px] text-zinc-600">{k + 1}.</span>
            <MoveToken san={wm.san} active={viewPly === wPly} clickable={mode === 'replay'} onClick={() => jump(wPly)} />
            {bm
              ? <MoveToken san={bm.san} active={viewPly === bPly} clickable={mode === 'replay'} onClick={() => jump(bPly)} />
              : <span className="flex-1" />}
          </div>
        );
      })}
    </div>
  );
}

export function SidePanel() {
  const mode = useGame(s => s.mode);
  const replayCode = useGame(s => s.replayCode);
  const replayResult = useGame(s => s.replayResult);
  const apiOnline = useGame(s => s.apiOnline);
  const playerColor = useGame(s => s.playerColor);
  const panelOpen = useGame(s => s.panelOpen);
  const togglePanel = useGame.getState().togglePanel;
  return (
    <>
      <AnimatePresence>
        {panelOpen && (
          <motion.aside
            initial={{ x: 316, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 316, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 380, damping: 34 }}
            className="absolute bottom-4 right-4 top-[68px] z-10 hidden w-[302px] flex-col overflow-hidden rounded-xl border border-[#27272a] bg-[#0c0c0f]/75 backdrop-blur-md lg:flex"
          >
            <header className="flex items-center justify-between border-b border-[#27272a] px-4 py-3">
              <div className="flex items-baseline gap-2">
                <span className="font-display text-[13px] font-semibold tracking-[0.22em] text-zinc-100">GAMBIT</span>
                <span className="font-mono text-[9px] tracking-[0.2em] text-zinc-600">3D CHESS</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`rounded-full border px-2 py-0.5 font-mono text-[9px] tracking-[0.15em] ${mode === 'replay' ? 'border-accent/40 text-accent' : 'border-[#27272a] text-zinc-500'}`}>
                  {mode === 'replay' ? (replayCode || 'REPLAY') : 'LIVE'}
                </span>
                <button onClick={togglePanel} title="Collapse panel"
                  className="grid h-6 w-6 place-items-center rounded-md text-zinc-600 transition hover:bg-zinc-800/70 hover:text-zinc-200">
                  <PanelRightClose size={14} />
                </button>
              </div>
            </header>
            <EvalChart />
            <MoveLog />
            {mode === 'replay' ? (
              <div className="border-t border-[#27272a] p-3"><ReplayControls /></div>
            ) : (
              <footer className="flex items-center justify-between border-t border-[#27272a] px-4 py-2.5">
                <span className="font-mono text-[10px] text-zinc-600">{apiOnline === false ? 'ENGINE — LOCAL' : 'ENGINE — API'}</span>
                <span className="font-mono text-[10px] text-zinc-700">YOU PLAY {playerColor === 'w' ? 'WHITE' : 'BLACK'}</span>
              </footer>
            )}
            {mode === 'replay' && (
              <footer className="flex items-center justify-between border-t border-[#27272a] px-4 py-2.5">
                <span className="font-mono text-[10px] text-zinc-600">RESULT — {replayResult || '—'}</span>
                <span className="font-mono text-[10px] text-zinc-700">SPACE · ←/→</span>
              </footer>
            )}
          </motion.aside>
        )}
      </AnimatePresence>
      {!panelOpen && (
        <button onClick={togglePanel} title="Open panel"
          className="absolute right-4 top-[68px] z-10 hidden h-9 w-9 place-items-center rounded-lg border border-[#27272a] bg-[#0c0c0f]/80 text-zinc-400 backdrop-blur-md transition hover:border-zinc-600 hover:text-zinc-100 lg:grid">
          <PanelRightOpen size={15} />
        </button>
      )}
    </>
  );
}

/** Mobile bottom sheet — the phone-sized stand-in for the side panel:
    eval chart + move log + replay transport, opened from the bottom bar. */
export function MobilePanel() {
  const mode = useGame(s => s.mode);
  const replayCode = useGame(s => s.replayCode);
  const replayResult = useGame(s => s.replayResult);
  const apiOnline = useGame(s => s.apiOnline);
  const playerColor = useGame(s => s.playerColor);
  const panelOpen = useGame(s => s.panelOpen);
  const togglePanel = useGame.getState().togglePanel;
  return (
    <AnimatePresence>
      {panelOpen && (
        <motion.div
          initial={{ y: '100%' }} animate={{ y: 0 }} exit={{ y: '100%' }}
          transition={{ type: 'spring', stiffness: 360, damping: 36 }}
          className="absolute inset-x-0 bottom-0 z-20 flex max-h-[64%] flex-col overflow-hidden rounded-t-2xl border-t border-[#27272a] bg-[#0c0c0f]/95 backdrop-blur-md lg:hidden"
        >
          <header className="flex items-center justify-between border-b border-[#27272a] px-4 py-3">
            <div className="flex items-baseline gap-2">
              <span className="font-display text-[13px] font-semibold tracking-[0.22em] text-zinc-100">GAMBIT</span>
              <span className={`rounded-full border px-2 py-0.5 font-mono text-[9px] tracking-[0.15em] ${mode === 'replay' ? 'border-accent/40 text-accent' : 'border-[#27272a] text-zinc-500'}`}>
                {mode === 'replay' ? (replayCode || 'REPLAY') : 'LIVE'}
              </span>
            </div>
            <button onClick={togglePanel} aria-label="Close panel"
              className="grid h-7 w-7 place-items-center rounded-md text-zinc-500 transition hover:bg-zinc-800/70 hover:text-zinc-200">
              <X size={15} />
            </button>
          </header>
          <EvalChart />
          <div className="flex min-h-0 flex-1 flex-col">
            <MoveLog />
          </div>
          {mode === 'replay' ? (
            <div className="border-t border-[#27272a] p-3"><ReplayControls /></div>
          ) : (
            <footer className="flex items-center justify-between border-t border-[#27272a] px-4 py-2.5">
              <span className="font-mono text-[10px] text-zinc-600">{apiOnline === false ? 'ENGINE — LOCAL' : 'ENGINE — API'}</span>
              <span className="font-mono text-[10px] text-zinc-700">YOU PLAY {playerColor === 'w' ? 'WHITE' : 'BLACK'}</span>
            </footer>
          )}
          {mode === 'replay' && (
            <footer className="flex items-center justify-between border-t border-[#27272a] px-4 py-2.5">
              <span className="font-mono text-[10px] text-zinc-600">RESULT — {replayResult || '—'}</span>
              <span className="font-mono text-[10px] text-zinc-700">TAP LOG TO JUMP</span>
            </footer>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function MobileBar() {
  const g = useChessGame();
  const mode = useGame(s => s.mode);
  const panelOpen = useGame(s => s.panelOpen);
  const togglePanel = useGame.getState().togglePanel;
  return (
    <div className="absolute inset-x-3 bottom-3 z-10 flex flex-col items-center gap-2 lg:hidden">
      {mode === 'replay' && (
        <div className="w-full rounded-xl border border-[#27272a] bg-[#0c0c0f]/80 p-2.5 backdrop-blur-md">
          <ReplayControls />
        </div>
      )}
      <div className="flex w-full items-center gap-2">
        <button onClick={togglePanel} title="Stats panel" aria-label="Toggle stats panel"
          className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg border backdrop-blur-md transition active:scale-95 ${
            panelOpen ? 'border-accent/50 bg-[#0c0c0f]/80 text-accent' : 'border-[#27272a] bg-[#0c0c0f]/80 text-zinc-400 hover:border-zinc-600 hover:text-zinc-100'}`}>
          <LineChart size={15} />
        </button>
        <div className="flex flex-1 items-center justify-center gap-3 rounded-full border border-[#27272a] bg-[#0c0c0f]/80 px-4 py-2 font-mono text-[11px] text-zinc-400 backdrop-blur-md">
          <span>MOVE {g.moveNumber}</span>
          <span className="text-zinc-800">|</span>
          <span>{g.turn === 'w' ? 'WHITE' : 'BLACK'}</span>
          {g.inCheck && <span className="text-[#ff8080]">CHECK</span>}
        </div>
      </div>
    </div>
  );
}

export function Hint() {
  const [show, setShow] = useState(true);
  const mode = useGame(s => s.mode);
  const viewMode = useGame(s => s.viewMode);
  useEffect(() => { const t = setTimeout(() => setShow(false), 9000); return () => clearTimeout(t); }, []);
  return (
    <AnimatePresence>
      {show && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ delay: 1.2 }}
          className="pointer-events-none absolute bottom-16 left-4 z-10 hidden font-mono text-[10px] tracking-wider text-zinc-600 md:lg:bottom-4 lg:block">
          {viewMode === '3d' ? 'DRAG — ORBIT · SCROLL — ZOOM · CLICK — MOVE' : 'CLICK — SELECT & MOVE'}
          {' · V — 2D/3D'}{mode === 'replay' ? ' · SPACE — PLAY · ←/→ — STEP' : ''}
        </motion.div>
      )}
    </AnimatePresence>
  );
}