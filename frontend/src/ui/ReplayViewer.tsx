import { useGame } from '../state/useChessGame';
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from 'lucide-react';

function TBtn({ icon: Icon, title, onClick }: { icon: any; title: string; onClick: () => void }) {
  return (
    <button title={title} onClick={onClick}
      className="grid h-8 w-8 place-items-center rounded-lg border border-[#27272a] text-zinc-400 transition hover:border-zinc-600 hover:text-zinc-100 active:scale-95 sm:h-9 sm:w-9">
      <Icon size={15} />
    </button>
  );
}

export function ReplayControls() {
  const moves = useGame(s => s.moves);
  const viewPly = useGame(s => s.viewPly);
  const playing = useGame(s => s.playing);
  const speed = useGame(s => s.speed);
  const setViewPly = useGame(s => s.setViewPly);
  const setPlaying = useGame(s => s.setPlaying);
  const setSpeed = useGame(s => s.setSpeed);
  const len = moves.length;
  return (
    <div className="space-y-2.5">
      <input
        className="scrub w-full" type="range" min={0} max={len} step={1} value={viewPly}
        onChange={e => { setPlaying(false); setViewPly(+e.target.value); }}
        aria-label="Replay timeline"
      />
      <div className="flex items-center justify-between gap-2">
        <span className="w-14 font-mono text-[11px] tabular-nums text-zinc-500">{viewPly}/{len}</span>
        <div className="flex items-center gap-1">
          <TBtn icon={SkipBack} title="Jump to start (Home)" onClick={() => { setPlaying(false); setViewPly(0); }} />
          <TBtn icon={ChevronLeft} title="Step back (←)" onClick={() => { setPlaying(false); setViewPly(viewPly - 1); }} />
          <button
            title="Play / pause (Space)"
            onClick={() => setPlaying(!playing)}
            className="grid h-8 w-8 place-items-center rounded-lg bg-zinc-100 text-zinc-950 transition hover:bg-white active:scale-95 sm:h-9 sm:w-9"
          >
            {playing ? <Pause size={15} /> : <Play size={15} className="translate-x-[1px]" />}
          </button>
          <TBtn icon={ChevronRight} title="Step forward (→)" onClick={() => { setPlaying(false); setViewPly(viewPly + 1); }} />
          <TBtn icon={SkipForward} title="Jump to end (End)" onClick={() => { setPlaying(false); setViewPly(len); }} />
        </div>
        <div className="flex overflow-hidden rounded-md border border-[#27272a]">
          {[1, 2, 4].map(x => (
            <button
              key={x}
              onClick={() => setSpeed(x as 1 | 2 | 4)}
              className={`px-1.5 py-1 font-mono text-[10px] transition ${speed === x ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
            >
              {x}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}