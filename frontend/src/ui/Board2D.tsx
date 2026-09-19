import { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useGame } from '../state/useChessGame';
import { buildPiecesAtPly } from '../lib/geometry';
import { PieceGlyph } from '../lib/glyphs';
import { wasDrag } from '../lib/input';
import type { VMove, PieceInstance } from '../types';

export function Board2D() {
  const moves = useGame(s => s.moves);
  const viewPly = useGame(s => s.viewPly);
  const gameKey = useGame(s => s.gameKey);
  const selected = useGame(s => s.selected);
  const targets = useGame(s => s.legalTargets);
  const lastMove = useGame(s => s.lastMove);
  const checkSquare = useGame(s => s.checkSquare);
  const turn = useGame(s => s.turn);
  const orientation = useGame(s => s.orientation);
  const playerColor = useGame(s => s.playerColor);
  const liveMode = useGame(s => s.mode === 'live');
  const thinking = useGame(s => s.thinking);
  const gameOver = useGame(s => s.gameOver);
  const onClick = useGame(s => s.onSquareClick);
  const clearSelection = useGame(s => s.clearSelection);
  const panelOpen = useGame(s => s.panelOpen);

  const pieces = useMemo(() => buildPiecesAtPly(moves, viewPly), [moves, viewPly]);
  const targetMap = useMemo(() => {
    const m = new Map<string, VMove>();
    targets.forEach(t => { if (!m.has(t.to)) m.set(t.to, t); });
    return m;
  }, [targets]);
  const bySquare = useMemo(() => {
    const m = new Map<string, PieceInstance>();
    pieces.forEach(p => { if (p.square) m.set(p.square, p); });
    return m;
  }, [pieces]);
  const interactive = liveMode && !thinking && !gameOver && turn === playerColor;

  // screen cell (col,row) → square, honoring the viewing orientation
  const cellSquare = (col: number, row: number): string => {
    const f = orientation === 'w' ? col : 7 - col;
    const rank = orientation === 'w' ? 8 - row : row + 1;
    return 'abcdefgh'[f] + rank;
  };
  // square → percent offsets, honoring the viewing orientation
  const sqToXY = (sq: string): [number, number] => {
    const f = sq.charCodeAt(0) - 97, r = +sq[1] - 1;
    const col = orientation === 'w' ? f : 7 - f;
    const row = orientation === 'w' ? 7 - r : r;
    return [col * 12.5, row * 12.5];
  };

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.22 }}
      className={`absolute inset-0 z-[5] flex items-center justify-center bg-[#09090b] pb-16 lg:pb-0 ${panelOpen ? 'lg:pr-[318px]' : ''}`}
      onClick={() => { if (!wasDrag()) clearSelection(); }}
    >
      <div
        className="relative aspect-square w-[min(92vw,74vh,620px)] overflow-hidden rounded-md shadow-[0_24px_70px_rgba(0,0,0,0.55)] ring-1 ring-[#27272a]"
        onClick={e => e.stopPropagation()}
      >
        <div className="grid h-full w-full grid-cols-8 grid-rows-[repeat(8,minmax(0,1fr))]">
          {Array.from({ length: 8 }, (_, row) =>
            Array.from({ length: 8 }, (_, col) => {
              const sq = cellSquare(col, row);
              const light = ((sq.charCodeAt(0) - 97) + (+sq[1] - 1)) % 2 === 1;
              const tgt = targetMap.get(sq);
              const pc = bySquare.get(sq);
              const clickable = !!tgt || (interactive && !!pc && pc.color === playerColor);
              const file = sq[0], rank = sq[1];
              return (
                <div
                  key={sq}
                  onClick={() => { if (!wasDrag()) onClick(sq); }}
                  className={`group relative ${clickable ? 'cursor-pointer' : ''}`}
                  style={{ background: light ? '#a49a8a' : '#37373f' }}
                >
                  {lastMove && (lastMove.from === sq || lastMove.to === sq) && (
                    <div className="pointer-events-none absolute inset-0 bg-accent/[0.18]" />
                  )}
                  {sq === selected && <div className="pointer-events-none absolute inset-0 bg-accent/40" />}
                  {sq === checkSquare && <div className="pointer-events-none absolute inset-0 animate-pulse bg-[#e5484d]/40" />}
                  {tgt && !tgt.captured && (
                    <div className="pointer-events-none absolute left-1/2 top-1/2 h-[24%] w-[24%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent/90 transition-transform duration-150 group-hover:scale-[1.35]" />
                  )}
                  {tgt && tgt.captured && (
                    <div className="pointer-events-none absolute inset-[8%] rounded-full border-[3.5px] border-accent/85 transition-transform duration-150 group-hover:scale-[1.08]" />
                  )}
                  {row === 7 && (
                    <span className="pointer-events-none absolute bottom-0 right-[3px] font-mono text-[9px] leading-4" style={{ color: light ? '#37373f' : '#a49a8a' }}>{file}</span>
                  )}
                  {col === 0 && (
                    <span className="pointer-events-none absolute left-[3px] top-0 font-mono text-[9px] leading-4" style={{ color: light ? '#37373f' : '#a49a8a' }}>{rank}</span>
                  )}
                </div>
              );
            })
          )}
        </div>
        <div className="pointer-events-none absolute inset-0">
          <AnimatePresence>
            {pieces.filter(p => p.square).map(p => {
              const [left, top] = sqToXY(p.square as string);
              return (
                <motion.div
                  key={gameKey + ':' + p.id}
                  className="absolute h-[12.5%] w-[12.5%] p-[0.5%]"
                  initial={{ left: left + '%', top: top + '%', opacity: 0, scale: 0.45 }}
                  animate={{ left: left + '%', top: top + '%', opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.35, transition: { duration: 0.16 } }}
                  transition={{
                    left: { duration: 0.22, ease: [0.2, 0.8, 0.2, 1] },
                    top: { duration: 0.22, ease: [0.2, 0.8, 0.2, 1] },
                    default: { duration: 0.2 },
                  }}
                >
                  <PieceGlyph type={p.type} color={p.color} />
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}