import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useGame } from '../state/useChessGame';
import {
  squarePos, trayPos, spawnDelay, buildPiecesAtPly,
  getParts, MAT_PIECE_W, MAT_PIECE_B, MAT_EYE_W, MAT_EYE_B,
} from '../lib/geometry';
import { wasDrag } from '../lib/input';
import type { PieceInstance } from '../types';

const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
const backOut = (t: number) => { const c = 1.70158, u = t - 1; return 1 + (c + 1) * u * u * u + c * u * u; };

function PieceModel({ type, color }: { type: string; color: 'w' | 'b' }) {
  const parts = getParts(type);
  const body = color === 'w' ? MAT_PIECE_W : MAT_PIECE_B;
  const eye = color === 'w' ? MAT_EYE_W : MAT_EYE_B;
  const rot = type === 'n' ? (color === 'w' ? Math.PI / 2 : -Math.PI / 2) : 0;
  return (
    // 0.004 lift keeps the lathe base from sitting coplanar with the tile
    // top (which caused z-fighting shimmer around every piece base).
    <group rotation-y={rot} position-y={0.004}>
      {parts.map((pt, i) => (
        <mesh
          key={i}
          geometry={pt.geo}
          material={pt.mat === 'eye' ? eye : body}
          position={pt.pos ?? [0, 0, 0]}
          rotation={pt.rot ?? [0, 0, 0]}
          castShadow
        />
      ))}
    </group>
  );
}

interface Piece3DProps {
  inst: PieceInstance;
  animDur: number;
  selected: boolean;
  captureTarget: boolean;
  onClickSquare: (sq: string) => void;
}

function Piece3D({ inst, animDur, selected, captureTarget, onClickSquare }: Piece3DProps) {
  const grp = useRef<THREE.Group>(null);
  const first = useRef(true);
  const lift = useRef(0);
  const trayScale = useRef(1);
  const spawn = useRef<{ start: number; done: boolean } | null>(null);
  const anim = useRef({ t: 1, dur: 0.45, from: new THREE.Vector3(), to: new THREE.Vector3(), arc: 0 });
  const [hovered, setHovered] = useState(false);

  const target: [number, number, number] = inst.square ? squarePos(inst.square) : trayPos(inst.color, inst.trayIndex);
  const targetKey = inst.square ?? 'cap' + inst.trayIndex;

  // Retarget the tween whenever this piece's logical destination changes.
  // The mesh itself is never remounted — that is the whole point.
  // On first mount we must seed the tween ENDPOINTS (from AND to), not just
  // the mesh position — useFrame recomposes the position from from/to every
  // single frame, so unseeded (0,0,0) vectors would drag every piece back to
  // the world origin. useLayoutEffect guarantees this runs before the first
  // painted frame.
  useLayoutEffect(() => {
    const g = grp.current;
    if (!g) return;
    const a = anim.current;
    if (first.current) {
      a.from.set(target[0], target[1], target[2]);
      a.to.set(target[0], target[1], target[2]);
      g.position.set(target[0], target[1], target[2]);
      a.t = 1;
      first.current = false;
      return;
    }
    a.from.copy(g.position);
    a.from.y -= lift.current; // strip the hover lift from the tween baseline
    a.to.set(target[0], target[1], target[2]);
    const dist = a.from.distanceTo(a.to);
    a.dur = Math.max(0.14, animDur);
    a.arc = inst.square ? Math.min(0.5, 0.14 + dist * 0.05) : 0.45;
    a.t = 0;
  }, [targetKey]);

  useEffect(() => () => { document.body.style.cursor = ''; }, []);

  useFrame((_, dtRaw) => {
    const g = grp.current;
    if (!g) return;
    const dt = Math.min(dtRaw, 0.05);
    const a = anim.current;
    if (a.t < 1) a.t = Math.min(1, a.t + dt / a.dur);
    const e = easeOutCubic(a.t);
    // Position is recomposed ABSOLUTELY from from/to each frame — no
    // incremental drift, and a piece can never get stuck between squares.
    g.position.set(
      a.from.x + (a.to.x - a.from.x) * e,
      a.from.y + (a.to.y - a.from.y) * e + Math.sin(Math.PI * a.t) * a.arc,
      a.from.z + (a.to.z - a.from.z) * e,
    );

    const wantLift = (hovered ? 0.05 : 0) + (selected ? 0.09 : 0);
    lift.current += (wantLift - lift.current) * Math.min(1, dt * 12);
    g.position.y += lift.current;

    // spawn pop, timed from the FIRST rendered frame
    if (!spawn.current) spawn.current = { start: performance.now() + spawnDelay(inst.id), done: false };
    const sc = spawn.current;
    let spawnS = 1;
    if (!sc.done) {
      const st = (performance.now() - sc.start) / 560;
      if (st <= 0) spawnS = 0.0001;
      else if (st >= 1) sc.done = true;
      else spawnS = backOut(st);
    }
    const wantTray = inst.square ? 1 : 0.72;
    trayScale.current += (wantTray - trayScale.current) * Math.min(1, dt * 8);
    g.scale.setScalar(spawnS * trayScale.current);
  });

  return (
    <group
      ref={grp}
      onPointerOver={(e: any) => {
        e.stopPropagation();
        if (inst.square) { setHovered(true); document.body.style.cursor = 'pointer'; }
      }}
      onPointerOut={() => { setHovered(false); document.body.style.cursor = ''; }}
      onClick={(e: any) => {
        e.stopPropagation();
        if (inst.square && !wasDrag()) onClickSquare(inst.square);
      }}
    >
      {(selected || captureTarget) && (
        // counter-offset so the ring stays on the square while the piece lifts
        <mesh rotation-x={-Math.PI / 2} position={[0, selected ? -0.062 : 0.024, 0]} renderOrder={2}>
          <ringGeometry args={[0.3, 0.36, 36]} />
          <meshBasicMaterial color={captureTarget ? '#e5484d' : '#e0a758'} transparent opacity={0.85} depthWrite={false} />
        </mesh>
      )}
      <PieceModel type={inst.type} color={inst.color} />
    </group>
  );
}

export function Pieces() {
  const moves = useGame(s => s.moves);
  const viewPly = useGame(s => s.viewPly);
  const gameKey = useGame(s => s.gameKey);
  const animDur = useGame(s => s.animDur);
  const selected = useGame(s => s.selected);
  const targets = useGame(s => s.legalTargets);
  const onClick = useGame(s => s.onSquareClick);
  // keys include gameKey → remount ONLY on game load, never on a move
  const pieces = useMemo(() => buildPiecesAtPly(moves, viewPly), [moves, viewPly]);
  const capSet = useMemo(() => new Set(targets.filter(m => m.captured).map(m => m.to)), [targets]);
  return (
    <group>
      {pieces.map(p => (
        <Piece3D
          key={gameKey + ':' + p.id}
          inst={p}
          animDur={animDur}
          selected={!!p.square && p.square === selected}
          captureTarget={!!p.square && capSet.has(p.square)}
          onClickSquare={onClick}
        />
      ))}
    </group>
  );
}