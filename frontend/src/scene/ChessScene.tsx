import { memo, useEffect, useMemo, useRef } from 'react';
import type { ReactElement } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, ContactShadows } from '@react-three/drei';
import { useGame, setHoverIfTarget } from '../state/useChessGame';
import { squareXY, TILE_GEO, TILE_MAT_L, TILE_MAT_D, boardLabelTexture } from '../lib/geometry';
import { wasDrag } from '../lib/input';
import { Pieces } from './Piece3D';

/* ── responsive framing ────────────────────────────────────────────
   The desktop pose (fov 42 @ ~11.5 units) cuts the board's sides on
   portrait screens: horizontal FOV = fov · aspect, and a 390×844 phone
   (aspect 0.46) sees under a third of the board's width. homePose()
   solves  tan(fov/2) · d · aspect ≥ halfBoard  for the distance, uses
   a wider fov + steeper (touch-friendlier) angle in portrait, and
   mirrors z for Black's side. */
const BOARD_HALF = 5.05; // plinth half-width (4.65) + margin

function homePose(orientation: 'w' | 'b', portrait: boolean): { pos: THREE.Vector3; fov: number } {
  const aspect = typeof window !== 'undefined'
    ? window.innerWidth / Math.max(1, window.innerHeight)
    : 1.6;
  if (!portrait) {
    return { pos: new THREE.Vector3(0, 7.6, orientation === 'b' ? -8.8 : 8.8), fov: 42 };
  }
  const fov = 62;
  const d = THREE.MathUtils.clamp(
    BOARD_HALF / (Math.tan((fov * Math.PI) / 360) * Math.max(aspect, 0.42)),
    13.5, 22,
  );
  const pos = new THREE.Vector3(0, 1.42, 1).normalize().multiplyScalar(d);
  if (orientation === 'b') pos.z = -pos.z;
  return { pos, fov };
}

/** Applies the correct home pose on mount and whenever the viewport
    crosses the portrait/landscape boundary. Also keeps OrbitControls'
    zoom limits in sync (portrait legitimately sits farther out). */
function ResponsiveCamera() {
  const size = useThree(s => s.size);
  const camera = useThree(s => s.camera) as THREE.PerspectiveCamera;
  const controls = useThree(s => s.controls) as any;
  const applied = useRef('');
  useEffect(() => {
    const portrait = size.width / size.height < 1;
    const key = (portrait ? 'p' : 'l') + (controls ? '+c' : '');
    if (applied.current === key) return;
    applied.current = key;
    const pose = homePose(useGame.getState().orientation, portrait);
    camera.fov = pose.fov;
    camera.position.copy(pose.pos);
    camera.lookAt(0, 0.2, 0);
    camera.updateProjectionMatrix();
    if (controls) {
      controls.maxDistance = portrait ? 24 : 17;
      controls.target.set(0, 0.2, 0);
      controls.update();
    }
  }, [size, camera, controls]);
  return null;
}

/** Static tile field — memoized so selection changes never re-render it. */
const Tiles = memo(function Tiles({ handler }: { handler: { current: (sq: string) => void } }) {
  const tiles = useMemo(() => {
    const out: ReactElement[] = [];
    for (let f = 0; f < 8; f++) {
      for (let r = 0; r < 8; r++) {
        const sq = 'abcdefgh'[f] + (r + 1);
        const light = (f + r) % 2 === 1;
        out.push(
          <mesh
            key={sq}
            geometry={TILE_GEO}
            material={light ? TILE_MAT_L : TILE_MAT_D}
            position={[f - 3.5, -0.06, 3.5 - r]}
            receiveShadow
            onClick={(e: any) => { e.stopPropagation(); handler.current(sq); }}
            onPointerOver={(e: any) => { e.stopPropagation(); setHoverIfTarget(sq); }}
            onPointerOut={() => setHoverIfTarget(null)}
          />
        );
      }
    }
    return out;
  }, [handler]);
  return <group>{tiles}</group>;
});

/** Procedural studio environment (PMREM from emissive strips — no network). */
function StudioEnv() {
  const gl = useThree(s => s.gl);
  const scene = useThree(s => s.scene);
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const env = new THREE.Scene();
    const created: THREE.Mesh[] = [];
    const strip = (w: number, h: number, rgb: [number, number, number], pos: [number, number, number], rot: [number, number, number]) => {
      const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
      mat.color.setRGB(rgb[0], rgb[1], rgb[2]);
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
      m.position.set(...pos);
      m.rotation.set(...rot);
      env.add(m);
      created.push(m);
    };
    strip(7, 3.2, [3.4, 2.7, 2.0], [0, 6.5, 4], [-1.9, 0, 0]);          // warm overhead key
    strip(9, 2.6, [0.9, 1.15, 1.6], [0, 4.5, -8], [1.85, 0, 0]);        // cool back rim
    strip(12, 12, [0.14, 0.12, 0.10], [0, -4, 0], [Math.PI / 2, 0, 0]); // dim floor bounce
    const rt = pmrem.fromScene(env, 0.05);
    scene.environment = rt.texture;
    pmrem.dispose();
    return () => {
      scene.environment = null;
      rt.dispose();
      created.forEach(m => { m.geometry.dispose(); (m.material as THREE.Material).dispose(); });
    };
  }, [gl, scene]);
  return null;
}

function TileGlow({ square, color, opacity }: { square: string; color: string; opacity: number }) {
  const [x, z] = squareXY(square);
  return (
    <mesh rotation-x={-Math.PI / 2} position={[x, 0.005, z]} renderOrder={1}>
      <planeGeometry args={[0.97, 0.97]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  );
}

function CheckGlow({ square }: { square: string }) {
  const mat = useRef<THREE.MeshBasicMaterial>(null);
  useFrame(({ clock }) => { if (mat.current) mat.current.opacity = 0.32 + 0.16 * Math.sin(clock.elapsedTime * 5.5); });
  const [x, z] = squareXY(square);
  return (
    <mesh rotation-x={-Math.PI / 2} position={[x, 0.007, z]} renderOrder={1}>
      <planeGeometry args={[0.97, 0.97]} />
      <meshBasicMaterial ref={mat} color="#e5484d" transparent opacity={0.4} depthWrite={false} />
    </mesh>
  );
}

function BoardOverlay() {
  const selected = useGame(s => s.selected);
  const targets = useGame(s => s.legalTargets);
  const lastMove = useGame(s => s.lastMove);
  const checkSquare = useGame(s => s.checkSquare);
  const hovered = useGame(s => s.hovered);
  const uniq = useMemo(() => {
    const seen = new Set<string>();
    return targets.filter(m => { if (seen.has(m.to)) return false; seen.add(m.to); return true; });
  }, [targets]);
  return (
    <group>
      {lastMove && lastMove.from !== selected && <TileGlow square={lastMove.from} color="#e0a758" opacity={0.14} />}
      {lastMove && <TileGlow square={lastMove.to} color="#e0a758" opacity={0.2} />}
      {selected && <TileGlow square={selected} color="#e0a758" opacity={0.34} />}
      {uniq.map(m => {
        const [x, z] = squareXY(m.to);
        const hot = hovered === m.to;
        return m.captured ? (
          <mesh key={'c' + m.to} rotation-x={-Math.PI / 2} position={[x, 0.022, z]} renderOrder={2}>
            <ringGeometry args={[0.33, 0.385, 40]} />
            <meshBasicMaterial color={hot ? '#f2c179' : '#e0a758'} transparent opacity={hot ? 0.95 : 0.6} depthWrite={false} />
          </mesh>
        ) : (
          <mesh key={'m' + m.to} position={[x, 0.022, z]} renderOrder={2}>
            <cylinderGeometry args={[hot ? 0.15 : 0.105, hot ? 0.15 : 0.105, 0.018, 20]} />
            <meshBasicMaterial color={hot ? '#f2c179' : '#e0a758'} transparent opacity={hot ? 0.95 : 0.55} depthWrite={false} />
          </mesh>
        );
      })}
      {checkSquare && <CheckGlow square={checkSquare} />}
    </group>
  );
}

function BoardBase() {
  const handler = useRef(useGame.getState().onSquareClick);
  handler.current = useGame.getState().onSquareClick;
  return (
    <group>
      <mesh position={[0, -0.2, 0]} castShadow receiveShadow>
        <boxGeometry args={[9.3, 0.28, 9.3]} />
        <meshStandardMaterial color="#1d1d22" roughness={0.7} metalness={0.08} envMapIntensity={0.2} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.055, 0]} renderOrder={2}>
        <planeGeometry args={[9.3, 9.3]} />
        <meshBasicMaterial map={boardLabelTexture()} transparent depthWrite={false} />
      </mesh>
      {[-1, 1].map(side => (
        <mesh key={side} rotation-x={-Math.PI / 2} position={[side * 5.75, -0.335, 0]} receiveShadow>
          <planeGeometry args={[1.7, 8.6]} />
          <meshStandardMaterial color="#0e0e12" roughness={0.95} />
        </mesh>
      ))}
      <Tiles handler={handler} />
    </group>
  );
}

function Ground() {
  return (
    <mesh rotation-x={-Math.PI / 2} position={[0, -0.34, 0]} receiveShadow>
      <circleGeometry args={[34, 48]} />
      <meshStandardMaterial color="#0b0b0e" roughness={1} metalness={0} />
    </mesh>
  );
}

/** smooth camera reset / 180° flip — spherical interpolation so the
    camera arcs AROUND the board, never over the pole. */
const _rigOff = new THREE.Vector3();
const _rigTgt = new THREE.Vector3();

function CameraRig() {
  const camCmd = useGame(s => s.camCmd);
  const camera = useThree(s => s.camera);
  const controls = useThree(s => s.controls) as any;
  const tw = useRef<{
    t: number;
    s0: THREE.Spherical;
    s1: THREE.Spherical;
    ft: THREE.Vector3;
    tt: THREE.Vector3;
  } | null>(null);
  useEffect(() => {
    if (!camCmd || !controls) return;
    const portrait = window.innerWidth < window.innerHeight;
    const curTarget = controls.target.clone();
    const s0 = new THREE.Spherical().setFromVector3(
      camera.position.clone().sub(curTarget)
    );
    if (s0.phi < 0.12) s0.phi = 0.12;

    let dest: THREE.Vector3;
    if (camCmd.type === 'flip') {
      dest = new THREE.Vector3(-camera.position.x, Math.max(4.5, camera.position.y), -camera.position.z);
    } else {
      const pose = homePose(useGame.getState().orientation, portrait);
      dest = pose.pos.clone();
      // reset also restores the correct fov for the current viewport
      (camera as THREE.PerspectiveCamera).fov = pose.fov;
      (camera as THREE.PerspectiveCamera).updateProjectionMatrix();
    }
    const s1 = new THREE.Spherical().setFromVector3(dest.sub(curTarget));
    if (s1.phi < 0.12) s1.phi = 0.12;

    // shortest arc around the board — never through the pole
    while (s1.theta - s0.theta > Math.PI) s1.theta -= Math.PI * 2;
    while (s1.theta - s0.theta < -Math.PI) s1.theta += Math.PI * 2;

    tw.current = { t: 0, s0, s1, ft: curTarget, tt: new THREE.Vector3(0, 0.2, 0) };
    controls.enabled = false;
  }, [camCmd, camera, controls]);
  useFrame((_, dtRaw) => {
    const a = tw.current;
    if (!a || !controls) return;
    const dt = Math.min(dtRaw, 0.05);
    a.t = Math.min(1, a.t + dt / 0.85);
    const k = a.t < 0.5 ? 4 * a.t * a.t * a.t : 1 - Math.pow(-2 * a.t + 2, 3) / 2;
    const phi = a.s0.phi + (a.s1.phi - a.s0.phi) * k;
    const theta = a.s0.theta + (a.s1.theta - a.s0.theta) * k;
    const radius = a.s0.radius + (a.s1.radius - a.s0.radius) * k;
    _rigTgt.copy(a.ft).lerp(a.tt, k);
    _rigOff.setFromSphericalCoords(radius, phi, theta);
    camera.position.copy(_rigTgt).add(_rigOff);
    camera.lookAt(_rigTgt);
    controls.target.copy(_rigTgt);
    if (a.t >= 1) {
      tw.current = null;
      controls.enabled = true;
      controls.update();
    }
  });
  return null;
}

export function ChessScene() {
  const clearSelection = useGame(s => s.clearSelection);
  const viewMode = useGame(s => s.viewMode);
  return (
    <div className="absolute inset-0">
      <Canvas
        shadows
        dpr={[1, 2]}
        frameloop={viewMode === '3d' ? 'always' : 'never'}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ position: [0, 7.6, 8.8], fov: 42, near: 0.1, far: 80 }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.26;
        }}
        onPointerMissed={() => { if (!wasDrag()) clearSelection(); }}
      >
        <color attach="background" args={['#0a0a0d']} />
        <fog attach="fog" args={['#0a0a0d', 15, 34]} />
        <StudioEnv />
        <hemisphereLight args={['#45454f', '#08080a', 0.75]} />
        <directionalLight
          castShadow position={[6, 10, 5]} intensity={1.35} color="#ffeeda"
          shadow-mapSize-width={2048} shadow-mapSize-height={2048}
          shadow-camera-left={-8} shadow-camera-right={8}
          shadow-camera-top={8} shadow-camera-bottom={-8}
          shadow-camera-near={2} shadow-camera-far={26}
          shadow-bias={-0.0004} shadow-normalBias={0.035}
        />
        <directionalLight position={[0, 4.5, 9]} intensity={0.42} color="#d8dee8" />
        <directionalLight position={[-7, 5, -5]} intensity={0.3} color="#93a1bd" />
        <directionalLight position={[-5, 6, -8]} intensity={0.6} color="#e8d5ae" />
        <BoardBase />
        <BoardOverlay />
        <Pieces />
        <ContactShadows position={[0, 0.012, 0]} scale={11.5} resolution={512} blur={2.6} far={0.85} opacity={0.5} color="#000000" />
        <Ground />
        <OrbitControls
          makeDefault enableDamping dampingFactor={0.08} enablePan={false}
          target={[0, 0.2, 0]} minDistance={5.5}
          /* maxDistance is managed by ResponsiveCamera (portrait needs more room) */
          minPolarAngle={0.32} maxPolarAngle={1.22}
        />
        <CameraRig />
        <ResponsiveCamera />
      </Canvas>
    </div>
  );
}