import * as THREE from 'three';
import type { PieceInstance, VMove } from '../types';

/** world XZ of a square; rank 1 sits at +Z (the default camera side). */
export function squareXY(sq: string): [number, number] {
  const f = sq.charCodeAt(0) - 97, r = +sq[1] - 1;
  return [f - 3.5, 3.5 - r];
}
export function squarePos(sq: string): [number, number, number] {
  const [x, z] = squareXY(sq);
  return [x, 0, z];
}
/** capture trays: black casualties right, white casualties left */
export function trayPos(color: 'w' | 'b', idx: number): [number, number, number] {
  const col = idx % 2, row = Math.floor(idx / 2);
  const x = (color === 'b' ? 5.35 : -5.35) + (color === 'b' ? col : -col) * 0.78;
  const z = (color === 'b' ? -1 : 1) * (3.4 - row * 0.78);
  return [x, -0.34, z];
}

export function spawnDelay(id: string): number {
  const f = id.charCodeAt(2) - 97;
  const r = +id[3] - 1;
  return f * 0.05 + (r === 1 ? 0 : 0.25); // pawns pop first, back rank follows
}

/** Pure derivation of the 32 piece instances at any ply.
    IDs are origin squares → identical React keys at every ply,
    so meshes NEVER remount while moving or scrubbing. */
export function buildPiecesAtPly(moves: VMove[], ply: number): PieceInstance[] {
  const pieces: PieceInstance[] = [];
  const back = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'];
  for (let c = 0; c < 2; c++) {
    const color: 'w' | 'b' = c === 0 ? 'w' : 'b';
    for (let f = 0; f < 8; f++) {
      const file = 'abcdefgh'[f];
      pieces.push({ id: color + back[f] + file + (c === 0 ? 1 : 8), type: back[f], color, square: file + (c === 0 ? 1 : 8), trayIndex: -1 });
      pieces.push({ id: color + 'p' + file + (c === 0 ? 2 : 7), type: 'p', color, square: file + (c === 0 ? 2 : 7), trayIndex: -1 });
    }
  }
  let capW = 0, capB = 0;
  for (let i = 0; i < ply && i < moves.length; i++) {
    const v = moves[i];
    const mover = pieces.find(p => p.square === v.from && p.color === v.color);
    if (!mover) continue;
    if (v.captured) {
      const capSq = v.flags.includes('e') ? v.to[0] + v.from[1] : v.to;
      const victim = pieces.find(p => p.square === capSq && p.color !== v.color);
      if (victim) {
        victim.square = null;
        victim.trayIndex = victim.color === 'w' ? capW++ : capB++;
      }
    }
    mover.square = v.to;
    if (v.promotion) mover.type = v.promotion;
    if (v.flags.includes('k') || v.flags.includes('q')) { // castling: move the rook too
      const rank = v.color === 'w' ? '1' : '8';
      const kingside = v.flags.includes('k');
      const rook = pieces.find(p => p.square === (kingside ? 'h' : 'a') + rank && p.color === v.color);
      if (rook) rook.square = (kingside ? 'f' : 'd') + rank;
    }
  }
  return pieces;
}

/* ── carved (lathe) Staunton profiles: [radius, height] ────────────── */

const PAWN_PROFILE: number[][] = [
  [0,0],[0.235,0],[0.235,0.02],[0.213,0.032],
  [0.182,0.048],[0.152,0.07],[0.148,0.09],
  [0.176,0.108],[0.192,0.128],[0.192,0.152],[0.176,0.172],   // base molding
  [0.138,0.196],[0.122,0.214],
  [0.114,0.30],[0.106,0.355],
  [0.128,0.378],[0.128,0.402],[0.104,0.428],                 // collar ring
  [0.078,0.455],
  [0.146,0.508],[0.156,0.545],[0.146,0.578],[0.118,0.606],[0.066,0.628],[0,0.648], // head
];
const ROOK_PROFILE: number[][] = [
  [0,0],[0.27,0],[0.27,0.02],[0.248,0.034],
  [0.212,0.05],[0.183,0.076],[0.178,0.094],
  [0.208,0.112],[0.224,0.134],[0.224,0.166],[0.208,0.188],   // base molding
  [0.162,0.212],[0.148,0.238],
  [0.142,0.30],[0.134,0.42],[0.128,0.54],[0.124,0.60],       // tapered stem
  [0.152,0.622],[0.184,0.646],[0.188,0.672],                 // cornice flare
  [0.188,0.740],[0.118,0.740],[0.118,0.700],[0,0.700],       // hollow parapet + inner floor
];
const BISHOP_PROFILE: number[][] = [
  [0,0],[0.25,0],[0.25,0.02],[0.228,0.034],
  [0.192,0.05],[0.163,0.076],[0.158,0.092],
  [0.188,0.11],[0.204,0.132],[0.204,0.162],[0.188,0.184],    // base molding
  [0.146,0.208],[0.132,0.232],
  [0.108,0.28],[0.094,0.34],[0.090,0.40],
  [0.112,0.418],[0.114,0.442],[0.096,0.462],                 // collar ring under mitre
  [0.138,0.508],[0.160,0.575],[0.164,0.632],
  [0.150,0.652],[0.146,0.668],[0.150,0.684],[0.160,0.700],   // carved slit groove
  [0.148,0.752],[0.112,0.800],[0.068,0.836],
  [0.048,0.846],[0.058,0.868],[0.034,0.886],[0.014,0.896],[0,0.902], // mitre + finial ball
];
/* QUEEN: deliberately much wider crown than the bishop — the flare reaches
   r=0.222 (bishop peaks at 0.164) and carries a spiked coronet with pearl
   tips, so the two silhouettes can't be confused at any angle. */
const QUEEN_PROFILE: number[][] = [
  [0,0],[0.29,0],[0.29,0.02],[0.267,0.034],
  [0.226,0.05],[0.191,0.078],[0.186,0.094],
  [0.215,0.112],[0.233,0.134],[0.233,0.166],[0.217,0.188],   // base molding
  [0.170,0.212],[0.152,0.240],
  [0.124,0.31],[0.106,0.43],[0.098,0.55],[0.094,0.61],       // long stem
  [0.120,0.638],[0.122,0.664],[0.100,0.682],                 // collar ring
  [0.158,0.714],[0.196,0.768],[0.216,0.832],[0.222,0.884],   // wide flared crown
  [0.198,0.925],[0.150,0.958],[0.088,0.975],[0,0.978],       // dome
];
const KING_PROFILE: number[][] = [
  [0,0],[0.30,0],[0.30,0.02],[0.276,0.034],
  [0.233,0.05],[0.196,0.078],[0.191,0.094],
  [0.222,0.112],[0.240,0.134],[0.240,0.166],[0.224,0.188],   // base molding
  [0.174,0.212],[0.156,0.240],
  [0.128,0.32],[0.110,0.46],[0.102,0.60],[0.098,0.68],
  [0.124,0.704],[0.126,0.730],[0.106,0.752],                 // collar ring
  [0.152,0.782],[0.174,0.85],[0.178,0.91],[0.164,0.962],[0.128,0.99],[0.086,1.006],[0.046,1.014],[0,1.016], // domed crown
];
const KNIGHT_BASE_PROFILE: number[][] = [
  [0,0],[0.265,0],[0.265,0.02],[0.243,0.034],
  [0.208,0.05],[0.178,0.076],[0.173,0.092],
  [0.203,0.11],[0.220,0.132],[0.220,0.164],[0.204,0.186],    // base molding
  [0.158,0.21],[0.144,0.238],[0.136,0.268],[0,0.268],
];

/** the knight head: extruded, beveled silhouette with serrated mane,
    ears, brow, nostril and jawline. */
function knightHeadGeo(): THREE.BufferGeometry {
  const s = new THREE.Shape();
  s.moveTo(-0.175, 0);
  s.quadraticCurveTo(-0.23, 0.10, -0.225, 0.20);            // lower back
  s.lineTo(-0.246, 0.245);                                  // serrated mane crest
  s.quadraticCurveTo(-0.228, 0.275, -0.246, 0.315);
  s.quadraticCurveTo(-0.226, 0.345, -0.242, 0.385);
  s.quadraticCurveTo(-0.218, 0.415, -0.228, 0.452);
  s.quadraticCurveTo(-0.20, 0.492, -0.188, 0.532);
  s.quadraticCurveTo(-0.172, 0.568, -0.138, 0.588);         // nape
  s.lineTo(-0.122, 0.675);                                  // back ear
  s.lineTo(-0.072, 0.598);
  s.lineTo(-0.048, 0.668);                                  // front ear
  s.quadraticCurveTo(-0.008, 0.638, 0.028, 0.602);          // brow
  s.quadraticCurveTo(0.098, 0.558, 0.158, 0.502);           // forehead
  s.quadraticCurveTo(0.238, 0.458, 0.288, 0.418);           // nose bridge
  s.quadraticCurveTo(0.322, 0.382, 0.312, 0.348);           // nose tip
  s.quadraticCurveTo(0.298, 0.318, 0.258, 0.318);           // muzzle underside
  s.quadraticCurveTo(0.232, 0.292, 0.198, 0.302);           // mouth notch
  s.quadraticCurveTo(0.158, 0.318, 0.122, 0.328);           // jaw
  s.quadraticCurveTo(0.152, 0.244, 0.163, 0.188);           // throat
  s.quadraticCurveTo(0.183, 0.098, 0.183, 0);               // chest
  s.closePath();
  const geo = new THREE.ExtrudeGeometry(s, {
    depth: 0.15, bevelEnabled: true, bevelThickness: 0.03, bevelSize: 0.028, bevelSegments: 3, curveSegments: 14,
  });
  geo.translate(0, 0, -0.075);
  return geo;
}

/** tilt a coronet spike outward, as an Euler triple */
function coronetTilt(a: number, t: number): [number, number, number] {
  const q = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(-Math.sin(a), 0, Math.cos(a)), t,
  );
  const e = new THREE.Euler().setFromQuaternion(q);
  return [e.x, e.y, e.z];
}

/* queen coronet geometry */
const SPIKE_H = 0.125, SPIKE_T = 0.34, SPIKE_R = 0.20, SPIKE_Y = 0.862;
function spikePos(a: number, along: number): [number, number, number] {
  const out = Math.sin(SPIKE_T) * along;
  const up = Math.cos(SPIKE_T) * along;
  return [Math.cos(a) * (SPIKE_R + out), SPIKE_Y + up, Math.sin(a) * (SPIKE_R + out)];
}

export interface PartDef {
  geo: THREE.BufferGeometry;
  pos?: [number, number, number];
  rot?: [number, number, number];
  mat?: 'eye';
}

let PARTS: Record<string, PartDef[]> | null = null;
export function getParts(type: string): PartDef[] {
  if (!PARTS) {
    // helpers typed as BufferGeometry so every part kind (lathe / box /
    // cylinder / cone / sphere) unifies into PartDef.geo
    const lathe = (pts: number[][], seg = 64): THREE.BufferGeometry =>
      new THREE.LatheGeometry(pts.map(p => new THREE.Vector2(p[0], p[1])), seg);
    const box = (w: number, h: number, d: number): THREE.BufferGeometry =>
      new THREE.BoxGeometry(w, h, d);
    const merlon: THREE.BufferGeometry = new THREE.CylinderGeometry(0.086, 0.105, 0.11, 4, 1);
    const eyeGeo: THREE.BufferGeometry = new THREE.SphereGeometry(0.016, 10, 8);
    const spikeGeo: THREE.BufferGeometry = new THREE.ConeGeometry(0.046, SPIKE_H, 12);
    const tipGeo: THREE.BufferGeometry = new THREE.SphereGeometry(0.027, 10, 8);

    const parts: Record<string, PartDef[]> = {
      p: [{ geo: lathe(PAWN_PROFILE) }],
      b: [{ geo: lathe(BISHOP_PROFILE) }],
      r: [
        { geo: lathe(ROOK_PROFILE) },
        ...[0, 1, 2, 3].map(i => {
          const a = (i * Math.PI) / 2;
          return {
            geo: merlon,
            pos: [Math.cos(a) * 0.152, 0.795, Math.sin(a) * 0.152] as [number, number, number],
            rot: [0, Math.PI / 4, 0] as [number, number, number],
          };
        }),
      ],
      n: [
        { geo: lathe(KNIGHT_BASE_PROFILE) },
        { geo: knightHeadGeo(), pos: [0, 0.262, 0] },
        { geo: eyeGeo, pos: [0.07, 0.782, 0.097], mat: 'eye' as const },
        { geo: eyeGeo, pos: [0.07, 0.782, -0.097], mat: 'eye' as const },
      ],
      q: [
        { geo: lathe(QUEEN_PROFILE) },
        ...Array.from({ length: 8 }, (_, i) => {
          const a = (i * Math.PI) / 4;
          return { geo: spikeGeo, pos: spikePos(a, SPIKE_H / 2), rot: coronetTilt(a, SPIKE_T) };
        }),
        ...Array.from({ length: 8 }, (_, i) => {
          const a = (i * Math.PI) / 4;
          return { geo: tipGeo, pos: spikePos(a, SPIKE_H) };
        }),
      ],
      k: [
        { geo: lathe(KING_PROFILE) },
        { geo: new THREE.SphereGeometry(0.034, 12, 10), pos: [0, 1.02, 0] },
        { geo: box(0.052, 0.20, 0.052), pos: [0, 1.11, 0] },
        { geo: box(0.16, 0.052, 0.052), pos: [0, 1.142, 0] },
      ],
    };
    PARTS = parts;
  }
  return PARTS[type] ?? PARTS.p;
}

/* ── materials ──
   Black is a readable espresso (#46362a) with high clearcoat + env response;
   combined with the front fill + warm rim in ChessScene it stays legible
   from every angle without washing out the dark theme. */
export const MAT_PIECE_W = new THREE.MeshPhysicalMaterial({
  color: '#ede6d6', roughness: 0.33, metalness: 0.02,
  clearcoat: 0.55, clearcoatRoughness: 0.28, envMapIntensity: 0.55,
});
export const MAT_PIECE_B = new THREE.MeshPhysicalMaterial({
  color: '#46362a', roughness: 0.36, metalness: 0.05,
  clearcoat: 0.7, clearcoatRoughness: 0.22, envMapIntensity: 1.05,
});
export const MAT_EYE_W = new THREE.MeshStandardMaterial({ color: '#241c15', roughness: 0.45 });
export const MAT_EYE_B = new THREE.MeshStandardMaterial({ color: '#b9a488', roughness: 0.5 });

export const TILE_GEO = new THREE.BoxGeometry(1, 0.12, 1);
export const TILE_MAT_L = new THREE.MeshStandardMaterial({ color: '#a49a8a', roughness: 0.6, metalness: 0.04, envMapIntensity: 0.28 });
export const TILE_MAT_D = new THREE.MeshStandardMaterial({ color: '#37373f', roughness: 0.62, metalness: 0.04, envMapIntensity: 0.28 });

let _labelTex: THREE.CanvasTexture | null = null;
/** coordinate ring (a-h / 1-8) drawn onto the board plinth via canvas */
export function boardLabelTexture(): THREE.CanvasTexture {
  if (_labelTex) return _labelTex;
  const S = 1024;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const ctx = cv.getContext('2d')!;
  const tile = S / 9.3, border = (S - tile * 8) / 2;
  const draw = () => {
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = '#5d5d68';
    ctx.font = '600 ' + Math.round(tile * 0.34) + 'px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let f = 0; f < 8; f++) {
      const x = border + (f + 0.5) * tile;
      ctx.fillText('abcdefgh'[f], x, S - border / 2);
      ctx.fillText('abcdefgh'[f], x, border / 2);
    }
    for (let r = 1; r <= 8; r++) {
      const y = S - (r + 0.15) * tile;
      ctx.fillText(String(r), border / 2, y);
      ctx.fillText(String(r), S - border / 2, y);
    }
  };
  draw();
  _labelTex = new THREE.CanvasTexture(cv);
  _labelTex.anisotropy = 4;
  if ((document as any).fonts?.ready) {
    document.fonts.ready.then(() => { draw(); _labelTex!.needsUpdate = true; });
  }
  return _labelTex;
}