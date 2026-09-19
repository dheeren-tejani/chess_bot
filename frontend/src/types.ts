export type Square = string;

/** Verbose chess.js move — the single source of truth for all rendering. */
export interface VMove {
  from: Square;
  to: Square;
  piece: string;
  color: 'w' | 'b';
  captured?: string;
  promotion?: string;
  flags: string;
  san: string;
}

/** A piece with a persistent identity: survives moves, captures, scrubs. */
export interface PieceInstance {
  id: string;            // e.g. "wpe2" — origin square, stable for the whole game
  type: string;          // p n b r q k (changes only on promotion)
  color: 'w' | 'b';
  square: Square | null; // null while it lives on a capture tray
  trayIndex: number;
}

export interface GamePayload {
  moves: string[];        // SAN list
  result: string;         // "1-0" | "0-1" | "1/2-1/2"
  evaluations: number[];  // entry i = centipawn eval (White-positive) after ply i+1
  player_color?: 'white' | 'black';
}

export interface GameOverInfo {
  result: string;
  reason: string;
  winner: 'w' | 'b' | null;
}