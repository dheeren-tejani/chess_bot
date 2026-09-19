import { useEffect } from 'react';
import { ChessScene } from './scene/ChessScene';
import { Board2D } from './ui/Board2D';
import { TopBar, ActionButtons, SidePanel, MobileBar, MobilePanel, Hint } from './ui/GameHUD';
import { Home } from './ui/Home';
import { PromotionModal, ResignModal, GameOverModal, Toasts } from './ui/Modals';
import { useGame } from './state/useChessGame';

function useKeyboard() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      const s = useGame.getState();
      if (e.key === 'v' || e.key === 'V') { s.toggleView(); return; }
      if (e.key === 'Escape') { s.setResignConfirm(false); return; }
      if (s.mode !== 'replay') return;
      switch (e.key) {
        case ' ': case 'Spacebar': e.preventDefault(); s.setPlaying(!s.playing); break;
        case 'ArrowRight': e.preventDefault(); s.setPlaying(false); s.setViewPly(s.viewPly + 1); break;
        case 'ArrowLeft': e.preventDefault(); s.setPlaying(false); s.setViewPly(s.viewPly - 1); break;
        case 'Home': e.preventDefault(); s.setPlaying(false); s.setViewPly(0); break;
        case 'End': e.preventDefault(); s.setPlaying(false); s.setViewPly(s.moves.length); break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

/** the replay heartbeat — advances one ply per tick at 1x/2x/4x */
function usePlayback() {
  const playing = useGame(s => s.playing);
  const speed = useGame(s => s.speed);
  const mode = useGame(s => s.mode);
  useEffect(() => {
    if (!playing || mode !== 'replay') return;
    const iv = window.setInterval(() => {
      const s = useGame.getState();
      if (s.viewPly >= s.moves.length) { s.setPlaying(false); return; }
      s.setViewPly(s.viewPly + 1);
    }, 1400 / speed);
    return () => window.clearInterval(iv);
  }, [playing, speed, mode]);
}

export default function App() {
  useKeyboard();
  usePlayback();
  const viewMode = useGame(s => s.viewMode);
  const screen = useGame(s => s.screen);
  useEffect(() => {
    const boot = document.getElementById('boot');
    if (boot) {
      boot.style.opacity = '0';
      setTimeout(() => boot.remove(), 750);
    }
  }, []);
  return (
    <div className="fixed inset-0 select-none overflow-hidden bg-[#09090b] font-display text-zinc-200">
      <ChessScene />
      {screen === 'game' && (
        <>
          {viewMode === '2d' && <Board2D />}
          <TopBar />
          <ActionButtons />
          <SidePanel />
          <MobileBar />
          <MobilePanel />
          <Hint />
          <PromotionModal />
          <ResignModal />
          <GameOverModal />
        </>
      )}
      <Home />
      <Toasts />
    </div>
  );
}