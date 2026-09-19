let dgX = 0, dgY = 0, dgDragged = false;
if (typeof window !== 'undefined') {
  window.addEventListener('pointerdown', (e) => { dgX = e.clientX; dgY = e.clientY; dgDragged = false; }, true);
  window.addEventListener('pointermove', (e) => {
    if (Math.abs(e.clientX - dgX) + Math.abs(e.clientY - dgY) > 8) dgDragged = true;
  }, true);
}

/** True when the current pointer sequence moved more than a few px —
    used to distinguish board clicks from orbit-drag releases. */
export function wasDrag(): boolean { return dgDragged; }