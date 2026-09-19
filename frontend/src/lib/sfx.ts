/** Web Audio, fully synthesized (zero asset latency). */

class SoundFX {
  private ctx: AudioContext | null = null;
  private noise: AudioBuffer | null = null;
  muted = false;

  private ensure(): AudioContext | null {
    if (!this.ctx) {
      const AC = (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!AC) return null;
      this.ctx = new AC();
      const sr = this.ctx.sampleRate;
      const len = Math.floor(sr * 0.3);
      const buf = this.ctx.createBuffer(1, len, sr);
      const d = buf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      this.noise = buf;
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume();
    return this.ctx;
  }
  private env(t0: number, peak: number, dur: number) {
    const g = this.ctx!.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(peak, t0 + 0.006);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    return g;
  }
  private tone(fa: number, fb: number, dur: number, peak: number, type: OscillatorType, at: number) {
    const ctx = this.ctx!;
    const o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(fa, at);
    o.frequency.exponentialRampToValueAtTime(Math.max(30, fb), at + dur);
    const g = this.env(at, peak, dur);
    o.connect(g).connect(ctx.destination);
    o.start(at); o.stop(at + dur + 0.02);
  }
  private thump(freq: number, dur: number, peak: number, at: number) {
    const ctx = this.ctx!;
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(freq, at);
    o.frequency.exponentialRampToValueAtTime(Math.max(40, freq * 0.45), at + dur);
    const g = this.env(at, peak, dur);
    o.connect(g).connect(ctx.destination);
    o.start(at); o.stop(at + dur + 0.02);
  }
  private noiseBurst(freq: number, q: number, dur: number, peak: number, at: number) {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const f = ctx.createBiquadFilter();
    f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = q;
    const g = this.env(at, peak, dur);
    src.connect(f).connect(g).connect(ctx.destination);
    src.start(at); src.stop(at + dur + 0.02);
  }
  play(kind: 'move' | 'capture' | 'check' | 'select' | 'end' | 'ui') {
    const ctx = this.ensure();
    if (!ctx || this.muted) return;
    const t = ctx.currentTime + 0.01;
    switch (kind) {
      case 'move':
        this.noiseBurst(2600, 1.2, 0.06, 0.12, t);
        this.thump(300, 0.09, 0.12, t);
        break;
      case 'capture':
        this.noiseBurst(900, 0.8, 0.11, 0.3, t);
        this.thump(180, 0.14, 0.22, t);
        this.tone(140, 70, 0.1, 0.08, 'square', t);
        break;
      case 'check':
        this.tone(660, 660, 0.1, 0.09, 'sine', t);
        this.tone(880, 880, 0.12, 0.09, 'sine', t + 0.11);
        break;
      case 'select':
        this.tone(2400, 2000, 0.03, 0.05, 'sine', t);
        break;
      case 'end':
        this.tone(392, 392, 0.4, 0.07, 'triangle', t);
        this.tone(523.25, 523.25, 0.4, 0.07, 'triangle', t + 0.09);
        this.tone(659.25, 659.25, 0.5, 0.07, 'triangle', t + 0.18);
        break;
      case 'ui':
        this.tone(1300, 1100, 0.035, 0.04, 'sine', t);
        break;
    }
  }
}
export const SFX = new SoundFX();