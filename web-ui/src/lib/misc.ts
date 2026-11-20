import { useEffect, useState, type RefObject } from 'react';

/* ── per-viewer preferences ───────────────────────────────────────
 * Only conveniences (which stage view, sound on/off). Storage may be blocked,
 * so every access is guarded. */
export function loadPref<T extends string>(key: string, fallback: T, allowed: readonly T[]): T {
  try {
    const v = localStorage.getItem(`crowdguard.${key}`);
    return v !== null && (allowed as readonly string[]).includes(v) ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}

export function savePref(key: string, value: string): void {
  try { localStorage.setItem(`crowdguard.${key}`, value); } catch { /* ignore */ }
}

/* ── element size ─────────────────────────────────────────────── */
export function useSize(ref: RefObject<HTMLElement>) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize((s) => (s.width === width && s.height === height ? s : { width, height }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

/* ── wall clock ───────────────────────────────────────────────── */
export function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return now;
}

/* ── alert chime ──────────────────────────────────────────────────
 * A short two-tone chime for EMERGENCY alerts. Browsers allow audio only
 * after the user has interacted with the page, so failure is silent. */
let audio: AudioContext | null = null;

export function chime(): void {
  try {
    audio ??= new AudioContext();
    const ctx = audio;
    if (ctx.state === 'suspended') void ctx.resume();
    const now = ctx.currentTime;
    [880, 660].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      const t = now + i * 0.18;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.18, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 0.18);
    });
  } catch {
    /* audio unavailable */
  }
}
