import type { Gate, Level, Snapshot } from './types';

export const LEVELS: Level[] = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];

export const LEVEL_LABEL: Record<Level, string> = {
  LOW: 'Low',
  MODERATE: 'Moderate',
  HIGH: 'High',
  CRITICAL: 'Critical',
};

export function levelRank(level: string): number {
  return Math.max(0, LEVELS.indexOf(level as Level));
}

/** A gate is live (playing, scored), staged (footage loaded, not started) or
 *  has no feed at all (a routing target on its stated capacity). */
export type GateMode = 'live' | 'staged' | 'nofeed';

export function gateMode(g: Gate, running: boolean): GateMode {
  if (!g.is_monitored) return 'nofeed';
  return running ? 'live' : 'staged';
}

export function levelVar(g: Gate, running: boolean): string {
  return gateMode(g, running) === 'live'
    ? `var(--lv-${g.congestion_level.toLowerCase()})`
    : 'var(--lv-none)';
}

export function naturalSort(ids: string[]): string[] {
  return [...ids].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

/** Gate ids in perimeter order. The snapshot's gate_order is a plain string
 *  sort (G1, G10, G11, G2...), which would scramble the ring past 9 gates. */
export function ringOrder(snap: Snapshot): string[] {
  const ids = Object.keys(snap.gates);
  const vg = snap.venue?.gates;
  if (vg && vg.length) {
    const ordered = [...vg]
      .sort((a, b) => a.position_index - b.position_index)
      .map((g) => g.gate_id)
      .filter((id) => id in snap.gates);
    if (ordered.length === ids.length) return ordered;
  }
  return naturalSort(ids);
}

export function fmtClock(sec: number): string {
  const t = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const p = (n: number) => String(n).padStart(2, '0');
  return h ? `${h}:${p(m)}:${p(s)}` : `${p(m)}:${p(s)}`;
}

export function fmtAge(sec: number): string {
  if (sec < 60) return `${Math.max(0, Math.round(sec))}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${Math.floor(sec / 3600)}h`;
}

export function actionLabel(g: Gate, running: boolean): string {
  const mode = gateMode(g, running);
  if (mode === 'nofeed') return running ? 'Routing target' : 'No footage';
  if (mode === 'staged') return 'Ready';
  switch (g.action) {
    case 'DIVERT': return g.target_gate ? `Divert → ${g.target_gate}` : 'Divert';
    case 'RESTRICT_INFLOW': return 'Restrict inflow';
    case 'MONITOR': return 'Monitor';
    case 'NO_ACTION': return 'Normal';
    case 'HOLD': return 'Hold';
    case 'INITIALIZING': return 'Starting…';
    case 'NO_DATA': return 'No data';
    default: return g.action.replace(/_/g, ' ').toLowerCase();
  }
}

/** Action family, which drives the chip colour. */
export function actionTone(g: Gate, running: boolean): 'divert' | 'restrict' | 'monitor' | 'calm' | 'idle' {
  if (gateMode(g, running) !== 'live') return 'idle';
  if (g.action === 'DIVERT') return 'divert';
  if (g.action === 'RESTRICT_INFLOW' || g.action === 'HOLD') return 'restrict';
  if (g.action === 'MONITOR') return 'monitor';
  return 'calm';
}

export function gateLabel(g: Gate): string {
  return g.name || g.gate_id;
}

export function pct(x: number): string {
  return `${Math.round((x || 0) * 100)}%`;
}
