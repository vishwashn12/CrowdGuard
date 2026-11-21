/* The venue as the operator thinks of it: gates on a perimeter ring, coloured
 * by congestion, with every active diversion drawn as an animated arc that
 * runs around the outer concourse - the way the crowd would actually walk.
 *
 * Everything is in one 1000x1000 viewBox and scales to the stage. */
import { memo } from 'react';
import type { Gate, Snapshot } from '../lib/types';
import { gateMode, levelVar, ringAngles, segmentLengths } from '../lib/format';

const C = 500;      // centre
const R = 318;      // gate ring radius
const TAU = Math.PI * 2;

const mod = (a: number) => ((a % TAU) + TAU) % TAU;
const pt = (a: number, r: number): [number, number] => [C + r * Math.cos(a), C + r * Math.sin(a)];
const f = (v: number) => v.toFixed(1);

interface Arc {
  key: string;
  d: string;
  labelAt: [number, number];
  label: string;
}

/** Direction the route walks around the ring: +1 clockwise, -1 counter.
 *  Taken from the engine's route; otherwise the shorter way in metres (angles
 *  are proportional to walking distance). */
function routeDirection(src: number, dst: number, n: number, route: string[], order: string[], angles: number[]): 1 | -1 {
  if (route.length >= 2) {
    const next = order.indexOf(route[1]);
    if (next === (src + 1) % n) return 1;
    if (next === (src - 1 + n) % n) return -1;
  }
  return mod(angles[dst] - angles[src]) <= Math.PI ? 1 : -1;
}

function diversionArc(
  g: Gate, order: string[], angles: number[], nodeR: number, lane: number,
): Arc | null {
  const n = order.length;
  const si = order.indexOf(g.gate_id);
  const ti = g.target_gate ? order.indexOf(g.target_gate) : -1;
  if (si < 0 || ti < 0 || si === ti) return null;

  const dir = routeDirection(si, ti, n, g.route || [], order, angles);
  const a0 = angles[si];
  const span = dir > 0 ? mod(angles[ti] - a0) : -mod(a0 - angles[ti]);
  const a1 = a0 + span;

  const rArc = R + nodeR + 44 + lane * 26;
  const rOut = R + nodeR + 8;
  const [x0, y0] = pt(a0, rOut);
  const [x1, y1] = pt(a0, rArc);
  const [x2, y2] = pt(a1, rArc);
  const [x3, y3] = pt(a1, rOut + 6);
  const large = Math.abs(span) > Math.PI ? 1 : 0;
  const sweep = dir > 0 ? 1 : 0;

  return {
    key: `${g.gate_id}-${g.target_gate}`,
    d: `M ${f(x0)} ${f(y0)} L ${f(x1)} ${f(y1)} A ${rArc} ${rArc} 0 ${large} ${sweep} ${f(x2)} ${f(y2)} L ${f(x3)} ${f(y3)}`,
    labelAt: pt(a0 + span / 2, rArc + 30),
    label: `${g.gate_id} → ${g.target_gate}${g.command?.share != null && g.command.share < 1 ? ` ${Math.round(g.command.share * 100)}%` : ''}`,
  };
}

interface Props {
  snap: Snapshot;
  order: string[];
  selected: string | null;
  onSelect: (id: string) => void;
}

function VenueMapImpl({ snap, order, selected, onSelect }: Props) {
  const running = snap.session.state === 'running';
  const n = order.length;
  const nodeR = n <= 6 ? 38 : n <= 10 ? 32 : n <= 16 ? 25 : 20;
  const showNames = n <= 10;

  const gates = order.map((id) => snap.gates[id]).filter(Boolean);
  const segs = segmentLengths(snap);
  const angles = ringAngles(n, segs);
  const live = gates.filter((g) => gateMode(g, running) === 'live');
  const totalPeople = live.reduce((s, g) => s + (g.crowd_count || 0), 0);
  const diverting = running ? gates.filter((g) => g.action === 'DIVERT' && g.target_gate) : [];
  const arcs = diverting
    .map((g, i) => diversionArc(g, order, angles, nodeR, i % 3))
    .filter((a): a is Arc => a !== null);
  const anomalies = live.reduce((s, g) => s + (g.anomaly_count || 0), 0);

  return (
    <svg className="venue-map" viewBox="30 30 940 940" preserveAspectRatio="xMidYMid meet" role="img"
      aria-label="Venue map: gates on the perimeter ring, coloured by congestion">
      <defs>
        <radialGradient id="vm-field" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--field-1)" />
          <stop offset="100%" stopColor="var(--field-2)" />
        </radialGradient>
        {['low', 'moderate', 'high', 'critical'].map((lv) => (
          <radialGradient key={lv} id={`vm-glow-${lv}`}>
            <stop offset="0%" stopColor={`var(--lv-${lv})`} stopOpacity={0.55} />
            <stop offset="100%" stopColor={`var(--lv-${lv})`} stopOpacity={0} />
          </radialGradient>
        ))}
        <marker id="vm-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--accent)" />
        </marker>
      </defs>

      {/* field and concourse */}
      <circle cx={C} cy={C} r={R - nodeR - 70} fill="url(#vm-field)" stroke="var(--line)" strokeWidth={1.5} />
      <circle cx={C} cy={C} r={R - nodeR - 110} fill="none" stroke="var(--line-soft)" strokeDasharray="4 10" />
      <line x1={C - 60} x2={C + 60} y1={C} y2={C} className="vm-cross" />
      <line y1={C - 60} y2={C + 60} x1={C} x2={C} className="vm-cross" />
      <circle cx={C} cy={C} r={R} fill="none" stroke="var(--ring-band)" strokeWidth={nodeR * 1.5} />
      <circle cx={C} cy={C} r={R + nodeR * 0.75} fill="none" stroke="var(--line)" strokeWidth={1} />
      <circle cx={C} cy={C} r={R - nodeR * 0.75} fill="none" stroke="var(--line)" strokeWidth={1} />
      {Array.from({ length: 120 }, (_, i) => {
        const a = (i / 120) * TAU;
        const long = i % 10 === 0;
        const [x1, y1] = pt(a, R + nodeR * 0.75 + 6);
        const [x2, y2] = pt(a, R + nodeR * 0.75 + (long ? 18 : 11));
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className={long ? 'vm-tick vm-tick-l' : 'vm-tick'} />;
      })}

      {/* walking distance along each segment of the perimeter */}
      {segs && n <= 8 && segs.map((d, i) => {
        const a = angles[i] + mod(angles[(i + 1) % n] - angles[i]) / 2;
        const [x, y] = pt(a, R + nodeR * 0.75 + 34);
        return <text key={`seg-${i}`} x={x} y={y + 6} className="vm-seg">{Math.round(d)} m</text>;
      })}

      {/* centre readout */}
      <g className="vm-center">
        <text x={C} y={C - 46} className="vm-kicker">{running ? 'PEOPLE IN VIEW' : snap.session.gates_with_footage ? 'READY TO START' : 'SETUP'}</text>
        <text x={C} y={C + 22} className="vm-big">{running ? Math.round(totalPeople).toLocaleString() : `${snap.session.gates_with_footage}/${n}`}</text>
        <text x={C} y={C + 62} className="vm-sub">
          {running ? `${live.length} of ${n} gates live` : 'gates with footage'}
        </text>
        {running && (
          <g transform={`translate(${C}, ${C + 112})`}>
            <text x={-70} y={0} className="vm-stat" data-on={arcs.length > 0}>{arcs.length}</text>
            <text x={-70} y={24} className="vm-stat-l">diverting</text>
            <text x={70} y={0} className="vm-stat vm-stat-anom" data-on={anomalies > 0}>{anomalies}</text>
            <text x={70} y={24} className="vm-stat-l">anomalies</text>
          </g>
        )}
      </g>

      {/* heat glow under busy gates */}
      {gates.map((g, i) => {
        if (gateMode(g, running) !== 'live' || g.congestion_level === 'LOW') return null;
        const [x, y] = pt(angles[i], R);
        const lv = g.congestion_level.toLowerCase();
        return (
          <circle key={`glow-${g.gate_id}`} cx={x} cy={y} r={nodeR * 3.1} fill={`url(#vm-glow-${lv})`}
            className={g.congestion_level === 'CRITICAL' ? 'vm-glow vm-glow-pulse' : 'vm-glow'} />
        );
      })}

      {/* diversions */}
      {arcs.map((arc, i) => (
        <g key={arc.key} className="vm-arc">
          <path d={arc.d} className="vm-arc-under" />
          <path d={arc.d} className="vm-arc-flow" markerEnd="url(#vm-arrow)" />
          {[0, 1, 2].map((k) => (
            <circle key={k} r={7} className="vm-particle">
              <animateMotion dur="2.6s" repeatCount="indefinite" begin={`${(k * 0.87 + i * 0.3).toFixed(2)}s`} path={arc.d} />
            </circle>
          ))}
          <g transform={`translate(${f(arc.labelAt[0])}, ${f(arc.labelAt[1])})`}>
            <rect x={-arc.label.length * 6.2 - 14} y={-19} width={arc.label.length * 12.4 + 28} height={38} rx={19} className="vm-arc-label-bg" />
            <text y={7} className="vm-arc-label">{arc.label}</text>
          </g>
        </g>
      ))}

      {/* gates */}
      {gates.map((g, i) => {
        const a = angles[i];
        const [x, y] = pt(a, R);
        const mode = gateMode(g, running);
        const isSel = selected === g.gate_id;
        const [lx, ly] = pt(a, R - nodeR - 44);
        const verb = running ? g.command?.verb : undefined;
        const chip = verb === 'STOP_ENTRY' ? 'STOP ENTRY' : verb === 'HOLD_OUTSIDE' ? 'HOLD OUTSIDE'
          : verb === 'GET_READY' ? 'GET READY' : null;
        const [rx, ry] = pt(a, R + nodeR + 34);
        return (
          <g key={g.gate_id} className={`vm-gate vm-gate-${mode}`} data-level={g.congestion_level}
            onClick={() => onSelect(g.gate_id)} role="button" tabIndex={0}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(g.gate_id); }}
            aria-label={`${g.gate_id} ${g.name ?? ''}`}>
            {isSel && <circle cx={x} cy={y} r={nodeR + 13} className="vm-sel" />}
            {mode === 'live' && g.congestion_level === 'CRITICAL' && (
              <circle cx={x} cy={y} r={nodeR} className="vm-ping" style={{ stroke: levelVar(g, running) }} />
            )}
            <circle cx={x} cy={y} r={nodeR} className="vm-node" style={{ stroke: levelVar(g, running) }} />
            <circle cx={x} cy={y} r={nodeR - 7} className="vm-node-in"
              style={{ fill: mode === 'live' ? levelVar(g, running) : 'transparent' }} />
            <text x={x} y={y + nodeR * 0.18} className="vm-node-id" style={{ fontSize: nodeR * 0.62 }}>{g.gate_id}</text>

            {/* inward label: count and name */}
            <text x={lx} y={ly + (showNames ? -4 : 6)} className="vm-count" style={{ fontSize: Math.max(17, nodeR * 0.72) }}>
              {mode === 'live' ? Math.round(g.crowd_count) : mode === 'staged' ? '•' : '—'}
            </text>
            {showNames && (
              <text x={lx} y={ly + 20} className="vm-name">
                {mode === 'live' && g.density_m2 != null ? `${g.density_m2.toFixed(1)} /m²` : (g.name ?? '').replace(/ Gate$/, '')}
              </text>
            )}

            {mode === 'live' && g.anomaly_count > 0 && (
              <g transform={`translate(${f(x + nodeR * 0.78)}, ${f(y - nodeR * 0.78)})`} className="vm-anom">
                <circle r={17} />
                <text y={7}>{g.anomaly_count}</text>
              </g>
            )}

            {chip && (
              <g transform={`translate(${f(rx)}, ${f(ry)})`} className={`vm-chip vm-chip-${verb}`}>
                <rect x={-chip.length * 6.4 - 12} y={-16} width={chip.length * 12.8 + 24} height={32} rx={8} />
                <text y={6}>{chip}</text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export const VenueMap = memo(VenueMapImpl);

/** Small static ring for the setup screen: shows the layout being chosen. */
export function RingPreview({ count, monitored, segments }: {
  count: number; monitored: Set<string>; segments?: number[];
}) {
  const angles = ringAngles(count, segments);
  const nodeR = count <= 6 ? 56 : count <= 10 ? 46 : count <= 16 ? 34 : 26;
  return (
    <svg viewBox="0 0 1000 1000" className="ring-preview" aria-label={`${count}-gate ring`}>
      <circle cx={C} cy={C} r={R} fill="none" stroke="var(--ring-band)" strokeWidth={nodeR * 1.6} />
      <circle cx={C} cy={C} r={R - nodeR - 80} fill="url(#rp-field)" stroke="var(--line)" />
      <defs>
        <radialGradient id="rp-field">
          <stop offset="0%" stopColor="var(--field-1)" />
          <stop offset="100%" stopColor="var(--field-2)" />
        </radialGradient>
      </defs>
      <text x={C} y={C + 30} className="vm-big" style={{ fontSize: 150 }}>{count}</text>
      <text x={C} y={C + 96} className="vm-sub" style={{ fontSize: 40 }}>gates</text>
      {Array.from({ length: count }, (_, i) => {
        const id = `G${i + 1}`;
        const [x, y] = pt(angles[i], R);
        const on = monitored.has(id);
        return (
          <g key={id}>
            <circle cx={x} cy={y} r={nodeR} className={`rp-node ${on ? 'on' : ''}`} />
            <text x={x} y={y + nodeR * 0.22} className="vm-node-id" style={{ fontSize: nodeR * 0.62 }}>{id}</text>
          </g>
        );
      })}
    </svg>
  );
}
