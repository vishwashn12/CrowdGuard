/* Small shared pieces: level badge, command chip, verb icon, sparkline,
 * meter, and the live video canvas (with the calibrated zone drawn on it). */
import { useEffect, useRef, useState } from 'react';
import {
  ArrowRightLeft, CircleCheck, Hand, Loader2, PauseCircle, PersonStanding, TimerReset, Undo2, Users, VideoOff,
} from 'lucide-react';
import type { Gate, Instruction, Verb } from '../lib/types';
import { actionLabel, actionTone, gateMode, LEVEL_LABEL } from '../lib/format';
import { useSize } from '../lib/misc';
import { videoHub } from '../lib/video';

export function LevelBadge({ g, running, size = 'md' }: { g: Gate; running: boolean; size?: 'sm' | 'md' }) {
  const mode = gateMode(g, running);
  const level = mode === 'live' ? g.congestion_level : 'NONE';
  const text = mode === 'live' ? LEVEL_LABEL[g.congestion_level] : mode === 'staged' ? 'Ready' : 'No feed';
  return (
    <span className={`badge badge-${size}`} data-level={level}>
      <span className="badge-dot" />
      {text}
    </span>
  );
}

/** One fixed icon per verb, so an instruction is recognisable before it is read. */
export function VerbIcon({ verb, kind, size = 16 }: { verb: Verb; kind?: string | null; size?: number }) {
  const props = { size, strokeWidth: 2.3 };
  switch (verb) {
    case 'STOP_ENTRY': return <Hand {...props} />;
    case 'HOLD_OUTSIDE': return <PauseCircle {...props} />;
    case 'DIVERT': return <ArrowRightLeft {...props} />;
    case 'GET_READY': return <TimerReset {...props} />;
    case 'SEND_STAFF':
      if (kind === 'counterflow') return <Undo2 {...props} />;
      if (kind === 'person_down') return <PersonStanding {...props} />;
      return <Users {...props} />;
  }
}

export function ActionChip({ g, running }: { g: Gate; running: boolean }) {
  const tone = actionTone(g, running);
  return (
    <span className="action-chip" data-tone={tone}>
      {g.command ? <VerbIcon verb={g.command.verb} size={13} /> : tone === 'calm' ? <CircleCheck size={13} /> : null}
      {actionLabel(g, running)}
    </span>
  );
}

export function InstructionText({ item }: { item: Instruction }) {
  return (
    <>
      <span className="verb">{item.verb_label}</span>{' '}
      <span className="where">{item.gate_id}</span>
      {item.target && <> <span className="arrow">→</span> <span className="where">{item.target}</span></>}
    </>
  );
}

/** Congestion score history with the HIGH / CRITICAL thresholds marked. */
export function Sparkline({
  values, width = 120, height = 32, max = 1, bands = true, color = 'var(--accent)',
}: {
  values: number[]; width?: number; height?: number; max?: number; bands?: boolean; color?: string;
}) {
  const n = values.length;
  const y = (v: number) => height - 2 - (Math.min(v, max) / max) * (height - 4);
  const x = (i: number) => (n <= 1 ? width : (i / (n - 1)) * width);
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {bands && (
        <>
          <rect x={0} y={0} width={width} height={y(0.8)} className="spark-band-crit" />
          <rect x={0} y={y(0.8)} width={width} height={y(0.65) - y(0.8)} className="spark-band-high" />
          <line x1={0} x2={width} y1={y(0.65)} y2={y(0.65)} className="spark-line-th" />
        </>
      )}
      {n > 1 && (
        <>
          <polyline points={`0,${height} ${pts} ${width},${height}`} fill={color} opacity={0.12} stroke="none" />
          <polyline points={pts} fill="none" stroke={color} strokeWidth={1.6} strokeLinejoin="round" />
          <circle cx={x(n - 1)} cy={y(values[n - 1])} r={2.4} fill={color} />
        </>
      )}
    </svg>
  );
}

/** A gate's live video. Several canvases can show the same gate at once.
 *  When the camera is calibrated, its ground zone is outlined on the picture. */
export function VideoCanvas({ g, className, showZone = true }: { g: Gate; className?: string; showZone?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const size = useSize(box);
  const [ready, setReady] = useState(false);
  const [aspect, setAspect] = useState<number | null>(null);
  const monitored = g.is_monitored;

  useEffect(() => {
    if (!monitored) {
      videoHub.forget(g.gate_id);
      setReady(false);
      return;
    }
    const canvas = ref.current;
    if (!canvas) return;
    setReady(false);
    return videoHub.attach(g.gate_id, canvas, () => {
      setReady(true);
      setAspect(canvas.height ? canvas.width / canvas.height : null);
    });
  }, [g.gate_id, monitored, g.camera_id]);

  // where the picture actually sits inside the box (object-fit: contain)
  let zone = null;
  const pts = g.calibration?.points;
  if (showZone && ready && pts && aspect && size.width && size.height) {
    const w = Math.min(size.width, size.height * aspect);
    const h = w / aspect;
    const ox = (size.width - w) / 2;
    const oy = (size.height - h) / 2;
    zone = (
      <svg className="zone-overlay" width={size.width} height={size.height}>
        <polygon className="zone-under" points={pts.map(([x, y]) => `${ox + x * w},${oy + y * h}`).join(' ')} />
        <polygon className="zone-line" points={pts.map(([x, y]) => `${ox + x * w},${oy + y * h}`).join(' ')} />
      </svg>
    );
  }

  return (
    <div className={`video ${className ?? ''}`} ref={box}>
      {monitored && <canvas ref={ref} aria-label={`${g.gate_id} video`} />}
      {zone}
      {!monitored && (
        <div className="video-empty">
          <VideoOff size={22} />
          <span>No feed</span>
        </div>
      )}
      {monitored && !ready && (
        <div className="video-empty">
          <Loader2 size={20} className="spin" />
          <span>Waiting for video…</span>
        </div>
      )}
    </div>
  );
}

export function Meter({ value, level }: { value: number; level: string }) {
  return (
    <div className="meter" data-level={level}>
      <div className="meter-fill" style={{ width: `${Math.min(100, Math.round((value || 0) * 100))}%` }} />
      <span className="meter-tick" style={{ left: '40%' }} />
      <span className="meter-tick" style={{ left: '65%' }} />
      <span className="meter-tick" style={{ left: '80%' }} />
    </div>
  );
}
