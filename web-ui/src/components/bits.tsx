/* Small shared pieces: level badge, action chip, sparkline, video canvas. */
import { useEffect, useRef, useState } from 'react';
import { ArrowRightLeft, Ban, Eye, CircleCheck, VideoOff, Loader2 } from 'lucide-react';
import type { Gate } from '../lib/types';
import { actionLabel, actionTone, gateMode, LEVEL_LABEL } from '../lib/format';
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

const ACTION_ICON = {
  divert: ArrowRightLeft,
  restrict: Ban,
  monitor: Eye,
  calm: CircleCheck,
  idle: null,
} as const;

export function ActionChip({ g, running }: { g: Gate; running: boolean }) {
  const tone = actionTone(g, running);
  const Icon = ACTION_ICON[tone];
  return (
    <span className="action-chip" data-tone={tone}>
      {Icon && <Icon size={13} strokeWidth={2.4} />}
      {actionLabel(g, running)}
    </span>
  );
}

/** Congestion score history with the HIGH / CRITICAL thresholds marked. */
export function Sparkline({
  values,
  width = 120,
  height = 32,
  max = 1,
  bands = true,
  color = 'var(--accent)',
}: {
  values: number[];
  width?: number;
  height?: number;
  max?: number;
  bands?: boolean;
  color?: string;
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

/** A gate's live video. Several canvases can show the same gate at once. */
export function VideoCanvas({ g, className }: { g: Gate; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);
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
    return videoHub.attach(g.gate_id, canvas, () => setReady(true));
  }, [g.gate_id, monitored, g.camera_id]);

  return (
    <div className={`video ${className ?? ''}`}>
      {monitored && <canvas ref={ref} aria-label={`${g.gate_id} video`} />}
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
