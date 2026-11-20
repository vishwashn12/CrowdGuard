/* Pipeline health, for the engineer rather than the operator. */
import { X } from 'lucide-react';
import type { Snapshot } from '../lib/types';

export function Diagnostics({ snap, onClose }: { snap: Snapshot; onClose: () => void }) {
  const p = snap.pipeline || {};
  const play = Object.values(p.playback?.cameras || {}) as { display_fps?: number; source_fps?: number }[];
  const displayFps = play.length ? play.reduce((s, c) => s + (c.display_fps || 0), 0) / play.length : 0;
  const sourceFps = play[0]?.source_fps ?? 0;
  const dropped = p.buffers?.anomaly?.dropped ?? 0;
  const breaks = p.posetrack?.continuity_breaks ?? 0;
  const valid = p.synchronizer?.valid_ratio ?? 0;

  const items: [string, string | number, boolean][] = [
    ['cameras', Object.keys(p.cameras || {}).length, false],
    ['playback fps', `${displayFps.toFixed(1)} / ${sourceFps || 0}`, false],
    ['pose latency', `${p.posetrack?.avg_inference_ms ?? 0} ms`, false],
    ['CSRNet latency', `${p.csrnet?.avg_inference_ms ?? 0} ms`, false],
    ['motion valid', `${Math.round(valid * 100)}%`, snap.session.state === 'running' && valid < 0.8],
    ['frames refused', dropped, dropped > 0],
    ['continuity breaks', breaks, breaks > 0],
    ['anomaly queue', p.buffers?.anomaly?.size ?? 0, false],
    ['decisions', snap.total_decisions, false],
  ];

  return (
    <div className="diag">
      <header>
        <strong>Pipeline diagnostics</strong>
        <span className="muted">{snap.device}</span>
        <span className="spacer" />
        <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={16} /></button>
      </header>
      <div className="diag-grid">
        {items.map(([label, value, bad]) => (
          <div key={label} className={`diag-item ${bad ? 'bad' : ''}`}>
            <b className="mono">{value}</b>
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
