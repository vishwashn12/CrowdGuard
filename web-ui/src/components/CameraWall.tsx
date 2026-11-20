/* Every gate's camera at once, tiled to fill the stage without scrolling.
 * Each tile carries just enough telemetry to be read across a room: gate,
 * level, people, anomalies, and what the system recommends. */
import { useRef } from 'react';
import { PersonStanding, Undo2, Users, AlertOctagon } from 'lucide-react';
import type { Snapshot } from '../lib/types';
import { gateMode } from '../lib/format';
import { useSize } from '../lib/misc';
import { videoHub } from '../lib/video';
import { ActionChip, LevelBadge, VideoCanvas } from './bits';

const GAP = 8;

/** Grid that gives the largest tiles of aspect `ar` for `k` tiles in w x h. */
function bestGrid(k: number, w: number, h: number, ar: number) {
  let best = { cols: 1, rows: Math.max(1, k), tw: 0, th: 0 };
  if (!w || !h || k === 0) return best;
  for (let cols = 1; cols <= k; cols++) {
    const rows = Math.ceil(k / cols);
    const cw = (w - GAP * (cols - 1)) / cols;
    const ch = (h - GAP * (rows - 1)) / rows;
    const tw = Math.min(cw, ch * ar);
    if (tw > best.tw) best = { cols, rows, tw, th: tw / ar };
  }
  return best;
}

interface Props {
  snap: Snapshot;
  order: string[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export function CameraWall({ snap, order, selected, onSelect }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const { width, height } = useSize(ref);
  const running = snap.session.state === 'running';

  const withFeed = order.filter((id) => snap.gates[id]?.is_monitored);
  const noFeed = order.filter((id) => snap.gates[id] && !snap.gates[id].is_monitored);
  // Size tiles to the footage itself, so overlays sit on the picture rather
  // than on letterbox bars. Before the first frame, assume 4:3.
  const ar = withFeed.map((id) => videoHub.aspect(id)).find((a): a is number => a !== null) ?? 4 / 3;
  const { cols, tw, th } = bestGrid(withFeed.length, width - 20, height - 20 - (noFeed.length ? 34 : 0), ar);

  return (
    <div className="wall" ref={ref}>
      {withFeed.length === 0 ? (
        <div className="wall-empty">No gate has footage yet.</div>
      ) : (
        <div className="wall-grid" style={{
          gridTemplateColumns: `repeat(${cols}, ${Math.floor(tw)}px)`,
          gridAutoRows: `${Math.floor(th)}px`,
        }}>
          {withFeed.map((id) => {
            const g = snap.gates[id];
            const live = gateMode(g, running) === 'live';
            return (
              <button key={id} type="button"
                className={`tile ${selected === id ? 'is-selected' : ''}`}
                data-level={live ? g.congestion_level : 'NONE'}
                onClick={() => onSelect(id)}>
                <VideoCanvas g={g} />
                <div className="tile-top">
                  <span className="tile-id">{g.gate_id}</span>
                  <span className="tile-name">{g.name}</span>
                  <span className="spacer" />
                  <LevelBadge g={g} running={running} size="sm" />
                </div>
                <div className="tile-bottom">
                  <span className="tile-stat" title="People (CSRNet)">
                    <Users size={14} />{live ? Math.round(g.crowd_count) : '—'}
                    <small>/ {g.capacity}</small>
                  </span>
                  {live && g.sitting_count > 0 && (
                    <span className="tile-stat tile-anom" title="Sitting or fallen"><PersonStanding size={14} />{g.sitting_count}</span>
                  )}
                  {live && g.opposite_count > 0 && (
                    <span className="tile-stat tile-counter" title="Counter-flow"><Undo2 size={14} />{g.opposite_count}</span>
                  )}
                  {live && g.is_stale && (
                    <span className="tile-stat tile-warn" title="No fresh observation for 5 s"><AlertOctagon size={14} />stale</span>
                  )}
                  <span className="spacer" />
                  <ActionChip g={g} running={running} />
                </div>
              </button>
            );
          })}
        </div>
      )}
      {noFeed.length > 0 && (
        <div className="wall-nofeed">
          <span>No camera</span>
          {noFeed.map((id) => (
            <button key={id} type="button" className="mini-gate" onClick={() => onSelect(id)}>{id}</button>
          ))}
          <span className="muted">· routing targets on stated capacity</span>
        </div>
      )}
    </div>
  );
}
