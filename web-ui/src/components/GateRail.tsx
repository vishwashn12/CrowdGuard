/* One compact card per gate under the stage: level, people, trend and the
 * recommendation. In map view each card also carries a live thumbnail, so the
 * cameras are never more than a glance away. */
import { TrendingDown, TrendingUp, Minus } from 'lucide-react';
import type { Snapshot } from '../lib/types';
import { gateMode, levelVar } from '../lib/format';
import { ActionChip, LevelBadge, Sparkline, VideoCanvas } from './bits';

const TREND_ICON = { INCREASING: TrendingUp, DECREASING: TrendingDown, STABLE: Minus } as const;

interface Props {
  snap: Snapshot;
  order: string[];
  selected: string | null;
  onSelect: (id: string) => void;
  thumbnails: boolean;
}

export function GateRail({ snap, order, selected, onSelect, thumbnails }: Props) {
  const running = snap.session.state === 'running';
  return (
    <div className="rail" role="list">
      {order.map((id) => {
        const g = snap.gates[id];
        if (!g) return null;
        const live = gateMode(g, running) === 'live';
        const T = TREND_ICON[g.trend] ?? Minus;
        return (
          <button key={id} type="button" role="listitem"
            className={`rail-card ${selected === id ? 'is-selected' : ''} ${thumbnails && g.is_monitored ? 'has-thumb' : ''}`}
            data-level={live ? g.congestion_level : 'NONE'}
            style={{ ['--lv' as string]: levelVar(g, running) }}
            onClick={() => onSelect(id)}>
            {thumbnails && g.is_monitored && <VideoCanvas g={g} className="rail-thumb" />}
            <div className="rail-body">
              <div className="rail-head">
                <span className="rail-id">{g.gate_id}</span>
                <span className="rail-name">{g.name}</span>
                <span className="spacer" />
                <LevelBadge g={g} running={running} size="sm" />
              </div>
              <div className="rail-mid">
                <span className="rail-count">
                  {live ? Math.round(g.crowd_count) : '—'}
                  <small>{g.calibration ? (live && g.density_m2 != null ? ` · ${g.density_m2.toFixed(1)}/m²` : ' · zone') : `/${g.capacity}`}</small>
                </span>
                {live && <T size={14} className={`trend trend-${g.trend.toLowerCase()}`} />}
                <span className="spacer" />
                <Sparkline values={live ? g.score_history : []} width={84} height={26} color={levelVar(g, running)} />
              </div>
              <ActionChip g={g} running={running} />
            </div>
          </button>
        );
      })}
    </div>
  );
}
