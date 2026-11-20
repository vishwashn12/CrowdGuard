/* Layout A, the command centre: a large stage (venue map, or the camera wall)
 * with the gate rail beneath it and the incident queue beside it. Selecting a
 * gate anywhere opens its focus drawer over the stage. */
import { useEffect } from 'react';
import { Camera, Map as MapIcon, Play, Settings2, UploadCloud } from 'lucide-react';
import type { Snapshot } from '../lib/types';
import { LEVELS, LEVEL_LABEL, ringOrder } from '../lib/format';
import { VenueMap } from '../components/VenueMap';
import { CameraWall } from '../components/CameraWall';
import { GateRail } from '../components/GateRail';
import { IncidentPanel } from '../components/IncidentPanel';
import { GateDrawer } from '../components/GateDrawer';

export type StageMode = 'map' | 'wall';

interface Props {
  snap: Snapshot;
  stage: StageMode;
  onStage: (m: StageMode) => void;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onSetup: () => void;
  onStart: () => void;
  busy: boolean;
}

export function MonitorView({ snap, stage, onStage, selected, onSelect, onSetup, onStart, busy }: Props) {
  const order = ringOrder(snap);
  const running = snap.session.state === 'running';
  const staged = snap.session.gates_with_footage;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA') return;
      if (e.key === 'Escape') onSelect(null);
      if (e.key === 'v' || e.key === 'V') onStage(stage === 'map' ? 'wall' : 'map');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [stage, onStage, onSelect]);

  const selectedGone = selected !== null && !snap.gates[selected];
  useEffect(() => { if (selectedGone) onSelect(null); }, [selectedGone, onSelect]);

  const toggle = (id: string) => onSelect(selected === id ? null : id);

  return (
    <main className="monitor">
      <section className="panel stage">
        <header className="stage-head">
          <div className="seg" role="tablist" aria-label="Stage view (V)">
            <button type="button" className={stage === 'map' ? 'on' : ''} onClick={() => onStage('map')} title="Venue map (V)">
              <MapIcon size={15} /> Venue map
            </button>
            <button type="button" className={stage === 'wall' ? 'on' : ''} onClick={() => onStage('wall')} title="Camera wall (V)">
              <Camera size={15} /> Camera wall
            </button>
          </div>
          <span className="spacer" />
          <div className="legend">
            {LEVELS.map((lv) => (
              <span key={lv}><span className="dot" data-level={lv} />{LEVEL_LABEL[lv]}</span>
            ))}
            <span><span className="dot" data-level="NONE" />No feed</span>
            {stage === 'map' && <span className="lg-arrow">diversion</span>}
          </div>
        </header>

        <div className="stage-body">
          {stage === 'map'
            ? <VenueMap snap={snap} order={order} selected={selected} onSelect={toggle} />
            : <CameraWall snap={snap} order={order} selected={selected} onSelect={toggle} />}

          {!running && (
            <div className="stage-overlay">
              {staged === 0 ? (
                <div className="cta">
                  <UploadCloud size={30} strokeWidth={1.5} />
                  <strong>No footage yet</strong>
                  <span>Choose how many gates the venue has and add a video to each. Gates without footage still take part in routing.</span>
                  <button type="button" className="btn btn-primary" onClick={onSetup}><Settings2 size={15} /> Set up venue</button>
                </div>
              ) : (
                <div className="cta cta-compact">
                  <strong>{staged} gate{staged === 1 ? '' : 's'} ready</strong>
                  <span>All gates start together at the footage's real frame rate.</span>
                  <button type="button" className="btn btn-primary" onClick={onStart} disabled={busy}><Play size={14} /> Start monitoring</button>
                </div>
              )}
            </div>
          )}

          {selected && snap.gates[selected] && (
            <GateDrawer snap={snap} gateId={selected} order={order} onClose={() => onSelect(null)} onSelect={onSelect} />
          )}
        </div>
      </section>

      <GateRail snap={snap} order={order} selected={selected} onSelect={toggle} thumbnails={stage === 'map'} />

      <IncidentPanel snap={snap} onSelect={(id) => onSelect(id)} />
    </main>
  );
}
