/* Everything about one gate: live video, the numbers behind its congestion
 * score, and why the engine recommends what it does - including every other
 * gate it considered and why each was or wasn't chosen. */
import { useEffect, useState } from 'react';
import { ArrowRight, Check, X, AlertTriangle, Radio, Ruler } from 'lucide-react';
import type { Decision, Snapshot } from '../lib/types';
import { api } from '../lib/api';
import { gateMode, levelVar, pct, speedText } from '../lib/format';
import { InstructionText, LevelBadge, Meter, Sparkline, VerbIcon, VideoCanvas } from './bits';

function Stat({ label, value, unit, tone }: { label: string; value: string | number; unit?: string; tone?: string }) {
  return (
    <div className="stat" data-tone={tone}>
      <b>{value}{unit && <small>{unit}</small>}</b>
      <span>{label}</span>
    </div>
  );
}

export function GateDrawer({ snap, gateId, order, onClose, onSelect }: {
  snap: Snapshot;
  gateId: string;
  order: string[];
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  const g = snap.gates[gateId];
  const running = snap.session.state === 'running';
  const [decision, setDecision] = useState<Decision | null>(null);

  useEffect(() => {
    setDecision(null);
    if (!running) return;
    let alive = true;
    const load = () => api.decisions()
      .then((all) => { if (alive) setDecision(all[gateId] ?? null); })
      .catch(() => {});
    void load();
    const t = setInterval(load, 1000);
    return () => { alive = false; clearInterval(t); };
  }, [gateId, running]);

  if (!g) return null;
  const live = gateMode(g, running) === 'live';
  const warnings = [
    live && g.is_stale && 'No fresh observation for over 5 s: decisions paused.',
    live && !g.continuity_ok && 'Frames are being skipped: tracks may break.',
    live && !g.classifier_active && 'Anomaly classifier inactive: rule-based only.',
    live && g.motion_status !== 'OK' && `Motion ${g.motion_status.toLowerCase()}: score uses density only.`,
  ].filter(Boolean) as string[];

  const candidates = decision?.candidates ?? [];
  const route = (g.route?.length ? g.route : decision?.route) ?? [];

  return (
    <aside className="drawer" aria-label={`${g.gate_id} details`}>
      <header className="drawer-head">
        <span className="drawer-id" style={{ borderColor: levelVar(g, running) }}>{g.gate_id}</span>
        <div className="drawer-title">
          <strong>{g.name}</strong>
          <span>
            {g.calibration
              ? <><Ruler size={11} /> zone {g.calibration.width_m}×{g.calibration.length_m} m · full at {g.capacity}</>
              : g.is_monitored ? `not calibrated · assumes ${g.capacity} people` : `no camera · assumes ${g.capacity} people`}
          </span>
        </div>
        <span className="spacer" />
        <LevelBadge g={g} running={running} />
        <button className="icon-btn" onClick={onClose} aria-label="Close (Esc)"><X size={18} /></button>
      </header>

      <div className="drawer-scroll">
        <VideoCanvas g={g} className="drawer-video" />

        {live && (
          <>
            <section className="drawer-sec">
              <div className="score-row">
                <div>
                  <span className="label">Congestion score</span>
                  <b className="score-big" style={{ color: levelVar(g, running) }}>{pct(g.congestion_score)}</b>
                </div>
                <Sparkline values={g.score_history} width={180} height={46} color={levelVar(g, running)} />
              </div>
              <Meter value={g.congestion_score} level={g.congestion_level} />
              <div className="meter-scale"><span>0</span><span style={{ left: '40%' }}>moderate</span><span style={{ left: '65%' }}>high</span><span style={{ left: '80%' }}>critical</span></div>
            </section>

            <section className="drawer-sec stats-grid">
              <Stat label={g.calibration ? 'people in zone' : 'people'} value={Math.round(g.crowd_count)} unit={` / ${g.capacity}`} />
              {g.density_m2 != null
                ? <Stat label="density" value={g.density_m2.toFixed(1)} unit=" /m²" tone={g.density_m2 >= 4 ? 'warn' : undefined} />
                : <Stat label="of assumed capacity" value={pct(g.utilization)} />}
              <Stat label="trend" value={g.trend.toLowerCase()} tone={g.trend === 'INCREASING' ? 'warn' : undefined} />
              <Stat label="speed" value={speedText(g).split(' ')[0]} unit={` ${g.velocity_unit}`} />
              <Stat label="flow coherence" value={pct(g.flow_consistency)} />
              <Stat label="tracks" value={g.track_count} />
              <Stat label="sitting / fallen" value={g.sitting_count} tone={g.sitting_count ? 'anom' : 'mute'} />
              <Stat label="counter-flow" value={g.opposite_count} tone={g.opposite_count ? 'counter' : 'mute'} />
              <Stat label="confidence" value={pct(g.confidence)} tone={g.confidence < 0.3 ? 'warn' : undefined} />
            </section>
          </>
        )}

        <section className="drawer-sec">
          <h3>Instruction</h3>
          {live && g.command ? (
            <div className="cmd-card" data-level={g.command.level}>
              <div className="cmd-head">
                <VerbIcon verb={g.command.verb} size={20} />
                <strong><InstructionText item={g.command} /></strong>
                <span className="ins-reason">{g.command.reason}</span>
              </div>
              <p className="ins-radio"><Radio size={14} /> “{g.command.radio}”</p>
              <p className="ins-detail">{g.command.detail}</p>
              {route.length > 2 && (
                <div className="route">
                  {route.map((id, i) => (
                    <span key={`${id}-${i}`} className="route-step">
                      {i > 0 && <ArrowRight size={12} />}
                      <button type="button" className="mini-gate" onClick={() => onSelect(id)}>{id}</button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <p className="why">
              {live ? 'No instruction. Keep watching.'
                : g.is_monitored ? 'Footage is staged. It plays when monitoring starts.'
                  : 'No camera on this gate. It can still receive diverted crowd, on an assumed load.'}
            </p>
          )}
          {live && g.reason && (
            <details className="why-more">
              <summary>Why the engine decided this</summary>
              <p className="why">{g.reason}</p>
            </details>
          )}
          {warnings.map((w) => (
            <div key={w} className="warn-line"><AlertTriangle size={13} />{w}</div>
          ))}
        </section>

        {live && candidates.length > 0 && (
          <section className="drawer-sec">
            <h3>Gates considered</h3>
            <table className="cand">
              <thead>
                <tr><th>Gate</th><th>Route score</th><th>Congestion</th><th>Dist</th><th /></tr>
              </thead>
              <tbody>
                {[...candidates]
                  .sort((a, b) => Number(b.is_viable) - Number(a.is_viable) || b.routing_score - a.routing_score)
                  .map((c) => (
                    <tr key={c.gate_id} className={c.gate_id === g.target_gate ? 'is-target' : c.is_viable ? '' : 'is-rejected'}>
                      <td>
                        <button type="button" className="mini-gate" onClick={() => onSelect(c.gate_id)}>{c.gate_id}</button>
                        {!c.is_monitored && <small className="muted"> no cam</small>}
                      </td>
                      <td>
                        <div className="bar"><div style={{ width: `${Math.max(0, Math.min(1, c.routing_score)) * 100}%` }} /></div>
                        <span className="mono">{c.routing_score.toFixed(2)}</span>
                      </td>
                      <td><span className="dot" data-level={c.congestion_level} /> {pct(c.congestion_score)}</td>
                      <td className="mono">{Math.round(c.distance_m)} m</td>
                      <td title={c.rejection_reason ?? ''}>
                        {c.is_viable ? <Check size={14} className="ok" /> : <span className="rej">{c.rejection_reason ?? 'rejected'}</span>}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </section>
        )}

        {live && (
          <section className="drawer-sec">
            <h3>Crowd count</h3>
            <Sparkline values={g.count_history} width={380} height={56} max={Math.max(g.capacity, ...g.count_history, 1)} bands={false} />
          </section>
        )}

        <nav className="drawer-nav">
          {order.map((id) => (
            <button key={id} type="button" className={`mini-gate ${id === gateId ? 'on' : ''}`}
              style={{ borderColor: levelVar(snap.gates[id], running) }} onClick={() => onSelect(id)}>{id}</button>
          ))}
        </nav>
      </div>
    </aside>
  );
}
