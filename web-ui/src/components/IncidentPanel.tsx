/* Alerts as a work queue, most urgent first. Congestion, individual-behaviour
 * anomalies and system health are visually distinct, so an operator can tell
 * "move the crowd" from "send staff to a person" from "check the camera". */
import { useState } from 'react';
import { ArrowRight, Cpu, PersonStanding, ShieldCheck, Users } from 'lucide-react';
import type { Alert, Snapshot } from '../lib/types';
import { fmtAge } from '../lib/format';

const CATEGORY_ICON: Record<string, typeof Users> = {
  CONGESTION: Users,
  ANOMALY: PersonStanding,
  SYSTEM: Cpu,
};

const SEVERITIES = ['EMERGENCY', 'ALERT', 'WARNING', 'INFO'] as const;

function AlertRow({ a, onSelect, historic }: { a: Alert; onSelect: (id: string) => void; historic?: boolean }) {
  const Icon = CATEGORY_ICON[a.category] ?? Cpu;
  const gate = a.gate_id && a.gate_id !== 'ALL' ? a.gate_id : null;
  const now = Date.now() / 1000;
  return (
    <li className={`inc ${historic ? 'inc-historic' : ''}`} data-sev={a.severity} data-cat={a.category}>
      <button type="button" className="inc-btn" onClick={() => gate && onSelect(gate)} disabled={!gate}>
        <span className="inc-icon"><Icon size={16} /></span>
        <div className="inc-body">
          <div className="inc-top">
            <span className="inc-sev">{a.severity}</span>
            {gate && <span className="inc-gate">{gate}</span>}
            {a.target_gate && (
              <span className="inc-target"><ArrowRight size={11} />{a.target_gate}</span>
            )}
            <span className="spacer" />
            <span className="inc-age">
              {fmtAge(historic ? now - a.timestamp : a.age_s)}
              {a.occurrences > 1 && <> · ×{a.occurrences}</>}
            </span>
          </div>
          <div className="inc-title">{a.title}</div>
          {a.action && <div className="inc-do">{a.action}</div>}
        </div>
      </button>
    </li>
  );
}

export function IncidentPanel({ snap, onSelect }: { snap: Snapshot; onSelect: (id: string) => void }) {
  const [tab, setTab] = useState<'active' | 'log'>('active');
  const running = snap.session.state === 'running';
  const active = snap.alerts || [];
  const log = [...(snap.incidents || [])].sort((a, b) => b.timestamp - a.timestamp);
  const counts = SEVERITIES.map((s) => [s, active.filter((a) => a.severity === s).length] as const);

  return (
    <section className="panel incidents">
      <header className="panel-head">
        <h2>Incidents</h2>
        <div className="sev-counts">
          {counts.filter(([, c]) => c > 0).map(([s, c]) => (
            <span key={s} className="sev-count" data-sev={s} title={s}>{c}</span>
          ))}
        </div>
        <span className="spacer" />
        <div className="seg seg-sm" role="tablist">
          <button type="button" className={tab === 'active' ? 'on' : ''} onClick={() => setTab('active')}>
            Active{active.length ? ` ${active.length}` : ''}
          </button>
          <button type="button" className={tab === 'log' ? 'on' : ''} onClick={() => setTab('log')}>Log</button>
        </div>
      </header>

      <div className="panel-scroll">
        {tab === 'active' && active.length === 0 && (
          <div className="all-clear">
            <ShieldCheck size={34} strokeWidth={1.5} />
            <strong>{running ? 'All clear' : 'Not monitoring'}</strong>
            <span>{running ? 'Nothing needs attention right now.' : 'Alerts appear here once monitoring starts.'}</span>
          </div>
        )}
        {tab === 'log' && log.length === 0 && (
          <div className="all-clear"><span>No incidents logged in this session.</span></div>
        )}
        <ul className="inc-list">
          {(tab === 'active' ? active : log).map((a) => (
            <AlertRow key={`${a.alert_id}-${a.timestamp}`} a={a} onSelect={onSelect} historic={tab === 'log'} />
          ))}
        </ul>
      </div>

      <footer className="panel-foot legend">
        <span><Users size={12} /> crowd</span>
        <span className="lg-anom"><PersonStanding size={12} /> person</span>
        <span><Cpu size={12} /> system</span>
      </footer>
    </section>
  );
}
