import { Activity, Bell, BellOff, Monitor, Play, Settings2, Square, Loader2 } from 'lucide-react';
import type { Snapshot } from '../lib/types';
import { fmtClock, gateMode, levelRank } from '../lib/format';
import { useClock } from '../lib/misc';

export type Tab = 'monitor' | 'setup';

interface Props {
  snap: Snapshot | null;
  connected: boolean;
  tab: Tab;
  onTab: (t: Tab) => void;
  busy: boolean;
  onSession: (action: 'start' | 'stop') => void;
  sound: boolean;
  onSound: () => void;
  diagOpen: boolean;
  onDiag: () => void;
}

/** One-phrase answer to "is anything wrong?" */
function overall(snap: Snapshot): { tone: string; text: string } {
  const running = snap.session.state === 'running';
  if (!running) {
    const n = snap.session.gates_with_footage;
    return n ? { tone: 'ready', text: `${n} gate${n === 1 ? '' : 's'} ready to start` } : { tone: 'idle', text: 'Set up the venue to begin' };
  }
  const live = Object.values(snap.gates).filter((g) => gateMode(g, true) === 'live');
  const worst = live.reduce((m, g) => Math.max(m, levelRank(g.congestion_level)), 0);
  const emergencies = (snap.alerts || []).filter((a) => a.severity === 'EMERGENCY').length;
  const hot = live.filter((g) => levelRank(g.congestion_level) >= 2).length;
  if (emergencies) return { tone: 'critical', text: `${emergencies} emergenc${emergencies === 1 ? 'y' : 'ies'}` };
  if (worst >= 3) return { tone: 'critical', text: `${hot} gate${hot === 1 ? '' : 's'} congested` };
  if (worst >= 2) return { tone: 'high', text: `${hot} gate${hot === 1 ? '' : 's'} congested` };
  if (worst >= 1) return { tone: 'moderate', text: 'Watching busy gates' };
  return { tone: 'low', text: 'All gates normal' };
}

export function TopBar({ snap, connected, tab, onTab, busy, onSession, sound, onSound, diagOpen, onDiag }: Props) {
  const now = useClock();
  const running = snap?.session.state === 'running';
  const ready = !running && (snap?.session.gates_with_footage ?? 0) > 0;
  const status = snap ? overall(snap) : { tone: 'idle', text: 'Connecting…' };

  return (
    <header className="topbar">
      <div className="brand">
        <svg viewBox="0 0 32 32" width={26} height={26} aria-hidden>
          <circle cx="16" cy="16" r="10" fill="none" stroke="var(--accent)" strokeWidth="2.4" />
          <circle cx="16" cy="6" r="3.2" fill="var(--accent)" />
          <circle cx="24.7" cy="21" r="3.2" fill="var(--lv-high)" />
          <circle cx="7.3" cy="21" r="3.2" fill="var(--lv-low)" />
        </svg>
        <div className="brand-text">
          <strong>CrowdGuard</strong>
          <span>{snap?.venue?.venue_name || 'Venue'}</span>
        </div>
      </div>

      <nav className="seg tabs" role="tablist" aria-label="Views">
        <button type="button" role="tab" className={tab === 'monitor' ? 'on' : ''} onClick={() => onTab('monitor')}>
          <Monitor size={15} /> Monitor
        </button>
        <button type="button" role="tab" className={tab === 'setup' ? 'on' : ''} onClick={() => onTab('setup')}>
          <Settings2 size={15} /> Setup
        </button>
      </nav>

      <div className="status-pill" data-tone={status.tone}>
        <span className="status-dot" />
        {status.text}
      </div>

      <span className="spacer" />

      {running && (
        <div className="live-chip">
          <span className="live-dot" /> LIVE <span className="mono">{fmtClock(snap!.session.running_s)}</span>
        </div>
      )}

      {(running || ready) && (
        <button type="button" className={`btn ${running ? 'btn-stop' : 'btn-primary'}`} disabled={busy}
          onClick={() => onSession(running ? 'stop' : 'start')}>
          {busy ? <Loader2 size={15} className="spin" /> : running ? <Square size={14} /> : <Play size={14} />}
          {running ? 'Stop' : 'Start monitoring'}
        </button>
      )}

      <div className="topbar-tools">
        <button type="button" className={`icon-btn ${diagOpen ? 'on' : ''}`} onClick={onDiag} title="Pipeline diagnostics">
          <Activity size={17} />
        </button>
        <button type="button" className="icon-btn" onClick={onSound} title={sound ? 'Emergency chime on' : 'Emergency chime off'}>
          {sound ? <Bell size={17} /> : <BellOff size={17} />}
        </button>
        <span className={`conn ${connected ? 'on' : 'off'}`} title={connected ? 'Connected' : 'Reconnecting…'} />
        <div className="clock">
          <span className="mono">{now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
          <small>{snap?.device ?? '—'}</small>
        </div>
      </div>
    </header>
  );
}
