import { useCallback, useEffect, useRef, useState } from 'react';
import { Loader2, WifiOff } from 'lucide-react';
import { useTelemetry } from './lib/telemetry';
import { api } from './lib/api';
import { useToast } from './lib/toast';
import { chime, loadPref, savePref } from './lib/misc';
import { videoHub } from './lib/video';
import { TopBar, type Tab } from './components/TopBar';
import { Banner } from './components/Banner';
import { Diagnostics } from './components/Diagnostics';
import { MonitorView, type StageMode } from './views/MonitorView';
import { SetupView } from './views/SetupView';

export default function App() {
  const { snap, connected } = useTelemetry();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>('monitor');
  const [stage, setStageState] = useState<StageMode>(() => loadPref('stage', 'map', ['map', 'wall'] as const));
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sound, setSound] = useState(() => loadPref('sound', 'on', ['on', 'off'] as const) === 'on');
  const [diagOpen, setDiagOpen] = useState(false);
  const seen = useRef<Set<string> | null>(null);

  useEffect(() => { videoHub.start(); }, []);

  const setStage = useCallback((m: StageMode) => { setStageState(m); savePref('stage', m); }, []);

  // Announce new urgent alerts once each.
  useEffect(() => {
    if (!snap) return;
    const alerts = snap.alerts || [];
    if (seen.current === null) { seen.current = new Set(alerts.map((a) => a.alert_id)); return; }
    for (const a of alerts) {
      if (seen.current.has(a.alert_id)) continue;
      seen.current.add(a.alert_id);
      if (snap.session.state !== 'running') continue;
      if (a.severity === 'EMERGENCY') {
        toast('err', `${a.gate_id && a.gate_id !== 'ALL' ? `${a.gate_id} · ` : ''}${a.title}`, a.action);
        if (sound) chime();
      } else if (a.severity === 'ALERT') {
        toast('warn', `${a.gate_id && a.gate_id !== 'ALL' ? `${a.gate_id} · ` : ''}${a.title}`, a.action);
      }
    }
  }, [snap, toast, sound]);

  const onSession = useCallback(async (action: 'start' | 'stop') => {
    if (busy) return;
    setBusy(true);
    try {
      const r = await api.session(action);
      if (action === 'start') {
        const n = r.gates?.length ?? 0;
        toast('ok', 'Monitoring started', `${n} gate${n === 1 ? '' : 's'} in lockstep`);
        setTab('monitor');
      } else {
        toast('ok', 'Monitoring stopped', 'Footage stays staged.');
      }
    } catch (err) {
      toast('err', action === 'start' ? 'Could not start' : 'Could not stop', (err as Error).message);
    } finally {
      setBusy(false);
    }
  }, [busy, toast]);

  const select = useCallback((id: string | null) => {
    setSelected(id);
    if (id) setTab('monitor');
  }, []);

  if (!snap) {
    return (
      <div className="boot">
        {connected ? <Loader2 className="spin" size={28} /> : <WifiOff size={28} />}
        <strong>CrowdGuard</strong>
        <span>{connected ? 'Loading…' : 'Connecting to the server…'}</span>
      </div>
    );
  }

  return (
    <div className="app">
      <TopBar snap={snap} connected={connected} tab={tab} onTab={setTab} busy={busy} onSession={onSession}
        sound={sound} onSound={() => { const v = !sound; setSound(v); savePref('sound', v ? 'on' : 'off'); if (v) chime(); }}
        diagOpen={diagOpen} onDiag={() => setDiagOpen((o) => !o)} />
      {!connected && <div className="offline"><WifiOff size={14} /> Connection lost. Showing the last known state while reconnecting…</div>}
      <Banner snap={snap} onSelect={select} />
      {tab === 'monitor'
        ? <MonitorView snap={snap} stage={stage} onStage={setStage} selected={selected} onSelect={select}
            onSetup={() => setTab('setup')} onStart={() => void onSession('start')} busy={busy} />
        : <SetupView snap={snap} busy={busy} onSession={(a) => void onSession(a)} />}
      {diagOpen && <Diagnostics snap={snap} onClose={() => setDiagOpen(false)} />}
    </div>
  );
}
