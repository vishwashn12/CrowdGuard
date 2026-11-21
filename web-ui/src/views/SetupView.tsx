/* Setup in three steps: how many gates, footage (and calibration) per gate,
 * start. Nothing to guess: capacity comes from each camera's calibration, or
 * a fixed default when a camera is not calibrated. */
import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import {
  CheckCircle2, FileVideo, Lock, Loader2, Play, RefreshCw, Ruler, Square, Trash2, TriangleAlert, UploadCloud,
} from 'lucide-react';
import type { Snapshot, VenueConfig } from '../lib/types';
import { api, uploadFootage, VIDEO_ACCEPT, VIDEO_SUFFIXES } from '../lib/api';
import { naturalSort } from '../lib/format';
import { useToast } from '../lib/toast';
import { RingPreview } from '../components/VenueMap';
import { CalibrationDialog } from '../components/CalibrationDialog';

const PRESETS = [2, 3, 4, 6, 8, 10, 12];

interface Props {
  snap: Snapshot | null;
  busy: boolean;
  onSession: (action: 'start' | 'stop') => void;
}

type UploadState = Record<string, { progress: number; name: string } | undefined>;

function isVideo(f: File): boolean {
  const name = f.name.toLowerCase();
  return VIDEO_SUFFIXES.some((s) => name.endsWith(s));
}

export function SetupView({ snap, busy, onSession }: Props) {
  const toast = useToast();
  const [cfg, setCfg] = useState<VenueConfig | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const [venueName, setVenueName] = useState('');
  const [applying, setApplying] = useState(false);
  const [uploads, setUploads] = useState<UploadState>({});
  const [dragOver, setDragOver] = useState<string | null>(null);
  const [calibrating, setCalibrating] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const pendingGate = useRef<string | null>(null);

  const running = snap?.session.state === 'running';
  const staged = snap?.session.gates_with_footage ?? 0;

  const load = useCallback(async (resetForm = false) => {
    try {
      const c = await api.venueConfig();
      setCfg(c);
      if (resetForm) {
        setCount(c.gate_count);
        setVenueName(c.venue_name || '');
      }
    } catch { /* retried on next change */ }
  }, []);

  useEffect(() => { void load(true); }, [load]);
  // Keep the list in step with the server (uploads, removals, resizes, calibration).
  const gateKey = snap ? Object.values(snap.gates)
    .map((g) => `${g.gate_id}:${g.camera_id}:${g.calibration?.area_m2 ?? ''}`).join(',') : '';
  useEffect(() => { if (gateKey) void load(false); }, [gateKey, load]);

  if (!cfg) {
    return <main className="setup"><div className="setup-loading"><Loader2 className="spin" /> Loading venue…</div></main>;
  }

  const pickedCount = count ?? cfg.gate_count;
  const dropsFeeds = cfg.gates.filter((g, i) => i >= pickedCount && g.is_monitored);
  const monitored = new Set(cfg.gates.filter((g) => g.is_monitored).map((g) => g.gate_id));
  const calibrated = cfg.gates.filter((g) => g.calibration).length;
  const layoutDirty = pickedCount !== cfg.gate_count || (venueName || '') !== (cfg.venue_name || '');
  const calGate = calibrating ? cfg.gates.find((g) => g.gate_id === calibrating) : undefined;

  const apply = async () => {
    setApplying(true);
    try {
      const r = await api.configureVenue({ gate_count: pickedCount, venue_name: venueName.trim() || undefined });
      const bits = [];
      if (r.cameras_stopped.length) bits.push(`${r.cameras_stopped.length} feed(s) stopped`);
      if (r.cameras_kept.length) bits.push(`${r.cameras_kept.length} kept`);
      toast('ok', `Venue set to ${r.gate_count} gates`, bits.join(' · ') || undefined);
      await load(true);
    } catch (err) {
      toast('err', 'Could not apply layout', (err as Error).message);
    } finally {
      setApplying(false);
    }
  };

  const upload = async (gateId: string, file: File) => {
    if (!isVideo(file)) {
      toast('err', `${gateId}: unsupported file`, `Use one of ${VIDEO_ACCEPT}`);
      return;
    }
    setUploads((u) => ({ ...u, [gateId]: { progress: 0, name: file.name } }));
    try {
      const r = await uploadFootage(gateId, file, (p) =>
        setUploads((u) => ({ ...u, [gateId]: { progress: p, name: file.name } })));
      toast('ok', `${gateId} ready`, `${r.filename} · calibrate it for real density`);
      await load(false);
    } catch (err) {
      toast('err', `${gateId}: upload failed`, (err as Error).message);
    } finally {
      setUploads((u) => ({ ...u, [gateId]: undefined }));
    }
  };

  /** Several files dropped on the list: fill empty gates first, in order. */
  const uploadMany = async (files: File[]) => {
    const videos = files.filter(isVideo);
    const empty = naturalSort(cfg.gates.filter((g) => !g.is_monitored && !uploads[g.gate_id]).map((g) => g.gate_id));
    if (!videos.length) { toast('err', 'No video files in the drop', `Use one of ${VIDEO_ACCEPT}`); return; }
    if (videos.length > empty.length) toast('warn', `Only ${empty.length} empty gate(s)`, 'Extra files were ignored. Drop a file on a gate to replace its footage.');
    for (let i = 0; i < Math.min(videos.length, empty.length); i++) {
      await upload(empty[i], videos[i]);
    }
  };

  const pick = (gateId: string) => {
    pendingGate.current = gateId;
    if (fileInput.current) { fileInput.current.value = ''; fileInput.current.click(); }
  };

  const onPicked = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    const gate = pendingGate.current;
    pendingGate.current = null;
    if (f && gate) void upload(gate, f);
  };

  const remove = async (cameraId: string, gateId: string) => {
    try {
      await api.removeCamera(cameraId);
      toast('ok', `${gateId}: footage removed`);
      await load(false);
    } catch (err) {
      toast('err', 'Could not remove footage', (err as Error).message);
    }
  };

  const dropProps = (key: string, onFiles: (files: File[]) => void) => ({
    onDragOver: (e: DragEvent) => { if (running) return; e.preventDefault(); e.stopPropagation(); setDragOver(key); },
    onDragLeave: (e: DragEvent) => { e.stopPropagation(); setDragOver((k) => (k === key ? null : k)); },
    onDrop: (e: DragEvent) => {
      e.preventDefault(); e.stopPropagation(); setDragOver(null);
      if (running) return;
      const files = Array.from(e.dataTransfer.files || []);
      if (files.length) onFiles(files);
    },
  });

  return (
    <main className="setup">
      <div className="setup-main">
        {running && (
          <div className="lockbar">
            <Lock size={16} />
            Monitoring is running, so setup is locked. Stop monitoring to change gates, footage or calibration.
            <span className="spacer" />
            <button type="button" className="btn btn-stop btn-sm" onClick={() => onSession('stop')} disabled={busy}>
              <Square size={13} /> Stop
            </button>
          </div>
        )}

        {/* ── 1. gates ──────────────────────────────── */}
        <section className={`panel step ${running ? 'is-locked' : ''}`}>
          <header className="step-head">
            <span className="step-num">1</span>
            <div>
              <h2>How many gates?</h2>
              <p>Gates sit evenly around the venue, so a crowded gate can send people along the outer path to another one.</p>
            </div>
          </header>

          <div className="chips">
            {PRESETS.filter((p) => p >= cfg.min_gates && p <= cfg.max_gates).map((p) => (
              <button key={p} type="button" className={`chip ${p === pickedCount ? 'on' : ''}`} onClick={() => setCount(p)} disabled={running}>{p}</button>
            ))}
          </div>

          <div className="fields fields-1">
            <label className="field">
              <span>Venue name (optional)</span>
              <input value={venueName} maxLength={120} onChange={(e) => setVenueName(e.target.value)} disabled={running} placeholder="Venue" />
            </label>
          </div>

          {dropsFeeds.length > 0 && (
            <div className="warn-line"><TriangleAlert size={13} /> Applying removes the footage on {dropsFeeds.map((g) => g.gate_id).join(', ')}.</div>
          )}

          <div className="step-actions">
            <button type="button" className="btn btn-primary" onClick={apply} disabled={running || applying || !layoutDirty}>
              {applying ? <Loader2 size={15} className="spin" /> : <RefreshCw size={15} />} Apply
            </button>
            {!layoutDirty && <span className="muted"><CheckCircle2 size={14} /> {cfg.gate_count} gates set</span>}
          </div>
        </section>

        {/* ── 2. footage + calibration ──────────────── */}
        <section className={`panel step ${running ? 'is-locked' : ''}`}>
          <header className="step-head">
            <span className="step-num">2</span>
            <div>
              <h2>Footage and calibration</h2>
              <p>Add a video per gate, then <b>calibrate</b> it: mark a rectangle on the ground and enter its size.
                Calibrated gates are measured in people per m² and m/s. Uncalibrated gates assume they are full at 200 people.</p>
            </div>
          </header>

          <div className={`dropzone ${dragOver === '*' ? 'over' : ''}`} {...dropProps('*', (fs) => void uploadMany(fs))}>
            <UploadCloud size={20} />
            <span>Drop videos here to fill empty gates · {VIDEO_ACCEPT.replace(/\./g, '').replace(/,/g, ', ')}</span>
          </div>

          <ul className="assign">
            {cfg.gates.map((g) => {
              const up = uploads[g.gate_id];
              return (
                <li key={g.gate_id} className={`arow ${g.is_monitored ? 'live' : ''} ${dragOver === g.gate_id ? 'over' : ''}`}
                  {...dropProps(g.gate_id, (fs) => void upload(g.gate_id, fs[0]))}>
                  <span className="arow-id">{g.gate_id}</span>
                  <div className="arow-main">
                    <strong>{g.name}</strong>
                    {up ? (
                      <div className="progress"><div style={{ width: `${Math.round(up.progress * 100)}%` }} /><span>{up.name} · {Math.round(up.progress * 100)}%</span></div>
                    ) : g.is_monitored ? (
                      <span className="arow-status">
                        <span className="ok"><FileVideo size={13} /> footage</span>
                        {g.calibration
                          ? <span className="cal-chip on"><Ruler size={12} /> {g.calibration.area_m2.toFixed(0)} m² · full at {g.effective_capacity}</span>
                          : <span className="cal-chip"><TriangleAlert size={12} /> not calibrated · assumes {g.effective_capacity}</span>}
                      </span>
                    ) : (
                      <span className="muted">No footage · routing target only, assumed {g.effective_capacity} people</span>
                    )}
                  </div>
                  <div className="arow-btns">
                    {g.is_monitored && (
                      <button type="button" className={`btn btn-sm ${g.calibration ? '' : 'btn-primary'}`}
                        onClick={() => setCalibrating(g.gate_id)} disabled={running || !!up}>
                        <Ruler size={14} /> {g.calibration ? 'Recalibrate' : 'Calibrate'}
                      </button>
                    )}
                    <button type="button" className="btn btn-sm" onClick={() => pick(g.gate_id)} disabled={running || !!up}>
                      <UploadCloud size={14} /> {g.is_monitored ? 'Replace' : 'Upload'}
                    </button>
                    {g.is_monitored && g.camera_id && (
                      <button type="button" className="icon-btn danger" title="Remove footage" disabled={running || !!up}
                        onClick={() => void remove(g.camera_id!, g.gate_id)}>
                        <Trash2 size={15} />
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>

        {/* ── 3. start ──────────────────────────────── */}
        <section className="panel step">
          <header className="step-head">
            <span className="step-num">3</span>
            <div>
              <h2>Start monitoring</h2>
              <p>Every gate with footage starts at the same instant and plays at its real frame rate. Clips loop when they end.</p>
            </div>
          </header>
          <div className="step-actions">
            {running ? (
              <button type="button" className="btn btn-stop btn-lg" onClick={() => onSession('stop')} disabled={busy}>
                <Square size={16} /> Stop monitoring
              </button>
            ) : (
              <button type="button" className="btn btn-primary btn-lg" onClick={() => onSession('start')} disabled={busy || staged === 0}>
                {busy ? <Loader2 size={16} className="spin" /> : <Play size={16} />} Start monitoring
              </button>
            )}
            <span className="muted">
              {running ? 'Running.' : staged ? `${staged} gate${staged === 1 ? '' : 's'} will start together.` : 'Upload footage to at least one gate first.'}
            </span>
          </div>
        </section>
      </div>

      <aside className="setup-side panel">
        <header className="panel-head"><h2>Preview</h2></header>
        <RingPreview count={pickedCount} monitored={monitored} />
        <dl className="summary">
          <div><dt>Gates</dt><dd>{pickedCount}{pickedCount !== cfg.gate_count && <small> (now {cfg.gate_count})</small>}</dd></div>
          <div><dt>With footage</dt><dd>{monitored.size}</dd></div>
          <div><dt>Calibrated</dt><dd>{calibrated}<small> / {monitored.size}</small></dd></div>
          <div><dt>Routing only</dt><dd>{Math.max(0, pickedCount - monitored.size)}</dd></div>
        </dl>
      </aside>

      <input ref={fileInput} type="file" accept={VIDEO_ACCEPT} hidden onChange={onPicked} />
      {calGate && (
        <CalibrationDialog gateId={calGate.gate_id} gateName={calGate.name} existing={calGate.calibration}
          onClose={() => setCalibrating(null)}
          onSaved={(m) => { toast('ok', m); void load(false); }} />
      )}
    </main>
  );
}
