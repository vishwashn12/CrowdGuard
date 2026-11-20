/* Setup as three steps - layout, footage, start - with a live preview of the
 * ring being configured. Videos can be dropped straight onto a gate, or
 * several at once onto the list to fill empty gates in order. */
import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { CheckCircle2, FileVideo, Lock, Play, Square, Trash2, UploadCloud, RefreshCw, Loader2 } from 'lucide-react';
import type { Snapshot, VenueConfig } from '../lib/types';
import { api, uploadFootage, VIDEO_ACCEPT, VIDEO_SUFFIXES } from '../lib/api';
import { naturalSort } from '../lib/format';
import { useToast } from '../lib/toast';
import { RingPreview } from '../components/VenueMap';

const PRESETS = [2, 3, 4, 6, 8, 10, 12, 16, 24];

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
  const [capacity, setCapacity] = useState('500');
  const [radius, setRadius] = useState('120');
  const [venueName, setVenueName] = useState('');
  const [applying, setApplying] = useState(false);
  const [uploads, setUploads] = useState<UploadState>({});
  const [dragOver, setDragOver] = useState<string | null>(null);
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
        setCapacity(String(c.gates.length ? Math.round(c.gates.reduce((s, g) => s + g.capacity, 0) / c.gates.length) : 500));
        setRadius(String(Math.round(c.radius_m)));
        setVenueName(c.venue_name || '');
      }
    } catch { /* retried on next change */ }
  }, []);

  useEffect(() => { void load(true); }, [load]);
  // Keep the footage list in step with the server (uploads, removals, resizes).
  const gateKey = snap ? `${Object.keys(snap.gates).length}:${staged}:${Object.values(snap.gates).map((g) => g.camera_id).join(',')}` : '';
  useEffect(() => { if (gateKey) void load(false); }, [gateKey, load]);

  if (!cfg) {
    return <main className="setup"><div className="setup-loading"><Loader2 className="spin" /> Loading venue…</div></main>;
  }

  const pickedCount = count ?? cfg.gate_count;
  const dropsFeeds = cfg.gates.filter((g, i) => i >= pickedCount && g.is_monitored);
  const monitored = new Set(cfg.gates.filter((g) => g.is_monitored).map((g) => g.gate_id));
  const layoutDirty = pickedCount !== cfg.gate_count
    || Number(radius) !== Math.round(cfg.radius_m)
    || (venueName || '') !== (cfg.venue_name || '')
    || Number(capacity) !== Math.round(cfg.gates.reduce((s, g) => s + g.capacity, 0) / Math.max(1, cfg.gates.length));

  const apply = async () => {
    setApplying(true);
    try {
      const r = await api.configureVenue({
        gate_count: pickedCount,
        default_capacity: parseInt(capacity, 10) || undefined,
        radius_m: parseFloat(radius) || undefined,
        venue_name: venueName.trim() || undefined,
      });
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
      toast('ok', `${gateId} ready`, `${r.filename}${r.source_fps ? ` · ${r.source_fps} fps` : ''}`);
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
            Monitoring is running, so setup is locked. Stop monitoring to change gates or footage.
            <span className="spacer" />
            <button type="button" className="btn btn-stop btn-sm" onClick={() => onSession('stop')} disabled={busy}>
              <Square size={13} /> Stop
            </button>
          </div>
        )}

        {/* ── 1. layout ─────────────────────────────── */}
        <section className={`panel step ${running ? 'is-locked' : ''}`}>
          <header className="step-head">
            <span className="step-num">1</span>
            <div>
              <h2>Venue layout</h2>
              <p>Gates sit evenly on a perimeter ring, so a congested gate can divert along the outer concourse. Growing the venue keeps existing gates and footage; shrinking it stops feeds on removed gates.</p>
            </div>
          </header>

          <div className="field-label">Number of gates</div>
          <div className="chips">
            {PRESETS.filter((p) => p >= cfg.min_gates && p <= cfg.max_gates).map((p) => (
              <button key={p} type="button" className={`chip ${p === pickedCount ? 'on' : ''}`} onClick={() => setCount(p)} disabled={running}>{p}</button>
            ))}
            <label className="chip-custom">
              or
              <input type="number" min={cfg.min_gates} max={cfg.max_gates} value={PRESETS.includes(pickedCount) ? '' : pickedCount}
                placeholder="#" disabled={running}
                onChange={(e) => {
                  const n = parseInt(e.target.value, 10);
                  if (Number.isFinite(n)) setCount(Math.max(cfg.min_gates, Math.min(cfg.max_gates, n)));
                }} />
            </label>
          </div>

          <div className="fields">
            <label className="field">
              <span>Venue name</span>
              <input value={venueName} maxLength={120} onChange={(e) => setVenueName(e.target.value)} disabled={running} placeholder="Venue" />
            </label>
            <label className="field">
              <span>Capacity per gate</span>
              <input type="number" min={1} step={10} value={capacity} onChange={(e) => setCapacity(e.target.value)} disabled={running} />
              <small>People a gate area can hold. Drives the congestion thresholds.</small>
            </label>
            <label className="field">
              <span>Venue radius (m)</span>
              <input type="number" min={5} step={5} value={radius} onChange={(e) => setRadius(e.target.value)} disabled={running} />
              <small>Sets walking distance between neighbouring gates.</small>
            </label>
          </div>

          {dropsFeeds.length > 0 && (
            <div className="warn-line">Applying stops the feed on {dropsFeeds.map((g) => g.gate_id).join(', ')}.</div>
          )}

          <div className="step-actions">
            <button type="button" className="btn btn-primary" onClick={apply} disabled={running || applying || !layoutDirty}>
              {applying ? <Loader2 size={15} className="spin" /> : <RefreshCw size={15} />} Apply layout
            </button>
            {!layoutDirty && <span className="muted"><CheckCircle2 size={14} /> Layout is applied</span>}
          </div>
        </section>

        {/* ── 2. footage ────────────────────────────── */}
        <section className={`panel step ${running ? 'is-locked' : ''}`}>
          <header className="step-head">
            <span className="step-num">2</span>
            <div>
              <h2>Footage</h2>
              <p>Drop a video on a gate, or drop several on the list to fill empty gates in order. Nothing plays until you start. A gate without footage remains a routing target.</p>
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
                    ) : (
                      <span className={g.is_monitored ? 'ok' : 'muted'}>
                        {g.is_monitored ? <><FileVideo size={13} /> {g.camera_id}</> : `No footage · capacity ${g.capacity} · routing target only`}
                      </span>
                    )}
                  </div>
                  <div className="arow-btns">
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
          <div><dt>Routing only</dt><dd>{Math.max(0, pickedCount - monitored.size)}</dd></div>
          <div><dt>Gate spacing</dt><dd>{Math.round((2 * Math.PI * (parseFloat(radius) || cfg.radius_m)) / pickedCount)} m</dd></div>
        </dl>
      </aside>

      <input ref={fileInput} type="file" accept={VIDEO_ACCEPT} hidden onChange={onPicked} />
    </main>
  );
}
