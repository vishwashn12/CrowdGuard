/* Calibrate a gate camera: mark four corners of a rectangle on the ground and
 * say how big it is. The rectangle becomes the gate's counting zone, and its
 * perspective gives walking speed in m/s and density in people/m^2. */
import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { Loader2, RotateCcw, Ruler, Trash2, X as CloseIcon } from 'lucide-react';
import type { Calibration } from '../lib/types';
import { api } from '../lib/api';
import { useSize } from '../lib/misc';

type Pt = [number, number];
const FULL_DENSITY = 5; // people per m^2 at which a zone counts as full (backend default)

export function CalibrationDialog({ gateId, gateName, existing, onClose, onSaved }: {
  gateId: string;
  gateName: string;
  existing: Calibration | null;
  onClose: () => void;
  onSaved: (message: string) => void;
}) {
  const [points, setPoints] = useState<Pt[]>(existing?.points ?? []);
  const [width, setWidth] = useState(existing ? String(existing.width_m) : '');
  const [length, setLength] = useState(existing ? String(existing.length_m) : '');
  const [src] = useState(() => api.stillUrl(gateId));
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const imgBox = useRef<HTMLDivElement>(null);
  const { width: bw, height: bh } = useSize(imgBox);
  const X = (x: number) => x * bw;
  const Y = (y: number) => y * bh;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const toNorm = (e: PointerEvent): Pt | null => {
    const r = imgBox.current?.getBoundingClientRect();
    if (!r) return null;
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    return [x, y];
  };

  const onDown = (e: PointerEvent) => {
    if (points.length >= 4 || dragging !== null) return;
    const p = toNorm(e);
    if (p) setPoints([...points, p]);
  };
  const onMove = (e: PointerEvent) => {
    if (dragging === null) return;
    const p = toNorm(e);
    if (p) setPoints(points.map((q, i) => (i === dragging ? p : q)));
  };

  const w = parseFloat(width);
  const l = parseFloat(length);
  const sizeOk = w >= 0.5 && l >= 0.5;
  const area = sizeOk ? w * l : null;
  const ready = points.length === 4 && sizeOk;

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.setCalibration(gateId, { points, width_m: w, length_m: l });
      onSaved(`${gateId} calibrated: ${(w * l).toFixed(0)} m² zone, full at ${r.capacity} people`);
      onClose();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    setBusy(true);
    try {
      const r = await api.clearCalibration(gateId);
      onSaved(`${gateId} calibration removed: assumes ${r.capacity} people`);
      onClose();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const edgeLabel = (a: Pt, b: Pt, text: string) => (
    <text x={X((a[0] + b[0]) / 2)} y={Y((a[1] + b[1]) / 2)} className="cal-edge" dy={-10}>{text}</text>
  );

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={`Calibrate ${gateId}`}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal cal">
        <header className="modal-head">
          <Ruler size={18} />
          <strong>Calibrate {gateId}</strong>
          <span className="muted">{gateName}</span>
          <span className="spacer" />
          <button className="icon-btn" onClick={onClose} aria-label="Close"><CloseIcon size={18} /></button>
        </header>

        <div className="cal-body">
          <div className="cal-stage">
            <div className="cal-img" ref={imgBox} onPointerDown={onDown} onPointerMove={onMove}
              onPointerUp={() => setDragging(null)} onPointerLeave={() => setDragging(null)}
              data-done={points.length >= 4}>
              <img src={src} alt={`${gateId} first frame`} onLoad={() => setLoaded(true)} draggable={false}
                onError={() => setError('Could not load a frame from this footage.')} />
              {!loaded && !error && <div className="cal-loading"><Loader2 className="spin" /> Loading frame…</div>}
              <svg className="cal-svg">
                {points.length >= 2 && (
                  <polygon points={points.map(([x, y]) => `${X(x)},${Y(y)}`).join(' ')} className="cal-fill" />
                )}
                {points.map((p, i) => i > 0 && (
                  <line key={`l${i}`} x1={X(points[i - 1][0])} y1={Y(points[i - 1][1])}
                    x2={X(p[0])} y2={Y(p[1])} className={`cal-line ${i === 1 ? 'w' : i === 2 ? 'l' : ''}`} />
                ))}
                {points.length === 4 && (
                  <line x1={X(points[3][0])} y1={Y(points[3][1])} x2={X(points[0][0])} y2={Y(points[0][1])} className="cal-line" />
                )}
                {points.length >= 2 && edgeLabel(points[0], points[1], `width ${sizeOk ? w : '?'} m`)}
                {points.length >= 3 && edgeLabel(points[1], points[2], `length ${sizeOk ? l : '?'} m`)}
                {points.map(([x, y], i) => (
                  <g key={i} className="cal-handle" onPointerDown={(e) => { e.stopPropagation(); setDragging(i); }}>
                    <circle cx={X(x)} cy={Y(y)} r={13} />
                    <text x={X(x)} y={Y(y)} dy={5}>{i + 1}</text>
                  </g>
                ))}
              </svg>
            </div>
          </div>

          <aside className="cal-side">
            <ol className="cal-steps">
              <li data-done={points.length >= 4}>
                <strong>Mark 4 corners on the ground</strong>
                <span>Click the corners of a rectangle on the floor of the queue area, in order around it.
                  Use where feet touch the ground, not heads. Drag a corner to adjust.</span>
                <span className="cal-count">{points.length}/4 marked</span>
              </li>
              <li data-done={sizeOk}>
                <strong>Enter its real size</strong>
                <span>Measure it, or estimate from floor tiles, barrier lengths or lane widths.</span>
                <div className="cal-inputs">
                  <label><span className="cal-key w">1 → 2</span> width
                    <input type="number" min={0.5} step={0.5} value={width} onChange={(e) => setWidth(e.target.value)} placeholder="m" />
                  </label>
                  <label><span className="cal-key l">2 → 3</span> length
                    <input type="number" min={0.5} step={0.5} value={length} onChange={(e) => setLength(e.target.value)} placeholder="m" />
                  </label>
                </div>
              </li>
            </ol>

            <div className="cal-result">
              {area ? (
                <>
                  <b>{area.toFixed(0)} m²</b>
                  <span>counting zone · full at {Math.round(area * FULL_DENSITY)} people ({FULL_DENSITY}/m²)</span>
                </>
              ) : <span className="muted">The zone's area and capacity appear here.</span>}
            </div>

            {error && <div className="cal-error">{error}</div>}

            <div className="cal-actions">
              <button type="button" className="btn btn-primary" disabled={!ready || busy} onClick={save}>
                {busy ? <Loader2 size={15} className="spin" /> : <Ruler size={15} />} Save calibration
              </button>
              <button type="button" className="btn" onClick={() => setPoints([])} disabled={!points.length || busy}>
                <RotateCcw size={14} /> Re-mark
              </button>
              {existing && (
                <button type="button" className="btn btn-ghost danger-text" onClick={clear} disabled={busy}>
                  <Trash2 size={14} /> Remove
                </button>
              )}
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
