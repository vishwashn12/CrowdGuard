import type { Decision, Snapshot, VenueConfig } from './types';

export function wsUrl(path: string): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}${path}`;
}

async function readError(r: Response): Promise<string> {
  const body = await r.json().catch(() => ({}));
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
  return `Request failed (${r.status})`;
}

async function json<T>(input: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(input, init);
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err;
    throw new Error('Could not reach the server. Is it still running?');
  }
  if (!r.ok) throw new Error(await readError(r));
  return r.json() as Promise<T>;
}

export const api = {
  status: () => json<Snapshot>('/api/status'),
  venueConfig: () => json<VenueConfig>('/api/venue/config'),
  decisions: () => json<Record<string, Decision>>('/api/decisions'),

  configureVenue: (body: { gate_count: number; venue_name?: string }) =>
    json<{ gate_count: number; cameras_kept: string[]; cameras_stopped: string[] }>(
      '/api/venue/configure',
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    ),

  /** Start or stop the session. Never hangs: gives up after `timeoutMs`. */
  async session(action: 'start' | 'stop', timeoutMs = 20000) {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), timeoutMs);
    try {
      return await json<{ gates?: string[] }>(`/api/session/${action}`, {
        method: 'POST',
        signal: abort.signal,
      });
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        throw new Error(
          `The server did not answer within ${timeoutMs / 1000} s. Check the server window, then try again.`,
        );
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  },

  setCalibration: (gateId: string, body: { points: [number, number][]; width_m: number; length_m: number }) =>
    json<{ capacity: number }>(`/api/gates/${encodeURIComponent(gateId)}/calibration`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }),

  clearCalibration: (gateId: string) =>
    json<{ capacity: number }>(`/api/gates/${encodeURIComponent(gateId)}/calibration`, { method: 'DELETE' }),

  stillUrl: (gateId: string) => `/api/gates/${encodeURIComponent(gateId)}/still?t=${Date.now()}`,

  removeCamera: (cameraId: string) =>
    json<{ gate_id: string }>(`/api/cameras/${encodeURIComponent(cameraId)}`, { method: 'DELETE' }),
};

export interface UploadResult {
  gate_id: string;
  camera_id: string;
  filename: string;
  size_bytes: number;
  source_fps: number | null;
  frame_count: number | null;
}

/** Upload footage to a gate, reporting progress (0..1). Uses XHR, because
 *  fetch has no upload-progress events. */
export function uploadFootage(
  gateId: string,
  file: File,
  onProgress: (fraction: number) => void,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/gates/${encodeURIComponent(gateId)}/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: { detail?: unknown } = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as unknown as UploadResult);
      else reject(new Error(typeof body.detail === 'string' ? body.detail : `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('Could not reach the server. Is it still running?'));
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

export const VIDEO_ACCEPT = '.mp4,.avi,.mov,.mkv,.m4v';
export const VIDEO_SUFFIXES = VIDEO_ACCEPT.split(',');
