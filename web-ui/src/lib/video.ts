import { wsUrl } from './api';

/* Every gate's video arrives over ONE WebSocket as tagged binary frames:
 *   [len(gate_id)][gate_id ascii][jpeg bytes]
 *
 * Browsers allow six HTTP connections per server; one long-lived request per
 * gate used them all up and ordinary requests (Start) then never left the
 * browser. A WebSocket sits outside that limit.
 *
 * The hub decodes each frame once and paints it onto every canvas currently
 * showing that gate (camera wall, rail thumbnail, focus drawer). Only the
 * newest pending frame per gate is kept, so a slow client skips frames rather
 * than drifting behind live. */

type OnFirst = (() => void) | null;

class VideoHub {
  private sinks = new Map<string, Map<HTMLCanvasElement, OnFirst>>();
  private latest = new Map<string, ImageBitmap>();
  private pending = new Map<string, Uint8Array>();
  private decoding = new Set<string>();
  private started = false;

  start(): void {
    if (this.started) return;
    this.started = true;
    this.connect();
  }

  private connect(): void {
    const sock = new WebSocket(wsUrl('/ws/video'));
    sock.binaryType = 'arraybuffer';
    sock.onmessage = (e) => {
      const bytes = new Uint8Array(e.data as ArrayBuffer);
      const n = bytes[0];
      let id = '';
      for (let i = 1; i <= n; i++) id += String.fromCharCode(bytes[i]);
      this.pending.set(id, bytes.subarray(1 + n));
      if (!this.decoding.has(id)) void this.decode(id);
    };
    sock.onclose = () => setTimeout(() => this.connect(), 2000);
  }

  private async decode(id: string): Promise<void> {
    const jpeg = this.pending.get(id);
    if (!jpeg) return;
    this.pending.delete(id);
    this.decoding.add(id);
    try {
      const bitmap = await createImageBitmap(new Blob([jpeg], { type: 'image/jpeg' }));
      const previous = this.latest.get(id);
      this.latest.set(id, bitmap);
      const sinks = this.sinks.get(id);
      sinks?.forEach((onFirst, canvas) => {
        draw(canvas, bitmap);
        if (onFirst) { onFirst(); sinks.set(canvas, null); }
      });
      previous?.close();
    } catch {
      /* a corrupt frame is skipped */
    } finally {
      this.decoding.delete(id);
      if (this.pending.has(id)) void this.decode(id);
    }
  }

  /** Show a gate on a canvas. Returns the detach function. */
  attach(id: string, canvas: HTMLCanvasElement, onFirstFrame?: () => void): () => void {
    let sinks = this.sinks.get(id);
    if (!sinks) { sinks = new Map(); this.sinks.set(id, sinks); }
    const bitmap = this.latest.get(id);
    if (bitmap) {
      draw(canvas, bitmap);
      onFirstFrame?.();
      sinks.set(canvas, null);
    } else {
      sinks.set(canvas, onFirstFrame ?? null);
    }
    return () => { this.sinks.get(id)?.delete(canvas); };
  }

  /** Width / height of a gate's video, once a frame has arrived. */
  aspect(id: string): number | null {
    const b = this.latest.get(id);
    return b && b.height ? b.width / b.height : null;
  }

  /** Drop a gate's last frame, e.g. after its footage is removed. */
  forget(id: string): void {
    this.latest.get(id)?.close();
    this.latest.delete(id);
  }
}

function draw(canvas: HTMLCanvasElement, bitmap: ImageBitmap): void {
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  canvas.getContext('2d')?.drawImage(bitmap, 0, 0);
}

export const videoHub = new VideoHub();
