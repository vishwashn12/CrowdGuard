import { useCallback, useEffect, useRef, useState } from 'react';
import { api, wsUrl } from './api';
import type { Snapshot } from './types';

const RECONNECT_MS = 2000;
const POLL_MS = 1000;

/** Live system snapshot, pushed over /ws twice a second. Falls back to REST
 *  polling while the socket is down, so the screen never freezes silently. */
export function useTelemetry() {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try { setSnap(await api.status()); } catch { /* keep trying */ }
  }, []);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: number | null = null;
    let disposed = false;

    const stopPolling = () => {
      if (pollRef.current !== null) { clearInterval(pollRef.current); pollRef.current = null; }
    };
    const startPolling = () => {
      if (pollRef.current === null) pollRef.current = window.setInterval(refresh, POLL_MS);
    };

    const connect = () => {
      socket = new WebSocket(wsUrl('/ws'));
      socket.onopen = () => { setConnected(true); stopPolling(); };
      socket.onmessage = (e) => {
        try { setSnap(JSON.parse(e.data as string)); } catch (err) { console.error(err); }
      };
      socket.onclose = () => {
        setConnected(false);
        if (disposed) return;
        startPolling();
        retry = window.setTimeout(connect, RECONNECT_MS);
      };
      socket.onerror = () => socket?.close();
    };

    void refresh();
    connect();
    return () => {
      disposed = true;
      stopPolling();
      if (retry !== null) clearTimeout(retry);
      socket?.close();
    };
  }, [refresh]);

  return { snap, connected, refresh };
}
