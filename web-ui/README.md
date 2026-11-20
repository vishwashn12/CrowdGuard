# CrowdGuard dashboard (React + TypeScript + Vite)

The operator dashboard. The FastAPI server serves the production build from
`web-ui/dist/` at `/`. The original vanilla-JS page is still at `/classic`.

```bash
cd web-ui
npm install          # once
npm run build        # writes dist/, then restart or reload the server page
npm run dev          # hot-reload dev server on :5173, proxying /api and /ws to :8000
```

Set `CROWDGUARD_BACKEND=http://host:port` to point `npm run dev` at another server.

## Layout

| Area | File |
| :--- | :--- |
| Top bar: overall status, Start/Stop, chime toggle, diagnostics, clock | `components/TopBar.tsx` |
| Banner: the single most urgent alert | `components/Banner.tsx` |
| Stage toggle (V key): venue ring with animated diversion arcs, or camera wall | `components/VenueMap.tsx`, `components/CameraWall.tsx` |
| Gate rail: per-gate level, count, score sparkline, recommendation | `components/GateRail.tsx` |
| Incident queue: active alerts and session log | `components/IncidentPanel.tsx` |
| Gate focus drawer (Esc closes): video, score breakdown, every candidate gate considered | `components/GateDrawer.tsx` |
| Setup: layout, drag-and-drop footage with progress, start | `views/SetupView.tsx` |

Data comes from the existing API only: `/ws` (telemetry, 2 Hz), `/ws/video`
(every gate's frames over one socket, painted by `lib/video.ts`), and the REST
endpoints in `crowdguard/web/server.py`. No backend behaviour changed.
