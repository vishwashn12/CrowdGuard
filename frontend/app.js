/* CrowdGuard dashboard client.
 *
 * Two screens. Monitor answers "is anything wrong, and where"; Setup answers
 * "what does my venue look like, and what footage is on it". Keeping them apart
 * is deliberate: an operator watching a crush does not need capacity inputs on
 * the same screen.
 *
 * Telemetry arrives over one WebSocket twice a second. Video arrives over a
 * second WebSocket that carries every gate's frames, drawn onto canvases.
 *
 * Video must not use one HTTP request per gate. Browsers allow six HTTP
 * connections per server; never-ending per-gate video streams used them all up,
 * and ordinary requests such as Start then queued inside the browser and never
 * reached the server. WebSockets sit outside that limit.
 */

'use strict';

const RECONNECT_MS = 2000;
const POLL_MS = 1000;

const S = {
  socket: null,
  cards: new Map(),      // gate_id -> {root, refs}
  view: 'monitor',
  session: 'setup',      // 'setup' | 'running'
  footage: 0,            // gates with footage staged
  sessionBusy: false,
  pickedGates: null,     // setup selection, null until the user chooses
  seenAlerts: new Set(),
  painted: false,
  pollTimer: null,
  pendingUpload: null,
  lastGateIds: '',
  video: null,              // the video WebSocket
  framesDrawn: {},          // gate_id -> frames drawn (for diagnostics)
};

const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function toast(msg, kind) {
  const n = el('div', `toast ${kind || ''}`, msg);
  $('toasts').appendChild(n);
  setTimeout(() => n.remove(), 6000);
}

function fmtUptime(sec) {
  const t = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  const p = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${p(m)}:${p(s)}` : `${p(m)}:${p(s)}`;
}

/* ── views ──────────────────────────────────────────────────────── */

function setView(name) {
  S.view = name;
  document.querySelectorAll('.viewswitch-btn').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.view === name);
  });
  $('view-monitor').classList.toggle('is-active', name === 'monitor');
  $('view-setup').classList.toggle('is-active', name === 'setup');
  if (name === 'setup') loadSetup();
}

/* ── connection ─────────────────────────────────────────────────── */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const sock = new WebSocket(`${proto}//${location.host}/ws`);
  S.socket = sock;

  sock.onopen = () => {
    $('connDot').className = 'link on';
    if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
  };
  sock.onmessage = (e) => {
    try { render(JSON.parse(e.data)); } catch (err) { console.error(err); }
  };
  sock.onclose = () => {
    $('connDot').className = 'link off';
    startPolling();
    setTimeout(connect, RECONNECT_MS);
  };
  sock.onerror = () => { $('connDot').className = 'link off'; };
}

/* ── video: every gate's frames over one WebSocket ──────────────── */

const videoPending = new Map();   // gate_id -> newest undrawn JPEG bytes
const videoDecoding = new Set();  // gates with a decode in flight

function connectVideo() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const sock = new WebSocket(`${proto}//${location.host}/ws/video`);
  sock.binaryType = 'arraybuffer';
  S.video = sock;

  /* Message layout: [len(gate_id)][gate_id ascii][jpeg bytes] */
  sock.onmessage = (e) => {
    const bytes = new Uint8Array(e.data);
    const n = bytes[0];
    const gateId = String.fromCharCode(...bytes.subarray(1, 1 + n));
    videoPending.set(gateId, bytes.subarray(1 + n));
    if (!videoDecoding.has(gateId)) drawNextFrame(gateId);
  };
  sock.onclose = () => setTimeout(connectVideo, RECONNECT_MS);
}

/* Decode off the main thread and draw. Only the newest pending frame per gate
   is kept, so if decoding ever falls behind, stale frames are skipped rather
   than queued - playback stays live instead of drifting late. */
async function drawNextFrame(gateId) {
  const jpeg = videoPending.get(gateId);
  if (!jpeg) return;
  videoPending.delete(gateId);
  videoDecoding.add(gateId);
  try {
    const bitmap = await createImageBitmap(new Blob([jpeg], { type: 'image/jpeg' }));
    const card = S.cards.get(gateId);
    if (card) {
      const canvas = card.refs.canvas;
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      S.framesDrawn[gateId] = (S.framesDrawn[gateId] || 0) + 1;
    }
    bitmap.close();
  } catch (_) {
    /* a corrupt frame is simply skipped */
  } finally {
    videoDecoding.delete(gateId);
    if (videoPending.has(gateId)) drawNextFrame(gateId);
  }
}

/* REST fallback so the page keeps updating if the socket cannot hold. */
function startPolling() {
  if (S.pollTimer) return;
  S.pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/status');
      if (r.ok) render(await r.json());
    } catch (_) { /* keep trying */ }
  }, POLL_MS);
}

/* ── session: setup → Start → running → Stop ─────────────────────── */

function renderSession(data) {
  const sess = data.session || {};
  S.session = sess.state || 'setup';
  S.footage = sess.gates_with_footage || 0;

  const running = S.session === 'running';
  const ready = !running && S.footage > 0;

  const chip = $('sessionChip');
  chip.dataset.state = running ? 'running' : (ready ? 'ready' : 'setup');
  chip.textContent = running
    ? `Live · ${fmtUptime(sess.running_s)}`
    : (ready ? `${S.footage} ready` : 'Setup');

  const btn = $('sessionBtn');
  btn.classList.toggle('is-hidden', !(running || ready));
  btn.textContent = running ? 'Stop' : 'Start';
  btn.className = `btn btn-sm ${running ? 'btn-stop' : 'btn-primary'}${running || ready ? '' : ' is-hidden'}`;
  btn.disabled = S.sessionBusy;

  $('readyBar').classList.toggle('is-hidden', !ready);
  $('readyTitle').textContent = `${S.footage} gate${S.footage === 1 ? '' : 's'} ready`;

  $('lockBar').classList.toggle('is-hidden', !running);
  $('stepLayout').classList.toggle('is-locked', running);
  $('stepFootage').classList.toggle('is-locked', running);

  const start = $('startBtn');
  const status = $('startStatus');
  if (running) {
    start.textContent = 'Stop monitoring';
    start.className = 'btn btn-lg btn-stop';
    start.dataset.action = 'stop';
    start.disabled = S.sessionBusy;
    status.className = 'setup-status ok';
    status.textContent = `Running for ${fmtUptime(sess.running_s)}`;
  } else {
    start.textContent = 'Start monitoring';
    start.className = 'btn btn-lg btn-primary';
    start.dataset.action = 'start';
    start.disabled = S.sessionBusy || S.footage === 0;
    status.className = 'setup-status';
    status.textContent = S.footage
      ? `${S.footage} gate${S.footage === 1 ? '' : 's'} will start together.`
      : 'Upload footage to at least one gate first.';
  }
}

const SESSION_TIMEOUT_MS = 20000;

async function sessionAction(action) {
  if (S.sessionBusy) return;
  S.sessionBusy = true;
  /* Never leave the buttons disabled indefinitely: if the server does not
     answer, say so and let the operator try again. */
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), SESSION_TIMEOUT_MS);
  try {
    const r = await fetch(`/api/session/${action}`, { method: 'POST', signal: abort.signal })
      .catch((err) => {
        throw new Error(err.name === 'AbortError'
          ? `The server did not answer within ${SESSION_TIMEOUT_MS / 1000} s. Check the server window, then try again.`
          : 'Could not reach the server. Is it still running?');
      });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `Failed (${r.status})`);

    if (action === 'start') {
      toast(`Monitoring started · ${body.gates.length} gate${body.gates.length === 1 ? '' : 's'} in lockstep`, 'ok');
      setView('monitor');
    } else {
      toast('Monitoring stopped', 'ok');
    }
  } catch (err) {
    toast(err.message, 'err');
  } finally {
    clearTimeout(timer);
    S.sessionBusy = false;
  }
}

function onSessionButton(ev) {
  const action = ev.currentTarget.dataset.action
    || (S.session === 'running' ? 'stop' : 'start');
  sessionAction(action);
}

/* ── banner: exactly one message, the worst one ─────────────────── */

function renderBanner(alerts) {
  const actionable = alerts.filter((a) => a.severity !== 'INFO');
  const banner = $('banner');

  if (!actionable.length || S.session !== 'running') {
    banner.classList.add('is-hidden');
    return;
  }

  const top = actionable[0];
  banner.classList.remove('is-hidden');
  banner.dataset.sev = top.severity;
  $('bannerSev').textContent = top.severity;
  $('bannerTitle').textContent = top.title;
  $('bannerAction').textContent = top.action;
  $('bannerCount').textContent = actionable.length > 1
    ? `+${actionable.length - 1} more` : '';
}

/* ── gate cards ─────────────────────────────────────────────────── */

function buildCard(gateId) {
  const root = el('article', 'gate');

  const top = el('div', 'gate-top');
  const name = el('div', 'gate-name');
  const title = el('b', null, gateId);
  const sub = el('span');
  name.append(title, sub);
  const pill = el('span', 'pill', 'LOW');
  top.append(name, pill);
  root.appendChild(top);

  const pane = el('div', 'pane');
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-label', `${gateId} video`);
  const idle = el('div', 'pane-idle');
  const idleText = el('div', null, 'No footage on this gate');
  const idleBtn = el('button', 'btn btn-sm btn-primary', 'Upload');
  idleBtn.type = 'button';
  idleBtn.onclick = () => pickFile(gateId, idleBtn);
  idle.append(idleText, idleBtn);
  pane.append(canvas, idle);
  root.appendChild(pane);

  /* Click the video to see the tracking at full width; click again to shrink. */
  pane.title = 'Click to enlarge';
  canvas.addEventListener('click', () => {
    const expanded = root.classList.toggle('expanded');
    pane.title = expanded ? 'Click to shrink' : 'Click to enlarge';
    if (expanded) root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  const stats = el('div', 'stats');
  const mk = (label, cls) => {
    const cell = el('div', `stat ${cls || ''}`);
    const v = el('b', null, '—');
    cell.append(v, el('span', null, label));
    stats.appendChild(cell);
    return { cell, v };
  };
  const sPeople = mk('people');
  const sDown = mk('on ground', 'hot');
  const sCounter = mk('counter-flow', 'cf');
  root.appendChild(stats);

  const meter = el('div', 'meter');
  const mTop = el('div', 'meter-top');
  const mLeft = el('span', null, 'congestion');
  const mRight = el('span', 'mono', '0%');
  mTop.append(mLeft, mRight);
  const track = el('div', 'meter-track');
  const fill = el('div', 'meter-fill');
  track.appendChild(fill);
  meter.append(mTop, track);
  root.appendChild(meter);

  const act = el('div', 'gate-act');
  const tag = el('span', 'tag', '—');
  const why = el('p', 'gate-why');
  act.append(tag, why);
  root.appendChild(act);

  const warn = el('div', 'gate-warn is-hidden');
  root.appendChild(warn);

  return {
    root,
    refs: { sub, pill, canvas, idle, sPeople, sDown, sCounter, mRight, fill, tag, why, warn },
  };
}

function patchCard(card, g) {
  const r = card.refs;
  const hasFootage = g.is_monitored;
  const staged = hasFootage && S.session !== 'running';
  const live = hasFootage && S.session === 'running';
  const anomalies = live ? (g.anomaly_count || 0) : 0;

  card.root.dataset.level = live ? g.congestion_level : 'NONE';
  card.root.classList.toggle('alarm', anomalies > 0);
  card.root.classList.toggle('idle', !hasFootage);

  r.sub.textContent = [g.name, `cap ${g.capacity}`].filter(Boolean).join(' · ');

  r.pill.dataset.level = live ? g.congestion_level : 'NONE';
  r.pill.textContent = live ? g.congestion_level : (staged ? 'READY' : 'NO FEED');

  /* Frames arrive over the video socket; with no footage, blank the canvas so
     the last frame of removed footage does not linger. */
  r.idle.classList.toggle('is-hidden', hasFootage);
  if (!hasFootage && r.canvas.width > 0) {
    r.canvas.getContext('2d').clearRect(0, 0, r.canvas.width, r.canvas.height);
  }

  r.sPeople.v.textContent = live ? Math.round(g.crowd_count) : '—';
  r.sDown.v.textContent = live ? g.sitting_count : '—';
  r.sCounter.v.textContent = live ? g.opposite_count : '—';
  r.sDown.cell.classList.toggle('mute', !live || !g.sitting_count);
  r.sCounter.cell.classList.toggle('mute', !live || !g.opposite_count);

  const pct = Math.round((g.congestion_score || 0) * 100);
  r.mRight.textContent = `${pct}%`;
  r.fill.style.width = `${pct}%`;
  r.fill.dataset.level = g.congestion_level;

  if (live) {
    r.tag.dataset.action = g.action;
    r.tag.textContent = (g.action || '—').replace(/_/g, ' ')
      + (g.target_gate ? ` → ${g.target_gate}` : '');
    r.why.textContent = g.reason || '';
  } else {
    r.tag.dataset.action = '';
    r.tag.textContent = staged ? 'READY' : 'NO FOOTAGE';
    r.why.textContent = staged
      ? 'Plays when monitoring starts.'
      : (S.session === 'running' ? 'Routing target on stated capacity.' : '');
  }

  const warnings = [];
  if (live && g.is_stale) warnings.push('feed stale');
  if (live && !g.continuity_ok) warnings.push('dropping frames');
  if (live && !g.classifier_active) warnings.push('rule-based only');
  r.warn.textContent = warnings.join(' · ');
  r.warn.classList.toggle('is-hidden', !warnings.length);
}

function renderGates(data) {
  const gates = data.gates || {};
  const order = data.gate_order || Object.keys(gates);
  const host = $('gates');

  /* Rebuild wholesale if the gate set changed, e.g. venue reconfigured. */
  const key = order.join(',');
  if (key !== S.lastGateIds) {
    S.lastGateIds = key;
    host.textContent = '';
    S.cards.clear();
  }

  order.forEach((id) => {
    const g = gates[id];
    if (!g) return;
    let card = S.cards.get(id);
    if (!card) {
      card = buildCard(id);
      S.cards.set(id, card);
      host.appendChild(card.root);
    }
    patchCard(card, g);
  });

  const withFootage = order.filter((id) => gates[id] && gates[id].is_monitored).length;
  $('gateSummary').textContent = S.session === 'running'
    ? `${withFootage} of ${order.length} live`
    : `${withFootage} of ${order.length} with footage`;

  const nothingStaged = withFootage === 0;
  $('firstRun').classList.toggle('is-hidden', !nothingStaged);
  $('monitorGrid').classList.toggle('is-hidden', nothingStaged);
}

/* ── map ────────────────────────────────────────────────────────── */

const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  return n;
};

const levelFill = (level, live) => live ? ({
  LOW: 'var(--low)', MODERATE: 'var(--moderate)',
  HIGH: 'var(--high)', CRITICAL: 'var(--critical)',
}[level] || 'var(--none)') : 'var(--none)';

function renderMap(data) {
  const host = $('map');
  const gates = data.gates || {};
  const order = data.gate_order || Object.keys(gates);
  host.textContent = '';
  if (!order.length) return;

  const cx = 150, cy = 150;
  const radius = order.length > 10 ? 104 : 96;

  host.appendChild(svg('circle', {
    cx, cy, r: radius, fill: 'none', stroke: 'var(--line)', 'stroke-width': 9,
  }));
  host.appendChild(svg('circle', {
    cx, cy, r: radius - 30, fill: 'rgba(255,255,255,.015)',
    stroke: 'var(--line-soft)', 'stroke-width': 1, 'stroke-dasharray': '3 5',
  }));

  const defs = svg('defs');
  const marker = svg('marker', {
    id: 'arw', viewBox: '0 0 10 10', refX: 9, refY: 5,
    markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse',
  });
  marker.appendChild(svg('path', { d: 'M0 0 L10 5 L0 10 z', fill: 'var(--high)' }));
  defs.appendChild(marker);
  host.appendChild(defs);

  const pos = new Map();
  order.forEach((id, i) => {
    const a = (i / order.length) * Math.PI * 2 - Math.PI / 2;
    pos.set(id, { x: cx + radius * Math.cos(a), y: cy + radius * Math.sin(a) });
  });

  /* diversion arrows, bowed outward so they read as going around */
  order.forEach((id) => {
    const g = gates[id];
    if (!g || g.action !== 'DIVERT' || !g.target_gate) return;
    const a = pos.get(id), b = pos.get(g.target_gate);
    if (!a || !b) return;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    host.appendChild(svg('path', {
      d: `M ${a.x} ${a.y} Q ${mx + (mx - cx) * .5} ${my + (my - cy) * .5} ${b.x} ${b.y}`,
      fill: 'none', stroke: 'var(--high)', 'stroke-width': 2.4,
      'stroke-dasharray': '6 4', 'marker-end': 'url(#arw)',
    }));
  });

  const nodeR = order.length > 12 ? 10 : 13;

  order.forEach((id) => {
    const g = gates[id], p = pos.get(id);
    if (!g || !p) return;

    if ((g.anomaly_count || 0) > 0) {
      host.appendChild(svg('circle', {
        cx: p.x, cy: p.y, r: nodeR + 5, fill: 'none',
        stroke: 'var(--sitting)', 'stroke-width': 2,
      }));
    }

    host.appendChild(svg('circle', {
      cx: p.x, cy: p.y, r: nodeR,
      fill: levelFill(g.congestion_level, g.is_monitored),
      stroke: 'var(--bg)', 'stroke-width': 2.5,
    }));

    const t = svg('text', {
      x: p.x, y: p.y + 3.5, 'text-anchor': 'middle', fill: '#fff',
      'font-size': order.length > 12 ? 8 : 9.5, 'font-weight': 700,
      'font-family': 'inherit',
    });
    t.textContent = id;
    host.appendChild(t);

    const out = 1 + (nodeR + 12) / radius;
    const c = svg('text', {
      x: cx + (p.x - cx) * out, y: cy + (p.y - cy) * out + 3,
      'text-anchor': 'middle', fill: 'var(--ink-3)',
      'font-size': 8.5, 'font-family': 'inherit',
    });
    c.textContent = g.is_monitored ? Math.round(g.crowd_count) : '·';
    host.appendChild(c);
  });
}

/* ── advisories ─────────────────────────────────────────────────── */

function renderAdvisories(data) {
  const alerts = data.alerts || [];
  const host = $('advisories');
  host.textContent = '';

  $('advisoryCount').textContent = alerts.length
    ? `${alerts.length} active` : '';

  if (!alerts.length) {
    host.appendChild(el('p', 'empty', 'Nothing to report.'));
    return;
  }

  alerts.forEach((a) => {
    const row = el('div', 'adv');
    row.dataset.sev = a.severity;

    const top = el('div', 'adv-top');
    if (a.gate_id && a.gate_id !== 'ALL') {
      top.appendChild(el('span', 'adv-gate', a.gate_id));
    }
    top.appendChild(el('span', 'adv-title', a.title));
    row.appendChild(top);
    row.appendChild(el('p', 'adv-do', a.action));
    host.appendChild(row);

    if (!S.seenAlerts.has(a.alert_id)) {
      S.seenAlerts.add(a.alert_id);
      if (S.painted && (a.severity === 'EMERGENCY' || a.severity === 'ALERT')) {
        toast(a.title, a.severity === 'EMERGENCY' ? 'err' : '');
      }
    }
  });
}

/* ── diagnostics ────────────────────────────────────────────────── */

function renderDiag(data) {
  if (!$('diag').open) return;
  const p = data.pipeline || {};
  const host = $('diagGrid');
  host.textContent = '';

  const dropped = p.buffers?.anomaly?.dropped ?? 0;
  const breaks = p.posetrack?.continuity_breaks ?? 0;

  const play = Object.values(p.playback?.cameras || {});
  const displayFps = play.length
    ? (play.reduce((s, c) => s + (c.display_fps || 0), 0) / play.length).toFixed(1)
    : '0';
  const sourceFps = play.length ? play[0].source_fps : 0;

  const items = [
    ['cameras', Object.keys(p.cameras || {}).length, false],
    ['playback fps', `${displayFps} / ${sourceFps}`, false],
    ['pose latency', `${p.posetrack?.avg_inference_ms ?? 0} ms`, false],
    ['csrnet latency', `${p.csrnet?.avg_inference_ms ?? 0} ms`, false],
    ['frames refused', dropped, dropped > 0],
    ['continuity breaks', breaks, breaks > 0],
    ['motion valid', `${Math.round((p.synchronizer?.valid_ratio ?? 0) * 100)}%`, false],
    ['anomaly queue', p.buffers?.anomaly?.size ?? 0, false],
    ['evaluations', p.decisions?.evaluations ?? 0, false],
  ];

  items.forEach(([label, value, bad]) => {
    const cell = el('div', `diag-item ${bad ? 'bad' : ''}`);
    cell.append(el('b', null, String(value)), el('span', null, label));
    host.appendChild(cell);
  });
}

/* ── render ─────────────────────────────────────────────────────── */

function render(data) {
  const v = data.venue || {};
  $('appbarMeta').textContent =
    `${v.venue_name || 'venue'} · ${data.device || '—'} · up ${fmtUptime(data.uptime_s)}`;

  renderSession(data);
  renderBanner(data.alerts || []);
  renderGates(data);
  renderMap(data);
  renderAdvisories(data);
  renderDiag(data);
  S.painted = true;
}

/* ── setup ──────────────────────────────────────────────────────── */

const PRESETS = [2, 3, 4, 6, 8, 10, 12];

async function loadSetup() {
  try {
    const r = await fetch('/api/venue/config');
    if (!r.ok) return;
    const cfg = await r.json();

    if (S.pickedGates === null) S.pickedGates = cfg.gate_count;

    $('capacityInput').value = cfg.gates.length
      ? Math.round(cfg.gates.reduce((s, g) => s + g.capacity, 0) / cfg.gates.length)
      : 500;
    $('radiusInput').value = Math.round(cfg.radius_m);
    $('gateCountInput').min = cfg.min_gates;
    $('gateCountInput').max = cfg.max_gates;

    renderGatePicker();
    renderAssign(cfg);
  } catch (err) {
    console.error(err);
  }
}

function renderGatePicker() {
  const host = $('gateCountPick');
  host.textContent = '';
  PRESETS.forEach((n) => {
    const b = el('button', `chip ${n === S.pickedGates ? 'on' : ''}`, String(n));
    b.type = 'button';
    b.onclick = () => { S.pickedGates = n; $('gateCountInput').value = ''; renderGatePicker(); };
    host.appendChild(b);
  });
  if (!PRESETS.includes(S.pickedGates)) {
    $('gateCountInput').value = S.pickedGates;
  }
}

function renderAssign(cfg) {
  const host = $('assign');
  host.textContent = '';

  cfg.gates.forEach((g) => {
    const row = el('div', `arow ${g.is_monitored ? 'live' : ''}`);
    row.appendChild(el('div', 'arow-id', g.gate_id));

    const main = el('div', 'arow-main');
    main.appendChild(el('div', 'arow-name', g.name || g.gate_id));
    const sub = el('div', `arow-sub ${g.is_monitored ? 'live' : ''}`);
    sub.textContent = g.is_monitored
      ? `live · ${g.camera_id}`
      : `no footage · capacity ${g.capacity} · routing target only`;
    main.appendChild(sub);
    row.appendChild(main);

    const btns = el('div', 'arow-btns');
    const up = el('button', 'btn btn-sm', g.is_monitored ? 'Replace' : 'Upload');
    up.type = 'button';
    up.onclick = () => pickFile(g.gate_id, up);
    btns.appendChild(up);

    if (g.is_monitored) {
      const rm = el('button', 'btn btn-sm btn-danger', 'Remove');
      rm.type = 'button';
      rm.onclick = () => removeCamera(g.camera_id, rm);
      btns.appendChild(rm);
    }
    row.appendChild(btns);
    host.appendChild(row);
  });
}

async function applyVenue() {
  const btn = $('applyVenue');
  const status = $('venueStatus');
  const custom = parseInt($('gateCountInput').value, 10);
  const count = Number.isFinite(custom) && custom > 0 ? custom : S.pickedGates;

  btn.disabled = true;
  status.className = 'setup-status';
  status.textContent = 'Applying…';

  try {
    const r = await fetch('/api/venue/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gate_count: count,
        default_capacity: parseInt($('capacityInput').value, 10) || undefined,
        radius_m: parseFloat($('radiusInput').value) || undefined,
      }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || `Failed (${r.status})`);

    S.pickedGates = body.gate_count;
    status.className = 'setup-status ok';
    const bits = [`${body.gate_count} gates`];
    if (body.cameras_stopped.length) bits.push(`${body.cameras_stopped.length} feed(s) stopped`);
    if (body.cameras_kept.length) bits.push(`${body.cameras_kept.length} kept`);
    status.textContent = bits.join(' · ');
    toast(`Venue set to ${body.gate_count} gates`, 'ok');

    loadSetup();
  } catch (err) {
    status.className = 'setup-status err';
    status.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

/* ── upload ─────────────────────────────────────────────────────── */

function pickFile(gateId, btn) {
  S.pendingUpload = { gateId, btn, label: btn.textContent };
  const input = $('fileInput');
  input.value = '';
  input.click();
}

async function onFileChosen(ev) {
  const file = ev.target.files && ev.target.files[0];
  const pending = S.pendingUpload;
  S.pendingUpload = null;
  if (!file || !pending) return;

  const { gateId, btn, label } = pending;
  btn.disabled = true;
  btn.textContent = 'Uploading…';

  const form = new FormData();
  form.append('file', file);

  try {
    const r = await fetch(`/api/gates/${encodeURIComponent(gateId)}/upload`, {
      method: 'POST', body: form,
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || `Upload failed (${r.status})`);

    toast(`${gateId} ready · ${body.filename} · press Start when all gates are loaded`, 'ok');
    if (S.view === 'setup') loadSetup();
  } catch (err) {
    toast(`${gateId}: ${err.message}`, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

async function removeCamera(cameraId, btn) {
  btn.disabled = true;
  try {
    const r = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}`, { method: 'DELETE' });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `Failed (${r.status})`);
    }
    toast('Feed removed', 'ok');
    loadSetup();
  } catch (err) {
    toast(err.message, 'err');
  } finally {
    btn.disabled = false;
  }
}

/* ── boot ───────────────────────────────────────────────────────── */

function init() {
  document.querySelectorAll('.viewswitch-btn').forEach((b) => {
    b.onclick = () => setView(b.dataset.view);
  });
  document.querySelectorAll('[data-goto]').forEach((b) => {
    b.onclick = () => setView(b.dataset.goto);
  });

  $('fileInput').addEventListener('change', onFileChosen);
  $('applyVenue').addEventListener('click', applyVenue);
  $('sessionBtn').addEventListener('click', onSessionButton);
  $('startBtn').addEventListener('click', onSessionButton);
  document.querySelectorAll('.readybar [data-action="start"]').forEach((b) => {
    b.addEventListener('click', () => sessionAction('start'));
  });
  $('gateCountInput').addEventListener('input', (e) => {
    const n = parseInt(e.target.value, 10);
    if (Number.isFinite(n)) { S.pickedGates = n; renderGatePicker(); }
  });
  $('diag').addEventListener('toggle', () => {
    if ($('diag').open) fetch('/api/status').then((r) => r.json()).then(renderDiag).catch(() => {});
  });

  connect();
  connectVideo();

  /* paint once immediately so the page is not blank before the first push */
  fetch('/api/status')
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d) render(d); })
    .catch(() => {});
}

document.addEventListener('DOMContentLoaded', init);
