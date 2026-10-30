// Room scan viewer v2: scan list + upload, GLB model + detected 3D boxes, review, Send to backend.
// Contract: app/API_V2.md ("Server HTTP API v2"). No build step; three.js is vendored under /static/vendor/three.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const ACTIVE = new Set(['queued', 'running']);
const enc = encodeURIComponent;

async function api(path, opts = {}) {
  const init = { ...opts, headers: { ...(opts.headers || {}) } };
  if (init.body !== undefined && typeof init.body !== 'string') {
    init.body = JSON.stringify(init.body);
    init.headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(path, init);
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) throw httpError(res.status, res.statusText, data);
  return data;
}

function httpError(status, statusText, data) {
  let detail = data && typeof data === 'object' && 'detail' in data ? data.detail : data;
  if (detail && typeof detail !== 'string') detail = JSON.stringify(detail);
  const err = new Error(`${status}${statusText ? ' ' + statusText : ''}${detail ? ': ' + detail : ''}`);
  err.status = status;
  err.detail = detail;
  return err;
}

// GET that maps 404 to null.
async function apiOpt(path) {
  try { return await api(path); } catch (err) { if (err.status === 404) return null; throw err; }
}

function toast(msg, kind = 'info', ms = 4500) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), kind === 'error' ? ms * 2 : ms);
  return el;
}

// Stable color per class name (FNV-1a hash -> hue).
const colorCache = new Map();
function classColor(name) {
  if (!colorCache.has(name)) {
    let h = 2166136261;
    for (let i = 0; i < name.length; i++) { h ^= name.charCodeAt(i); h = Math.imul(h, 16777619); }
    h >>>= 0;
    const hue = (h % 360) / 360;
    const sat = 0.65 + ((h >>> 9) % 20) / 100;
    const lit = 0.55 + ((h >>> 17) % 10) / 100;
    colorCache.set(name, new THREE.Color().setHSL(hue, sat, lit, THREE.SRGBColorSpace));
  }
  return colorCache.get(name);
}
const cssColor = (name) => '#' + classColor(name).getHexString(THREE.SRGBColorSpace);
const fmt = (v, d = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '–');
const pct = (p) => `${Math.round(Math.max(0, Math.min(1, Number(p) || 0)) * 100)}%`;

function fmtTime(iso, withDate = true) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d)) return String(iso);
  const opts = { hour: '2-digit', minute: '2-digit' };
  if (withDate) Object.assign(opts, { month: 'short', day: 'numeric' });
  return d.toLocaleString(undefined, opts);
}

function parsePrompts(text) {
  const seen = new Set();
  const out = [];
  for (const p of String(text || '').split(/[\n,]+/)) {
    const t = p.trim();
    if (t && !seen.has(t.toLowerCase())) { seen.add(t.toLowerCase()); out.push(t); }
  }
  return out;
}

// ---------------------------------------------------------------- three setup
const host = $('canvas-host');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0f1114);
host.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.className = 'label-layer';
host.appendChild(labelRenderer.domElement);

const scene3 = new THREE.Scene();
scene3.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0)); // only matters for non-unlit materials
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
camera.position.set(3, 3, 3);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;
let userMovedCamera = false;
controls.addEventListener('start', () => { userMovedCamera = true; cancelFly(); });

// Everything in scan-native coordinates lives under "world".
const world = new THREE.Group();
world.name = 'world';
scene3.add(world);
const modelRoot = new THREE.Group();
modelRoot.name = 'model';
const boxesRoot = new THREE.Group();
boxesRoot.name = 'boxes';
const measureRoot = new THREE.Group();   // measure-tool markers, line and label
measureRoot.name = 'measure';
world.add(modelRoot, boxesRoot, measureRoot);

function resize() {
  const w = host.clientWidth || 1, h = host.clientHeight || 1;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
resize();

let fly = null;
function cancelFly() { fly = null; }
function flyTo(target, position, ms = 600) {
  fly = { t0: performance.now(), ms, fromT: controls.target.clone(), toT: target.clone(), fromP: camera.position.clone(), toP: position.clone() };
}

function animate() {
  requestAnimationFrame(animate);
  if (fly) {
    const k = Math.min(1, (performance.now() - fly.t0) / fly.ms);
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    controls.target.lerpVectors(fly.fromT, fly.toT, e);
    camera.position.lerpVectors(fly.fromP, fly.toP, e);
    if (k >= 1) fly = null;
  }
  controls.update();
  renderer.render(scene3, camera);
  labelRenderer.render(scene3, camera);
}
animate();

// ---------------------------------------------------------------- state
const state = {
  config: { default_prompts: [], max_upload_mb: 500 },
  scans: [],                 // summaries from GET /api/scans
  scansLoaded: false,
  scanId: null,              // selected scan
  record: null,              // GET /api/scans/{id}
  job: null,                 // GET /api/scans/{id}/job (null if none)
  rawDoc: null,              // objects?version=raw
  reviewedDoc: null,         // objects?version=reviewed (only if it matches the raw run)
  upAxis: 'Y',
  objects: [],               // entries, see buildEntry()
  baseline: new Map(),       // object_id -> {name, rejected}: the last sent (or raw) state
  classVisible: new Map(),
  threshold: 0,
  selectedId: null,
  showLabels: true,
  modelLoaded: false,
  modelVersion: Date.now(),
  sending: false,
  uploading: false,
  busyAction: false,         // flip / run request in flight
  loadToken: 0,
};

const summary = () => state.scans.find((s) => s.scan_id === state.scanId) || null;
// Shelf scans come from store videos: their 3D model exists only after the convert job.
const isShelf = () => (summary()?.scan_type || state.record?.scan_type) === 'shelf';
const hasModel = () => summary()?.has_model !== false;
const SHELF_VIDEO = /\.(mov|mp4|m4v)$/i;
const scanStatus = () => summary()?.status || state.record?.status || null;
function isBusy() {
  const st = scanStatus();
  return ACTIVE.has(st) || ACTIVE.has(state.job?.state);
}
function isStale() {
  const s = summary();
  return !!(s?.stale || state.record?.stale);
}
function orientation() {
  return summary()?.orientation || state.record?.orientation || null;
}

// ---------------------------------------------------------------- framing
function sceneBox() {
  world.updateMatrixWorld(true);
  const box = new THREE.Box3();
  if (state.modelLoaded) box.setFromObject(modelRoot);
  const sb = state.rawDoc?.scan?.scene_bounds;
  if (box.isEmpty() && sb?.min && sb?.max) {
    box.copy(new THREE.Box3(new THREE.Vector3(...sb.min), new THREE.Vector3(...sb.max)).applyMatrix4(world.matrixWorld));
  }
  if (box.isEmpty() && state.objects.length) box.setFromObject(boxesRoot);
  if (box.isEmpty()) box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  return box;
}

function fitView(animated = false) {
  const box = sceneBox();
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.1);
  let dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.95;
  let dir = new THREE.Vector3(0.55, 0.6, 0.9).normalize();
  if (isShelf()) {
    // shelf scans are exported with +Z toward the aisle: look at the facade face-on
    const size = box.getSize(new THREE.Vector3());
    const vt = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    dist = Math.max(size.y / 2 / vt, size.x / 2 / (vt * camera.aspect)) * 1.08 + size.z / 2;
    dir = new THREE.Vector3(0, 0.12, 1).normalize();
  }
  const pos = center.clone().addScaledVector(dir, dist);
  camera.near = Math.max(radius / 1000, 0.001);
  camera.far = radius * 100;
  camera.updateProjectionMatrix();
  controls.maxDistance = radius * 20;
  if (animated) flyTo(center, pos);
  else { cancelFly(); controls.target.copy(center); camera.position.copy(pos); controls.update(); }
}

function focusEntry(entry) {
  const wb = new THREE.Box3().setFromObject(entry.pick);
  const c = wb.getCenter(new THREE.Vector3());
  const s = wb.getSize(new THREE.Vector3());
  const r = Math.max(s.length() / 2, 0.15);
  const dist = Math.max(r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.3, 0.6);
  const dir = camera.position.clone().sub(controls.target);
  if (dir.lengthSq() < 1e-8) dir.set(0.55, 0.6, 0.9);
  dir.normalize();
  flyTo(c, c.clone().addScaledVector(dir, dist));
}

function applyUpAxis(axis) {
  state.upAxis = String(axis || 'Y').toUpperCase();
  world.rotation.set(state.upAxis === 'Z' ? -Math.PI / 2 : 0, 0, 0);
  world.updateMatrixWorld(true);
}

// ---------------------------------------------------------------- boxes
// obj: one entry of the v2 export schema `objects[]`.
function buildEntry(obj) {
  const bb = obj.bounding_box || {};
  const mn = new THREE.Vector3(...(bb.min || [0, 0, 0]));
  const mx = new THREE.Vector3(...(bb.max || [0, 0, 0]));
  const size = mx.clone().sub(mn);
  const center = mn.clone().add(mx).multiplyScalar(0.5);
  const geomSize = size.clone().max(new THREE.Vector3(1e-3, 1e-3, 1e-3));

  const group = new THREE.Group();
  group.position.copy(center);

  const boxGeom = new THREE.BoxGeometry(geomSize.x, geomSize.y, geomSize.z);
  const edges = new THREE.EdgesGeometry(boxGeom);
  const solidMat = new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.95 });
  const dashMat = new THREE.LineDashedMaterial({ color: 0x8a8f96, dashSize: 0.05, gapSize: 0.04, depthTest: false, transparent: true, opacity: 0.6 });
  const lines = new THREE.LineSegments(edges, solidMat);
  lines.computeLineDistances();
  lines.renderOrder = 10;

  const pickMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false, depthTest: false, side: THREE.DoubleSide });
  const pick = new THREE.Mesh(boxGeom, pickMat);
  pick.renderOrder = 9;

  const el = document.createElement('div');
  el.className = 'box-label-wrap';
  const inner = document.createElement('div');
  inner.className = 'box-label';
  const nameSpan = document.createElement('span');
  const confSpan = document.createElement('span');
  confSpan.className = 'conf';
  inner.append(nameSpan, confSpan);
  el.appendChild(inner);
  const label = new CSS2DObject(el);
  label.center.set(0.5, 1);
  const top = new THREE.Vector3();
  if (state.upAxis === 'Z') top.z = size.z / 2; else top.y = size.y / 2;
  label.position.copy(top);

  group.add(pick, lines, label);
  boxesRoot.add(group);

  const p = obj.position || {};
  const entry = {
    id: obj.object_id, raw: obj,
    origName: obj.class_name, name: obj.class_name,
    confidence: typeof obj.confidence === 'number' ? obj.confidence : 0,
    rejected: false,
    position: [p.x, p.y, p.z],
    center, size: Array.isArray(bb.size) ? new THREE.Vector3(...bb.size) : size,
    volume: geomSize.x * geomSize.y * geomSize.z,
    group, lines, pick, solidMat, dashMat, pickMat, label, labelEl: inner, nameSpan, confSpan,
  };
  pick.userData.entry = entry;
  inner.addEventListener('pointerdown', (e) => e.stopPropagation());
  inner.addEventListener('click', (e) => { e.stopPropagation(); select(entry.id); });
  return entry;
}

function disposeEntries() {
  for (const e of state.objects) {
    e.label.element.remove();
    e.group.removeFromParent();
    e.lines.geometry.dispose(); e.pick.geometry.dispose();
    e.solidMat.dispose(); e.dashMat.dispose(); e.pickMat.dispose();
  }
  state.objects = [];
}

function isFiltered(e) {
  return e.confidence < state.threshold || state.classVisible.get(e.name) === false;
}

function styleEntry(e) {
  const selected = e.id === state.selectedId;
  const col = classColor(e.name);
  e.group.visible = !isFiltered(e);
  e.lines.material = e.rejected ? e.dashMat : e.solidMat;
  e.solidMat.color.copy(selected ? new THREE.Color(0xffffff) : col);
  e.pickMat.color.copy(col);
  e.pickMat.opacity = selected ? (e.rejected ? 0.06 : 0.1) : 0;
  e.lines.renderOrder = selected ? 11 : 10;
  e.label.visible = state.showLabels || selected;
  e.nameSpan.textContent = e.name;
  e.confSpan.textContent = fmt(e.confidence, 2);
  e.labelEl.style.setProperty('--c', cssColor(e.name));
  e.labelEl.classList.toggle('selected', selected);
  e.labelEl.classList.toggle('rejected', e.rejected);
  e.label.element.style.zIndex = selected ? '1000' : '';
}

function restyleAll() { state.objects.forEach(styleEntry); }

// The review baseline: raw objects are the base list; a matching reviewed doc restores decisions.
function computeBaseline(rawObjs, reviewed) {
  const base = new Map();
  const rejectedIds = new Set(reviewed?.review_summary?.rejected_object_ids || []);
  const revById = new Map((reviewed?.objects || []).map((o) => [o.object_id, o]));
  for (const o of rawObjs) {
    if (!reviewed) base.set(o.object_id, { name: o.class_name, rejected: false });
    else if (revById.has(o.object_id)) base.set(o.object_id, { name: revById.get(o.object_id).class_name || o.class_name, rejected: false });
    else base.set(o.object_id, { name: o.class_name, rejected: true }); // in rejected_object_ids, or simply absent
    if (reviewed && rejectedIds.has(o.object_id)) base.get(o.object_id).rejected = true;
  }
  return base;
}

function setObjectsFromDocs() {
  disposeEntries();
  state.selectedId = null;
  const rawObjs = state.rawDoc?.objects || [];
  state.baseline = computeBaseline(rawObjs, state.reviewedDoc);
  state.objects = rawObjs.map((o) => {
    const e = buildEntry(o);
    const b = state.baseline.get(e.id);
    e.name = b.name; e.rejected = b.rejected;
    return e;
  });
  for (const e of state.objects) if (!state.classVisible.has(e.name)) state.classVisible.set(e.name, true);
  restyleAll();
  renderObjectsPanel();
}

const byId = (id) => state.objects.find((e) => e.id === id);

// ---------------------------------------------------------------- picking
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downPos = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downPos = { x: e.clientX, y: e.clientY, b: e.button }; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downPos || downPos.b !== 0) return;
  const moved = Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y);
  downPos = null;
  if (moved > 4) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  if (measure) { measureClick(); return; }
  const candidates = state.objects.filter((o) => o.group.visible).map((o) => o.pick);
  const hits = raycaster.intersectObjects(candidates, false);
  if (!hits.length) { select(null); return; }
  // Nested boxes are common: prefer the smallest box under the cursor.
  let best = null;
  for (const h of hits) {
    const en = h.object.userData.entry;
    if (!best || en.volume < best.volume) best = en;
  }
  select(best.id);
});
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (measure) { stopMeasure(); return; }
  if (!$('json-modal').classList.contains('hidden')) { closeJsonModal(); return; }
  if (!(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) select(null);
});

function select(id, { focus = false } = {}) {
  state.selectedId = id;
  restyleAll();
  renderList();
  renderSelected();
  const e = byId(id);
  if (e && focus) focusEntry(e);
}

// ---------------------------------------------------------------- object list
function renderList() {
  const listEl = $('object-list');
  const scroll = listEl.scrollTop;
  listEl.innerHTML = '';
  const total = state.objects.length;
  const shown = state.objects.filter((e) => !isFiltered(e)).length;
  const rejected = state.objects.filter((e) => e.rejected).length;
  $('obj-count').textContent = total ? `${shown}/${total} shown${rejected ? ` · ${rejected} rejected` : ''}` : '';

  if (!total) {
    const d = document.createElement('div');
    d.className = 'list-empty';
    const st = scanStatus();
    if (isShelf()) {
      d.textContent = ACTIVE.has(st) ? 'The store video is being converted to 3D…'
        : st === 'error' ? 'The conversion failed. See the log above.'
        : 'Empty-shelf (gap) detection for shelf scans is not built yet.';
    } else {
      d.textContent = !state.rawDoc
        ? (ACTIVE.has(st) ? 'The pipeline is running…' : st === 'error' ? 'The last run failed. See the log above.' : 'No results yet. Run the pipeline.')
        : 'The pipeline found no objects.';
    }
    listEl.appendChild(d);
    return;
  }

  const groups = new Map();
  for (const e of state.objects) {
    if (!groups.has(e.name)) groups.set(e.name, []);
    groups.get(e.name).push(e);
  }
  const names = [...groups.keys()].sort((a, b) => a.localeCompare(b));
  for (const name of names) {
    const items = groups.get(name).sort((a, b) => b.confidence - a.confidence);
    const g = document.createElement('div');
    g.className = 'class-group';

    const head = document.createElement('div');
    head.className = 'class-head';
    const lab = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = state.classVisible.get(name) !== false;
    cb.addEventListener('change', () => {
      state.classVisible.set(name, cb.checked);
      if (!cb.checked && byId(state.selectedId)?.name === name) state.selectedId = null;
      restyleAll(); renderList(); renderSelected();
    });
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = cssColor(name);
    const nm = document.createElement('span');
    nm.className = 'cname';
    nm.textContent = name;
    const cnt = document.createElement('span');
    cnt.className = 'count';
    cnt.textContent = `(${items.length})`;
    lab.append(cb, sw, nm, cnt);
    head.appendChild(lab);
    g.appendChild(head);

    for (const e of items) {
      const row = document.createElement('div');
      row.className = 'obj-item';
      row.dataset.id = e.id;
      row.classList.toggle('selected', e.id === state.selectedId);
      row.classList.toggle('rejected', e.rejected);
      row.classList.toggle('filtered', isFiltered(e));
      row.title = isFiltered(e) ? 'Hidden by filters (its decision is still sent)' : 'Click to select and focus';
      const n = document.createElement('span');
      n.className = 'obj-name';
      n.textContent = `${e.id} `;
      if (e.name !== e.origName) {
        const r = document.createElement('span');
        r.className = 'renamed';
        r.textContent = `renamed from "${e.origName}"`;
        n.appendChild(r);
      }
      const c = document.createElement('span');
      c.className = 'conf mono';
      c.textContent = fmt(e.confidence, 2);
      const tog = document.createElement('button');
      tog.type = 'button';
      tog.className = 'tog ' + (e.rejected ? 'rejected' : 'accepted');
      tog.textContent = e.rejected ? 'Rejected' : 'Accepted';
      tog.title = e.rejected ? 'Click to accept again' : 'Click to reject';
      tog.addEventListener('click', (ev) => { ev.stopPropagation(); setRejected(e, !e.rejected); });
      row.append(n, c, tog);
      row.addEventListener('click', () => {
        if (isFiltered(e)) toast('This object is hidden by the class/confidence filters.');
        select(e.id, { focus: !isFiltered(e) });
      });
      g.appendChild(row);
    }
    listEl.appendChild(g);
  }
  listEl.scrollTop = scroll;
}

// ---------------------------------------------------------------- selected panel
function renderSelected() {
  const e = byId(state.selectedId);
  $('selected-empty').classList.toggle('hidden', !!e);
  $('selected-details').classList.toggle('hidden', !e);
  if (!e) return;
  $('sel-swatch').style.background = cssColor(e.name);
  $('sel-class').textContent = e.name;
  $('sel-id').textContent = `${e.id}${e.name !== e.origName ? ` · was "${e.origName}"` : ''}`;
  $('sel-conf').textContent = fmt(e.confidence, 3);
  $('sel-pos').textContent = e.position.map((v) => fmt(v)).join(', ');
  $('sel-size').textContent = `${fmt(e.size.x)} × ${fmt(e.size.y)} × ${fmt(e.size.z)}`;
  $('sel-views').textContent = e.raw.num_views ?? '–';
  const ri = $('rename-input');
  if (document.activeElement !== ri) ri.value = e.name;
  $('rename-revert').disabled = e.name === e.origName;
  $('sel-accept').classList.toggle('on-accept', !e.rejected);
  $('sel-reject').classList.toggle('on-reject', e.rejected);
}

function setRejected(e, rejected) {
  e.rejected = rejected;
  styleEntry(e); renderList(); renderSelected(); updateDirty();
}

function rename(e, newName) {
  const n = (newName || '').trim() || e.origName;
  if (n === e.name) return;
  e.name = n;
  if (!state.classVisible.has(n)) state.classVisible.set(n, true);
  restyleAll(); renderList(); renderSelected(); updateDirty(); updateDatalist();
}

$('rename-input').addEventListener('change', () => { const e = byId(state.selectedId); if (e) rename(e, $('rename-input').value); });
$('rename-input').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') { ev.preventDefault(); $('rename-input').blur(); }
  if (ev.key === 'Escape') { const e = byId(state.selectedId); if (e) $('rename-input').value = e.name; $('rename-input').blur(); }
});
$('rename-revert').addEventListener('click', () => { const e = byId(state.selectedId); if (e) { rename(e, e.origName); $('rename-input').value = e.name; } });
$('sel-accept').addEventListener('click', () => { const e = byId(state.selectedId); if (e) setRejected(e, false); });
$('sel-reject').addEventListener('click', () => { const e = byId(state.selectedId); if (e) setRejected(e, true); });
$('sel-focus').addEventListener('click', () => { const e = byId(state.selectedId); if (e) focusEntry(e); });

function updateDatalist() {
  const opts = new Set([...parsePrompts($('prompts').value), ...(state.config.default_prompts || [])]);
  for (const e of state.objects) { opts.add(e.name); opts.add(e.origName); }
  const dl = $('class-options');
  dl.innerHTML = '';
  for (const o of [...opts].sort((a, b) => a.localeCompare(b))) {
    const op = document.createElement('option');
    op.value = o;
    dl.appendChild(op);
  }
}
$('prompts').addEventListener('input', updateDatalist);

// ---------------------------------------------------------------- review header, dirty, send
function isDirty() {
  return state.objects.some((e) => {
    const b = state.baseline.get(e.id);
    return !b || b.rejected !== e.rejected || b.name !== e.name;
  });
}

function renderReviewHead() {
  const h = $('review-head');
  if (!state.rawDoc) {
    h.className = 'review-head none';
    h.textContent = ACTIVE.has(scanStatus()) ? 'Waiting for the pipeline…' : 'No result yet';
  } else if (state.reviewedDoc) {
    h.className = 'review-head reviewed';
    const at = state.reviewedDoc.scan?.reviewed_at;
    h.textContent = `Reviewed and sent at ${at ? fmtTime(at) : '(unknown time)'}`;
    h.title = at || '';
  } else {
    h.className = 'review-head raw';
    h.textContent = 'Raw result (unreviewed)';
    h.title = '';
  }
}

function updateDirty() {
  const d = isDirty();
  $('dirty-indicator').classList.toggle('hidden', !d);
  const busy = isBusy();
  const btn = $('send-btn');
  btn.disabled = !state.rawDoc || busy || state.sending;
  btn.textContent = state.sending ? 'Sending…' : 'Send to backend';
  btn.title = !state.rawDoc ? 'No result to send yet'
    : busy ? 'Disabled while this scan is queued or running'
    : state.reviewedDoc ? (d ? 'Send the changed review (overwrites the previous reviewed JSON)' : 'Re-send the same review')
    : 'Write the reviewed JSON to the backend outbox';
}

function renderObjectsPanel() {
  renderReviewHead();
  renderList();
  renderSelected();
  updateDirty();
  updateDatalist();
  $('dl-raw').disabled = !state.rawDoc;
  $('dl-reviewed').disabled = !state.reviewedDoc;
  $('view-json').disabled = !state.rawDoc && !state.reviewedDoc;
}

window.addEventListener('beforeunload', (ev) => {
  if (isDirty()) { ev.preventDefault(); ev.returnValue = ''; }
});

$('send-btn').addEventListener('click', async () => {
  if (!state.rawDoc || isBusy() || state.sending) return;
  const id = state.scanId;
  const decisions = state.objects.map((e) => ({ object_id: e.id, decision: e.rejected ? 'reject' : 'accept', class_name: e.name }));
  state.sending = true; updateDirty();
  try {
    const res = await api(`/api/scans/${enc(id)}/send`, { method: 'POST', body: { decisions } });
    const rejected = decisions.filter((d) => d.decision === 'reject').length;
    toast(`Sent to backend: ${res?.num_objects ?? decisions.length - rejected} object(s)${rejected ? `, ${rejected} rejected` : ''}. Written to ${res?.outbox_path || res?.path || 'the outbox'}`, 'ok', 8000);
    if (state.scanId === id) {
      state.reviewedDoc = await apiOpt(`/api/scans/${enc(id)}/objects?version=reviewed`);
      const keepSel = state.selectedId;
      state.baseline = computeBaseline(state.rawDoc.objects || [], state.reviewedDoc);
      // Keep the current entries (they already show the sent decisions); only the baseline moves.
      state.selectedId = keepSel;
      renderObjectsPanel();
    }
    refreshScans();
  } catch (err) {
    toast(`Send failed: ${err.message}`, 'error');
  } finally {
    state.sending = false; updateDirty();
  }
});

// ---------------------------------------------------------------- filters / toggles
$('threshold').addEventListener('input', () => {
  state.threshold = parseFloat($('threshold').value);
  $('threshold-val').textContent = state.threshold.toFixed(2);
  const sel = byId(state.selectedId);
  if (sel && isFiltered(sel)) state.selectedId = null;
  restyleAll(); renderList(); renderSelected();
});
$('show-labels').addEventListener('change', () => { state.showLabels = $('show-labels').checked; restyleAll(); renderLabels(); renderGaps(); });
$('reset-view').addEventListener('click', () => fitView(true));

// ---------------------------------------------------------------- JSON download + modal
async function fetchJsonText(version) {
  const res = await fetch(`/api/scans/${enc(state.scanId)}/objects?version=${version}`);
  const text = await res.text();
  if (!res.ok) {
    let data = text; try { data = JSON.parse(text); } catch { /* keep text */ }
    throw httpError(res.status, res.statusText, data);
  }
  return text;
}

async function downloadJson(version) {
  try {
    const text = await fetchJsonText(version);
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${state.scanId}_${version}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (err) {
    toast(`Download failed: ${err.message}`, 'error');
  }
}
$('dl-raw').addEventListener('click', () => downloadJson('raw'));
$('dl-reviewed').addEventListener('click', () => downloadJson('reviewed'));

let modalVersion = 'raw';
async function showJsonVersion(version) {
  modalVersion = version;
  $('json-tab-raw').classList.toggle('on', version === 'raw');
  $('json-tab-reviewed').classList.toggle('on', version === 'reviewed');
  $('json-tab-raw').disabled = !state.rawDoc;
  $('json-tab-reviewed').disabled = !state.reviewedDoc;
  $('json-title').textContent = `${state.scanId} · ${version}`;
  const pre = $('json-pre');
  pre.textContent = 'Loading…';
  try {
    const text = await fetchJsonText(version);
    let pretty = text;
    try { pretty = JSON.stringify(JSON.parse(text), null, 2); } catch { /* show as is */ }
    if (modalVersion === version) pre.textContent = pretty;
  } catch (err) {
    if (modalVersion === version) pre.textContent = `Could not load: ${err.message}`;
  }
}
function openJsonModal() {
  $('json-modal').classList.remove('hidden');
  showJsonVersion(state.reviewedDoc ? 'reviewed' : 'raw');
}
function closeJsonModal() { $('json-modal').classList.add('hidden'); }
$('view-json').addEventListener('click', openJsonModal);
$('json-close').addEventListener('click', closeJsonModal);
$('json-modal').addEventListener('click', (e) => { if (e.target === $('json-modal')) closeJsonModal(); });
$('json-tab-raw').addEventListener('click', () => showJsonVersion('raw'));
$('json-tab-reviewed').addEventListener('click', () => showJsonVersion('reviewed'));
$('json-copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('json-pre').textContent); toast('Copied to the clipboard.', 'ok', 2000); }
  catch (err) { toast(`Copy failed: ${err.message}`, 'error'); }
});

// ---------------------------------------------------------------- scan list
let listTimer = null;

function renderScanList() {
  const el = $('scan-list');
  el.innerHTML = '';
  $('empty-state').classList.toggle('hidden', !state.scansLoaded || state.scans.length > 0);
  if (!state.scans.length) {
    const d = document.createElement('div');
    d.className = 'list-empty scans';
    d.textContent = state.scansLoaded ? 'No scans yet. Add a .usdz scan to start.' : 'Loading…';
    el.appendChild(d);
    return;
  }
  for (const s of state.scans) {
    const st = s.status || 'uploaded';
    const row = document.createElement('div');
    row.className = 'scan-row';
    row.dataset.id = s.scan_id;
    row.classList.toggle('selected', s.scan_id === state.scanId);
    row.classList.toggle('error', st === 'error');
    row.title = s.scan_id;

    const r1 = document.createElement('div');
    r1.className = 'r1';
    const fn = document.createElement('span');
    fn.className = 'fname ellipsis';
    fn.textContent = s.source_file || s.scan_id;
    if (s.scan_type === 'shelf') {
      const tag = document.createElement('span');
      tag.className = 'type-tag';
      tag.textContent = 'shelf';
      tag.title = 'Shelf scan made from a store video';
      fn.prepend(tag);
    }
    const badge = document.createElement('span');
    badge.className = `badge ${st === 'running' ? 'running' : st}`;
    badge.textContent = st;
    r1.append(fn, badge);

    const r2 = document.createElement('div');
    r2.className = 'r2';
    const t = document.createElement('span');
    t.textContent = fmtTime(s.uploaded_at);
    r2.appendChild(t);
    if (typeof s.num_objects === 'number' && (s.has_raw || s.num_objects > 0)) {
      const n = document.createElement('span');
      n.textContent = `· ${s.num_objects} object${s.num_objects === 1 ? '' : 's'}`;
      r2.appendChild(n);
    }
    const stale = s.stale || (s.scan_id === state.scanId && state.record?.stale);
    if (stale) {
      const w = document.createElement('span');
      w.className = 'stale';
      w.textContent = '· stale results';
      w.title = 'The orientation changed after the last run. Re-run to update the results.';
      r2.appendChild(w);
    }
    row.append(r1, r2);

    if (ACTIVE.has(st)) {
      const pr = document.createElement('div');
      pr.className = 'progress';
      const bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.width = `${Math.max(0, Math.min(1, Number(s.progress) || 0)) * 100}%`;
      pr.appendChild(bar);
      row.appendChild(pr);
    }
    if ((ACTIVE.has(st) || st === 'error') && s.message) {
      const m = document.createElement('div');
      m.className = 'smsg ellipsis';
      m.textContent = ACTIVE.has(st) ? `${pct(s.progress)} · ${s.message}` : s.message;
      m.title = s.message;
      row.appendChild(m);
    }
    row.addEventListener('click', () => { if (s.scan_id !== state.scanId) selectScan(s.scan_id); });
    el.appendChild(row);
  }
}

async function refreshScans() {
  clearTimeout(listTimer);
  const prev = summary();
  try {
    const list = await api('/api/scans');
    state.scans = Array.isArray(list) ? list : (list?.scans || []);
    state.scansLoaded = true;
  } catch (err) {
    console.warn('GET /api/scans failed:', err.message);
  }
  renderScanList();
  const cur = summary();
  if (state.scanId && cur) {
    renderScanPanels();
    // The selected scan became active elsewhere (e.g. queued by another tab): start following its job.
    if (ACTIVE.has(cur.status) && !jobFollowing) followJob();
    // The selected scan finished without our job poll noticing (e.g. a missed transition).
    if (prev && ACTIVE.has(prev.status) && !ACTIVE.has(cur.status) && !jobFollowing) reloadResults(true);
  }
  const anyActive = state.scans.some((s) => ACTIVE.has(s.status)) || state.uploading;
  listTimer = setTimeout(refreshScans, anyActive ? 2000 : 15000);
}

// ---------------------------------------------------------------- per-scan panels (orientation, run)
function renderScanPanels() {
  const s = summary();
  const has = !!state.scanId;
  $('no-scan').classList.toggle('hidden', has);
  $('scan-panels').classList.toggle('hidden', !has);
  if (!has) { renderVideo(); return; }
  const busy = isBusy();

  // orientation
  const o = orientation();
  if (o) {
    const flipped = !!o.flip_applied;
    let txt;
    if (!flipped) txt = 'as uploaded';
    else if (o.source === 'manual') txt = 'flipped manually (180°)';
    else txt = 'auto-flipped (180°)';
    $('orient-text').textContent = `Orientation: ${txt}`;
    const rec = o.flip_recommended === undefined || o.flip_recommended === null ? null : (o.flip_recommended ? 'flip recommended' : 'no flip recommended');
    $('orient-detail').textContent = `Heuristic: ${rec || 'n/a'}${o.confidence ? ` · confidence ${o.confidence}` : ''}${o.source ? ` · source ${o.source}` : ''}`;
  } else {
    $('orient-text').textContent = 'Orientation: unknown';
    $('orient-detail').textContent = '';
  }
  $('flip-btn').disabled = busy || state.busyAction || !o;
  $('flip-btn').title = busy ? 'Disabled while this scan is queued or running' : 'Rotate the scan 180° about X and rebuild the model';
  const stale = isStale() && (!!state.rawDoc || !!s?.has_raw);
  $('stale-warn').classList.toggle('hidden', !stale);
  $('stale-rerun').disabled = busy || state.busyAction;

  // run
  const job = state.job;
  const st = (job && job.state) || s?.status || 'uploaded';
  const badge = $('job-state');
  badge.textContent = s?.status === 'reviewed' && st === 'done' ? 'reviewed' : st;
  badge.className = 'badge ' + badge.textContent;
  const progress = job ? job.progress : s?.progress;
  const p = st === 'done' ? 1 : Math.max(0, Math.min(1, Number(progress) || 0));
  $('job-pct').textContent = job || ACTIVE.has(st) ? pct(p) : '';
  $('job-bar').style.width = `${(job || ACTIVE.has(st) ? p : 0) * 100}%`;
  $('job-bar').className = 'bar' + (st === 'error' ? ' error' : st === 'done' ? ' done' : '');
  $('job-stage').textContent = job?.stage && ACTIVE.has(st) ? `stage: ${job.stage}` : '';
  const qp = job?.queue_position;
  $('job-queue').textContent = st === 'queued' && typeof qp === 'number' ? `queue #${qp}` : '';
  let msg = job?.message || s?.message || '';
  if (!job && !ACTIVE.has(st) && !s?.has_raw && st !== 'error') msg = msg || 'Not run yet.';
  $('job-message').textContent = msg;
  const log = Array.isArray(job?.log_tail) ? job.log_tail.join('\n') : (job?.log_tail || '');
  const pre = $('job-log');
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
  pre.textContent = log || '(no output yet)';
  if (atBottom) pre.scrollTop = pre.scrollHeight;
  const shelf = isShelf();
  renderScale(shelf, busy);
  const jobKind = job?.kind || (shelf ? 'convert' : 'pipeline');
  $('run-title').textContent = jobKind === 'gaps' ? 'Gap detection run' : shelf ? 'Video → 3D conversion' : 'Pipeline run';
  $('rerun-wrap').classList.toggle('hidden', shelf);
  $('convert-row').classList.toggle('hidden', !shelf || busy || st !== 'error' || jobKind !== 'convert');
  $('objects-section').classList.toggle('hidden', shelf);
  $('selected-section').classList.toggle('hidden', shelf);
  if (shelf) { $('gaps-run').disabled = busy || state.busyAction; }
  $('convert-btn').disabled = busy || state.busyAction;
  if (shelf && !state.modelLoaded && !hasModel()) showNoModel();
  $('run-btn').disabled = busy || state.busyAction;
  $('run-btn').textContent = busy ? (st === 'queued' ? 'Queued…' : 'Running…') : (s?.has_raw || state.rawDoc ? 'Re-run' : 'Run');
  updateDirty();
  renderVideo();
}

// ---------------------------------------------------------------- job polling (selected scan)
let jobTimer = null;
let jobFollowing = false;   // a followJob loop is alive (including while a fetch is in flight)
let jobFailures = 0;

async function fetchJob(id) {
  return apiOpt(`/api/scans/${enc(id)}/job`);
}

function stopJobPoll() { clearTimeout(jobTimer); jobTimer = null; jobFollowing = false; followGen++; }
let followGen = 0;

function followJob() {
  stopJobPoll();
  jobFollowing = true;
  const gen = followGen;
  const id = state.scanId;
  const token = state.loadToken;
  const alive = () => gen === followGen && state.scanId === id && state.loadToken === token;
  const tick = async () => {
    jobTimer = null;
    if (!alive()) return;
    let job;
    try {
      job = await fetchJob(id);
      jobFailures = 0;
    } catch (err) {
      jobFailures++;
      $('job-message').textContent = `Lost contact with the server (${err.message}); retrying…`;
      if (jobFailures < 60) jobTimer = setTimeout(tick, 2000); else jobFollowing = false;
      return;
    }
    if (!alive()) return;
    state.job = job;
    renderScanPanels();
    if (ACTIVE.has(job?.state)) { jobTimer = setTimeout(tick, 1000); return; }
    jobFollowing = false;
    {
      // terminal: refresh the summary and pick up the new results
      await refreshScans();
      if (job?.state === 'done' && job.kind === 'convert') {
        await reloadResults(false);
        if (!alive()) return;
        state.modelVersion = Date.now();
        userMovedCamera = false;
        await loadModel(id, token);
        if (alive() && !userMovedCamera) fitView(true);
        lab.keyframes = null;
        loadLabels();
        toast('Conversion finished: the 3D shelf scan is ready.', 'ok');
      } else if (job?.state === 'done' && job.kind === 'gaps') {
        await refreshScans();
        await loadGaps();
        const d = gapState.doc?.methods || {};
        toast(`Gap detection finished: ${Object.entries(d).map(([k, m]) => `${(METHOD[k] || { short: k }).short} ${m.gaps.length}`).join(', ')} gaps.`, 'ok');
      } else if (job?.state === 'done') {
        const hadEdits = isDirty();
        await reloadResults(false);
        toast(`Pipeline finished: ${state.objects.length} object${state.objects.length === 1 ? '' : 's'}.${hadEdits ? ' Unsent review changes were discarded.' : ''}`, 'ok');
      } else if (job?.state === 'error') {
        $('job-log-wrap').open = true;
        toast(`${job.kind === 'convert' ? 'Conversion' : job.kind === 'gaps' ? 'Gap detection' : 'Pipeline'} failed: ${job.message || 'see the log'}`, 'error');
        await reloadResults(false);
        if (job.kind === 'convert' && !hasModel()) showNoModel();
      }
    }
  };
  tick();
}

// ---------------------------------------------------------------- model
function clearModel() {
  modelRoot.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      for (const m of [].concat(o.material)) {
        for (const k of Object.keys(m)) if (m[k] && m[k].isTexture) m[k].dispose();
        m.dispose();
      }
    }
  });
  modelRoot.clear();
  state.modelLoaded = false;
}

// Shelf scan whose video has not been converted yet: explain instead of a model-load error.
function showNoModel() {
  clearModel();
  const st = scanStatus();
  const s = summary();
  if (ACTIVE.has(st)) {
    $('model-error').classList.add('hidden');
    $('loading-text').textContent = st === 'queued'
      ? 'Store video uploaded. The video → 3D conversion is queued…'
      : `Converting the store video to 3D… ${pct(s?.progress)}${s?.message ? ` · ${s.message}` : ''}`;
    $('loading').classList.remove('hidden');
  } else {
    $('loading').classList.add('hidden');
    const b = $('model-error');
    b.textContent = st === 'error'
      ? 'The video → 3D conversion failed, so there is no model yet. See the log, then use Retry conversion.'
      : 'This shelf scan has no 3D model yet.';
    b.classList.remove('hidden');
  }
  return false;
}

function loadModel(id, token) {
  clearModel();
  $('model-error').classList.add('hidden');
  return new Promise((resolve) => {
    const loader = new GLTFLoader();
    $('loading-text').textContent = 'Loading model…';
    $('loading').classList.remove('hidden');
    const url = `/api/scans/${enc(id)}/model.glb?v=${state.modelVersion}`;
    loader.load(url, (gltf) => {
      if (token !== state.loadToken) { resolve(false); return; }
      clearModel();
      modelRoot.add(gltf.scene);
      state.modelLoaded = true;
      $('loading').classList.add('hidden');
      renderScanPanels();   // panels that need the model (Scale) appear now
      resolve(true);
    }, (ev) => {
      if (token !== state.loadToken) return;
      const mb = (ev.loaded / 1048576).toFixed(1);
      $('loading-text').textContent = ev.total
        ? `Loading model… ${Math.round((ev.loaded / ev.total) * 100)}% (${mb} MB)`
        : `Loading model… ${mb} MB`;
    }, (err) => {
      if (token !== state.loadToken) { resolve(false); return; }
      $('loading').classList.add('hidden');
      const msg = err?.message || (err?.target?.status ? `HTTP ${err.target.status}` : String(err));
      console.warn('model load failed:', msg);
      const b = $('model-error');
      b.textContent = `Model failed to load (${msg}).${state.objects.length ? ' The detected boxes are still drawn without the mesh.' : ''}`;
      b.classList.remove('hidden');
      resolve(false);
    });
  });
}

// ---------------------------------------------------------------- results (raw + reviewed)
async function reloadResults(keepEdits) {
  const id = state.scanId;
  const token = state.loadToken;
  if (!id) return;
  if (keepEdits && isDirty()) return; // never clobber unsent edits silently on a background refresh
  let raw = null, reviewed = null, record = state.record;
  try {
    const s = summary();
    const wantRaw = !(s && s.has_raw === false);
    [raw, record] = await Promise.all([
      wantRaw ? apiOpt(`/api/scans/${enc(id)}/objects?version=raw`) : Promise.resolve(null),
      apiOpt(`/api/scans/${enc(id)}`),
    ]);
    if (raw && (s?.has_reviewed || record?.has_reviewed || s?.status === 'reviewed' || record?.status === 'reviewed' || s?.has_reviewed === undefined)) {
      reviewed = await apiOpt(`/api/scans/${enc(id)}/objects?version=reviewed`);
    }
  } catch (err) {
    toast(`Could not load the results: ${err.message}`, 'error');
  }
  if (state.scanId !== id || state.loadToken !== token) return;
  // A reviewed doc from an older run does not apply to the current raw object ids.
  if (reviewed && raw && reviewed.scan?.job_id && raw.scan?.job_id && reviewed.scan.job_id !== raw.scan.job_id) reviewed = null;
  state.record = record || state.record;
  state.rawDoc = raw;
  state.reviewedDoc = reviewed;
  const up = raw?.scan?.coordinate_frame?.up_axis || state.record?.coordinate_frame?.up_axis || state.record?.up_axis || 'Y';
  if (String(up).toUpperCase() !== state.upAxis) { applyUpAxis(up); if (!userMovedCamera) fitView(); }
  setObjectsFromDocs();
  fillRunPrompts(false);
  renderScanPanels();
  renderScanList();
}

// ---------------------------------------------------------------- scan selection
let promptsTouched = false;
$('prompts').addEventListener('input', () => { promptsTouched = true; });

function fillRunPrompts(force) {
  if (promptsTouched && !force) return;
  const p = state.rawDoc?.scan?.prompts || state.record?.prompts || state.config.default_prompts || [];
  $('prompts').value = p.join(', ');
  promptsTouched = false;
  updateDatalist();
}

async function selectScan(id, { force = false } = {}) {
  if (!force && id !== state.scanId && isDirty()
      && !window.confirm('You have review changes that were not sent. Discard them and switch scans?')) return false;
  stopJobPoll();
  stopMeasure();
  const token = ++state.loadToken;
  state.scanId = id;
  state.record = null; state.job = null; state.rawDoc = null; state.reviewedDoc = null;
  state.classVisible.clear();
  state.modelVersion = Date.now();
  disposeEntries();
  state.baseline = new Map();
  state.selectedId = null;
  userMovedCamera = false;
  promptsTouched = false;
  try { history.replaceState(null, '', id ? `#scan=${enc(id)}` : location.pathname); } catch { /* ignore */ }
  const s = summary();
  $('scene-title').textContent = s?.source_file ? `${s.source_file}` : (id || 'No scan selected');
  $('scene-title').title = id || '';
  document.title = s?.source_file ? `${s.source_file} · Room Scan Viewer` : 'Room Scan Viewer';
  $('json-modal').classList.add('hidden');
  $('job-log-wrap').open = false;
  $('rerun-wrap').open = false;
  renderScanList();
  renderObjectsPanel();
  renderScanPanels();
  if (!id) { clearModel(); return true; }

  const modelP = hasModel() ? loadModel(id, token) : Promise.resolve(showNoModel());
  lab.list = []; lab.available = false; lab.keyframes = null; closeLabelModal(); renderLabels();
  gapState.doc = null; renderGaps();
  await reloadResults(false);
  loadLabels().then(loadGaps);
  if (token !== state.loadToken) return true;
  const st0 = summary()?.status;
  if (st0 === 'uploaded' && !summary()?.has_raw) state.job = null; // never run: no job record to fetch
  else { try { state.job = await fetchJob(id); } catch { state.job = null; } }
  if (token !== state.loadToken) return true;
  if (state.job?.state === 'error') $('job-log-wrap').open = true;
  if (!state.job && !state.rawDoc && !ACTIVE.has(summary()?.status)) $('rerun-wrap').open = true; // never run: show the Run button
  renderScanPanels();
  if (ACTIVE.has(state.job?.state) || ACTIVE.has(summary()?.status)) followJob();
  fitView();
  await modelP;
  if (token === state.loadToken && !userMovedCamera) fitView();
  return true;
}

// ---------------------------------------------------------------- orientation flip
$('flip-btn').addEventListener('click', async () => {
  const id = state.scanId;
  const o = orientation();
  if (!id || !o || isBusy() || state.busyAction) return;
  state.busyAction = true; renderScanPanels();
  const t = toast('Flipping and rebuilding the model…', 'info', 30000);
  try {
    const res = await api(`/api/scans/${enc(id)}/orientation`, { method: 'POST', body: { flip: !o.flip_applied } });
    t.remove();
    if (state.scanId !== id) return;
    if (res && res.scan_id === id && res.status) {
      // the response is a summary or record; merge what we can
      const i = state.scans.findIndex((x) => x.scan_id === id);
      if (i >= 0 && res.orientation) state.scans[i] = { ...state.scans[i], ...res };
    }
    state.record = (await apiOpt(`/api/scans/${enc(id)}`)) || state.record;
    await refreshScans();
    state.modelVersion = Date.now();
    userMovedCamera = false;
    await loadModel(id, state.loadToken);
    if (!userMovedCamera) fitView(true);
    const nowFlipped = orientation()?.flip_applied;
    toast(`Orientation: ${nowFlipped ? 'flipped 180°' : 'as uploaded'}.${state.rawDoc ? ' The existing results are now stale; re-run to update them.' : ''}`, state.rawDoc ? 'info' : 'ok', 6000);
  } catch (err) {
    t.remove();
    toast(`Flip failed: ${err.message}`, 'error');
  } finally {
    state.busyAction = false; renderScanPanels(); renderScanList();
  }
});

// ---------------------------------------------------------------- run / re-run
async function startRun() {
  const id = state.scanId;
  if (!id || isBusy() || state.busyAction) return;
  const prompts = parsePrompts($('prompts').value);
  if (!prompts.length) { toast('Enter at least one prompt.', 'error'); $('rerun-wrap').open = true; return; }
  if (isDirty() && !window.confirm('You have review changes that were not sent. A new run replaces the raw result and discards them. Continue?')) return;
  state.busyAction = true; renderScanPanels();
  try {
    const job = await api(`/api/scans/${enc(id)}/run`, { method: 'POST', body: { prompts } });
    if (state.scanId !== id) return;
    state.job = job && job.state ? job : { state: 'queued', progress: 0, message: 'Queued', log_tail: [] };
    const i = state.scans.findIndex((x) => x.scan_id === id);
    if (i >= 0) state.scans[i] = { ...state.scans[i], status: 'queued', progress: 0 };
    toast(`Run queued with ${prompts.length} prompt${prompts.length === 1 ? '' : 's'}.`, 'ok');
    state.busyAction = false;
    renderScanPanels(); renderScanList();
    followJob();
    refreshScans();
  } catch (err) {
    toast(`Run failed: ${err.message}`, 'error');
    state.busyAction = false;
    renderScanPanels();
  }
}
$('run-btn').addEventListener('click', startRun);
$('convert-btn').addEventListener('click', async () => {
  const id = state.scanId;
  if (!id || isBusy() || state.busyAction) return;
  state.busyAction = true; renderScanPanels();
  try {
    const job = await api(`/api/scans/${enc(id)}/convert`, { method: 'POST' });
    if (state.scanId !== id) return;
    state.job = job;
    const i = state.scans.findIndex((x) => x.scan_id === id);
    if (i >= 0) state.scans[i] = { ...state.scans[i], status: 'queued', progress: 0 };
    toast('Conversion queued.', 'ok');
    state.busyAction = false;
    renderScanPanels(); renderScanList();
    followJob();
    refreshScans();
  } catch (err) {
    toast(`Retry failed: ${err.message}`, 'error');
    state.busyAction = false;
    renderScanPanels();
  }
});
$('stale-rerun').addEventListener('click', startRun);
$('prompts-reset').addEventListener('click', () => { $('prompts').value = (state.config.default_prompts || []).join(', '); promptsTouched = true; updateDatalist(); });

// ---------------------------------------------------------------- empty-slot labels (shelf scans: ground truth)
// Boxes are drawn on the original keyframes (2D is the source of truth); the server places
// each on the shelf front in 3D and reports its size in the scan's current scale.
const KIND = { empty: { name: 'empty', color: '#ff4fd8' }, pushed_back: { name: 'pushed back', color: '#ffa63d' } };
// Clear an overlay group. CSS2DObject removes its DOM element only when it is itself removed
// from its parent; ours sit inside groups, so remove the elements explicitly first.
function clearOverlay(root) {
  root.traverse((o) => {
    if (o.isCSS2DObject && o.element) o.element.remove();
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  root.clear();
}
const labelsRoot = new THREE.Group();
labelsRoot.name = 'slot-labels';
world.add(labelsRoot);
const lab = { list: [], available: false, keyframes: null, kfIndex: 0, kind: 'empty', selected: null, drag: null };

async function loadLabels() {
  const id = state.scanId;
  if (!id || !isShelf() || !hasModel()) { lab.list = []; lab.available = false; renderLabels(); return; }
  try {
    const r = await api(`/api/scans/${enc(id)}/labels`);
    if (state.scanId !== id) return;
    lab.list = r.labels || [];
    lab.available = true;
  } catch (err) {
    if (state.scanId !== id) return;
    lab.list = []; lab.available = false;
    if (err.status !== 409) toast(`Could not load the labels: ${err.message}`, 'error');
  }
  renderLabels();
}

const approxScale = () => scaleInfo()?.status !== 'calibrated';
const sizeText = (l) => (l.error ? 'cannot place' : `${approxScale() ? '≈ ' : ''}${fmt(l.width_cm, 1)} × ${fmt(l.height_cm, 1)} cm`);

function renderLabels() {
  // 3D boxes
  clearOverlay(labelsRoot);
  for (const l of lab.list) {
    if (!l.bbox) continue;
    const mn = new THREE.Vector3(...l.bbox.min), mx = new THREE.Vector3(...l.bbox.max);
    const size = mx.clone().sub(mn).max(new THREE.Vector3(1e-3, 1e-3, 1e-3));
    const c = mn.clone().add(mx).multiplyScalar(0.5);
    const col = new THREE.Color((KIND[l.kind] || KIND.empty).color);
    const g = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: col }));
    const fill = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.18, depthWrite: false }));
    const grp = new THREE.Group();
    grp.position.copy(c); grp.add(fill, edges);
    if (state.showLabels) {
      const el = document.createElement('div');
      el.className = 'slot-label';
      el.style.setProperty('--c', (KIND[l.kind] || KIND.empty).color);
      el.textContent = `${(KIND[l.kind] || KIND.empty).name} ${sizeText(l)}`;
      const tag = new CSS2DObject(el);
      tag.position.set(0, size.y / 2, 0);
      grp.add(tag);
    }
    labelsRoot.add(grp);
  }
  // sidebar
  const show = isShelf() && lab.available;
  $('labels-section').classList.toggle('hidden', !show);
  $('label-open').classList.toggle('hidden', !show);
  $('labels-count').textContent = show ? `${lab.list.length}` : '';
  const listEl = $('labels-list');
  listEl.innerHTML = '';
  for (const l of lab.list) {
    const row = document.createElement('div');
    row.className = 'lab-row';
    const dot = document.createElement('span');
    dot.className = 'dot'; dot.style.background = (KIND[l.kind] || KIND.empty).color;
    const t = document.createElement('span');
    t.className = 'ellipsis';
    t.textContent = `${l.id} · ${(KIND[l.kind] || KIND.empty).name} · ${sizeText(l)}`;
    if (l.error) { t.classList.add('err'); t.title = l.error; }
    const kf = document.createElement('span');
    kf.className = 'muted mono'; kf.textContent = l.keyframe;
    const sp = document.createElement('span'); sp.className = 'spacer';
    const edit = document.createElement('button');
    edit.type = 'button'; edit.className = 'small-btn'; edit.textContent = 'Edit';
    edit.title = 'Open this keyframe in the labelling window';
    edit.addEventListener('click', (ev) => { ev.stopPropagation(); openLabelModal(l.keyframe, l.id); });
    row.append(dot, t, sp, kf, edit);
    row.title = 'Click to show it in 3D';
    row.addEventListener('click', () => {
      if (!l.bbox) return;
      const b = new THREE.Box3(new THREE.Vector3(...l.bbox.min), new THREE.Vector3(...l.bbox.max)).applyMatrix4(world.matrixWorld);
      const cc = b.getCenter(new THREE.Vector3());
      const r = Math.max(b.getSize(new THREE.Vector3()).length() / 2, 0.15);
      const dist = Math.max(r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.6, 0.5);
      flyTo(cc, cc.clone().add(new THREE.Vector3(0, 0.1, 1).normalize().multiplyScalar(dist)));
    });
    listEl.appendChild(row);
  }
  if (!$('label-modal').classList.contains('hidden')) drawKeyframe();
}

// ---- labelling window
const curKf = () => lab.keyframes?.keyframes?.[lab.kfIndex] || null;

async function openLabelModal(kfName = null, selectId = null) {
  const id = state.scanId;
  if (!id) return;
  try {
    if (!lab.keyframes || lab.keyframes.scanId !== id) {
      lab.keyframes = { ...(await api(`/api/scans/${enc(id)}/keyframes`)), scanId: id };
    }
  } catch (err) { toast(`Keyframes unavailable: ${err.message}`, 'error'); return; }
  const kfs = lab.keyframes.keyframes;
  if (!kfs.length) { toast('This scan has no keyframes.', 'error'); return; }
  let i = kfName ? kfs.findIndex((k) => k.name === kfName) : -1;
  if (i < 0) {   // nearest keyframe to the paused video
    const t = $('video-el').currentTime || 0;
    i = kfs.reduce((b, k, j) => (Math.abs(k.time_s - t) < Math.abs(kfs[b].time_s - t) ? j : b), 0);
  }
  $('video-el').pause();
  lab.selected = selectId;
  $('label-modal').classList.remove('hidden');
  setKeyframe(i);
}

function closeLabelModal() { $('label-modal').classList.add('hidden'); lab.drag = null; lab.selected = null; }

function setKeyframe(i) {
  const kfs = lab.keyframes.keyframes;
  lab.kfIndex = Math.max(0, Math.min(kfs.length - 1, i));
  const k = curKf();
  $('kf-name').textContent = `${k.name} · ${fmt(k.time_s, 2)} s · ${lab.kfIndex + 1}/${kfs.length}`;
  $('kf-prev').disabled = lab.kfIndex === 0;
  $('kf-next').disabled = lab.kfIndex === kfs.length - 1;
  const img = $('kf-img');
  img.onload = () => drawKeyframe();
  img.src = `/api/scans/${enc(state.scanId)}/keyframes/${k.name}.jpg`;
  if (lab.selected && !lab.list.some((l) => l.id === lab.selected && l.keyframe === k.name)) lab.selected = null;
  drawKeyframe();
}

function kfScale() {
  const img = $('kf-img');
  return img.naturalWidth ? img.clientWidth / img.naturalWidth : 1;
}

function drawKeyframe() {
  const img = $('kf-img'), cv = $('kf-canvas'), k = curKf();
  if (!img.naturalWidth || !k) return;
  const stage = $('label-stage').getBoundingClientRect(), r = img.getBoundingClientRect();
  cv.width = Math.round(r.width); cv.height = Math.round(r.height);
  cv.style.left = `${r.left - stage.left}px`; cv.style.top = `${r.top - stage.top}px`;
  const g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  const sc = kfScale();
  const here = lab.list.filter((l) => l.keyframe === k.name);
  for (const l of here) {
    const [x0, y0, x1, y1] = l.box_px.map((v) => v * sc);
    const col = (KIND[l.kind] || KIND.empty).color;
    g.lineWidth = l.id === lab.selected ? 3 : 2;
    g.strokeStyle = col; g.fillStyle = l.id === lab.selected ? `${col}55` : `${col}22`;
    g.fillRect(x0, y0, x1 - x0, y1 - y0); g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    g.font = '12px system-ui'; g.fillStyle = col;
    g.fillText(`${l.id} ${sizeText(l)}`, x0 + 3, Math.max(12, y0 - 4));
  }
  if (lab.drag) {
    const { x0, y0, x1, y1 } = lab.drag;
    g.setLineDash([6, 4]); g.lineWidth = 2; g.strokeStyle = KIND[lab.kind].color;
    g.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)); g.setLineDash([]);
  }
  $('label-delete').disabled = !lab.selected;
  const total = lab.list.length;
  $('label-status').textContent = `${here.length} box${here.length === 1 ? '' : 'es'} on this keyframe · ${total} in total`
    + (approxScale() ? ' · sizes use the approximate scale (calibrate in the Scale panel)' : '');
}

function setKind(kind) {
  lab.kind = kind;
  $('kind-empty').classList.toggle('on', kind === 'empty');
  $('kind-pushed').classList.toggle('on', kind === 'pushed_back');
  const sel = lab.list.find((l) => l.id === lab.selected);
  if (sel && sel.kind !== kind) patchLabel(sel.id, { kind });
}

async function patchLabel(labelId, body) {
  try {
    const r = await api(`/api/scans/${enc(state.scanId)}/labels/${enc(labelId)}`, { method: 'PATCH', body });
    lab.list = lab.list.map((l) => (l.id === labelId ? r : l));
    renderLabels();
  } catch (err) { toast(`Update failed: ${err.message}`, 'error'); }
}

async function deleteSelectedLabel() {
  const labelId = lab.selected;
  if (!labelId) return;
  try {
    await api(`/api/scans/${enc(state.scanId)}/labels/${enc(labelId)}`, { method: 'DELETE' });
    lab.list = lab.list.filter((l) => l.id !== labelId);
    lab.selected = null;
    renderLabels();
    if (gapState.doc) loadGaps();
  } catch (err) { toast(`Delete failed: ${err.message}`, 'error'); }
}

const kfCanvas = $('kf-canvas');
const canvasXY = (e) => { const r = kfCanvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
kfCanvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  const [x, y] = canvasXY(e);
  const sc = kfScale(), k = curKf();
  const hit = lab.list.filter((l) => l.keyframe === k?.name).reverse().find((l) => {
    const [x0, y0, x1, y1] = l.box_px.map((v) => v * sc);
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
  });
  if (hit) { lab.selected = hit.id; setKind(hit.kind); drawKeyframe(); return; }
  lab.selected = null;
  lab.drag = { x0: x, y0: y, x1: x, y1: y };
  kfCanvas.setPointerCapture(e.pointerId);
  drawKeyframe();
});
kfCanvas.addEventListener('pointermove', (e) => {
  if (!lab.drag) return;
  const [x, y] = canvasXY(e);
  lab.drag.x1 = Math.max(0, Math.min(kfCanvas.width, x)); lab.drag.y1 = Math.max(0, Math.min(kfCanvas.height, y));
  drawKeyframe();
});
kfCanvas.addEventListener('pointerup', async () => {
  const d = lab.drag;
  lab.drag = null;
  if (!d) return;
  if (Math.abs(d.x1 - d.x0) < 6 || Math.abs(d.y1 - d.y0) < 6) { drawKeyframe(); return; }
  const sc = kfScale(), k = curKf(), id = state.scanId;
  const box = [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)].map((v) => v / sc);
  $('label-status').textContent = 'Saving and placing the box on the shelf…';
  try {
    const r = await api(`/api/scans/${enc(id)}/labels`, { method: 'POST', body: { keyframe: k.name, box_px: box, kind: lab.kind } });
    if (state.scanId !== id) return;
    lab.list.push(r);
    lab.selected = r.id;
    renderLabels();
    if (gapState.doc) loadGaps();   // scores change with the labels
    if (r.error) toast(`Saved ${r.id}, but it could not be placed in 3D: ${r.error}`, 'error');
  } catch (err) { toast(`Saving the box failed: ${err.message}`, 'error'); drawKeyframe(); }
});

$('label-open').addEventListener('click', () => openLabelModal());
$('labels-open').addEventListener('click', () => openLabelModal());
$('label-close').addEventListener('click', closeLabelModal);
$('label-modal').addEventListener('click', (e) => { if (e.target === $('label-modal')) closeLabelModal(); });
$('kf-prev').addEventListener('click', () => setKeyframe(lab.kfIndex - 1));
$('kf-next').addEventListener('click', () => setKeyframe(lab.kfIndex + 1));
$('kind-empty').addEventListener('click', () => setKind('empty'));
$('kind-pushed').addEventListener('click', () => setKind('pushed_back'));
$('label-delete').addEventListener('click', deleteSelectedLabel);
window.addEventListener('resize', () => { if (!$('label-modal').classList.contains('hidden')) drawKeyframe(); });
window.addEventListener('keydown', (e) => {
  if ($('label-modal').classList.contains('hidden')) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); setKeyframe(lab.kfIndex - 1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); setKeyframe(lab.kfIndex + 1); }
  else if ((e.key === 'Delete' || e.key === 'Backspace') && lab.selected) { e.preventDefault(); deleteSelectedLabel(); }
  else if (e.key === 'Escape') { e.stopImmediatePropagation(); closeLabelModal(); }
}, true);

// ---------------------------------------------------------------- gap detection (shelf scans)
// GET /api/scans/{id}/gaps: both baselines in the current scene frame, scored against the labels.
const METHOD = {
  // detectors run on the ORIGINAL VIDEO KEYFRAMES, boxes placed on the shelf front in 3D
  frames_rfdetr: { name: 'RF-DETR trained · on video frames', short: 'RF-DETR', color: '#a78bfa' },
  frames_yolo: { name: 'YOLOv8s trained · on video frames', short: 'YOLOv8s', color: '#fde047' },
  frames_yolo_dl: { name: 'Downloaded YOLOv8 · on video frames', short: 'YOLO-dl', color: '#f87171' },
  // detector run on face-on RENDERS of the 3D model
  yolo: { name: 'Downloaded YOLOv8 · on 3D renders', short: 'void', color: '#ff8c00' },
  dino: { name: 'AI (Grounding DINO + SAM)', short: 'AI', color: '#22d3ee' },
  depth: { name: 'Depth (behind shelf front)', short: 'depth', color: '#a3e635' },
};
const gapsRoot = new THREE.Group();
gapsRoot.name = 'gaps';
world.add(gapsRoot);
// the trained RF-DETR on video frames is shown first; every other method is off until ticked
const gapState = { doc: null, show: { frames_rfdetr: true, frames_yolo: false, frames_yolo_dl: false, yolo: false, dino: false, depth: false } };

async function loadGaps() {
  const id = state.scanId;
  gapState.doc = null;
  if (!id || !isShelf() || !hasModel()) { renderGaps(); return; }
  try {
    const r = await api(`/api/scans/${enc(id)}/gaps`);
    if (state.scanId !== id) return;
    gapState.doc = r;
  } catch (err) {
    if (state.scanId !== id) return;
    if (err.status !== 404 && err.status !== 409) toast(`Could not load the gap results: ${err.message}`, 'error');
  }
  renderGaps();
}

function focusBox(bbox) {
  const b = new THREE.Box3(new THREE.Vector3(...bbox.min), new THREE.Vector3(...bbox.max)).applyMatrix4(world.matrixWorld);
  const cc = b.getCenter(new THREE.Vector3());
  const r = Math.max(b.getSize(new THREE.Vector3()).length() / 2, 0.15);
  const dist = Math.max(r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.6, 0.5);
  flyTo(cc, cc.clone().add(new THREE.Vector3(0, 0.1, 1).normalize().multiplyScalar(dist)));
}

const pctTxt = (v) => (v === null || v === undefined ? '–' : `${Math.round(v * 100)}%`);

function renderGaps() {
  clearOverlay(gapsRoot);
  const doc = gapState.doc;
  const shelf = isShelf() && hasModel() && lab.available;
  $('gaps-section').classList.toggle('hidden', !shelf);
  if (!shelf) return;
  const busy = isBusy() || state.busyAction;
  $('gaps-run').disabled = busy;
  $('gaps-run').textContent = doc ? 'Detect again' : 'Detect gaps';
  const info = $('gaps-info');
  const box = $('gaps-methods');
  box.innerHTML = '';
  if (!doc) {
    info.textContent = 'Not run yet. Runs two baselines on the original keyframes: AI (Grounding DINO + SAM, prompts like "empty shelf space") and depth (surfaces more than 10 cm behind the shelf front). Takes a few minutes.';
    return;
  }
  const nl = doc.labelled_slots || 0;
  info.textContent = `Last run ${fmtTime(doc.created_at)} on ${doc.num_keyframes} keyframes. `
    + (nl ? `Scored against your ${nl} labelled slot${nl === 1 ? '' : 's'} (match = overlap on the shelf front ≥ 25%).`
          : 'Label real empty slots to see how well each method does.');
  if (doc.scale_changed_since_run) {
    const w = document.createElement('div');
    w.className = 'gaps-warn small';
    w.textContent = 'The scale changed after this run: the depth method used the old scale for its 10 cm threshold. Detect again for a consistent result.';
    box.appendChild(w);
  }
  const ORDER = ['frames_rfdetr', 'frames_yolo', 'frames_yolo_dl', 'yolo', 'dino', 'depth'];
  const order = (k) => (ORDER.includes(k) ? ORDER.indexOf(k) : ORDER.length);
  for (const [key, m] of Object.entries(doc.methods || {}).sort((a, b) => order(a[0]) - order(b[0]))) {
    const meta = METHOD[key] || { name: key, short: key, color: '#ffffff' };
    const sc = (doc.scores || {})[key] || {};
    const card = document.createElement('div');
    card.className = 'gm';
    const head = document.createElement('label');
    head.className = 'gm-head';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = gapState.show[key] !== false;
    cb.addEventListener('change', () => { gapState.show[key] = cb.checked; renderGaps(); });
    const dot = document.createElement('span'); dot.className = 'dot'; dot.style.background = meta.color;
    const nm = document.createElement('span'); nm.textContent = meta.name;
    const sp = document.createElement('span'); sp.className = 'spacer';
    const n = document.createElement('span'); n.className = 'muted small'; n.textContent = `${m.gaps.length} gap${m.gaps.length === 1 ? '' : 's'}`;
    head.append(cb, dot, nm, sp, n);
    card.appendChild(head);
    const score = document.createElement('div');
    score.className = 'gm-score';
    score.innerHTML = nl
      ? `Found <b>${sc.found}/${sc.labelled_slots}</b> labelled slots (${pctTxt(sc.recall)}) · <b>${sc.false_gaps}</b> false · `
        + `precision ${pctTxt(sc.precision)}${sc.mean_abs_width_err_cm !== null && sc.mean_abs_width_err_cm !== undefined ? ` · width error ${fmt(sc.mean_abs_width_err_cm, 1)} cm` : ''}`
      : `<span class="muted">No labels to compare with yet.</span>`;
    card.appendChild(score);
    if (m.gaps.length) {
      const det = document.createElement('details');
      const sum = document.createElement('summary'); sum.textContent = 'Gaps';
      det.appendChild(sum);
      const matched = new Set((sc.matches || []).map((x) => x.gap_id));
      for (const g of m.gaps.slice(0, 200)) {
        const r = document.createElement('div');
        r.className = 'gap-row';
        r.textContent = `${g.id} · ${approxScale() ? '≈ ' : ''}${fmt(g.width_cm, 1)} × ${fmt(g.height_cm, 1)} cm · ${g.num_views} views${matched.has(g.id) ? ' · ✓ matches a label' : ''}`;
        r.addEventListener('click', () => focusBox(g.bbox));
        det.appendChild(r);
      }
      card.appendChild(det);
    }
    box.appendChild(card);
    // 3D
    if (gapState.show[key] === false) continue;
    const col = new THREE.Color(meta.color);
    for (const g of m.gaps) {
      const mn = new THREE.Vector3(...g.bbox.min), mx = new THREE.Vector3(...g.bbox.max);
      const size = mx.clone().sub(mn).max(new THREE.Vector3(1e-3, 1e-3, 1e-3));
      const geo = new THREE.BoxGeometry(size.x, size.y, size.z);
      const grp = new THREE.Group();
      grp.position.copy(mn.clone().add(mx).multiplyScalar(0.5));
      grp.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: col })));
      grp.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.12, depthWrite: false })));
      if (state.showLabels) {
        const el = document.createElement('div');
        el.className = 'slot-label';
        el.style.setProperty('--c', meta.color);
        el.textContent = `${meta.short} ${fmt(g.width_cm, 0)}×${fmt(g.height_cm, 0)} cm`;
        const tag = new CSS2DObject(el);
        tag.position.set(0, -size.y / 2, 0);
        grp.add(tag);
      }
      gapsRoot.add(grp);
    }
  }
}

$('gaps-run').addEventListener('click', async () => {
  const id = state.scanId;
  if (!id || isBusy() || state.busyAction) return;
  state.busyAction = true; renderScanPanels();
  try {
    const job = await api(`/api/scans/${enc(id)}/gaps/run`, { method: 'POST' });
    if (state.scanId !== id) return;
    state.job = job;
    const i = state.scans.findIndex((x) => x.scan_id === id);
    if (i >= 0) state.scans[i] = { ...state.scans[i], status: 'queued', progress: 0 };
    toast('Gap detection queued (a few minutes).', 'ok');
    state.busyAction = false;
    renderScanPanels(); renderScanList();
    followJob();
    refreshScans();
  } catch (err) {
    toast(`Gap detection failed to start: ${err.message}`, 'error');
    state.busyAction = false;
    renderScanPanels();
  }
});

// ---------------------------------------------------------------- scale + measure tool (shelf scans)
// Shelf scans have no metric scale: they start "approximate" (assumed shelf height). The
// measure tool is a ruler in the current scale; typing the real distance of the two points
// calibrates the whole scan (POST /api/scans/{id}/scale).
let measure = null;   // {points: [THREE.Vector3 in scan coords]} while measuring

function scaleInfo() { return state.record?.scale || null; }

function renderScale(shelf, busy) {
  const show = shelf && hasModel() && state.modelLoaded;
  $('scale-section').classList.toggle('hidden', !show);
  if (!show) { if (measure) stopMeasure(); return; }
  const sc = scaleInfo() || {};
  const t = $('scale-text');
  if (sc.status === 'calibrated') {
    t.textContent = 'Scale: calibrated';
    t.className = 'calibrated';
    const m = sc.measured || {};
    $('scale-detail').textContent = `From your measurement: ${fmt(m.real_cm, 1)} cm between two points${sc.updated_at ? ` · ${fmtTime(sc.updated_at)}` : ''}.`;
  } else {
    t.textContent = 'Scale: approximate';
    t.className = 'approximate';
    $('scale-detail').textContent = `Assumes the scanned shelf is ${fmt(sc.assumed_height_m ?? 1.9, 2)} m tall. Sizes can be off by 20% or more until you calibrate with one known distance.`;
  }
  $('measure-btn').disabled = busy || state.busyAction;
  $('scale-reset').disabled = busy || state.busyAction || sc.status !== 'calibrated';
}

function markerRadius() {
  const s = sceneBox().getSize(new THREE.Vector3()).length();
  return Math.max(s * 0.004, 0.004);
}

function drawMeasure() {
  clearOverlay(measureRoot);
  if (!measure) return;
  const r = markerRadius();
  for (const p of measure.points) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0x4c9aff, depthTest: false }));
    m.position.copy(p); m.renderOrder = 10;
    measureRoot.add(m);
  }
  if (measure.points.length === 2) {
    const [a, b] = measure.points;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
      new THREE.LineBasicMaterial({ color: 0x4c9aff, depthTest: false }));
    line.renderOrder = 10;
    measureRoot.add(line);
    const el = document.createElement('div');
    el.className = 'measure-label';
    el.textContent = measureText(a.distanceTo(b));
    const lab = new CSS2DObject(el);
    lab.position.copy(a.clone().add(b).multiplyScalar(0.5));
    measureRoot.add(lab);
  }
}

function measureText(d) {
  const approx = scaleInfo()?.status !== 'calibrated';
  return `${approx ? '≈ ' : ''}${(d * 100).toFixed(1)} cm${approx ? ' (approximate scale)' : ''}`;
}

function startMeasure() {
  if (!state.modelLoaded) return;
  measure = { points: [] };
  select(null);
  $('viewport').classList.add('measuring');
  $('measure-panel').classList.remove('hidden');
  $('measure-cal').classList.add('hidden');
  $('measure-hint').textContent = 'Click the first point on the model (e.g. one end of a shelf bay or price label).';
  $('measure-dist').textContent = '';
  $('measure-btn').classList.add('on');
  drawMeasure();
}

function stopMeasure() {
  if (!measure) return;
  measure = null;
  $('viewport').classList.remove('measuring');
  $('measure-panel').classList.add('hidden');
  $('measure-btn').classList.remove('on');
  drawMeasure();
}

function measureClick() {
  const hits = raycaster.intersectObject(modelRoot, true);
  if (!hits.length) { toast('Click on the model surface.', 'info', 2000); return; }
  const p = world.worldToLocal(hits[0].point.clone());   // scan (scene) coordinates
  if (measure.points.length >= 2) measure.points = [];
  measure.points.push(p);
  if (measure.points.length === 1) {
    $('measure-hint').textContent = 'Click the second point.';
    $('measure-dist').textContent = '';
    $('measure-cal').classList.add('hidden');
  } else {
    const d = measure.points[0].distanceTo(measure.points[1]);
    $('measure-hint').textContent = 'Distance in the current scale. Click again to start a new measurement.';
    $('measure-dist').textContent = measureText(d);
    $('measure-cal').classList.remove('hidden');
    $('measure-cm').focus();
  }
  drawMeasure();
}

async function applyScale(body, okMsg) {
  const id = state.scanId;
  if (!id || state.busyAction) return;
  state.busyAction = true; renderScanPanels();
  try {
    const res = await api(`/api/scans/${enc(id)}/scale`, { method: 'POST', body });
    if (state.scanId !== id) return;
    stopMeasure();
    state.record = (await apiOpt(`/api/scans/${enc(id)}`)) || state.record;
    await refreshScans();
    state.modelVersion = Date.now();
    userMovedCamera = false;
    await loadModel(id, state.loadToken);
    if (!userMovedCamera) fitView(true);
    loadLabels().then(loadGaps);   // label and gap sizes follow the new scale
    toast(okMsg(res?.scale || {}), 'ok', 6000);
  } catch (err) {
    toast(`Scale change failed: ${err.message}`, 'error');
  } finally {
    state.busyAction = false; renderScanPanels();
  }
}

$('measure-btn').addEventListener('click', () => (measure ? stopMeasure() : startMeasure()));
$('measure-cancel').addEventListener('click', stopMeasure);
$('measure-apply').addEventListener('click', () => {
  const cm = parseFloat($('measure-cm').value);
  if (!measure || measure.points.length !== 2) return;
  if (!(cm >= 0.5 && cm <= 100000)) { toast('Enter the real distance in centimetres (0.5 to 100000).', 'error'); return; }
  const [a, b] = measure.points;
  const prev = scaleInfo()?.factor || 1;
  applyScale({ method: 'measured', p1: a.toArray(), p2: b.toArray(), real_cm: cm },
    (sc) => `Scale calibrated: the scan was resized ×${(sc.factor / prev).toFixed(3)} so the two points are ${cm} cm apart.`);
});
$('measure-cm').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('measure-apply').click(); } });
$('scale-reset').addEventListener('click', () => {
  if (!window.confirm('Go back to the approximate scale (assumed shelf height)? Your calibration is discarded.')) return;
  applyScale({ method: 'reset' }, () => 'Scale reset to the approximate default.');
});

// ---------------------------------------------------------------- upload
function resetUploadPrompts() { $('upload-prompts').value = (state.config.default_prompts || []).join(', '); }
$('upload-prompts-reset').addEventListener('click', resetUploadPrompts);

function validateFile(file) {
  if (!file) return 'No file.';
  if (!/\.usdz$/i.test(file.name) && !SHELF_VIDEO.test(file.name)) return `"${file.name}" is not a .usdz scan or a store video (.mov, .mp4, .m4v).`;
  const max = Number(state.config.max_upload_mb) || 500;
  if (file.size > max * 1024 * 1024) return `"${file.name}" is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${max} MB.`;
  if (file.size === 0) return `"${file.name}" is empty.`;
  return null;
}

function setUploadUi(name, p, msg, kind = '') {
  $('upload-status').classList.remove('hidden');
  $('upload-name').textContent = name;
  $('upload-name').title = name;
  $('upload-pct').textContent = p === null ? '' : pct(p);
  $('upload-bar').style.width = `${(p === null ? 1 : p) * 100}%`;
  $('upload-bar').className = 'bar' + (kind ? ' ' + kind : '');
  $('upload-msg').textContent = msg;
}

function uploadFile(file) {
  const problem = validateFile(file);
  if (problem) { toast(problem, 'error'); return; }
  if (state.uploading) { toast('An upload is already in progress.', 'error'); return; }
  const isVideo = SHELF_VIDEO.test(file.name);   // store video -> new shelf scan (converted on the server)
  const prompts = parsePrompts($('upload-prompts').value);
  if (!isVideo && !prompts.length) { toast('Enter at least one prompt in the upload settings.', 'error'); $('upload-opts').open = true; return; }
  const autoRun = $('auto-run').checked;
  const fd = new FormData();
  fd.append('file', file, file.name);
  if (!isVideo) {
    fd.append('prompts', prompts.join(','));
    fd.append('auto_run', autoRun ? 'true' : 'false');
  }

  state.uploading = true;
  $('add-scan-btn').disabled = true;
  setUploadUi(file.name, 0, 'Uploading…');
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/scans');
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) setUploadUi(file.name, ev.loaded / ev.total, `Uploading… ${(ev.loaded / 1048576).toFixed(1)} / ${(ev.total / 1048576).toFixed(1)} MB`);
  };
  xhr.upload.onload = () => setUploadUi(file.name, 1, isVideo ? 'Uploaded. Creating the shelf scan…' : 'Uploaded. Preparing the scan on the server (orientation, model)…');
  const finish = () => { state.uploading = false; $('add-scan-btn').disabled = false; };
  xhr.onerror = () => {
    finish();
    setUploadUi(file.name, 1, 'Upload failed: network error.', 'error');
    toast('Upload failed: network error.', 'error');
  };
  xhr.onload = async () => {
    finish();
    let data = null;
    try { data = xhr.responseText ? JSON.parse(xhr.responseText) : null; } catch { data = xhr.responseText; }
    if (xhr.status < 200 || xhr.status >= 300) {
      const err = httpError(xhr.status, xhr.statusText, data);
      const msg = xhr.status === 413 ? `The file is too large (${err.detail || 'limit exceeded'}).` : (err.detail || err.message);
      setUploadUi(file.name, 1, `Upload failed: ${msg}`, 'error');
      toast(`Upload failed: ${msg}`, 'error');
      return;
    }
    const queuedMsg = isVideo ? ' The video → 3D conversion is queued (about 8–15 min).' : autoRun ? ' The pipeline run is queued.' : '';
    setUploadUi(file.name, 1, `Uploaded.${queuedMsg}`, 'done');
    setTimeout(() => { if (!state.uploading) $('upload-status').classList.add('hidden'); }, 8000);
    toast(`Uploaded ${file.name}.${queuedMsg}`, 'ok');
    const id = data?.scan_id;
    if (id && !state.scans.some((s) => s.scan_id === id)) state.scans.unshift(data);
    await refreshScans();
    if (id) {
      if (isDirty() && id !== state.scanId && !window.confirm('Select the new scan? Your unsent review changes on the current scan will be discarded.')) return;
      await selectScan(id, { force: true });
    }
  };
  xhr.send(fd);
}

$('add-scan-btn').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', () => {
  const f = $('file-input').files?.[0];
  $('file-input').value = '';
  if (f) uploadFile(f);
});
$('drop-zone').addEventListener('click', () => $('file-input').click());
$('drop-zone').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file-input').click(); } });

// drag and drop anywhere on the page
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  $('drop-overlay').classList.remove('hidden');
});
window.addEventListener('dragover', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});
window.addEventListener('dragleave', (e) => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $('drop-overlay').classList.add('hidden');
});
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $('drop-overlay').classList.add('hidden');
  const files = [...(e.dataTransfer.files || [])];
  if (!files.length) return;
  if (files.length > 1) toast('Only one file is uploaded at a time; using the first.', 'info');
  if (/\.webm$/i.test(files[0].name)) uploadVideo(files[0]);   // .webm: attach to this scan
  else uploadFile(files[0]);                                      // .usdz scan, or store video -> shelf scan
});

// ---------------------------------------------------------------- source video (the capture footage the model was made from)
// One optional video per scan (GET/POST/DELETE /api/scans/{id}/video). Display only; the pipeline never reads it.
const VIDEO_EXT = /\.(mp4|m4v|webm)$/i;
function pref(key, fallback) { try { const v = localStorage.getItem(`ov3d.${key}`); return v === null ? fallback : v; } catch { return fallback; } }
function setPref(key, value) { try { localStorage.setItem(`ov3d.${key}`, String(value)); } catch { /* private mode etc. */ } }

let videoOpen = pref('videoOpen', '1') === '1';
let videoSrcKey = null;   // scan + attach time the <video> element currently holds
let videoUpload = null;   // {scanId, p, msg, kind} while uploading, or after a failure on that scan
const videoMeta = () => state.record?.source_video || null;

function renderVideo() {
  const has = !!state.scanId;
  const meta = videoMeta();
  const btn = $('video-btn');
  btn.disabled = !has;
  btn.classList.toggle('on', has && videoOpen);
  btn.title = meta ? `Show the capture video (${meta.file})` : 'Show the capture video, or attach one to this scan';
  $('video-panel').classList.toggle('hidden', !has || !videoOpen);

  const el = $('video-el');
  const key = meta ? `${state.scanId}|${meta.attached_at}` : null;
  if (key !== videoSrcKey) {
    el.pause();
    // #t=0.1 makes the browser show a real frame instead of a black box before Play
    if (meta) el.src = `${meta.url}?v=${enc(meta.attached_at || '')}#t=0.1`;
    else { el.removeAttribute('src'); el.load(); }
    videoSrcKey = key;
  }
  const up = videoUpload && videoUpload.scanId === state.scanId ? videoUpload : null;
  const busy = !!up && !up.kind;
  el.classList.toggle('hidden', !meta);
  $('video-caption').classList.toggle('hidden', !meta);
  $('video-empty').classList.toggle('hidden', !!meta || busy);
  $('video-replace').classList.toggle('hidden', !meta);
  $('video-replace').disabled = busy;
  $('video-remove').disabled = busy;
  $('video-attach').disabled = busy;
  // shelf scan still converting: its video is attached automatically when the conversion ends
  const pendingShelf = isShelf() && !hasModel() && !meta;
  $('video-empty-title').textContent = pendingShelf ? 'The store video is being converted.' : 'No source video for this scan.';
  $('video-empty-hint').textContent = pendingShelf
    ? 'It appears here automatically (as a browser-playable H.264 copy without location data) when the 3D conversion finishes.'
    : 'Attach the capture video (.mp4 or .webm) that the 3D model was made from.';
  $('video-attach').classList.toggle('hidden', pendingShelf);
  if (meta) {
    $('video-name').textContent = `${meta.file} · ${(meta.bytes / 1048576).toFixed(1)} MB`;
    $('video-name').title = meta.file;
  }
  $('video-upload').classList.toggle('hidden', !up);
  if (up) {
    $('video-bar').style.width = `${up.p * 100}%`;
    $('video-bar').className = 'bar' + (up.kind ? ` ${up.kind}` : '');
    $('video-msg').textContent = up.msg;
  }
}

function setVideoOpen(open) {
  videoOpen = open;
  setPref('videoOpen', open ? '1' : '0');
  if (!open) $('video-el').pause();
  renderVideo();
}

function uploadVideo(file) {
  const id = state.scanId;
  if (!id) { toast('Select a scan first, then attach its video.', 'error'); return; }
  if (!VIDEO_EXT.test(file.name)) { toast(`"${file.name}" is not an .mp4 or .webm video.`, 'error'); return; }
  const max = Number(state.config.max_upload_mb) || 500;
  if (file.size > max * 1024 * 1024) { toast(`"${file.name}" is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${max} MB.`, 'error'); return; }
  if (!file.size) { toast(`"${file.name}" is empty.`, 'error'); return; }
  if (videoUpload && !videoUpload.kind) { toast('A video upload is already in progress.', 'error'); return; }

  const set = (p, msg, kind = '') => { videoUpload = { scanId: id, p, msg, kind }; renderVideo(); };
  setVideoOpen(true);
  set(0, `Uploading ${file.name}…`);
  const fd = new FormData();
  fd.append('file', file, file.name);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/api/scans/${enc(id)}/video`);
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) set(ev.loaded / ev.total, `Uploading… ${(ev.loaded / 1048576).toFixed(1)} / ${(ev.total / 1048576).toFixed(1)} MB`);
  };
  xhr.onerror = () => { set(1, 'Upload failed: network error.', 'error'); toast('Video upload failed: network error.', 'error'); };
  xhr.onload = () => {
    let data = null;
    try { data = xhr.responseText ? JSON.parse(xhr.responseText) : null; } catch { data = xhr.responseText; }
    if (xhr.status < 200 || xhr.status >= 300) {
      const err = httpError(xhr.status, xhr.statusText, data);
      const msg = xhr.status === 413 ? `the file is too large (${err.detail || 'limit exceeded'})` : (err.detail || err.message);
      set(1, `Upload failed: ${msg}`, 'error');
      toast(`Video upload failed: ${msg}`, 'error');
      return;
    }
    videoUpload = null;
    if (state.scanId === id && state.record) state.record = { ...state.record, source_video: data };
    toast(`Attached ${file.name}.`, 'ok');
    renderVideo();
    refreshScans();
  };
  xhr.send(fd);
}

$('video-btn').addEventListener('click', () => setVideoOpen(!videoOpen));
$('video-close').addEventListener('click', () => setVideoOpen(false));
$('video-attach').addEventListener('click', () => $('video-input').click());
$('video-replace').addEventListener('click', () => $('video-input').click());
$('video-input').addEventListener('change', () => {
  const f = $('video-input').files?.[0];
  $('video-input').value = '';
  if (f) uploadVideo(f);
});
$('video-remove').addEventListener('click', async () => {
  const id = state.scanId;
  const meta = videoMeta();
  if (!id || !meta) return;
  if (!window.confirm(`Remove "${meta.file}" from this scan? The scan and its results are kept.`)) return;
  try {
    await api(`/api/scans/${enc(id)}/video`, { method: 'DELETE' });
    if (state.scanId === id && state.record) state.record = { ...state.record, source_video: null };
    toast('Video removed from the scan.', 'ok');
    refreshScans();
  } catch (err) {
    toast(`Remove failed: ${err.message}`, 'error');
  }
  renderVideo();
});
$('video-el').addEventListener('error', () => {
  if ($('video-el').getAttribute('src')) toast('This browser cannot play the video. Use H.264 .mp4 or .webm.', 'error', 6000);
});

// remember the panel width the user dragged to
const savedVideoW = Number(pref('videoWidth', '0'));
if (savedVideoW >= 180) $('video-panel').style.width = `${savedVideoW}px`;
new ResizeObserver(() => {
  const w = $('video-panel').offsetWidth;
  if (w >= 180) setPref('videoWidth', w);
}).observe($('video-panel'));

// ---------------------------------------------------------------- init
async function init() {
  try {
    const cfg = await api('/api/config');
    state.config = { ...state.config, ...(cfg || {}) };
  } catch (err) {
    toast(`Could not load /api/config: ${err.message}`, 'error');
  }
  resetUploadPrompts();
  renderScanPanels();
  renderObjectsPanel();
  await refreshScans();
  const m = /scan=([^&]+)/.exec(location.hash);
  const want = m ? decodeURIComponent(m[1]) : null;
  const pick = state.scans.find((s) => s.scan_id === want) || state.scans[0];
  if (pick) await selectScan(pick.scan_id, { force: true });
  else fitView();
}

init();
// for debugging and tests from the console
window.__viewer = { state, scene: scene3, camera, controls, world, isDirty };
