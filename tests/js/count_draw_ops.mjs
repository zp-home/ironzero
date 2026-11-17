// Count canvas draw calls for one sandbox frame, before/after the optimisations.
//
// `render.js`/`sprites.js`/`hud.js` are real modules; only the 2D context is
// faked, and it records how many of each expensive operation happen per frame.
// This measures the actual change in work per frame rather than trusting that
// "removing shadowBlur must be faster".
//
// The directory holding the modules is taken from argv[2] so the A/B harness can
// point this at a checkout of the *old* modules and compare like for like.

import path from 'node:path';
import { pathToFileURL } from 'node:url';

const jsDir = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.resolve(import.meta.dirname, '..', '..', 'web', 'frontend', 'js');const load = (name) => import(pathToFileURL(path.join(jsDir, name)).href);

const counts = {
  fill: 0, stroke: 0, fillRect: 0, strokeRect: 0, drawImage: 0,
  createRadialGradient: 0, createLinearGradient: 0,
  shadowBlurSets: 0, arcs: 0, hypot: 0, atan2: 0,
};

function makeContext() {
  // Named `ctxState`, not `state`: the module-level `state` import below is the
  // app state and shadowing it here silently broke the stub.
  const ctxState = { shadowBlur: 0 };
  const handler = {
    get(target, prop) {
      if (prop === 'shadowBlur') return ctxState.shadowBlur;
      if (prop === 'canvas') return { width: 900, height: 600 };
      if (prop === 'createRadialGradient') {
        return () => { counts.createRadialGradient += 1; return { addColorStop() {} }; };
      }
      if (prop === 'createLinearGradient') {
        return () => { counts.createLinearGradient += 1; return { addColorStop() {} }; };
      }
      if (prop === 'arc') return () => { counts.arcs += 1; };
      if (prop === 'fill') return () => { counts.fill += 1; };
      if (prop === 'stroke') return () => { counts.stroke += 1; };
      if (prop === 'fillRect') return () => { counts.fillRect += 1; };
      if (prop === 'strokeRect') return () => { counts.strokeRect += 1; };
      if (prop === 'drawImage') return () => { counts.drawImage += 1; };
      if (prop === 'getContext') return () => ctx;
      if (prop === 'measureText') return () => ({ width: 10 });
      return typeof prop === 'string' ? () => {} : undefined;
    },
    set(target, prop, value) {
      if (prop === 'shadowBlur') {
        if (value > 0) counts.shadowBlurSets += 1;
        ctxState.shadowBlur = value;
      }
      return true;
    },
  };
  return new Proxy({}, handler);
}
const ctx = makeContext();

// --- DOM stubs -------------------------------------------------------------
const canvasEl = { width: 900, height: 600, getContext: () => ctx, addEventListener() {}, style: {} };
globalThis.window = { devicePixelRatio: 1, location: { hash: '', search: '' }, history: { replaceState() {} }, addEventListener() {} };
globalThis.document = {
  readyState: 'complete',
  getElementById: (id) => (id === 'sandbox-canvas' ? canvasEl : (id === 'stage-wrap' ? { clientWidth: 900, clientHeight: 600 } : null)),
  querySelectorAll: () => [],
  createElement: () => ({ getContext: () => ctx, width: 0, height: 0, style: {}, appendChild() {}, addEventListener() {} }),
  addEventListener() {},
};
globalThis.performance = { now: () => 1000 };
globalThis.requestAnimationFrame = () => 0;

// Count Math.hypot / atan2 calls too: they were the hot ops in the O(n^2) path.
// Patched BEFORE the modules are imported so intra-module calls go through it.
const realHypot = Math.hypot;
const realAtan2 = Math.atan2;
Math.hypot = (...args) => { counts.hypot += 1; return realHypot(...args); };
Math.atan2 = (...args) => { counts.atan2 += 1; return realAtan2(...args); };
const { normalizeFrame } = await load('normalize.js');
const { state } = await load('state.js');
const { drawSandbox } = await load('render.js');
const { els } = await load('state.js');

// --- a representative 149-unit frame --------------------------------------
const units = [];
for (let index = 0; index < 149; index += 1) {
  const own = index % 2 === 0;
  units.push({
    id: index + 1,
    // `team` (0 = own, 1 = enemy) is the field the recorders emit; normalizeUnit
    // ignores `team_id`, so using it would put every unit in one bucket and hide
    // the nearest-enemy path entirely.
    team: own ? 0 : 1,
    x: (index % 13) * 180 + 100,
    y: Math.floor(index / 13) * 160 + 120,
    hp: 100,
    max_hp: 100,
    unit_type: own ? 'tank' : 'hover_tank',
    is_building: index % 17 === 0,
  });
}
const frame = normalizeFrame({ tick: 100, time: 1.6, units, events: [] });

els.sandboxCanvas = canvasEl;
els.stageWrap = { clientWidth: 900, clientHeight: 600 };
els.viewZoom = null;
els.viewResetBtn = null;

state.sandbox.mapDetail = null;          // abstract/vector path
state.sandbox.terrainView = null;
state.sandbox.simMode = 'arena';
state.sandbox.viewMode = 'battle';
state.sandbox.frames = [frame];
state.sandbox.playback = 0;
state.sandbox.playing = true;
state.sandbox.fogEnabled = false;
state.sandbox.map = { width: 2600, height: 2600 };
state.sandbox.camera = { zoom: 1, panX: 0, panY: 0 };
state.sandbox.effects = [];
state.sandbox.selectedId = null;
state.sandbox.mapId = '';

drawSandbox();
console.log(JSON.stringify({ units: units.length, counts }, null, 2));
console.log(`\nper frame: ${counts.shadowBlurSets} shadowBlur sets, `
  + `${counts.fill + counts.stroke} fills/strokes, ${counts.hypot} hypot, `
  + `${counts.createRadialGradient} radial gradients`);
