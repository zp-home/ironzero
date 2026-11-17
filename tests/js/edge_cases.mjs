// Edge cases for the bounded follow / paged loader / chart throttle.
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const jsDir = path.resolve(import.meta.dirname, '..', '..', 'web', 'frontend', 'js');
const load = (name) => import(pathToFileURL(path.join(jsDir, name)).href);

globalThis.window = { devicePixelRatio: 1, location: { hash: '', search: '' }, history: { replaceState() {} }, addEventListener() {} };
globalThis.document = {
  readyState: 'complete',
  getElementById: () => null,
  querySelectorAll: () => [],
  createElement: () => ({ getContext: () => null, width: 0, height: 0, style: {}, appendChild() {}, addEventListener() {} }),
  addEventListener() {},
};
globalThis.performance = { now: () => Date.now() };
globalThis.requestAnimationFrame = () => 0;

const results = [];
const check = (name, fn) => {
  try { results.push({ name, ok: true, detail: fn() ?? null }); }
  catch (error) { results.push({ name, ok: false, error: String(error.message || error) }); }
};
const assert = (cond, msg) => { if (!cond) throw new Error(msg); };

const feInfer = await load('feInfer.js');
const feInferUi = await load('feInferUi.js');
const { state } = await load('state.js');

check('FE_LOAD_MAX_PAGES covers a longest realistic episode', () => {
  // Worst case the service allows is 216000 ticks; at DECISION_INTERVAL 10 that
  // is 21600 frames, ~43 pages of 500. The cap must be comfortably above it.
  const worstFrames = 216000 / 10;
  const pagesNeeded = Math.ceil(worstFrames / 500);
  assert(
    feInfer.FE_LOAD_MAX_PAGES >= pagesNeeded,
    `FE_LOAD_MAX_PAGES=${feInfer.FE_LOAD_MAX_PAGES} < ${pagesNeeded} pages needed `
      + `for a ${worstFrames}-frame episode`,
  );
  return { worstFrames, pagesNeeded, cap: feInfer.FE_LOAD_MAX_PAGES };
});

check('follow limit is a server-chosen bound, not unbounded', () => {
  // 0 means "server picks a bounded chunk". undefined/null would mean whole-tail.
  assert(feInfer.FE_FOLLOW_FRAME_LIMIT !== undefined, 'follow limit must be set');
  assert(feInfer.FE_FOLLOW_FRAME_LIMIT !== null, 'follow limit must not be null');
  assert(feInfer.FE_FOLLOW_FRAME_LIMIT === 0, `expected 0, got ${feInfer.FE_FOLLOW_FRAME_LIMIT}`);
  return { limit: feInfer.FE_FOLLOW_FRAME_LIMIT };
});

check('chart throttle allows an immediate first draw, then rate-limits', () => {
  const fe = state.fe;
  fe.chartsAt = 0;
  assert(feInferUi.feChartsDue(false) === true, 'first draw must not be throttled');
  assert(feInferUi.feChartsDue(false) === false, 'a second immediate draw must be throttled');
  // force always wins, for tab switches and run completion.
  assert(feInferUi.feChartsDue(true) === true, 'force must bypass the throttle');
  return { interval: feInferUi.FE_CHARTS_MIN_INTERVAL_MS };
});

check('chart throttle releases once the interval elapses', () => {
  const fe = state.fe;
  fe.chartsAt = 0;
  feInferUi.feChartsDue(false);
  // Pretend the last draw happened a full interval ago.
  fe.chartsAt = Date.now() - feInferUi.FE_CHARTS_MIN_INTERVAL_MS - 1;
  assert(feInferUi.feChartsDue(false) === true, 'draw must resume after the interval');
  return null;
});

console.log(JSON.stringify({ results, failed: results.filter((r) => !r.ok).length }, null, 2));
process.exit(results.some((r) => !r.ok) ? 1 : 0);
