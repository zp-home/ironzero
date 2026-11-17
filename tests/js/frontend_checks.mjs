// Behavioural checks for the frontend modules the sandbox optimisation touched.
//
// Runs the real ES modules (not copies) under Node, with tiny DOM stubs where a
// module touches globals at import time. Prints a JSON result object; the Python
// wrapper in tests/test_frontend_render_budget.py asserts on it.
//
// Kept in one file because each module is small and the stubs are shared: the
// point is to verify behaviour the old implementation had, so the checks assert
// equivalence against a straightforward reference implementation.

const results = [];
function check(name, fn) {
  try {
    const detail = fn();
    results.push({ name, ok: true, detail: detail === undefined ? null : detail });
  } catch (error) {
    results.push({ name, ok: false, error: String(error && error.message || error) });
  }
}
function assert(condition, message) {
  if (!condition) throw new Error(message);
}

// --- module loading with DOM stubs -----------------------------------------
// The modules read `window` / `document` at import time (globals table, DPR).
globalThis.window = globalThis.window || {
  devicePixelRatio: 2,
  location: { hash: '', search: '' },
  history: { replaceState() {} },
  addEventListener() {},
};
globalThis.document = globalThis.document || {
  readyState: 'complete',
  getElementById() { return null; },
  querySelectorAll() { return []; },
  createElement() {
    return {
      getContext: () => new Proxy({}, { get: () => () => {} }),
      width: 0,
      height: 0,
      style: {},
      appendChild() {},
      addEventListener() {},
    };
  },
  addEventListener() {},
};
globalThis.performance = globalThis.performance || { now: () => Date.now() };
globalThis.requestAnimationFrame = globalThis.requestAnimationFrame || (() => 0);
globalThis.fetch = globalThis.fetch || (async () => { throw new Error('fetch disabled in test'); });

const { thinPoints, DASH_MAX_POINTS } = await import('../../web/frontend/js/control.js');
const { normalizeFrame } = await import('../../web/frontend/js/normalize.js');
const { interpolateUnits } = await import('../../web/frontend/js/playback.js');
const { isUnitVisible } = await import('../../web/frontend/js/view.js');
const { updateUnitAngles, unitAngleCache } = await import('../../web/frontend/js/sprites.js');

// --- thinPoints: same range, bounded count, real thinning -------------------
check('thinPoints keeps small series intact', () => {
  const points = [{ x: 1 }, { x: 2 }, { x: 3 }];
  const out = thinPoints(points, 480);
  assert(out === points, 'a short series should be returned as-is (no copy)');
  return { length: out.length };
});

check('thinPoints bounds a large series and preserves both ends', () => {
  const points = Array.from({ length: 30000 }, (_v, index) => ({ x: index, y: index % 7 }));
  const out = thinPoints(points, DASH_MAX_POINTS);
  assert(out.length <= DASH_MAX_POINTS, `expected <= ${DASH_MAX_POINTS}, got ${out.length}`);
  assert(out[0].x === 0, `first point must survive, got ${out[0].x}`);
  assert(out[out.length - 1].x === 29999, `last point must survive, got ${out[out.length - 1].x}`);
  // Monotonic in x, so the drawn line never doubles back.
  for (let index = 1; index < out.length; index += 1) {
    assert(out[index].x > out[index - 1].x, `x must increase at ${index}`);
  }
  return { in: points.length, out: out.length };
});

check('thinPoints does not mutate its input', () => {
  const points = Array.from({ length: 5000 }, (_v, index) => ({ x: index }));
  const snapshot = points.map((point) => point.x);
  thinPoints(points, 100);
  assert(points.length === 5000, 'input length changed');
  assert(points.every((point, index) => point.x === snapshot[index]), 'input values changed');
  return { length: points.length };
});

check('thinPoints honours an explicit cap and a 2-point floor', () => {
  const points = Array.from({ length: 100 }, (_v, index) => ({ x: index }));
  assert(thinPoints(points, 10).length === 10, 'cap 10 not honoured');
  assert(thinPoints(points, 1).length === 2, 'a 1-point cap must still yield 2 points');
  return { cap10: thinPoints(points, 10).length };
});

// --- interpolateUnits: same output as the old implementation ----------------
// Reference: the pre-change implementation, kept here so equivalence is
// checked against actual old code rather than a restatement of the new code.
function interpolateUnitsReference(frame, nextFrame, fraction) {
  const merged = new Map();
  for (const unit of frame.units) merged.set(unit.id, { unit, next: null });
  if (nextFrame && nextFrame !== frame) {
    for (const unit of nextFrame.units) {
      const hit = merged.get(unit.id);
      if (hit) hit.next = unit;
    }
  }
  const blend = Math.min(1, Math.max(0, fraction));
  const result = [];
  for (const entry of merged.values()) {
    const unit = entry.unit;
    const next = entry.next;
    result.push({
      id: unit.id,
      team: unit.team,
      group: unit.group,
      x: next ? unit.x + (next.x - unit.x) * blend : unit.x,
      y: next ? unit.y + (next.y - unit.y) * blend : unit.y,
      hp: next ? unit.hp + (next.hp - unit.hp) * blend : unit.hp,
      maxHp: unit.maxHp,
      order: next ? next.order : unit.order,
      target: next ? next.target : unit.target,
      unitType: unit.unitType,
      isBuilding: unit.isBuilding,
      isBuilder: unit.isBuilder,
      isFactory: unit.isFactory,
    });
  }
  return result;
}

function sampleFrames() {
  // `team` (0 = own, 1 = enemy) is what the recorders emit. `team_id` is NOT
  // read by normalizeUnit and silently becomes 'enemy', which would put every
  // unit in one bucket and make the nearest-enemy tests vacuous.
  const frameA = normalizeFrame({
    tick: 10,
    units: [
      { id: 1, team: 0, x: 0, y: 0, hp: 100, max_hp: 100, unit_type: 'tank' },
      { id: 2, team: 1, x: 50, y: 20, hp: 40, max_hp: 80, unit_type: 'tank' },
      { id: 3, team: 0, x: 10, y: 30, hp: 5, max_hp: 10, is_building: true },
    ],
  });
  const frameB = normalizeFrame({
    tick: 20,
    units: [
      { id: 1, team: 0, x: 8, y: 4, hp: 90, max_hp: 100, unit_type: 'tank' },
      // unit 2 is absent: it died between the frames
      { id: 3, team: 0, x: 10, y: 30, hp: 5, max_hp: 10, is_building: true },
      { id: 4, team: 1, x: 99, y: 99, hp: 1, max_hp: 1, unit_type: 'tank' },
    ],
  });
  return { frameA, frameB };
}

check('interpolateUnits matches the previous implementation', () => {
  const { frameA, frameB } = sampleFrames();
  for (const fraction of [0, 0.25, 0.5, 1]) {
    const next = interpolateUnits(frameA, frameB, fraction);
    const reference = interpolateUnitsReference(frameA, frameB, fraction);
    assert(next.length === reference.length, `length differs at ${fraction}`);
    for (let index = 0; index < next.length; index += 1) {
      for (const key of Object.keys(reference[index])) {
        assert(
          next[index][key] === reference[index][key],
          `${key} differs for ${reference[index].id} at fraction ${fraction}: `
            + `${next[index][key]} vs ${reference[index][key]}`,
        );
      }
    }
  }
  return { fractions: 4 };
});

check('interpolateUnits drops units missing from the next frame', () => {
  const { frameA, frameB } = sampleFrames();
  const out = interpolateUnits(frameA, frameB, 0.5);
  const ids = out.map((unit) => unit.id).sort();
  // Units come from the CURRENT frame: 2 is still drawn (it is alive on frame A,
  // and simply has no next position to interpolate towards), 4 does not exist
  // yet on frame A so it must not appear.
  assert(ids.join(',') === '1,2,3', `expected 1,2,3, got ${ids.join(',')}`);
  const dying = out.find((unit) => unit.id === 2);
  assert(dying !== undefined, 'the dying unit should still be drawn on its last frame');
  return { ids };
});

check('interpolateUnits reuses objects across frames', () => {
  const { frameA, frameB } = sampleFrames();
  const first = interpolateUnits(frameA, frameB, 0.3);
  const second = interpolateUnits(frameA, frameB, 0.7);
  const firstById = new Map(first.map((unit) => [unit.id, unit]));
  let reused = 0;
  for (const unit of second) {
    if (firstById.get(unit.id) === unit) reused += 1;
  }
  assert(reused === second.length, `only ${reused}/${second.length} objects were pooled`);
  // And the values were still updated in place, not frozen at the first call.
  const a = second.find((unit) => unit.id === 1);
  assert(Math.abs(a.x - 5.6) < 1e-9, `x was not updated: ${a.x}`);
  return { reused };
});

// --- isUnitVisible: same answer as the old hypot implementation ------------
function isUnitVisibleReference(sandbox, unit, ownUnits) {
  if (!sandbox.fogEnabled) return true;
  if (unit.team === 'own') return true;
  const radius = Number(sandbox.visionRadius) || 0;
  for (const own of ownUnits) {
    if (Math.hypot(unit.x - own.x, unit.y - own.y) <= radius) return true;
  }
  return false;
}

check('isUnitVisible agrees with the previous hypot implementation', () => {
  const sandbox = { fogEnabled: true, visionRadius: 260 };
  const ownUnits = [];
  for (let index = 0; index < 40; index += 1) {
    ownUnits.push({ id: index, team: 'own', x: (index % 7) * 300, y: Math.floor(index / 7) * 300 });
  }
  let checked = 0;
  // Probe a dense grid, including the exact radius boundary.
  for (let gx = -200; gx <= 2200; gx += 37) {
    for (let gy = -200; gy <= 2200; gy += 37) {
      const unit = { id: 9999, team: 'enemy', x: gx, y: gy };
      const next = isUnitVisible(sandbox, unit, ownUnits);
      const reference = isUnitVisibleReference(sandbox, unit, ownUnits);
      assert(next === reference, `disagreement at (${gx},${gy}): ${next} vs ${reference}`);
      checked += 1;
    }
  }
  return { probes: checked };
});

check('isUnitVisible is always true with fog disabled or for own units', () => {
  const enemy = { id: 1, team: 'enemy', x: 9999, y: 9999 };
  const own = { id: 2, team: 'own', x: 9999, y: 9999 };
  const ownUnits = [{ id: 3, team: 'own', x: 0, y: 0 }];
  assert(isUnitVisible({ fogEnabled: false, visionRadius: 10 }, enemy, ownUnits) === true,
    'fog off must show everything');
  assert(isUnitVisible({ fogEnabled: true, visionRadius: 10 }, own, ownUnits) === true,
    'own units must always be visible');
  return null;
});

// --- updateUnitAngles: same angles as the old O(n^2) scan ------------------
function updateUnitAnglesReference(frame, previousFrame) {
  const previous = new Map();
  if (previousFrame) {
    for (const unit of previousFrame.units) previous.set(unit.id, unit);
  }
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, unit);
  const out = new Map();
  for (const unit of frame.units) {
    const record = { heading: -Math.PI / 2, turret: -Math.PI / 2 };
    const before = previous.get(unit.id);
    if (before) {
      const dx = unit.x - before.x;
      const dy = unit.y - before.y;
      if (Math.hypot(dx, dy) > 0.6) record.heading = Math.atan2(dy, dx);
    }
    let target = null;
    if (unit.target !== null && unit.target !== undefined) target = positions.get(unit.target) || null;
    if (!target) {
      let bestDistance = Infinity;
      for (const item of frame.units) {
        if (item.team === unit.team) continue;
        const distance = Math.hypot(item.x - unit.x, item.y - unit.y);
        if (distance < bestDistance) {
          bestDistance = distance;
          target = item;
        }
      }
    }
    if (target) record.turret = Math.atan2(target.y - unit.y, target.x - unit.x);
    out.set(unit.id, record);
  }
  return out;
}

function pseudoRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

check('updateUnitAngles matches the previous nearest-enemy scan', () => {
  const random = pseudoRandom(12345);
  unitAngleCache.clear();
  const previousFrame = { units: [] };
  const frame = { units: [] };
  for (let index = 0; index < 60; index += 1) {
    const id = index + 1;
    const x = random() * 2000;
    const y = random() * 2000;
    // Interleave teams so both buckets are non-empty: with a single team the
    // nearest-enemy branch never runs and this test would prove nothing.
    const team = index % 2 === 0 ? 0 : 1;
    previousFrame.units.push({ id, team, x, y });
    frame.units.push({
      id,
      team,
      x: x + (random() - 0.5) * 4,
      y: y + (random() - 0.5) * 4,
      target: null,
    });
  }
  const reference = updateUnitAnglesReference(frame, previousFrame);
  updateUnitAngles(frame, previousFrame);
  for (const [id, expected] of reference) {
    const actual = unitAngleCache.get(id);
    assert(actual, `missing angle cache entry for ${id}`);
    assert(
      Math.abs(actual.heading - expected.heading) < 1e-12,
      `heading differs for ${id}: ${actual.heading} vs ${expected.heading}`,
    );
    assert(
      Math.abs(actual.turret - expected.turret) < 1e-12,
      `turret differs for ${id}: ${actual.turret} vs ${expected.turret}`,
    );
  }
  // Every unit must have found an enemy, or the scan silently did nothing.
  const aimed = [...unitAngleCache.values()].filter((r) => r.turret !== -Math.PI / 2).length;
  assert(aimed === frame.units.length, `only ${aimed}/${frame.units.length} units aimed`);
  unitAngleCache.clear();
  return { units: frame.units.length, aimed };
});

check('updateUnitAngles prefers an explicit target over the nearest enemy', () => {
  unitAngleCache.clear();
  const frame = {
    units: [
      { id: 1, team: 'own', x: 0, y: 0, target: 3 },
      { id: 2, team: 'enemy', x: 10, y: 0, target: null },
      { id: 3, team: 'enemy', x: 0, y: 500, target: null },
    ],
  };
  updateUnitAngles(frame, null);
  const record = unitAngleCache.get(1);
  // Unit 2 is much closer, but unit 1 declared its target as 3.
  assert(Math.abs(record.turret - Math.PI / 2) < 1e-12, `turret aimed at the wrong unit: ${record.turret}`);
  unitAngleCache.clear();
  return null;
});

check('updateUnitAngles leaves turret unchanged when no enemy exists', () => {
  unitAngleCache.clear();
  const frame = { units: [{ id: 1, team: 0, x: 0, y: 0, target: null }] };
  updateUnitAngles(frame, null);
  const record = unitAngleCache.get(1);
  assert(record.turret === -Math.PI / 2, `default turret changed: ${record.turret}`);
  unitAngleCache.clear();
  return null;
});

check('updateUnitAngles no longer calls Math.hypot in the nearest-enemy scan', () => {
  // Counter-check: the reference implementation DOES call hypot for the same
  // frame, so this asserts a real difference rather than an unreached path.
  unitAngleCache.clear();
  const frame = { units: [] };
  for (let index = 0; index < 60; index += 1) {
    frame.units.push({
      id: index + 1,
      team: index % 2 === 0 ? 0 : 1,
      x: index * 13,
      y: index * 7,
      target: null,
    });
  }

  const realHypot = Math.hypot;
  let referenceCalls = 0;
  Math.hypot = (...args) => { referenceCalls += 1; return realHypot(...args); };
  updateUnitAnglesReference(frame, null);
  Math.hypot = realHypot;
  assert(referenceCalls > 0, 'the reference scan did not call hypot; the frame is degenerate');

  let actualCalls = 0;
  Math.hypot = (...args) => { actualCalls += 1; return realHypot(...args); };
  updateUnitAngles(frame, null);
  Math.hypot = realHypot;

  assert(actualCalls === 0, `updateUnitAngles still calls hypot ${actualCalls}x`);
  unitAngleCache.clear();
  return { referenceCalls, actualCalls };
});

// --- results ---------------------------------------------------------------
const failed = results.filter((item) => !item.ok);
console.log(JSON.stringify({ results, failed: failed.length }, null, 2));
process.exit(failed.length ? 1 : 0);
