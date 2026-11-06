'use strict';

/* ============================================================
 * IronZero · 铁零 — 训练 / 推理监控大屏
 * 纯静态单文件应用：无 CDN、无外部字体、无构建步骤。
 *
 * 【API 适配层】本文件所有网络请求都经由文件顶部的 api() 适配层，
 * 后端字段 / 路径如有调整只需修改这一区。默认使用相对路径 /api/...，
 * 同源部署（FastAPI :8000 / Docker）无需修改代码。
 * 调试时可用 ?api=http://127.0.0.1:8100 或
 * window.IRONZERO_API_BASE 指定其它后端地址。
 * ============================================================ */

const API_DEFAULT_BASE = '';
const API_URL_OVERRIDE = (() => {
  try {
    return new URLSearchParams(window.location.search).get('api') || '';
  } catch (err) {
    return '';
  }
})();

class ApiError extends Error {
  constructor(message, path, status) {
    super(message);
    this.name = 'ApiError';
    this.path = path;
    this.status = status === undefined ? 0 : status;
  }
}

function apiBase() {
  const explicit = typeof window.IRONZERO_API_BASE === 'string' && window.IRONZERO_API_BASE
    ? window.IRONZERO_API_BASE
    : API_URL_OVERRIDE;
  return String(explicit || API_DEFAULT_BASE).replace(/\/+$/, '');
}

async function api(path, options) {
  const opts = options || {};
  const method = opts.method || 'GET';
  const hasBody = opts.body !== undefined && opts.body !== null;
  const controller = typeof AbortController === 'function' ? new AbortController() : null;
  const timeoutMs = opts.timeout || 8000;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const headers = { Accept: 'application/json' };
  if (hasBody) headers['Content-Type'] = 'application/json';
  try {
    const response = await fetch(apiBase() + path, {
      method: method,
      headers: headers,
      body: hasBody ? JSON.stringify(opts.body) : undefined,
      signal: controller ? controller.signal : undefined,
      cache: 'no-store',
    });
    if (!response.ok) {
      let detail = '';
      try {
        detail = (await response.text()).slice(0, 240);
      } catch (err) {
        detail = '';
      }
      throw new ApiError(method + ' ' + path + ' → HTTP ' + response.status + (detail ? ' · ' + detail : ''), path, response.status);
    }
    if (response.status === 204) return null;
    const text = await response.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch (err) {
      throw new ApiError(method + ' ' + path + ' → 响应不是合法 JSON', path, response.status);
    }
  } catch (error) {
    if (error instanceof ApiError) throw error;
    const isAbort = error && error.name === 'AbortError';
    throw new ApiError(isAbort ? method + ' ' + path + ' → 请求超时' : method + ' ' + path + ' → ' + (error && error.message ? error.message : error), path, 0);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** 将含 / 的 id 逐段编码（回放 id 是相对路径） */
function encodePathId(value) {
  return String(value == null ? '' : value).split('/').map(encodeURIComponent).join('/');
}

const Api = {
  health: () => api('/api/health'),
  runs: () => api('/api/runs'),
  run: (id) => api('/api/runs/' + encodeURIComponent(id)),
  startRun: (config) => api('/api/runs', { method: 'POST', body: config }),
  stopRun: (id) => api('/api/runs/' + encodeURIComponent(id) + '/stop', { method: 'POST' }),
  runEvents: (id, after) => api('/api/runs/' + encodeURIComponent(id) + '/events?after=' + Math.max(0, Math.floor(after || 0))),
  checkpoints: () => api('/api/checkpoints'),
  startInfer: (config) => api('/api/infer', { method: 'POST', body: config }),
  infer: (id) => api('/api/infer/' + encodeURIComponent(id)),
  inferFrames: (id, after) => api('/api/infer/' + encodeURIComponent(id) + '/frames?after=' + Math.max(0, Math.floor(after || 0))),
  replays: () => api('/api/replays'),
  replay: (id) => api('/api/replays/' + encodePathId(id)),
  deleteReplay: (id) => api('/api/replays/' + encodePathId(id), { method: 'DELETE' }),
  maps: () => api('/api/maps'),
  mapDetail: (id) => api('/api/maps/' + encodeURIComponent(id)),
  assetUrl: (relative) => apiBase() + '/api/assets/' + String(relative || '').split('/').map(encodeURIComponent).join('/'),
};

/* ============================================================
 * 通用工具
 * ============================================================ */

const BASE_FPS = 10; // 沙盘播放基准帧率（帧 = 决策步）

function pickVal(object, ...keys) {
  if (!object || typeof object !== 'object') return undefined;
  for (const key of keys) {
    if (object[key] !== undefined && object[key] !== null) return object[key];
  }
  return undefined;
}

function num(value, fallback) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : (fallback === undefined ? 0 : fallback);
}

function numOrNull(value) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clampInt(value, min, max, fallback) {
  const parsed = numOrNull(value);
  if (parsed === null) return fallback === undefined ? min : fallback;
  return clamp(Math.round(parsed), min, max);
}

function fmtNum(value, digits) {
  const parsed = numOrNull(value);
  if (parsed === null) return '—';
  const places = digits === undefined ? 2 : digits;
  return parsed.toFixed(places);
}

function fmtPct(value, digits) {
  const parsed = numOrNull(value);
  if (parsed === null) return '—';
  return (parsed * 100).toFixed(digits === undefined ? 1 : digits) + '%';
}

function pad2(value) {
  return String(value).padStart(2, '0');
}

function clockString() {
  const now = new Date();
  return pad2(now.getHours()) + ':' + pad2(now.getMinutes()) + ':' + pad2(now.getSeconds());
}

function fmtDateTime(milliseconds) {
  const date = new Date(milliseconds);
  if (Number.isNaN(date.getTime())) return '—';
  return date.getFullYear() + '-' + pad2(date.getMonth() + 1) + '-' + pad2(date.getDate())
    + ' ' + pad2(date.getHours()) + ':' + pad2(date.getMinutes()) + ':' + pad2(date.getSeconds());
}

function fmtShortTime(value) {
  if (!value && value !== 0) return '';
  const text = String(value);
  const match = text.match(/(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/);
  if (match) return match[2] + '-' + match[3] + ' ' + match[4] + ':' + match[5];
  return text.length > 19 ? text.slice(0, 19) : text;
}

function basename(path) {
  if (!path) return '—';
  const text = String(path).replace(/[\\/]+$/, '');
  const parts = text.split(/[\\/]/);
  return parts[parts.length - 1] || text;
}

function safeStringify(value) {
  try {
    const text = JSON.stringify(value);
    return text && text.length > 300 ? text.slice(0, 300) + '…' : text || '';
  } catch (err) {
    return String(value);
  }
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ============================================================
 * 数据归一化（后端字段适配集中在这里）
 * ============================================================ */

function normalizeMap(raw) {
  if (raw && typeof raw === 'object') {
    return {
      width: Math.max(1, num(pickVal(raw, 'width', 'w'), 1000)),
      height: Math.max(1, num(pickVal(raw, 'height', 'h'), 1000)),
    };
  }
  return { width: 1000, height: 1000 };
}

/** 0 / 'own' / 'blue' → 'own'；1 / 'enemy' / 'red' → 'enemy' */
function normalizeTeam(value) {
  if (value === undefined || value === null) return 'enemy';
  if (typeof value === 'number') return value === 0 ? 'own' : 'enemy';
  const text = String(value).toLowerCase();
  if (text === '0' || text === 'own' || text === 'ally' || text === 'blue' || text === 'player' || text === 'ours') return 'own';
  return 'enemy';
}

function normalizeUnit(raw) {
  const unit = raw && typeof raw === 'object' ? raw : {};
  const maxHp = Math.max(1, num(pickVal(unit, 'max_hp', 'maxHp', 'hp_max'), 100));
  const hp = clamp(num(pickVal(unit, 'hp', 'health'), maxHp), 0, maxHp);
  return {
    id: pickVal(unit, 'id', 'entity_id', 'uid') === undefined ? 'u' : pickVal(unit, 'id', 'entity_id', 'uid'),
    team: normalizeTeam(pickVal(unit, 'team', 'side')),
    group: Math.round(num(pickVal(unit, 'group', 'kind_id'), 0)),
    x: num(unit.x, 0),
    y: num(unit.y, 0),
    hp: hp,
    maxHp: maxHp,
    order: pickVal(unit, 'order', 'action') || null,
    target: pickVal(unit, 'target', 'target_id') === undefined ? null : pickVal(unit, 'target', 'target_id'),
  };
}

function normalizeEvent(raw) {
  const event = raw && typeof raw === 'object' ? raw : {};
  let type = String(pickVal(event, 'type', 'kind', 'event_type') || '').toLowerCase();
  if (type === 'hit' || type === 'damage') type = 'shot';
  return {
    type: type || 'event',
    source: pickVal(event, 'source', 'src', 'from') === undefined ? null : pickVal(event, 'source', 'src', 'from'),
    target: pickVal(event, 'target', 'dst', 'to') === undefined ? null : pickVal(event, 'target', 'dst', 'to'),
    amount: num(pickVal(event, 'amount', 'damage'), 0),
    team: pickVal(event, 'team'),
    x: numOrNull(pickVal(event, 'x', 'tx')),
    y: numOrNull(pickVal(event, 'y', 'ty')),
    tick: num(pickVal(event, 'tick'), 0),
  };
}

function normalizeFrame(raw) {
  const frame = raw && typeof raw === 'object' ? raw : {};
  const stats = frame.stats && typeof frame.stats === 'object' ? frame.stats : {};
  const units = (Array.isArray(frame.units) ? frame.units : []).map(normalizeUnit);
  const events = (Array.isArray(frame.events) ? frame.events : []).map(normalizeEvent);
  return {
    episode: Math.round(num(frame.episode, 0)),
    decision: Math.round(num(pickVal(frame, 'decision', 'step'), 0)),
    tick: Math.round(num(frame.tick, 0)),
    time: num(frame.time, 0),
    reward: num(pickVal(frame, 'reward', 'r'), 0),
    units: units,
    events: events,
    stats: {
      own: clamp(num(pickVal(stats, 'own_hp_frac', 'own'), 0), 0, 1),
      enemy: clamp(num(pickVal(stats, 'enemy_hp_frac', 'enemy'), 0), 0, 1),
      ownAlive: numOrNull(pickVal(stats, 'own_alive')),
      enemyAlive: numOrNull(pickVal(stats, 'enemy_alive')),
    },
    result: pickVal(frame, 'result', 'outcome') || null,
    rewardTotal: num(pickVal(frame, 'reward_total', 'total_reward'), 0),
    map: normalizeMap(frame.map),
  };
}

function normalizeResult(raw) {
  const item = raw && typeof raw === 'object' ? raw : {};
  return {
    episode: Math.round(num(item.episode, 0)),
    result: pickVal(item, 'result', 'outcome') || null,
    ret: numOrNull(pickVal(item, 'return', 'ret')),
    decisions: numOrNull(pickVal(item, 'decisions', 'frames')),
    replayPath: pickVal(item, 'replay_path', 'path') || null,
  };
}

function normalizeReplay(raw) {
  const item = raw && typeof raw === 'object' ? raw : {};
  return {
    id: item.id || item.name || '',
    name: item.name || (item.id ? basename(item.id) : ''),
    checkpoint: pickVal(item, 'checkpoint') || null,
    opponent: pickVal(item, 'opponent') || null,
    seed: numOrNull(item.seed),
    result: pickVal(item, 'result') || null,
    ret: numOrNull(pickVal(item, 'return', 'ret')),
    episodes: numOrNull(pickVal(item, 'episodes')),
    frames: numOrNull(pickVal(item, 'frames')),
    created_at: pickVal(item, 'created_at', 'created') || null,
  };
}

const STATUS_META = {
  running: { label: '运行中', cls: 'running' },
  queued: { label: '排队中', cls: 'warn' },
  pending: { label: '排队中', cls: 'warn' },
  succeeded: { label: '已完成', cls: 'done' },
  finished: { label: '已完成', cls: 'done' },
  completed: { label: '已完成', cls: 'done' },
  done: { label: '已完成', cls: 'done' },
  stopped: { label: '已停止', cls: 'idle' },
  failed: { label: '失败', cls: 'bad' },
  error: { label: '失败', cls: 'bad' },
};

function statusMeta(status) {
  const key = String(status || '').toLowerCase();
  return STATUS_META[key] || { label: status ? String(status) : '未知', cls: 'idle' };
}

function resultMeta(result) {
  const text = String(result || '').toLowerCase();
  if (text === 'victory' || text === 'win' || text === 'won' || text === 'success') return { label: '胜利', cls: 'win' };
  if (text === 'defeat' || text === 'loss' || text === 'lost' || text === 'failure') return { label: '失败', cls: 'loss' };
  if (text === 'timeout' || text === 'draw' || text === 'tie') return { label: '平局', cls: 'draw' };
  return { label: result ? String(result) : '—', cls: 'unknown' };
}

/* ============================================================
 * DOM / 状态
 * ============================================================ */

const els = {};

function cacheEls() {
  const ids = [
    'clock', 'conn-badge', 'conn-label', 'conn-sub', 'demo-banner', 'demo-banner-text', 'demo-retry',
    'runs-count', 'run-list', 'start-form', 'start-btn',
    'kpi-update', 'kpi-update-foot', 'kpi-win', 'kpi-win-foot', 'kpi-return', 'kpi-return-foot', 'kpi-entropy', 'kpi-entropy-foot',
    'legend-win', 'chart-win', 'tip-win', 'legend-return', 'chart-return', 'tip-return',
    'run-status', 'stop-btn', 'meta-run', 'meta-kind', 'meta-created', 'meta-config', 'meta-eval',
    'log-count', 'log-panel',
    'sandbox-demo-chip', 'infer-status', 'record-btn', 'stage-wrap', 'sandbox-canvas', 'rec-badge',
    'map-select', 'view-select', 'map-status',
    'play-btn', 'latest-btn', 'speed-select', 'seek', 'frame-label', 'frame-sub',
    'infer-form', 'ckpt-select', 'episodes-input', 'infer-opponent', 'infer-seed', 'infer-mode', 'infer-btn', 'infer-hint',
    'score-own', 'score-enemy', 'hp-own', 'hp-own-val', 'hp-enemy', 'hp-enemy-val',
    'sb-ckpt', 'sb-opponent', 'sb-episode', 'sb-frames', 'event-feed',
    'replay-count', 'replay-refresh', 'replay-body', 'toast-host',
  ];
  for (const id of ids) {
    const key = id.replace(/-([a-z])/g, (match, chr) => chr.toUpperCase());
    els[key] = document.getElementById(id);
  }
}

function createSandboxState() {
  return {
    source: 'none', // none | infer | replay | demo
    inferId: null,
    frames: [],
    total: 0,
    playback: 0,
    playing: false,
    follow: false,
    streaming: false,
    speed: 1,
    fetchCursor: 0,
    status: 'idle',
    inferDone: false,
    pollErrors: 0,
    polling: false,
    map: { width: 1000, height: 1000 },
    mapId: '',
    mapDetail: null,
    terrain: null,
    terrainEntry: null,
    terrainLoading: false,
    terrainError: null,
    viewMode: 'battle',
    mapLoadToken: 0,
    meta: {},
    checkpoint: null,
    opponent: null,
    seed: null,
    episodes: 1,
    results: [],
    eventLog: [],
    effects: [],
    selectedId: null,
    view: null,
    lastUiAt: 0,
    feedCount: -1,
    recorder: null,
    recStream: null,
    recChunks: [],
    recCancel: false,
    wasPlaying: false,
  };
}

const state = {
  tab: 'train',
  mode: 'boot', // boot | live | demo
  connected: false,
  health: null,
  runs: [],
  currentRunId: null,
  runDetail: null,
  runErrors: 0,
  series: { win: [], retTrain: [], retEval: [], entropy: [], trainWin: [] },
  latest: {},
  lastEval: null,
  evalCount: 0,
  totalUpdates: null,
  eventCursor: 0,
  logs: [],
  autoScroll: true,
  logTailLoaded: false,
  dirtyCharts: true,
  checkpoints: [],
  replays: [],
  maps: { ready: false, available: false, error: null, list: [], units: {}, attempts: 0 },
  sandbox: createSandboxState(),
  mainBusy: false,
  inferTimer: null,
};

/* ============================================================
 * 演示数据（仅在 fetch 失败、后端不可用时使用，界面明确标注）
 * ============================================================ */

const DEMO = {
  ready: false,
  runs: [],
  events: {},
  details: {},
  checkpoints: [],
  replays: [],
  battles: {},
};

function buildDemoEvents(options) {
  const opts = options || {};
  const seed = num(opts.seed, 11);
  const updates = Math.max(8, Math.round(num(opts.updates, 96)));
  const evalEvery = Math.max(1, Math.round(num(opts.evalEvery, 8)));
  const random = mulberry32(seed * 7919 + 13);
  const noise = (scale) => (random() - 0.5) * scale;
  const events = [];
  let seq = 0;
  const push = (kind, payload) => {
    seq += 1;
    events.push({ seq: seq, kind: kind, payload: payload });
  };

  push('start', {
    config: {
      total_updates: updates,
      num_envs: 8,
      rollout_steps: 64,
      epochs: 4,
      lr: 0.0003,
      seed: seed,
      eval_every: evalEvery,
      eval_episodes: 8,
      opponent: 'scripted',
      hidden: 256,
    },
    device: 'cpu',
    random_baseline: { episodes: 8, win_rate: 0.0, mean_return: -1.31 },
    created_at: fmtDateTime(Date.now() - updates * 3400),
  });

  for (let update = 1; update <= updates; update += 1) {
    const winTrend = clamp(0.24 + 0.52 * (1 - Math.exp(-update / 30)) + noise(0.04), 0.02, 0.97);
    const returnTrend = -1.05 + 2.3 * (1 - Math.exp(-update / 26)) + noise(0.09);
    const entropy = 1.52 * Math.exp(-update / 55) + 0.74 + noise(0.02);
    const policyLoss = 0.32 * Math.exp(-update / 38) + 0.046 + random() * 0.014;
    const valueLoss = 0.9 * Math.exp(-update / 42) + 0.09 + random() * 0.025;
    push('update', {
      update: update,
      policy_loss: Number(policyLoss.toFixed(5)),
      value_loss: Number(valueLoss.toFixed(5)),
      entropy: Number(entropy.toFixed(4)),
      clip_fraction: Number((random() * 0.12).toFixed(4)),
      episodes: 8 * 64,
      mean_return: Number(returnTrend.toFixed(4)),
      win_rate: Number(winTrend.toFixed(4)),
      elapsed_s: Number((update * 3.4 + random()).toFixed(2)),
    });
    if (update % evalEvery === 0) {
      const evalWin = clamp(winTrend + noise(0.035), 0, 1);
      push('eval', {
        update: update,
        episodes: 8,
        win_rate: Number(evalWin.toFixed(4)),
        loss_rate: Number((1 - evalWin).toFixed(4)),
        timeout_rate: 0,
        mean_return: Number((returnTrend + noise(0.06)).toFixed(4)),
        mean_own_hp_frac: Number(clamp(evalWin * 0.92, 0, 1).toFixed(4)),
        mean_enemy_hp_frac: Number(clamp(1 - evalWin * 0.92, 0, 1).toFixed(4)),
        mean_decisions: Math.round(28 + random() * 8),
      });
    }
    if (update % 4 === 0) {
      push('log', {
        line: '[train] update ' + update + '/' + updates
          + '  return=' + returnTrend.toFixed(3)
          + '  win=' + winTrend.toFixed(2)
          + '  entropy=' + entropy.toFixed(2)
          + '  sps=' + Math.round(300 + random() * 90),
      });
    }
  }

  const finalEval = events.filter((event) => event.kind === 'eval').pop();
  push('done', {
    final_eval: finalEval ? finalEval.payload : null,
    total_seconds: Number((updates * 3.3).toFixed(1)),
    checkpoint: 'runs/demo/checkpoint.pt',
  });
  return events;
}

function demoDetailFromEvents(config, events) {
  let latest = {};
  let startPayload = null;
  let finalEval = null;
  const evals = [];
  for (const event of events) {
    const payload = event.payload || {};
    if (event.kind === 'update' && numOrNull(payload.update) !== null) latest = Object.assign({}, latest, payload);
    else if (event.kind === 'eval') evals.push(payload);
    else if (event.kind === 'start') startPayload = payload;
    else if (event.kind === 'done') finalEval = payload.final_eval || finalEval;
  }
  if (evals.length) {
    const last = evals[evals.length - 1];
    latest = Object.assign({}, latest, {
      win_rate: last.win_rate,
      mean_return: last.mean_return,
    });
  }
  const startConfig = (startPayload && startPayload.config) || {};
  const detail = {
    id: config.id,
    name: config.name,
    kind: 'train',
    status: config.status,
    created_at: fmtDateTime(Date.now() - config.createdAgo),
    config: Object.assign({}, startConfig, {
      updates: config.updates,
      total_updates: config.updates,
      envs: 8,
      rollout_steps: 64,
      opponent: 'scripted',
      seed: config.seed,
      eval_every: config.evalEvery,
      eval_episodes: 8,
    }),
    latest: latest,
    evals: evals,
    final_eval: finalEval,
    events: events.length,
    log_tail: buildDemoLogTail(evals, config),
  };
  return detail;
}

function buildDemoLogTail(evals, config) {
  const lines = [
    '[' + fmtShortTime(fmtDateTime(Date.now() - config.createdAgo)) + '] ironzero.train | device=cpu envs=8 updates=' + config.updates,
    '[' + fmtShortTime(fmtDateTime(Date.now() - config.createdAgo)) + '] opponent=scripted seed=' + config.seed + ' eval_every=' + config.evalEvery,
  ];
  const recent = evals.slice(-3);
  for (const item of recent) {
    lines.push('eval update=' + item.update + ' win_rate=' + Number(item.win_rate).toFixed(3) + ' mean_return=' + Number(item.mean_return).toFixed(3));
  }
  return lines;
}

function buildDemoBattle(options) {
  const opts = options || {};
  const seed = num(opts.seed, 11);
  const ownPower = num(opts.ownPower, 1);
  const random = mulberry32(seed * 104729 + 7);
  const MAP = { width: 1000, height: 1000 };
  const TEAM_SIZE = 8;
  const MAX_HP = 210;
  const TICK_RATE = 60;
  const DECISION_INTERVAL = 30;
  const MAX_DECISIONS = 150;
  const units = [];

  for (let index = 0; index < TEAM_SIZE; index += 1) {
    const group = Math.floor(index / 2);
    const baseY = 120 + group * 220;
    units.push({
      id: index + 1, team: 0, group: group,
      x: 170 + (index % 2) * 30 + random() * 18,
      y: baseY + (random() - 0.5) * 40,
      hp: MAX_HP, maxHp: MAX_HP, alive: true, target: null, cooldown: 0,
    });
    units.push({
      id: index + 9, team: 1, group: group,
      x: 830 - (index % 2) * 30 - random() * 18,
      y: baseY + (random() - 0.5) * 40,
      hp: MAX_HP, maxHp: MAX_HP, alive: true, target: null, cooldown: 0,
    });
  }

  const frames = [];
  const eventLog = [];
  let result = null;
  const stepMove = 16;
  const engageRange = 230;

  for (let decision = 1; decision <= MAX_DECISIONS; decision += 1) {
    const shotPlans = [];
    for (const unit of units) {
      if (!unit.alive) continue;
      const foes = units.filter((other) => other.alive && other.team !== unit.team);
      if (!foes.length) break;
      let target = null;
      let bestDistance = Infinity;
      for (const foe of foes) {
        const distance = Math.hypot(foe.x - unit.x, foe.y - unit.y);
        if (distance < bestDistance) {
          bestDistance = distance;
          target = foe;
        }
      }
      if (!target) break;
      unit.target = target.id;
      const angle = Math.atan2(target.y - unit.y, target.x - unit.x);
      if (bestDistance > engageRange) {
        unit.x = clamp(unit.x + Math.cos(angle) * stepMove, 0, MAP.width);
        unit.y = clamp(unit.y + Math.sin(angle) * stepMove, 0, MAP.height);
      }
      if (unit.cooldown > 0) unit.cooldown -= 1;
      if (bestDistance <= engageRange + 24 && unit.cooldown <= 0) {
        unit.cooldown = 4;
        shotPlans.push({ source: unit, target: target });
      }
    }

    const events = [];
    for (const plan of shotPlans) {
      const target = plan.target;
      if (!target.alive) continue;
      const power = target.team === 1 ? ownPower : 1;
      const amount = Math.round(25 * power * (0.9 + random() * 0.2));
      target.hp = Math.max(0, target.hp - amount);
      events.push({
        type: 'shot', source: plan.source.id, target: target.id,
        team: plan.source.team, amount: amount, tick: decision * DECISION_INTERVAL,
      });
      if (target.hp <= 0) {
        target.alive = false;
        events.push({
          type: 'kill', source: plan.source.id, target: target.id, team: plan.source.team,
          x: Number(target.x.toFixed(1)), y: Number(target.y.toFixed(1)), tick: decision * DECISION_INTERVAL,
        });
      }
    }

    const ownAlive = units.filter((unit) => unit.team === 0 && unit.alive);
    const enemyAlive = units.filter((unit) => unit.team === 1 && unit.alive);
    if (!enemyAlive.length) result = 'victory';
    else if (!ownAlive.length) result = 'defeat';

    const ownHp = units.filter((unit) => unit.team === 0).reduce((sum, unit) => sum + Math.max(0, unit.hp), 0) / (TEAM_SIZE * MAX_HP);
    const enemyHp = units.filter((unit) => unit.team === 1).reduce((sum, unit) => sum + Math.max(0, unit.hp), 0) / (TEAM_SIZE * MAX_HP);
    const frame = {
      episode: 0,
      decision: decision,
      tick: decision * DECISION_INTERVAL,
      time: Number(((decision * DECISION_INTERVAL) / TICK_RATE).toFixed(3)),
      reward: Number(((ownHp - enemyHp) / 8).toFixed(4)),
      units: units.filter((unit) => unit.alive).map((unit) => ({
        id: unit.id, team: unit.team, group: unit.group,
        x: Number(unit.x.toFixed(1)), y: Number(unit.y.toFixed(1)),
        hp: Number(unit.hp.toFixed(1)), max_hp: unit.maxHp,
        order: unit.target === null ? 'attack_move' : 'attack_target', target: unit.target,
      })),
      events: events,
      stats: {
        own_hp_frac: Number(ownHp.toFixed(4)),
        enemy_hp_frac: Number(enemyHp.toFixed(4)),
        own_alive: ownAlive.length,
        enemy_alive: enemyAlive.length,
      },
      result: result,
      reward_total: Number((((ownHp - enemyHp) / 4)).toFixed(4)),
    };
    frames.push(normalizeFrame(frame));
    for (const event of events) {
      eventLog.push({
        kind: event.type,
        time: frame.time,
        text: event.type === 'shot'
          ? '命中 #' + event.source + ' → #' + event.target + '  -' + Math.round(event.amount) + ' HP'
          : '击毁 #' + event.target + '（#' + event.source + '）',
      });
    }
    if (result) break;
  }

  const lastFrame = frames.length ? frames[frames.length - 1] : null;
  const totalReward = lastFrame ? numOrNull(lastFrame.rewardTotal) : null;
  const meta = {
    schema: 'ironzero.replay/1',
    checkpoint: 'runs/demo/checkpoint.pt',
    opponent: 'scripted',
    seed: seed,
    episodes: 1,
    mode: 'greedy',
    created_at: fmtDateTime(Date.now()),
    map: MAP,
    tick_rate: TICK_RATE,
    decision_interval: DECISION_INTERVAL,
    units_per_team: TEAM_SIZE,
    groups: 4,
    episode: 0,
    result: result || 'timeout',
    return: totalReward === null ? 0 : totalReward,
    decisions: frames.length,
  };
  return { meta: meta, frames: frames, eventLog: eventLog };
}

function initDemoData() {
  if (DEMO.ready) return;
  DEMO.ready = true;
  const configs = [
    { id: 'demo-microduel-a', name: 'micro_duel · 演示运行 A', status: 'running', seed: 11, updates: 96, evalEvery: 8, createdAgo: 5 * 60 * 1000 },
    { id: 'demo-microduel-b', name: 'micro_duel · 演示运行 B', status: 'succeeded', seed: 29, updates: 64, evalEvery: 8, createdAgo: 42 * 60 * 1000 },
  ];
  for (const config of configs) {
    const events = buildDemoEvents(config);
    const detail = demoDetailFromEvents(config, events);
    DEMO.events[config.id] = events;
    DEMO.details[config.id] = detail;
    DEMO.runs.push({
      id: detail.id,
      name: detail.name,
      kind: 'train',
      status: detail.status,
      created_at: detail.created_at,
      config: detail.config,
      latest: detail.latest,
      final_eval: detail.final_eval,
      eval_count: detail.evals.length,
    });
  }

  DEMO.checkpoints = [
    { name: 'demo/micro_duel/checkpoint.pt', path: 'runs/demo/micro_duel/checkpoint.pt', update: 96, eval: { win_rate: 0.76 }, mtime: fmtDateTime(Date.now()), size_mb: 0.66 },
    { name: 'demo/micro_probe/checkpoint.pt', path: 'runs/demo/micro_probe/checkpoint.pt', update: 64, eval: { win_rate: 0.63 }, mtime: fmtDateTime(Date.now() - 3600000), size_mb: 0.65 },
  ];

  const battleA = buildDemoBattle({ seed: 11, ownPower: 1.12 });
  const battleB = buildDemoBattle({ seed: 23, ownPower: 0.82 });
  DEMO.battles['demo-replay-a'] = battleA;
  DEMO.battles['demo-replay-b'] = battleB;
  DEMO.replays = [
    {
      id: 'demo-replay-a', name: 'demo-victory-scripted-seed11.json',
      checkpoint: battleA.meta.checkpoint, opponent: battleA.meta.opponent, seed: 11,
      result: battleA.meta.result, ret: battleA.meta.return, episodes: 1,
      frames: battleA.frames.length, created_at: battleA.meta.created_at,
    },
    {
      id: 'demo-replay-b', name: 'demo-defeat-scripted-seed23.json',
      checkpoint: battleB.meta.checkpoint, opponent: battleB.meta.opponent, seed: 23,
      result: battleB.meta.result, ret: battleB.meta.return, episodes: 1,
      frames: battleB.frames.length, created_at: battleB.meta.created_at,
    },
  ];
}

function demoReplayPayload(id) {
  initDemoData();
  const battle = DEMO.battles[id];
  if (!battle) return null;
  return { meta: battle.meta, frames: battle.frames.map((frame) => frame), eventLog: battle.eventLog };
}

/* ============================================================
 * 连接状态 / 降级
 * ============================================================ */

let clockTimer = null;

function startClock() {
  if (clockTimer) clearInterval(clockTimer);
  const tick = () => {
    if (els.clock) els.clock.textContent = clockString();
  };
  tick();
  clockTimer = setInterval(tick, 1000);
}

function renderConn() {
  const badge = els.connBadge;
  if (!badge) return;
  let modeName = 'down';
  let label = '后端未连接';
  let sub = '等待 /api/health';
  if (state.mode === 'boot') {
    modeName = 'connecting';
    label = '连接中…';
    sub = '检测 /api/health';
  } else if (state.connected) {
    modeName = 'ok';
    label = '已连接';
    sub = 'v' + ((state.health && state.health.version) || '—')
      + ' · jobs ' + ((state.health && state.health.jobs_running) !== undefined ? state.health.jobs_running : '—');
  } else if (state.mode === 'demo') {
    modeName = 'demo';
    label = '后端未连接';
    sub = '演示数据 · 仅界面预览';
  } else {
    modeName = 'down';
    label = '连接中断';
    sub = '显示最后一次数据';
  }
  badge.dataset.state = modeName;
  if (els.connLabel) els.connLabel.textContent = label;
  if (els.connSub) els.connSub.textContent = sub;

  const banner = els.demoBanner;
  if (!banner) return;
  const showBanner = state.mode === 'demo' || (state.mode === 'live' && !state.connected);
  banner.hidden = !showBanner;
  if (showBanner && els.demoBannerText) {
    els.demoBannerText.textContent = state.mode === 'demo'
      ? '后端未连接，当前使用内置演示数据渲染界面，仅用于视觉预览，不代表真实训练结果。'
      : '后端连接中断，页面显示最后一次获取的数据；恢复后会自动继续刷新。';
  }
}

function showDemoChips(visible) {
  const chips = document.querySelectorAll('.demo-chip');
  for (const chip of chips) chip.hidden = !visible;
}

async function checkHealth() {
  try {
    const health = await Api.health();
    state.connected = !!health && health.ok !== false;
    state.health = health || {};
  } catch (err) {
    state.connected = false;
    state.health = null;
  }
  return state.connected;
}

async function retryConnection() {
  const ok = await checkHealth();
  renderConn();
  if (ok) {
    toast('后端已连接，正在加载真实数据…', 'ok');
    tickMain();
  } else {
    toast('仍然无法连接后端（/api/health）', 'error');
  }
}

function enterDemo() {
  const firstTime = state.mode !== 'demo';
  state.mode = 'demo';
  initDemoData();
  showDemoChips(true);
  if (!firstTime) return;
  state.runs = DEMO.runs.slice();
  renderRunList();
  state.replays = DEMO.replays.slice();
  renderReplays();
  renderCheckpoints(DEMO.checkpoints);
  if (els.inferHint) els.inferHint.textContent = '后端未连接：可播放内置演示对局（演示数据，仅用于界面预览）。';
  if (DEMO.runs.length) selectDemoRun(DEMO.runs[0].id);
  syncMapStatus();
  if (state.tab === 'sandbox') ensureSandboxContent();
}

function exitDemo() {
  state.mode = 'live';
  showDemoChips(false);
  state.currentRunId = null;
  state.runDetail = null;
  resetRunData(true);
  state.runs = [];
  renderRunList();
  state.checkpoints = [];
  renderCheckpoints([]);
  state.replays = [];
  renderReplays();
  resetSandbox();
  if (els.inferHint) els.inferHint.textContent = '选择 checkpoint 后开始推理，沙盘将实时回放决策帧。';
  renderConn();
}

/* ============================================================
 * 训练监控
 * ============================================================ */

function upsertPoint(series, x, y) {
  const valueX = numOrNull(x);
  const valueY = numOrNull(y);
  if (valueX === null || valueY === null) return;
  const last = series[series.length - 1];
  if (last && last.x === valueX) {
    last.y = valueY;
    return;
  }
  if (last && valueX < last.x) {
    let low = 0;
    let high = series.length;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (series[mid].x < valueX) low = mid + 1;
      else high = mid;
    }
    if (series[low] && series[low].x === valueX) series[low].y = valueY;
    else series.splice(low, 0, { x: valueX, y: valueY });
    return;
  }
  series.push({ x: valueX, y: valueY });
  if (series.length > 4000) series.splice(0, series.length - 4000);
}

function resetRunData(clearSelection) {
  state.series = { win: [], retTrain: [], retEval: [], entropy: [], trainWin: [] };
  state.latest = {};
  state.lastEval = null;
  state.evalCount = 0;
  state.eventCursor = 0;
  state.totalUpdates = null;
  state.logs = [];
  state.logTailLoaded = false;
  state.dirtyCharts = true;
  if (clearSelection) {
    state.currentRunId = null;
    state.runDetail = null;
  }
  if (els.logPanel) els.logPanel.innerHTML = '<div class="log-empty">等待日志…</div>';
  if (els.logCount) els.logCount.textContent = '0';
  updateKpis();
  updateRunHeader();
  drawCharts();
}

function makeLogNode(entry) {
  const line = document.createElement('div');
  line.className = 'log-line kind-' + (entry.kind || 'log');
  if (entry.time) {
    const ts = document.createElement('span');
    ts.className = 'ts';
    ts.textContent = '[' + entry.time + ']';
    line.appendChild(ts);
  }
  line.appendChild(document.createTextNode(entry.text || ''));
  return line;
}

function appendLog(entry) {
  if (!entry) return;
  const item = {
    text: String(entry.text === undefined ? '' : entry.text),
    kind: entry.kind || 'log',
    time: entry.time || clockString(),
  };
  state.logs.push(item);
  if (state.logs.length > 800) state.logs.splice(0, state.logs.length - 800);
  const panel = els.logPanel;
  if (!panel) return;
  const empty = panel.querySelector('.log-empty');
  if (empty) empty.remove();
  panel.appendChild(makeLogNode(item));
  while (panel.childElementCount > 800) panel.removeChild(panel.firstElementChild);
  if (state.autoScroll) panel.scrollTop = panel.scrollHeight;
  if (els.logCount) els.logCount.textContent = String(state.logs.length);
}

function syncLogTail(lines) {
  if (!Array.isArray(lines) || !lines.length) return;
  if (state.logTailLoaded) return;
  state.logTailLoaded = true;
  const tail = lines.slice(-80);
  for (const line of tail) {
    const text = String(line);
    if (!text.trim()) continue;
    appendLog({ text: text, kind: 'stdout' });
  }
}

function handleStartEvent(payload) {
  const config = (payload && payload.config) || {};
  const totalUpdates = numOrNull(config.total_updates) !== null ? config.total_updates : numOrNull(config.updates);
  if (totalUpdates !== null) state.totalUpdates = totalUpdates;
  const parts = [];
  if (totalUpdates !== null) parts.push('updates=' + totalUpdates);
  if (numOrNull(config.num_envs) !== null) parts.push('envs=' + config.num_envs);
  if (numOrNull(config.rollout_steps) !== null) parts.push('rollout=' + config.rollout_steps);
  if (config.opponent) parts.push('opponent=' + config.opponent);
  if (numOrNull(config.seed) !== null) parts.push('seed=' + config.seed);
  const baseline = payload && payload.random_baseline;
  if (baseline && numOrNull(baseline.win_rate) !== null) parts.push('随机基线胜率=' + fmtPct(baseline.win_rate));
  appendLog({ text: '训练启动 · ' + (parts.join(' · ') || '配置已载入'), kind: 'status' });
  state.dirtyCharts = true;
}

function handleUpdateEvent(payload) {
  if (!payload) return;
  const update = numOrNull(payload.update);
  state.latest = Object.assign({}, state.latest, payload);
  if (update !== null) {
    if (numOrNull(payload.mean_return) !== null) upsertPoint(state.series.retTrain, update, payload.mean_return);
    if (numOrNull(payload.win_rate) !== null) upsertPoint(state.series.trainWin, update, payload.win_rate);
    if (numOrNull(payload.entropy) !== null) upsertPoint(state.series.entropy, update, payload.entropy);
  }
  state.dirtyCharts = true;
}

function handleEvalEvent(payload) {
  if (!payload) return;
  const update = numOrNull(payload.update);
  const isNewEval = update === null || !state.series.win.some((point) => point.x === update);
  state.lastEval = Object.assign({}, payload);
  if (isNewEval) state.evalCount += 1;
  if (update !== null) {
    if (numOrNull(payload.win_rate) !== null) upsertPoint(state.series.win, update, payload.win_rate);
    if (numOrNull(payload.mean_return) !== null) upsertPoint(state.series.retEval, update, payload.mean_return);
  }
  appendLog({
    text: '评估 update=' + (update === null ? '—' : update)
      + ' · 胜率 ' + fmtPct(payload.win_rate)
      + ' · 回报 ' + fmtNum(payload.mean_return, 3)
      + ' · 对局 ' + (numOrNull(payload.episodes) === null ? '—' : payload.episodes),
    kind: 'eval',
  });
  state.dirtyCharts = true;
}

function handleDoneEvent(payload) {
  const finalEval = payload && payload.final_eval;
  appendLog({
    text: '训练完成' + (finalEval ? ' · final win_rate=' + fmtPct(finalEval.win_rate) + ' · return=' + fmtNum(finalEval.mean_return, 3) : '')
      + (payload && numOrNull(payload.total_seconds) !== null ? ' · 用时 ' + fmtNum(payload.total_seconds, 1) + 's' : ''),
    kind: 'done',
  });
  if (finalEval) {
    // 不重复打日志：final_eval 与最后一次 eval 事件内容一致
    state.lastEval = Object.assign({}, finalEval);
    const update = numOrNull(finalEval.update);
    if (update !== null) {
      if (numOrNull(finalEval.win_rate) !== null) upsertPoint(state.series.win, update, finalEval.win_rate);
      if (numOrNull(finalEval.mean_return) !== null) upsertPoint(state.series.retEval, update, finalEval.mean_return);
    }
  }
  state.dirtyCharts = true;
}

function applyEvent(event) {
  if (!event || typeof event !== 'object') return;
  const kind = String(pickVal(event, 'kind', 'type', 'event') || '').toLowerCase();
  const payload = event.payload && typeof event.payload === 'object' ? event.payload : event;
  const seq = numOrNull(pickVal(event, 'seq', 'sequence'));
  if (seq !== null) state.eventCursor = Math.max(state.eventCursor, seq);

  if (kind === 'start') {
    handleStartEvent(payload);
    return;
  }
  if (kind === 'update' || kind === 'metrics' || kind === 'train') {
    handleUpdateEvent(payload);
    return;
  }
  if (kind === 'eval' || kind === 'evaluation' || kind === 'test') {
    handleEvalEvent(payload);
    return;
  }
  if (kind === 'done' || kind === 'finish' || kind === 'finished') {
    handleDoneEvent(payload);
    return;
  }
  if (kind === 'error') {
    const message = pickVal(payload, 'message', 'error', 'detail');
    appendLog({ text: '错误：' + (message ? String(message) : safeStringify(payload)), kind: 'error' });
    return;
  }
  if (kind === 'log' || kind === 'info' || kind === 'stdout') {
    const text = pickVal(payload, 'line', 'message', 'text');
    appendLog({ text: text ? String(text) : safeStringify(payload), kind: 'log' });
    if (numOrNull(payload.update) !== null) handleUpdateEvent(payload);
    return;
  }
  // 未知 kind：带指标按指标处理，否则作为普通日志
  const hasMetrics = numOrNull(payload.update) !== null
    && (numOrNull(payload.win_rate) !== null || numOrNull(payload.mean_return) !== null || numOrNull(payload.entropy) !== null);
  if (hasMetrics) {
    handleUpdateEvent(payload);
    return;
  }
  const text = pickVal(payload, 'line', 'message', 'text');
  if (text) appendLog({ text: String(text), kind: 'log' });
  else if (kind) appendLog({ text: '[' + kind + '] ' + safeStringify(payload), kind: 'log' });
}

function applyEvents(events, next) {
  if (Array.isArray(events)) {
    for (const event of events) applyEvent(event);
  }
  if (next !== undefined && next !== null) state.eventCursor = Math.max(state.eventCursor, num(next, state.eventCursor));
  updateKpis();
  updateRunHeader();
  if (state.dirtyCharts) drawCharts();
}

function applyDetailExtras(detail) {
  if (!detail || typeof detail !== 'object') return;
  const config = detail.config && typeof detail.config === 'object' ? detail.config : {};
  const totalUpdates = numOrNull(config.total_updates) !== null ? config.total_updates : numOrNull(config.updates);
  if (totalUpdates !== null) state.totalUpdates = totalUpdates;
  const latest = detail.latest;
  if (latest && typeof latest === 'object' && Object.keys(latest).length) {
    state.latest = Object.assign({}, state.latest, latest);
    const update = numOrNull(latest.update);
    if (update !== null) {
      if (numOrNull(latest.mean_return) !== null) upsertPoint(state.series.retTrain, update, latest.mean_return);
      if (numOrNull(latest.win_rate) !== null) upsertPoint(state.series.trainWin, update, latest.win_rate);
    }
  }
  const evals = Array.isArray(detail.evals) ? detail.evals : [];
  if (evals.length) {
    state.evalCount = Math.max(state.evalCount, evals.length);
    for (const item of evals) {
      if (!item || typeof item !== 'object') continue;
      const update = numOrNull(item.update);
      if (update === null) continue;
      if (numOrNull(item.win_rate) !== null) upsertPoint(state.series.win, update, item.win_rate);
      if (numOrNull(item.mean_return) !== null) upsertPoint(state.series.retEval, update, item.mean_return);
    }
    state.lastEval = Object.assign({}, evals[evals.length - 1]);
  }
  state.dirtyCharts = true;
}

function describeConfig(config) {
  const parts = [];
  if (numOrNull(config.updates) !== null) parts.push(config.updates + ' upd');
  if (numOrNull(config.envs) !== null) parts.push(config.envs + ' envs');
  if (numOrNull(config.rollout_steps) !== null) parts.push('rollout ' + config.rollout_steps);
  if (config.opponent) parts.push('vs ' + config.opponent);
  if (numOrNull(config.seed) !== null) parts.push('seed ' + config.seed);
  return parts.length ? parts.join(' · ') : '—';
}

function describeEvalPolicy(config) {
  const parts = [];
  if (numOrNull(config.eval_every) !== null) parts.push('每 ' + config.eval_every + ' upd');
  if (numOrNull(config.eval_episodes) !== null) parts.push(config.eval_episodes + ' 局');
  if (numOrNull(config.lr) !== null) parts.push('lr ' + config.lr);
  return parts.length ? parts.join(' · ') : '—';
}

function updateRunHeader() {
  const detail = state.runDetail || {};
  const listRun = state.runs.find((run) => run.id === state.currentRunId);
  const status = detail.status || (listRun && listRun.status);
  const meta = statusMeta(status);
  if (els.runStatus) {
    els.runStatus.className = 'status-pill ' + meta.cls;
    els.runStatus.textContent = state.currentRunId ? meta.label : '未选择';
  }
  if (els.stopBtn) {
    const active = status && ['running', 'queued', 'pending'].indexOf(String(status).toLowerCase()) >= 0;
    els.stopBtn.disabled = !active;
  }
  if (!els.metaRun) return;
  if (!state.currentRunId) {
    els.metaRun.textContent = '—';
    els.metaKind.textContent = '—';
    els.metaCreated.textContent = '—';
    els.metaConfig.textContent = '—';
    els.metaEval.textContent = '—';
    return;
  }
  const config = detail.config && typeof detail.config === 'object' ? detail.config : {};
  els.metaRun.textContent = detail.name || detail.id || state.currentRunId;
  els.metaKind.textContent = detail.kind || 'train';
  els.metaCreated.textContent = fmtShortTime(detail.created_at) || '—';
  els.metaConfig.textContent = describeConfig(config);
  els.metaEval.textContent = describeEvalPolicy(config);
}

function updateKpis() {
  if (!els.kpiUpdate) return;
  const latest = state.latest || {};
  const lastEval = state.lastEval;
  const config = (state.runDetail && state.runDetail.config) || {};
  const totalUpdates = numOrNull(state.totalUpdates) !== null
    ? state.totalUpdates
    : (numOrNull(config.total_updates) !== null ? config.total_updates : numOrNull(config.updates));
  const updateValue = numOrNull(latest.update) !== null
    ? numOrNull(latest.update)
    : (state.series.win.length ? state.series.win[state.series.win.length - 1].x : null);

  els.kpiUpdate.textContent = updateValue === null ? '—' : String(Math.round(updateValue));
  if (els.kpiUpdateFoot) {
    els.kpiUpdateFoot.textContent = totalUpdates !== null
      ? '共 ' + Math.round(totalUpdates) + ' updates · 评估 ' + state.evalCount + ' 次'
      : (state.evalCount ? '评估 ' + state.evalCount + ' 次' : '等待首个 update');
  }

  const winValue = lastEval && numOrNull(lastEval.win_rate) !== null ? lastEval.win_rate : numOrNull(latest.win_rate);
  els.kpiWin.textContent = winValue === null || winValue === undefined ? '—' : fmtPct(winValue);
  if (els.kpiWinFoot) {
    els.kpiWinFoot.textContent = lastEval
      ? '评估 ' + state.evalCount + ' 次 · 每 ' + (numOrNull(config.eval_every) === null ? '—' : config.eval_every) + ' upd'
      : '尚无评估数据';
  }

  const returnValue = lastEval && numOrNull(lastEval.mean_return) !== null ? lastEval.mean_return : numOrNull(latest.mean_return);
  els.kpiReturn.textContent = returnValue === null || returnValue === undefined ? '—' : fmtNum(returnValue, 3);
  if (els.kpiReturnFoot) {
    els.kpiReturnFoot.textContent = numOrNull(latest.policy_loss) === null
      ? 'policy_loss —'
      : 'policy_loss ' + fmtNum(latest.policy_loss, 4);
  }

  els.kpiEntropy.textContent = numOrNull(latest.entropy) === null ? '—' : fmtNum(latest.entropy, 3);
  if (els.kpiEntropyFoot) {
    els.kpiEntropyFoot.textContent = numOrNull(latest.value_loss) === null
      ? 'value_loss —'
      : 'value_loss ' + fmtNum(latest.value_loss, 4);
  }
}

function renderRunList() {
  const box = els.runList;
  if (!box) return;
  box.innerHTML = '';
  const runs = state.runs || [];
  if (els.runsCount) els.runsCount.textContent = String(runs.length);
  if (!runs.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = state.mode === 'demo' ? '暂无演示运行' : '暂无训练运行，填写左侧表单启动一次训练';
    box.appendChild(empty);
    return;
  }
  for (const run of runs) {
    const item = document.createElement('div');
    item.className = 'run-item' + (run.id === state.currentRunId ? ' active' : '');
    item.tabIndex = 0;
    item.setAttribute('role', 'button');
    const top = document.createElement('div');
    top.className = 'r-top';
    const name = document.createElement('span');
    name.className = 'run-name';
    name.textContent = run.name || run.id || '未命名';
    name.title = run.name || run.id || '';
    const pill = document.createElement('span');
    const meta = statusMeta(run.status);
    pill.className = 'status-pill ' + meta.cls;
    pill.textContent = meta.label;
    top.appendChild(name);
    top.appendChild(pill);

    const sub = document.createElement('div');
    sub.className = 'run-sub';
    const update = run.latest && numOrNull(run.latest.update) !== null ? run.latest.update : null;
    sub.textContent = 'update ' + (update === null ? '—' : update)
      + ' · 评估 ' + (numOrNull(run.eval_count) === null ? (Array.isArray(run.evals) ? run.evals.length : '—') : run.eval_count)
      + ' · ' + (fmtShortTime(run.created_at) || '—');

    item.appendChild(top);
    item.appendChild(sub);
    item.addEventListener('click', () => { selectRun(run.id); });
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectRun(run.id);
      }
    });
    box.appendChild(item);
  }
}

async function refreshRuns() {
  if (state.mode === 'demo') return;
  try {
    const data = await Api.runs();
    const list = Array.isArray(data) ? data : (data && Array.isArray(data.runs) ? data.runs : []);
    state.runs = list.filter((item) => item && typeof item === 'object');
    state.runErrors = 0;
    renderRunList();
    if (!state.currentRunId && state.runs.length) await selectRun(state.runs[0].id);
    else if (state.currentRunId && !state.runs.some((run) => run.id === state.currentRunId) && state.runs.length) {
      await selectRun(state.runs[0].id);
    }
  } catch (error) {
    state.runErrors += 1;
    if (state.runErrors === 2) toast('读取训练运行列表失败：' + error.message, 'error');
  }
}

async function selectRun(id) {
  if (!id) return;
  if (state.mode === 'demo') {
    selectDemoRun(id);
    return;
  }
  state.currentRunId = id;
  resetRunData(false);
  renderRunList();
  updateRunHeader();
  try {
    const detail = await Api.run(id);
    if (state.currentRunId !== id) return;
    state.runDetail = detail || {};
    syncLogTail(detail && detail.log_tail ? detail.log_tail : []);
    applyDetailExtras(detail);
    updateRunHeader();
    updateKpis();
    const eventsResponse = await Api.runEvents(id, 0);
    if (state.currentRunId !== id) return;
    applyEvents(eventsResponse && eventsResponse.events, eventsResponse && eventsResponse.next);
    drawCharts();
  } catch (error) {
    toast('加载运行详情失败：' + error.message, 'error');
  }
}

function selectDemoRun(id) {
  const run = DEMO.runs.find((item) => item.id === id) || DEMO.runs[0];
  if (!run) return;
  state.currentRunId = run.id;
  state.runDetail = DEMO.details[run.id] || null;
  resetRunData(false);
  renderRunList();
  updateRunHeader();
  const detail = state.runDetail;
  syncLogTail(detail && detail.log_tail ? detail.log_tail : []);
  applyDetailExtras(detail);
  const events = DEMO.events[run.id] || [];
  const lastSeq = events.length ? events[events.length - 1].seq : 0;
  applyEvents(events, lastSeq);
  drawCharts();
}

async function refreshCurrentRun() {
  const id = state.currentRunId;
  if (!id || state.mode !== 'live') return;
  try {
    const detail = await Api.run(id);
    if (state.currentRunId !== id) return;
    state.runDetail = detail || state.runDetail;
    syncLogTail(detail && detail.log_tail ? detail.log_tail : []);
    applyDetailExtras(detail);
    updateRunHeader();
    const eventsResponse = await Api.runEvents(id, state.eventCursor);
    if (state.currentRunId !== id) return;
    applyEvents(eventsResponse && eventsResponse.events, eventsResponse && eventsResponse.next);
    state.runErrors = 0;
  } catch (error) {
    state.runErrors += 1;
    if (state.runErrors === 3) toast('读取运行数据失败：' + error.message, 'error');
  }
}

async function stopCurrentRun() {
  const id = state.currentRunId;
  if (!id) return;
  if (state.mode === 'demo') {
    toast('演示数据：后端未连接，无法停止真实训练', 'error');
    return;
  }
  if (els.stopBtn) els.stopBtn.disabled = true;
  try {
    await Api.stopRun(id);
    toast('已发送停止指令', 'ok');
    await refreshRuns();
    await refreshCurrentRun();
  } catch (error) {
    toast('停止训练失败：' + error.message, 'error');
  } finally {
    updateRunHeader();
  }
}

async function submitStartRun(event) {
  event.preventDefault();
  if (state.mode === 'demo') {
    toast('后端未连接：无法启动真实训练（当前为演示数据）', 'error');
    return;
  }
  const form = els.startForm;
  if (!form) return;
  const data = new FormData(form);
  const getValue = (name) => (data.get(name) === null ? '' : String(data.get(name)));
  const payload = {
    name: getValue('name').trim(),
    updates: clampInt(getValue('updates'), 1, 5000, 200),
    envs: clampInt(getValue('envs'), 1, 64, 8),
    rollout_steps: clampInt(getValue('rollout_steps'), 8, 512, 64),
    opponent: getValue('opponent') || 'scripted',
    seed: clampInt(getValue('seed'), 0, 999999, 0),
    eval_every: clampInt(getValue('eval_every'), 1, 1000, 25),
    eval_episodes: clampInt(getValue('eval_episodes'), 1, 128, 16),
  };
  if (!payload.name) delete payload.name;
  if (els.startBtn) {
    els.startBtn.disabled = true;
    els.startBtn.textContent = '启动中…';
  }
  try {
    const response = await Api.startRun(payload);
    const id = response && (response.id || response.run_id);
    toast('训练任务已启动' + (id ? ' · ' + id : ''), 'ok');
    await refreshRuns();
    if (id) await selectRun(id);
  } catch (error) {
    toast('启动训练失败：' + error.message, 'error');
  } finally {
    if (els.startBtn) {
      els.startBtn.disabled = false;
      els.startBtn.textContent = '启动训练';
    }
  }
}

/* ============================================================
 * 图表（canvas 自绘折线 + 坐标轴 + 网格 + 图例 + 悬停提示）
 * ============================================================ */

function niceTicks(minValue, maxValue, targetCount) {
  let min = minValue;
  let max = maxValue;
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 1;
  }
  if (max - min < 1e-9) {
    max = min + 1;
  }
  const rough = (max - min) / Math.max(1, targetCount || 4);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const normalized = rough / magnitude;
  let step;
  if (normalized <= 1) step = 1;
  else if (normalized <= 2) step = 2;
  else if (normalized <= 2.5) step = 2.5;
  else if (normalized <= 5) step = 5;
  else step = 10;
  step *= magnitude;
  const low = Math.floor(min / step) * step;
  const high = Math.ceil(max / step) * step;
  const ticks = [];
  for (let value = low; value <= high + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return { min: low, max: Math.max(high, low + step), step: step, ticks: ticks };
}

function drawLineChart(canvas, tip, config) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const padLeft = 46;
  const padRight = 18;
  const padTop = 14;
  const padBottom = 26;
  const x0 = padLeft;
  const x1 = Math.max(padLeft + 20, width - padRight);
  const y0 = padTop;
  const y1 = Math.max(padTop + 20, height - padBottom);

  const series = (config.series || []).filter((item) => item && Array.isArray(item.points));
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = config.yMin !== undefined ? config.yMin : Infinity;
  let yMax = config.yMax !== undefined ? config.yMax : -Infinity;
  let totalPoints = 0;
  for (const item of series) {
    for (const point of item.points) {
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
      totalPoints += 1;
      xMin = Math.min(xMin, point.x);
      xMax = Math.max(xMax, point.x);
      if (config.yMin === undefined) yMin = Math.min(yMin, point.y);
      if (config.yMax === undefined) yMax = Math.max(yMax, point.y);
    }
  }
  if (!totalPoints) {
    xMin = 0;
    xMax = 1;
    if (config.yMin === undefined) yMin = 0;
    if (config.yMax === undefined) yMax = 1;
  }
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) {
    xMin = 0;
    xMax = 1;
  }
  if (xMax - xMin < 1e-9) {
    xMin = Math.max(0, xMin - 1);
    xMax = xMax + 1;
  }
  const xPad = (xMax - xMin) * 0.03;
  xMin -= xPad;
  xMax += xPad;
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = 0;
    yMax = 1;
  }
  if (config.yMin === undefined) {
    const span = Math.max(1e-6, yMax - yMin);
    yMin -= span * 0.12;
    if (yMax > 0 && yMin < 0 && config.includeZero !== false) yMin = Math.min(0, yMin);
  }
  if (config.yMax === undefined) {
    const span = Math.max(1e-6, yMax - yMin);
    yMax += span * 0.1;
  }
  if (config.xFloor !== undefined && config.xFloor !== null) xMin = Math.max(xMin, config.xFloor);
  const yTicks = niceTicks(yMin, yMax, 4);
  const xTicks = niceTicks(xMin, xMax, 4);

  const mapX = (value) => x0 + ((value - xTicks.min) / (xTicks.max - xTicks.min)) * (x1 - x0);
  const mapY = (value) => y1 - ((value - yTicks.min) / (yTicks.max - yTicks.min)) * (y1 - y0);

  // 绘图区背景
  ctx.fillStyle = 'rgba(4, 10, 18, 0.55)';
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);

  // 网格
  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = '10px Consolas, "Segoe UI", monospace';
  ctx.textBaseline = 'middle';
  for (const value of yTicks.ticks) {
    const y = mapY(value);
    if (y < y0 - 1 || y > y1 + 1) continue;
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.08)';
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(124, 147, 173, 0.95)';
    ctx.textAlign = 'right';
    ctx.fillText(config.yFormat ? config.yFormat(value) : String(value), x0 - 8, y);
  }
  ctx.textBaseline = 'top';
  for (const value of xTicks.ticks) {
    const x = mapX(value);
    if (x < x0 - 1 || x > x1 + 1) continue;
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.06)';
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
    if (value >= xTicks.min && value <= xTicks.max) {
      ctx.fillStyle = 'rgba(124, 147, 173, 0.95)';
      ctx.textAlign = 'center';
      ctx.fillText(config.xFormat ? config.xFormat(value) : String(value), clamp(x, x0 + 10, x1 - 10), y1 + 8);
    }
  }
  ctx.strokeStyle = 'rgba(31, 58, 89, 0.9)';
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.restore();

  if (!totalPoints) {
    ctx.save();
    ctx.fillStyle = 'rgba(93, 115, 144, 0.85)';
    ctx.font = '12px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('暂无数据', (x0 + x1) / 2, (y0 + y1) / 2);
    ctx.restore();
    canvas.__chart = { x0: x0, x1: x1, y0: y0, y1: y1, xMin: xTicks.min, xMax: xTicks.max, series: series, cfg: config };
    return;
  }

  // 参考线
  if (config.refLine !== undefined && config.refLine !== null) {
    const y = mapY(config.refLine);
    if (y >= y0 && y <= y1) {
      ctx.save();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = 'rgba(255, 195, 92, 0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      ctx.setLineDash([]);
      if (config.refLabel) {
        ctx.fillStyle = 'rgba(255, 195, 92, 0.75)';
        ctx.font = '10px Consolas, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(config.refLabel, x0 + 6, y - 3);
      }
      ctx.restore();
    }
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, x1 - x0, y1 - y0);
  ctx.clip();

  for (const item of series) {
    const points = item.points;
    if (!points.length) continue;
    const color = item.color || '#38e1ff';
    // 面积填充
    if (points.length > 1) {
      const gradient = ctx.createLinearGradient(0, y0, 0, y1);
      gradient.addColorStop(0, hexToRgba(color, 0.22));
      gradient.addColorStop(1, hexToRgba(color, 0.01));
      ctx.beginPath();
      ctx.moveTo(mapX(points[0].x), y1);
      for (const point of points) ctx.lineTo(mapX(point.x), mapY(point.y));
      ctx.lineTo(mapX(points[points.length - 1].x), y1);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();
    }
    // 折线
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.shadowColor = hexToRgba(color, 0.75);
    ctx.shadowBlur = 8;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = mapX(point.x);
      const y = mapY(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();

    // 评估点 / 稀疏点标记
    if (points.length <= 80) {
      for (const point of points) {
        ctx.beginPath();
        ctx.arc(mapX(point.x), mapY(point.y), 2.2, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
    // 末端高亮
    const lastPoint = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(mapX(lastPoint.x), mapY(lastPoint.y), 3.4, 0, Math.PI * 2);
    ctx.fillStyle = '#eafcff';
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.restore();

  canvas.__chart = {
    x0: x0, x1: x1, y0: y0, y1: y1,
    xMin: xTicks.min, xMax: xTicks.max,
    series: series, cfg: config,
  };
  if (tip) tip.hidden = true;
}

function hexToRgba(hex, alpha) {
  const text = String(hex || '#38e1ff').replace('#', '');
  const full = text.length === 3 ? text.split('').map((chr) => chr + chr).join('') : text;
  const value = parseInt(full, 16);
  if (!Number.isFinite(value)) return 'rgba(56, 225, 255, ' + alpha + ')';
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
}

function chartSeriesConfigs() {
  const winPoints = state.series.win.length ? state.series.win : state.series.trainWin;
  const winName = state.series.win.length ? '评估胜率' : '训练胜率';
  return {
    win: {
      series: [{ name: winName, color: '#38e1ff', points: winPoints, format: (value) => fmtPct(value) }],
      yMin: 0,
      yMax: 1,
      refLine: 0.5,
      refLabel: '50% 基线',
      yFormat: (value) => Math.round(value * 100) + '%',
      xFormat: (value) => 'u' + Math.round(value),
      xFloor: 0,
      includeZero: true,
    },
    ret: {
      series: [
        { name: '训练回报', color: '#ffc35c', points: state.series.retTrain, format: (value) => fmtNum(value, 3) },
        { name: '评估回报', color: '#39e08b', points: state.series.retEval, format: (value) => fmtNum(value, 3) },
      ],
      yFormat: (value) => fmtNum(value, 2),
      xFormat: (value) => 'u' + Math.round(value),
      xFloor: 0,
      includeZero: true,
    },
  };
}

function renderLegend(host, series) {
  if (!host) return;
  host.innerHTML = '';
  for (const item of series) {
    const span = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.style.background = item.color;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(item.name));
    const value = document.createElement('b');
    const points = item.points || [];
    value.textContent = points.length
      ? (item.format ? item.format(points[points.length - 1].y) : fmtNum(points[points.length - 1].y, 3))
      : '—';
    span.appendChild(value);
    host.appendChild(span);
  }
}

function drawCharts() {
  if (!els.chartWin || !els.chartReturn) return;
  const configs = chartSeriesConfigs();
  drawLineChart(els.chartWin, els.tipWin, configs.win);
  drawLineChart(els.chartReturn, els.tipReturn, configs.ret);
  renderLegend(els.legendWin, configs.win.series);
  renderLegend(els.legendReturn, configs.ret.series);
  state.dirtyCharts = false;
}

function nearestPoint(points, targetX) {
  if (!Array.isArray(points) || !points.length) return null;
  let best = points[0];
  let bestDistance = Math.abs(best.x - targetX);
  for (const point of points) {
    const distance = Math.abs(point.x - targetX);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = point;
    }
  }
  return best;
}

function handleChartHover(canvas, tip, clientX, clientY) {
  if (!canvas || !tip) return;
  const chart = canvas.__chart;
  if (!chart) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = clientX - rect.left;
  if (mouseX < chart.x0 || mouseX > chart.x1) {
    tip.hidden = true;
    return;
  }
  const primary = chart.series.length ? chart.series[0] : null;
  if (!primary || !primary.points.length) {
    tip.hidden = true;
    return;
  }
  const targetX = chart.xMin + ((mouseX - chart.x0) / Math.max(1, chart.x1 - chart.x0)) * (chart.xMax - chart.xMin);
  const anchor = nearestPoint(primary.points, targetX);
  if (!anchor) {
    tip.hidden = true;
    return;
  }
  tip.innerHTML = '';
  const header = document.createElement('div');
  header.textContent = (chart.cfg.xName || 'update') + ' ' + Math.round(anchor.x);
  header.style.color = '#8ce6ff';
  header.style.marginBottom = '2px';
  tip.appendChild(header);
  for (const item of chart.series) {
    const point = nearestPoint(item.points, anchor.x);
    if (!point) continue;
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '6px';
    const swatch = document.createElement('i');
    swatch.style.display = 'inline-block';
    swatch.style.width = '10px';
    swatch.style.height = '3px';
    swatch.style.borderRadius = '2px';
    swatch.style.background = item.color;
    const label = document.createElement('span');
    label.textContent = item.name + '：';
    const value = document.createElement('b');
    value.textContent = item.format ? item.format(point.y) : fmtNum(point.y, 3);
    value.style.color = '#eafcff';
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(value);
    tip.appendChild(row);
  }
  tip.hidden = false;
  tip.style.left = clamp(mouseX, 60, rect.width - 60) + 'px';
  tip.style.top = clamp(clientY - rect.top - 8, 28, rect.height) + 'px';
}

/* ============================================================
 * Checkpoint / 回放库
 * ============================================================ */

function renderCheckpoints(list) {
  const select = els.ckptSelect;
  if (!select) return;
  const items = Array.isArray(list) ? list : [];
  const previous = select.value;
  select.innerHTML = '';
  const randomOption = document.createElement('option');
  randomOption.value = 'random';
  randomOption.textContent = items.length
    ? 'random · 随机策略（无 checkpoint）'
    : '暂无 checkpoint · 使用随机策略';
  select.appendChild(randomOption);
  for (const item of items) {
    if (!item) continue;
    const path = pickVal(item, 'path', 'file') || item.name || '';
    if (!path) continue;
    const option = document.createElement('option');
    option.value = path;
    const update = numOrNull(item.update);
    const winRate = item.eval && numOrNull(item.eval.win_rate);
    option.textContent = (item.name || basename(path))
      + (update === null ? '' : ' · u' + update)
      + (winRate === null ? '' : ' · win ' + fmtPct(winRate));
    select.appendChild(option);
  }
  if (previous) {
    const matched = Array.prototype.some.call(select.options, (option) => option.value === previous);
    if (matched) select.value = previous;
  }
}

async function loadCheckpoints() {
  if (state.mode === 'demo') {
    renderCheckpoints(DEMO.checkpoints);
    return;
  }
  try {
    const data = await Api.checkpoints();
    state.checkpoints = Array.isArray(data) ? data : (data && Array.isArray(data.checkpoints) ? data.checkpoints : []);
    renderCheckpoints(state.checkpoints);
  } catch (error) {
    // 加载失败保持原列表，连接状态由顶部指示灯体现
  }
}

function renderReplays() {
  const body = els.replayBody;
  if (!body) return;
  body.innerHTML = '';
  const items = state.replays || [];
  if (els.replayCount) els.replayCount.textContent = String(items.length);
  if (!items.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 9;
    cell.className = 'empty';
    cell.textContent = state.mode === 'demo' ? '暂无演示回放' : '尚无回放；运行一次推理后会自动入库';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const item of items) {
    const row = document.createElement('tr');
    row.className = 'data-row';
    row.addEventListener('click', () => openReplay(item.id));

    const cellName = document.createElement('td');
    cellName.className = 'cell-name';
    cellName.textContent = item.name || basename(item.id);
    cellName.title = item.id || item.name || '';

    const cellCheckpoint = document.createElement('td');
    cellCheckpoint.className = 'cell-path';
    cellCheckpoint.textContent = item.checkpoint ? basename(item.checkpoint) : '—';
    cellCheckpoint.title = item.checkpoint || '';

    const cellOpponent = document.createElement('td');
    cellOpponent.textContent = item.opponent || '—';

    const cellResult = document.createElement('td');
    const badge = document.createElement('span');
    const meta = resultMeta(item.result);
    badge.className = 'result-badge ' + meta.cls;
    badge.textContent = meta.label;
    cellResult.appendChild(badge);

    const cellReturn = document.createElement('td');
    cellReturn.textContent = item.ret === null || item.ret === undefined ? '—' : fmtNum(item.ret, 3);

    const cellEpisodes = document.createElement('td');
    cellEpisodes.textContent = item.episodes === null || item.episodes === undefined ? '—' : String(item.episodes);

    const cellFrames = document.createElement('td');
    cellFrames.textContent = item.frames === null || item.frames === undefined ? '—' : String(item.frames);

    const cellTime = document.createElement('td');
    cellTime.textContent = fmtShortTime(item.created_at) || '—';

    const cellActions = document.createElement('td');
    const actions = document.createElement('div');
    actions.className = 'row-actions';
    const playButton = document.createElement('button');
    playButton.type = 'button';
    playButton.className = 'btn btn-sm';
    playButton.textContent = '播放';
    playButton.addEventListener('click', (event) => {
      event.stopPropagation();
      openReplay(item.id);
    });
    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'btn btn-danger btn-sm';
    deleteButton.textContent = '删除';
    deleteButton.addEventListener('click', (event) => {
      event.stopPropagation();
      removeReplay(item.id);
    });
    actions.appendChild(playButton);
    actions.appendChild(deleteButton);
    cellActions.appendChild(actions);

    row.appendChild(cellName);
    row.appendChild(cellCheckpoint);
    row.appendChild(cellOpponent);
    row.appendChild(cellResult);
    row.appendChild(cellReturn);
    row.appendChild(cellEpisodes);
    row.appendChild(cellFrames);
    row.appendChild(cellTime);
    row.appendChild(cellActions);
    body.appendChild(row);
  }
}

async function loadReplays() {
  if (state.mode === 'demo') {
    state.replays = DEMO.replays.slice();
    renderReplays();
    return;
  }
  try {
    const data = await Api.replays();
    const list = Array.isArray(data) ? data : (data && Array.isArray(data.replays) ? data.replays : []);
    state.replays = list.map(normalizeReplay);
    renderReplays();
  } catch (error) {
    // 保留上一次列表
  }
}

async function openReplay(id) {
  try {
    let payload = null;
    if (state.mode === 'demo') {
      payload = demoReplayPayload(id);
      if (!payload) throw new Error('演示回放不存在');
    } else {
      payload = await Api.replay(id);
    }
    const rawFrames = payload && Array.isArray(payload.frames) ? payload.frames : [];
    const frames = rawFrames.map(normalizeFrame).filter((frame) => frame.units.length || frame.events.length);
    if (!frames.length) {
      toast('该回放没有可播放的帧数据', 'error');
      return;
    }
    const meta = (payload && payload.meta) || {};
    resetSandbox();
    const sandbox = state.sandbox;
    sandbox.source = 'replay';
    sandbox.frames = frames;
    sandbox.meta = meta;
    sandbox.map = normalizeMap(meta.map);
    sandbox.total = frames.length;
    sandbox.playback = 0;
    sandbox.playing = true;
    sandbox.follow = false;
    sandbox.streaming = false;
    sandbox.status = 'replay';
    sandbox.checkpoint = meta.checkpoint || null;
    sandbox.opponent = meta.opponent || null;
    sandbox.seed = numOrNull(meta.seed);
    sandbox.episodes = numOrNull(meta.episodes) === null ? 1 : meta.episodes;
    sandbox.results = [{ episode: num(meta.episode, 0), result: meta.result || null, ret: numOrNull(meta.return) }];
    sandbox.eventLog = collectFrameEvents(frames);
    if (els.sandboxDemoChip) els.sandboxDemoChip.hidden = state.mode !== 'demo';
    if (meta.map_id) {
      if (!state.maps.ready) await loadMaps();
      if (state.maps.available) {
        await selectMap(meta.map_id, { silent: true });
      } else {
        syncMapSelect();
        syncMapStatus();
      }
    } else {
      syncMapSelect();
      syncMapStatus();
    }
    switchTab('sandbox');
    updatePlayerUi(true);
    toast('已加载回放 · ' + frames.length + ' 帧', 'ok');
  } catch (error) {
    toast('加载回放失败：' + error.message, 'error');
  }
}

function collectFrameEvents(frames) {
  const log = [];
  for (const frame of frames) {
    for (const event of frame.events) {
      if (event.type === 'shot') {
        log.push({ kind: 'shot', time: frame.time, text: '命中 #' + event.source + ' → #' + event.target + '  -' + Math.round(event.amount) + ' HP' });
      } else if (event.type === 'kill') {
        log.push({ kind: 'kill', time: frame.time, text: '击毁 #' + event.target + '（#' + event.source + '）' });
      }
    }
  }
  return log.slice(-300);
}

async function removeReplay(id) {
  if (state.mode === 'demo') {
    DEMO.replays = DEMO.replays.filter((item) => item.id !== id);
    state.replays = DEMO.replays.slice();
    renderReplays();
    toast('已移除演示回放', 'ok');
    return;
  }
  if (!window.confirm('确认删除该回放文件？此操作不可撤销。')) return;
  try {
    await Api.deleteReplay(id);
    toast('已删除回放', 'ok');
    await loadReplays();
  } catch (error) {
    toast('删除回放失败：' + error.message, 'error');
  }
}

/* ============================================================
 * 对局沙盘：推理 / 帧流 / 播放 / 渲染 / 录制
 * ============================================================ */

function applyInferStart(meta) {
  const sandbox = state.sandbox;
  if (!meta || typeof meta !== 'object') return;
  sandbox.meta = meta;
  sandbox.map = normalizeMap(meta.map);
  sandbox.checkpoint = meta.checkpoint || sandbox.checkpoint;
  sandbox.opponent = meta.opponent || sandbox.opponent;
  sandbox.seed = numOrNull(meta.seed) !== null ? meta.seed : sandbox.seed;
  sandbox.episodes = numOrNull(meta.episodes) !== null ? meta.episodes : sandbox.episodes;
  const mapId = meta.map_id === undefined || meta.map_id === null ? null : String(meta.map_id);
  if (mapId && mapId !== sandbox.mapId) {
    if (state.maps.ready) {
      if (state.maps.available) void selectMap(mapId, { silent: true });
    } else {
      void loadMaps().then(() => {
        if (state.maps.available && mapId !== state.sandbox.mapId) {
          void selectMap(mapId, { silent: true });
        }
      });
    }
  }
}

function applyInferBatch(batch, next) {
  if (!Array.isArray(batch) || !batch.length) return 0;
  const sandbox = state.sandbox;
  const previousLength = sandbox.frames.length;
  let added = 0;
  for (const item of batch) {
    if (!item || typeof item !== 'object') continue;
    const type = item.type;
    if (type === 'start') {
      applyInferStart(item.meta);
      continue;
    }
    if (type === 'frame') {
      const frame = normalizeFrame(item);
      sandbox.frames.push(frame);
      pushFrameEvents(frame);
      added += 1;
      continue;
    }
    if (type === 'episode_end') {
      sandbox.results.push(normalizeResult(item));
      continue;
    }
    if (type === 'done') {
      sandbox.inferDone = true;
      if (Array.isArray(item.episodes)) sandbox.results = item.episodes.map(normalizeResult);
      continue;
    }
    if (Array.isArray(item.units)) {
      const frame = normalizeFrame(item);
      sandbox.frames.push(frame);
      pushFrameEvents(frame);
      added += 1;
    }
  }
  if (next !== undefined && next !== null) sandbox.fetchCursor = Math.max(sandbox.fetchCursor, num(next, sandbox.fetchCursor + batch.length));
  else sandbox.fetchCursor += batch.length;
  if (added) {
    sandbox.total = Math.max(sandbox.total, sandbox.frames.length);
    if (sandbox.follow && sandbox.playing && sandbox.playback >= previousLength - 1.5) {
      sandbox.playback = Math.max(0, previousLength - 4);
    }
    updatePlayerUi(true);
  }
  return added;
}

function pushFrameEvents(frame) {
  const sandbox = state.sandbox;
  for (const event of frame.events) {
    if (event.type === 'shot') {
      sandbox.eventLog.push({ kind: 'shot', time: frame.time, text: '命中 #' + event.source + ' → #' + event.target + '  -' + Math.round(event.amount) + ' HP' });
    } else if (event.type === 'kill') {
      sandbox.eventLog.push({ kind: 'kill', time: frame.time, text: '击毁 #' + event.target + '（#' + event.source + '）' });
    }
  }
  if (sandbox.eventLog.length > 300) sandbox.eventLog.splice(0, sandbox.eventLog.length - 300);
}

async function submitInfer(event) {
  event.preventDefault();
  if (state.mode === 'demo') {
    startDemoInference();
    return;
  }
  const checkpoint = els.ckptSelect ? els.ckptSelect.value : '';
  if (!checkpoint) {
    toast('请选择 checkpoint（或选择随机策略）', 'error');
    return;
  }
  const payload = {
    checkpoint: checkpoint,
    episodes: clampInt(els.episodesInput && els.episodesInput.value, 1, 50, 1),
    opponent: (els.inferOpponent && els.inferOpponent.value) || 'scripted',
    seed: clampInt(els.inferSeed && els.inferSeed.value, 0, 999999, 11),
    mode: (els.inferMode && els.inferMode.value) || 'greedy',
    max_decisions: 400,
    map_id: state.sandbox.mapId || '',
  };
  if (els.inferBtn) {
    els.inferBtn.disabled = true;
    els.inferBtn.textContent = '启动中…';
  }
  try {
    const response = await Api.startInfer(payload);
    const id = response && (response.id || response.job_id);
    if (!id) throw new Error('后端未返回推理任务 id');
    resetSandbox({ keepMap: true });
    const sandbox = state.sandbox;
    sandbox.source = 'infer';
    sandbox.inferId = id;
    sandbox.status = 'running';
    sandbox.streaming = true;
    sandbox.playing = true;
    sandbox.follow = true;
    sandbox.checkpoint = checkpoint;
    sandbox.opponent = payload.opponent;
    sandbox.seed = payload.seed;
    sandbox.episodes = payload.episodes;
    if (els.sandboxDemoChip) els.sandboxDemoChip.hidden = true;
    if (els.inferHint) els.inferHint.textContent = '推理任务已启动（' + id + '），正在接收实时帧…';
    switchTab('sandbox');
    startInferPolling();
    toast('推理任务已启动 · ' + id, 'ok');
  } catch (error) {
    toast('启动推理失败：' + error.message, 'error');
  } finally {
    if (els.inferBtn) {
      els.inferBtn.disabled = false;
      els.inferBtn.textContent = '开始推理';
    }
  }
}

function startDemoInference() {
  initDemoData();
  const payload = demoReplayPayload('demo-replay-a');
  if (!payload) return;
  resetSandbox();
  const sandbox = state.sandbox;
  sandbox.source = 'demo';
  sandbox.frames = payload.frames.map((frame) => frame);
  sandbox.meta = payload.meta;
  sandbox.map = normalizeMap(payload.meta.map);
  sandbox.total = sandbox.frames.length;
  sandbox.playback = 0;
  sandbox.playing = true;
  sandbox.follow = false;
  sandbox.streaming = false;
  sandbox.status = 'demo';
  sandbox.checkpoint = 'runs/demo/checkpoint.pt';
  sandbox.opponent = 'scripted';
  sandbox.seed = 11;
  sandbox.episodes = 1;
  sandbox.results = [{ episode: 0, result: payload.meta.result, ret: payload.meta.return }];
  sandbox.eventLog = payload.eventLog ? payload.eventLog.slice(-300) : collectFrameEvents(sandbox.frames);
  if (els.sandboxDemoChip) els.sandboxDemoChip.hidden = false;
  if (els.inferHint) els.inferHint.textContent = '演示模式：正在播放内置演示对局（演示数据，非真实推理）。';
  switchTab('sandbox');
  updatePlayerUi(true);
  toast('演示模式：播放内置演示对局', 'ok');
}

function startInferPolling() {
  stopInferPolling();
  state.inferTimer = setInterval(pollInfer, 700);
  pollInfer();
}

function stopInferPolling() {
  if (state.inferTimer) {
    clearInterval(state.inferTimer);
    state.inferTimer = null;
  }
}

async function pollInfer() {
  const sandbox = state.sandbox;
  if (state.mode !== 'live' || !sandbox.inferId || sandbox.polling) return;
  sandbox.polling = true;
  try {
    let meta = null;
    try {
      meta = await Api.infer(sandbox.inferId);
    } catch (error) {
      // 帧接口可能仍可用，继续尝试
    }
    if (meta && typeof meta === 'object') {
      if (meta.status) sandbox.status = meta.status;
      if (numOrNull(meta.frames_total) !== null) sandbox.total = Math.max(sandbox.total, meta.frames_total);
      if (numOrNull(meta.episodes) !== null && numOrNull(meta.episodes) > 0) sandbox.episodes = meta.episodes;
      if (Array.isArray(meta.results) && meta.results.length) sandbox.results = meta.results.map(normalizeResult);
      if (meta.done) sandbox.inferDone = true;
      if (meta.checkpoint) sandbox.checkpoint = meta.checkpoint;
      if (meta.opponent) sandbox.opponent = meta.opponent;
      if (numOrNull(meta.seed) !== null) sandbox.seed = meta.seed;
    }
    const framesResponse = await Api.inferFrames(sandbox.inferId, sandbox.fetchCursor);
    if (framesResponse) {
      applyInferBatch(framesResponse.frames, framesResponse.next);
      if (framesResponse.status) sandbox.status = framesResponse.status;
    }
    sandbox.pollErrors = 0;
    updatePlayerUi(true);
    const statusText = String(sandbox.status || '').toLowerCase();
    if (sandbox.inferDone || ['succeeded', 'failed', 'stopped', 'finished', 'completed', 'done', 'error'].indexOf(statusText) >= 0) {
      await handleInferFinished();
    }
  } catch (error) {
    sandbox.pollErrors += 1;
    if (sandbox.pollErrors === 1) toast('推理帧轮询失败：' + error.message, 'error');
    if (sandbox.pollErrors >= 6) {
      stopInferPolling();
      sandbox.streaming = false;
      sandbox.status = 'failed';
      if (els.inferHint) els.inferHint.textContent = '推理帧轮询中断，请检查后端日志。';
      updatePlayerUi(true);
    }
  } finally {
    sandbox.polling = false;
  }
}

async function handleInferFinished() {
  const sandbox = state.sandbox;
  stopInferPolling();
  sandbox.streaming = false;
  sandbox.follow = false;
  const statusText = String(sandbox.status || '').toLowerCase();
  const failed = ['failed', 'error', 'stopped'].indexOf(statusText) >= 0;
  if (failed) {
    sandbox.status = 'failed';
    toast('推理任务失败，请检查 checkpoint 或后端日志', 'error');
    if (els.inferHint) els.inferHint.textContent = '推理失败，请检查 checkpoint 与后端日志。';
  } else {
    sandbox.status = 'done';
    const results = sandbox.results || [];
    const wins = results.filter((item) => resultMeta(item.result).cls === 'win').length;
    toast('推理完成 · ' + results.length + ' 局 · 胜利 ' + wins + ' 局，回放已入库', 'ok');
    if (els.inferHint) els.inferHint.textContent = '推理完成：' + results.length + ' 局，回放已写入回放库。';
    loadReplays();
  }
  updatePlayerUi(true);
}

function playerStatus(sandbox) {
  if (sandbox.source === 'replay') return { label: '回放', cls: 'done' };
  if (sandbox.source === 'demo') return { label: '演示', cls: 'warn' };
  if (sandbox.source === 'infer') {
    if (sandbox.status === 'failed') return { label: '失败', cls: 'bad' };
    if (sandbox.status === 'done') return { label: '已完成', cls: 'done' };
    if (sandbox.streaming || sandbox.status === 'running') return { label: '推理中', cls: 'running' };
    return { label: '推理', cls: 'warn' };
  }
  return { label: '空闲', cls: 'idle' };
}

function resetSandbox(options) {
  cancelRecording();
  const previous = state.sandbox;
  state.sandbox = createSandboxState();
  if (options && options.keepMap && previous.mapId && previous.mapDetail) {
    state.sandbox.mapId = previous.mapId;
    state.sandbox.mapDetail = previous.mapDetail;
    state.sandbox.terrainEntry = previous.terrainEntry;
    state.sandbox.terrain = previous.terrain;
    state.sandbox.viewMode = previous.viewMode;
    state.sandbox.terrainLoading = false;
  }
  if (els.sandboxDemoChip) els.sandboxDemoChip.hidden = true;
  syncMapSelect();
  syncMapStatus();
  updatePlayerUi(true);
  drawSandbox();
}

function loadDemoBattle() {
  initDemoData();
  const payload = demoReplayPayload('demo-replay-a');
  if (!payload) return;
  const sandbox = state.sandbox;
  sandbox.source = 'demo';
  sandbox.frames = payload.frames.map((frame) => frame);
  sandbox.meta = payload.meta;
  sandbox.map = normalizeMap(payload.meta.map);
  sandbox.total = sandbox.frames.length;
  sandbox.playing = false;
  sandbox.follow = false;
  sandbox.streaming = false;
  sandbox.status = 'demo';
  sandbox.checkpoint = 'runs/demo/checkpoint.pt';
  sandbox.opponent = 'scripted';
  sandbox.seed = 11;
  sandbox.episodes = 1;
  sandbox.results = [{ episode: 0, result: payload.meta.result, ret: payload.meta.return }];
  sandbox.eventLog = payload.eventLog ? payload.eventLog.slice(-300) : collectFrameEvents(sandbox.frames);
  sandbox.playback = pickDemoFrameIndex(sandbox.frames);
  if (els.sandboxDemoChip) els.sandboxDemoChip.hidden = false;
  updatePlayerUi(true);
  drawSandbox();
}

function ensureSandboxContent() {
  const sandbox = state.sandbox;
  if (sandbox.frames.length) return;
  if (state.mode === 'demo') loadDemoBattle();
}

function pickDemoFrameIndex(frames) {
  if (!frames.length) return 0;
  let best = Math.round((frames.length - 1) * 0.6);
  let bestScore = -Infinity;
  const start = Math.floor(frames.length * 0.35);
  for (let index = start; index < frames.length; index += 1) {
    const events = frames[index].events || [];
    const alive = frames[index].units.length;
    let score = alive * 0.6;
    for (const event of events) {
      if (event.type === 'kill') score += 4;
      else if (event.type === 'shot') score += 2;
    }
    if (score > bestScore) {
      bestScore = score;
      best = index;
    }
  }
  return clamp(best, 0, frames.length - 1);
}

function currentFrame() {
  const frames = state.sandbox.frames;
  if (!frames.length) return null;
  return frames[clamp(Math.round(state.sandbox.playback), 0, frames.length - 1)];
}

function updatePlayerUi(force) {
  const sandbox = state.sandbox;
  if (!force) {
    const now = performance.now();
    if (now - sandbox.lastUiAt < 90) return;
    sandbox.lastUiAt = now;
  }
  const frames = sandbox.frames;
  const count = frames.length;
  const maxIndex = Math.max(0, count - 1);
  const index = count ? clamp(Math.round(sandbox.playback), 0, maxIndex) : 0;
  const frame = count ? frames[index] : null;
  const totalFrames = Math.max(sandbox.total || 0, count);

  if (els.seek) {
    els.seek.max = String(maxIndex);
    els.seek.value = String(Math.min(index, maxIndex));
  }
  if (els.frameLabel) {
    els.frameLabel.textContent = count ? '帧 ' + (index + 1) + ' / ' + (totalFrames || count) : '帧 — / —';
  }
  if (els.frameSub) {
    if (!frame) {
      els.frameSub.textContent = sandbox.status === 'running' || sandbox.streaming ? '等待首帧…' : '—';
    } else {
      const result = frame.result ? resultMeta(frame.result) : null;
      els.frameSub.textContent = 't=' + fmtNum(frame.time, 1) + 's · 第 ' + (frame.episode + 1)
        + '/' + Math.max(1, sandbox.episodes || 1) + ' 局' + (result ? ' · ' + result.label : '');
    }
  }
  if (els.playBtn) {
    const label = sandbox.playing ? '暂停' : '播放';
    if (els.playBtn.textContent !== label) els.playBtn.textContent = label;
  }
  if (els.inferStatus) {
    const status = playerStatus(sandbox);
    els.inferStatus.className = 'status-pill ' + status.cls;
    els.inferStatus.textContent = status.label;
  }

  const stats = frame ? frame.stats : null;
  const ownAlive = stats && stats.ownAlive !== null ? stats.ownAlive : (frame ? frame.units.filter((unit) => unit.team === 'own').length : null);
  const enemyAlive = stats && stats.enemyAlive !== null ? stats.enemyAlive : (frame ? frame.units.filter((unit) => unit.team === 'enemy').length : null);
  if (els.scoreOwn) els.scoreOwn.textContent = ownAlive === null ? '—' : String(ownAlive);
  if (els.scoreEnemy) els.scoreEnemy.textContent = enemyAlive === null ? '—' : String(enemyAlive);
  const ownFraction = stats ? clamp(stats.own, 0, 1) : 0;
  const enemyFraction = stats ? clamp(stats.enemy, 0, 1) : 0;
  if (els.hpOwn) els.hpOwn.style.width = (ownFraction * 100).toFixed(1) + '%';
  if (els.hpOwnVal) els.hpOwnVal.textContent = Math.round(ownFraction * 100) + '%';
  if (els.hpEnemy) els.hpEnemy.style.width = (enemyFraction * 100).toFixed(1) + '%';
  if (els.hpEnemyVal) els.hpEnemyVal.textContent = Math.round(enemyFraction * 100) + '%';

  if (els.sbCkpt) els.sbCkpt.textContent = sandbox.checkpoint ? basename(sandbox.checkpoint) : '—';
  if (els.sbOpponent) {
    els.sbOpponent.textContent = sandbox.opponent
      ? sandbox.opponent + (sandbox.seed === null || sandbox.seed === undefined ? '' : ' / seed ' + sandbox.seed)
      : '—';
  }
  if (els.sbEpisode) {
    const done = sandbox.results.length ? sandbox.results.length : (frame ? frame.episode + 1 : 0);
    els.sbEpisode.textContent = sandbox.episodes ? Math.min(done, sandbox.episodes) + ' / ' + sandbox.episodes + ' 局' : '—';
  }
  if (els.sbFrames) els.sbFrames.textContent = count ? count + ' / ' + (totalFrames || count) : '—';
  renderEventFeed();
}

function renderEventFeed() {
  const box = els.eventFeed;
  if (!box) return;
  const sandbox = state.sandbox;
  const log = sandbox.eventLog || [];
  if (sandbox.feedCount === log.length) return;
  sandbox.feedCount = log.length;
  box.innerHTML = '';
  const items = log.slice(-14);
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = '暂无事件';
    box.appendChild(empty);
    return;
  }
  for (const item of items) {
    const line = document.createElement('div');
    line.className = 'ev-line ' + (item.kind || 'info');
    const time = document.createElement('span');
    time.className = 'ev-ts';
    time.textContent = 't=' + fmtNum(item.time, 2) + 's';
    line.appendChild(time);
    line.appendChild(document.createTextNode(item.text || ''));
    box.appendChild(line);
  }
  box.scrollTop = box.scrollHeight;
}

/* ---------------- 沙盘渲染 ---------------- */

const SANDBOX_COLORS = {
  own: '#4da3ff',
  enemy: '#ff5f6d',
  grid: 'rgba(56, 225, 255, 0.07)',
  gridStrong: 'rgba(56, 225, 255, 0.15)',
};

/* ============================================================
 * 实机地图：目录 / 地形预渲染 / 相机 / 单位贴图
 * ============================================================ */

const UNIT_TILE_WIDTH = 16;
const UNIT_TILE_HEIGHT = 30;
const TURRET_TILE_WIDTH = 10;
const TURRET_TILE_HEIGHT = 40;

const spriteCache = { tank: null, turret: null, pending: null, failed: false };
const unitAngleCache = new Map();
const terrainCache = new Map();

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('图片加载失败：' + url));
    image.src = url;
  });
}

async function ensureUnitSprites() {
  if (spriteCache.tank && spriteCache.turret) return true;
  if (spriteCache.pending) return spriteCache.pending;
  spriteCache.pending = (async () => {
    const units = state.maps.units || {};
    const tankRel = units.tank || 'units/tanks/tank.png';
    const turretRel = units.tank_turret || 'units/tanks/tank_turret.png';
    try {
      const loaded = await Promise.all([loadImage(Api.assetUrl(tankRel)), loadImage(Api.assetUrl(turretRel))]);
      spriteCache.tank = loaded[0];
      spriteCache.turret = loaded[1];
      spriteCache.failed = false;
      return true;
    } catch (error) {
      spriteCache.failed = true;
      return false;
    } finally {
      spriteCache.pending = null;
    }
  })();
  return spriteCache.pending;
}

async function loadMaps(force) {
  if (state.maps.ready && !force) return;
  state.maps.attempts += 1;
  if (!force && state.maps.attempts > 3) return;
  try {
    const payload = await Api.maps();
    const list = payload && Array.isArray(payload.maps) ? payload.maps : [];
    state.maps.available = !!(payload && payload.available);
    state.maps.list = list.filter((item) => item && !item.broken);
    state.maps.units = (payload && payload.units) || {};
    state.maps.error = null;
  } catch (error) {
    state.maps.available = false;
    state.maps.list = [];
    state.maps.error = error.message || String(error);
  }
  state.maps.ready = true;
  renderMapOptions();
  return;
}

function renderMapOptions() {
  const select = els.mapSelect;
  if (!select) return;
  const current = state.sandbox.mapId || '';
  while (select.options.length > 1) select.remove(1);
  if (state.maps.available) {
    const folderRank = { skirmish: 0, normal: 1, survival: 2, challenge: 3, testing: 4, menu_background: 5 };
    const sorted = state.maps.list.slice().sort((left, right) => {
      const rankLeft = folderRank[left.folder] === undefined ? 9 : folderRank[left.folder];
      const rankRight = folderRank[right.folder] === undefined ? 9 : folderRank[right.folder];
      if (rankLeft !== rankRight) return rankLeft - rankRight;
      const playersLeft = num(left.players, 0);
      const playersRight = num(right.players, 0);
      if (playersLeft !== playersRight) return playersRight - playersLeft;
      return String(left.name || '').localeCompare(String(right.name || ''));
    });
    for (const item of sorted) {
      const option = document.createElement('option');
      option.value = item.id;
      const people = item.players ? item.players + 'P · ' : '';
      option.textContent = item.name + ' · ' + people + (item.folder || 'map');
      option.title = (item.relative || '') + ' · ' + item.width_px + '×' + item.height_px;
      select.appendChild(option);
    }
  }
  const exists = [...select.options].some((option) => option.value === current);
  select.value = exists ? current : '';
  syncMapStatus();
}

function setMapStatus(text, tone) {
  if (!els.mapStatus) return;
  els.mapStatus.textContent = text;
  els.mapStatus.className = 'map-status' + (tone ? ' ' + tone : '');
}

function syncMapStatus() {
  const sandbox = state.sandbox;
  if (state.mode === 'demo') {
    setMapStatus('演示模式 · 抽象沙盘');
    return;
  }
  if (!state.maps.available) {
    setMapStatus(
      state.maps.ready
        ? '未安装游戏资源缓存：请运行 web/scripts/prepare_assets.ps1'
        : '正在读取地图列表…',
      state.maps.ready ? 'warn' : ''
    );
    return;
  }
  if (sandbox.terrainLoading) {
    setMapStatus('地图加载中…');
    return;
  }
  if (sandbox.terrainError) {
    setMapStatus('地图加载失败，已回退抽象渲染 · ' + sandbox.terrainError, 'bad');
    return;
  }
  if (sandbox.mapDetail) {
    setMapStatus('实机地图渲染 · 地形仅视觉 · ' + sandbox.mapDetail.name, 'ok');
    return;
  }
  setMapStatus('竞技场模式 · 抽象渲染');
}

function mapCrop(detail) {
  if (detail.crop && detail.crop.width && detail.crop.height) {
    return {
      x: num(detail.crop.x, 0),
      y: num(detail.crop.y, 0),
      width: num(detail.crop.width, 1200),
      height: num(detail.crop.height, 1200),
    };
  }
  const tileW = num(detail.tile_width, 20);
  const tileH = num(detail.tile_height, 20);
  return {
    x: 0,
    y: 0,
    width: Math.max(1, num(detail.width_px, num(detail.width, 1000) * tileW)),
    height: Math.max(1, num(detail.height_px, num(detail.height, 1000) * tileH)),
  };
}

function engineToMapPixel(x, y, detail) {
  const crop = mapCrop(detail);
  return {
    x: crop.x + (clamp(num(x, 0), 0, 1000) / 1000) * crop.width,
    y: crop.y + (clamp(num(y, 0), 0, 1000) / 1000) * crop.height,
  };
}

function mapViewRegion(sandbox, detail) {
  const crop = mapCrop(detail);
  if (sandbox.viewMode === 'full') {
    return {
      x: 0,
      y: 0,
      width: Math.max(1, num(detail.width_px, crop.width)),
      height: Math.max(1, num(detail.height_px, crop.height)),
    };
  }
  return { x: crop.x, y: crop.y, width: crop.width, height: crop.height };
}

function mapCamera(width, height, region, padding) {
  const pad = num(padding, 0);
  const rw = Math.max(1, region.width);
  const rh = Math.max(1, region.height);
  const scale = Math.max(0.0001, Math.min((width - pad * 2) / rw, (height - pad * 2) / rh));
  return {
    region: { x: region.x, y: region.y, width: rw, height: rh },
    scale: scale,
    ox: (width - rw * scale) / 2 - region.x * scale,
    oy: (height - rh * scale) / 2 - region.y * scale,
    width: width,
    height: height,
  };
}

function findTileset(tilesets, gid) {
  for (const set of tilesets) {
    if (gid >= set.firstgid) return set;
  }
  return null;
}

function renderTerrain(detail, region, scale, images) {
  const tileW = num(detail.tile_width, 20);
  const tileH = num(detail.tile_height, 20);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.ceil(region.width * scale));
  canvas.height = Math.max(1, Math.ceil(region.height * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  ctx.imageSmoothingEnabled = scale < 1;
  ctx.fillStyle = '#0a141f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const tilesets = (detail.tilesets || []).slice().sort((left, right) => right.firstgid - left.firstgid);
  const columns = Math.max(1, num(detail.width, 1));
  const rows = Math.max(1, num(detail.height, 1));
  const tileX0 = Math.max(0, Math.floor(region.x / tileW));
  const tileY0 = Math.max(0, Math.floor(region.y / tileH));
  const tileX1 = Math.min(columns - 1, Math.ceil((region.x + region.width) / tileW));
  const tileY1 = Math.min(rows - 1, Math.ceil((region.y + region.height) / tileH));

  for (const layer of detail.layers || []) {
    const data = layer && layer.data;
    if (!Array.isArray(data)) continue;
    for (let ty = tileY0; ty <= tileY1; ty += 1) {
      const rowOffset = ty * columns;
      for (let tx = tileX0; tx <= tileX1; tx += 1) {
        const raw = data[rowOffset + tx];
        if (!raw) continue;
        const gid = raw & 0x1fffffff;
        if (!gid) continue;
        const set = findTileset(tilesets, gid);
        if (!set) continue;
        const image = images.get(set.firstgid);
        if (!image) continue;
        const local = gid - set.firstgid;
        const col = local % Math.max(1, set.columns);
        const row = Math.floor(local / Math.max(1, set.columns));
        if (row >= Math.max(1, set.rows) || col >= Math.max(1, set.columns)) continue;
        const sw = num(set.tile_width, tileW);
        const sh = num(set.tile_height, tileH);
        const dx = (tx * tileW - region.x) * scale;
        const dy = (ty * tileH - region.y) * scale;
        ctx.drawImage(image, col * sw, row * sh, sw, sh, dx, dy, tileW * scale, tileH * scale);
      }
    }
  }
  return canvas;
}

async function buildTerrainEntry(detail) {
  const cacheKey = detail.id || String(detail.relative || detail.name);
  const cached = terrainCache.get(cacheKey);
  const images = new Map();
  await Promise.all(
    (detail.tilesets || []).map(async (set) => {
      if (!set || !set.image) return;
      try {
        images.set(set.firstgid, await loadImage(Api.assetUrl(set.image)));
      } catch (error) {
        // 单个 tileset 失败不阻塞整图渲染，缺失区域保留底色
      }
    })
  );
  const crop = mapCrop(detail);
  const entry = cached || {};
  entry.detail = detail;
  entry.images = images;
  if (!entry.crop) entry.crop = renderTerrain(detail, crop, 1, images);
  return entry;
}

function ensureFullTerrain(entry, detail) {
  if (entry.full) return entry.full;
  const width = Math.max(1, num(detail.width_px, 1000));
  const height = Math.max(1, num(detail.height_px, 1000));
  const scale = Math.min(1, 2048 / Math.max(width, height));
  entry.full = renderTerrain(detail, { x: 0, y: 0, width: width, height: height }, scale, entry.images);
  entry.fullScale = scale;
  return entry.full;
}

async function selectMap(mapId, options) {
  const sandbox = state.sandbox;
  const opts = options || {};
  const nextId = mapId || '';
  if (!nextId) {
    sandbox.mapId = '';
    sandbox.mapDetail = null;
    sandbox.terrain = null;
    sandbox.terrainEntry = null;
    sandbox.terrainLoading = false;
    sandbox.terrainError = null;
    sandbox.viewMode = 'battle';
    if (els.mapSelect) els.mapSelect.value = '';
    if (els.viewSelect) {
      els.viewSelect.value = 'battle';
      els.viewSelect.disabled = true;
    }
    syncMapStatus();
    drawSandbox();
    return;
  }
  if (!state.maps.available) {
    toast('地图资源不可用：请先准备游戏资源缓存', 'error');
    if (els.mapSelect) els.mapSelect.value = '';
    return;
  }
  const token = (sandbox.mapLoadToken || 0) + 1;
  sandbox.mapLoadToken = token;
  sandbox.mapId = nextId;
  sandbox.mapDetail = null;
  sandbox.terrain = null;
  sandbox.terrainEntry = null;
  sandbox.terrainError = null;
  sandbox.terrainLoading = true;
  if (els.mapSelect) els.mapSelect.value = nextId;
  if (els.viewSelect) els.viewSelect.disabled = false;
  syncMapStatus();
  drawSandbox();
  try {
    const detail = await Api.mapDetail(nextId);
    if (sandbox.mapLoadToken !== token) return;
    const entry = await buildTerrainEntry(detail);
    if (sandbox.mapLoadToken !== token) return;
    await ensureUnitSprites();
    if (sandbox.mapLoadToken !== token) return;
    sandbox.mapDetail = detail;
    sandbox.terrainEntry = entry;
    sandbox.terrain = entry.crop;
    sandbox.terrainLoading = false;
    if (sandbox.viewMode === 'full') {
      setMapStatus('地形渲染中…');
      await new Promise((resolve) => setTimeout(resolve, 16));
      sandbox.terrain = ensureFullTerrain(entry, detail);
    }
    syncMapStatus();
    drawSandbox();
    if (!opts.silent) toast('已加载地图 · ' + (detail.name || nextId), 'ok');
  } catch (error) {
    if (sandbox.mapLoadToken !== token) return;
    sandbox.terrainLoading = false;
    sandbox.terrainError = error.message || String(error);
    sandbox.mapDetail = null;
    sandbox.mapId = '';
    sandbox.terrainEntry = null;
    if (els.mapSelect) els.mapSelect.value = '';
    if (els.viewSelect) els.viewSelect.disabled = true;
    syncMapStatus();
    drawSandbox();
    toast('地图加载失败，已回退抽象渲染：' + error.message, 'error');
  }
}

async function applyViewMode(mode) {
  const sandbox = state.sandbox;
  sandbox.viewMode = mode === 'full' ? 'full' : 'battle';
  if (els.viewSelect) els.viewSelect.value = sandbox.viewMode;
  const entry = sandbox.terrainEntry;
  if (sandbox.mapDetail && entry && sandbox.viewMode === 'full' && !entry.full) {
    setMapStatus('地形渲染中…');
    await new Promise((resolve) => setTimeout(resolve, 16));
    sandbox.terrain = ensureFullTerrain(entry, sandbox.mapDetail);
  } else if (entry) {
    sandbox.terrain = sandbox.viewMode === 'full' ? entry.full : entry.crop;
  }
  syncMapStatus();
  drawSandbox();
}

function syncMapSelect() {
  if (!els.mapSelect) return;
  const id = state.sandbox.mapId || '';
  const exists = [...els.mapSelect.options].some((option) => option.value === id);
  els.mapSelect.value = exists ? id : '';
  if (els.viewSelect) {
    els.viewSelect.disabled = !state.sandbox.mapDetail;
    els.viewSelect.value = state.sandbox.viewMode || 'battle';
  }
}

function updateUnitAngles(frame, previousFrame) {
  const previous = new Map();
  if (previousFrame) {
    for (const unit of previousFrame.units) previous.set(unit.id, unit);
  }
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, unit);
  for (const unit of frame.units) {
    const record = unitAngleCache.get(unit.id) || { heading: -Math.PI / 2, turret: -Math.PI / 2 };
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
    unitAngleCache.set(unit.id, record);
  }
}

function drawSandboxSpriteUnit(ctx, unit, x, y, selected, leader, time, options) {
  const own = unit.team === 'own';
  const color = own ? SANDBOX_COLORS.own : SANDBOX_COLORS.enemy;
  const size = Math.max(8, num(options && options.size, 30));
  const bodyWidth = size * (UNIT_TILE_WIDTH / UNIT_TILE_HEIGHT);
  const hpFraction = clamp(unit.hp / Math.max(1, unit.maxHp), 0, 1);
  const angles = unitAngleCache.get(unit.id) || { heading: -Math.PI / 2, turret: -Math.PI / 2 };

  // 阵营色环（脚下）
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(x, y + size * 0.2, size * 0.42, size * 0.26, 0, 0, Math.PI * 2);
  ctx.strokeStyle = own ? 'rgba(63, 169, 255, 0.9)' : 'rgba(255, 90, 90, 0.9)';
  ctx.lineWidth = Math.max(1, size * 0.055);
  ctx.shadowColor = color;
  ctx.shadowBlur = size * 0.25;
  ctx.stroke();
  ctx.restore();

  if (selected || leader) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, size * 0.62, 0, Math.PI * 2);
    ctx.strokeStyle = selected ? 'rgba(234, 252, 255, 0.95)' : 'rgba(56, 225, 255, 0.32)';
    ctx.lineWidth = selected ? 1.8 : 1;
    if (selected) {
      ctx.setLineDash([5, 4]);
      ctx.lineDashOffset = -time * 14;
    } else {
      ctx.setLineDash([3, 5]);
      ctx.lineDashOffset = time * 8;
    }
    ctx.stroke();
    ctx.restore();
  }

  if (spriteCache.tank) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angles.heading + Math.PI / 2);
    ctx.drawImage(spriteCache.tank, 0, 0, UNIT_TILE_WIDTH, UNIT_TILE_HEIGHT, -bodyWidth / 2, -size / 2, bodyWidth, size);
    ctx.restore();
    if (spriteCache.turret) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angles.turret + Math.PI / 2);
      const turretWidth = size * (TURRET_TILE_WIDTH / TURRET_TILE_HEIGHT) * 1.05;
      const turretHeight = size * 1.02;
      ctx.drawImage(
        spriteCache.turret,
        0, 0, TURRET_TILE_WIDTH, TURRET_TILE_HEIGHT,
        -turretWidth / 2, -turretHeight * 0.7, turretWidth, turretHeight
      );
      ctx.restore();
    }
  } else {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, size * 0.32, 0, Math.PI * 2);
    ctx.fillStyle = own ? '#2f6fbf' : '#b5303c';
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.restore();
  }

  // 血条
  const barWidth = Math.max(14, size * 0.66);
  const barHeight = Math.max(2.5, size * 0.09);
  const barX = x - barWidth / 2;
  const barY = y - size * 0.62;
  ctx.fillStyle = 'rgba(2, 6, 12, 0.8)';
  ctx.fillRect(barX - 1, barY - 1, barWidth + 2, barHeight + 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.fillRect(barX, barY, barWidth, barHeight);
  ctx.fillStyle = hpFraction > 0.6 ? '#39e08b' : (hpFraction > 0.3 ? '#ffc35c' : '#ff5f6d');
  ctx.fillRect(barX, barY, barWidth * hpFraction, barHeight);
}

/* ============================================================
 * 对局沙盘渲染
 * ============================================================ */

function drawSandbox() {
  const canvas = els.sandboxCanvas;
  const wrap = els.stageWrap;
  if (!canvas || !wrap) return;
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;
  if (width < 8 || height < 8) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const background = ctx.createLinearGradient(0, 0, 0, height);
  background.addColorStop(0, '#060d17');
  background.addColorStop(1, '#03070d');
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  const sandbox = state.sandbox;
  const detail = sandbox.mapDetail;
  const mapMode = !!(detail && sandbox.terrain);
  const region = detail ? mapViewRegion(sandbox, detail) : { x: 0, y: 0, width: 1000, height: 1000 };
  const cam = mapCamera(width, height, region, mapMode ? 0 : 20);
  const offsetX = cam.ox + cam.region.x * cam.scale;
  const offsetY = cam.oy + cam.region.y * cam.scale;
  const drawnWidth = cam.region.width * cam.scale;
  const drawnHeight = cam.region.height * cam.scale;
  const mapX = (x) => (detail ? cam.ox + engineToMapPixel(x, 0, detail).x * cam.scale : cam.ox + num(x, 0) * cam.scale);
  const mapY = (y) => (detail ? cam.oy + engineToMapPixel(0, y, detail).y * cam.scale : cam.oy + num(y, 0) * cam.scale);
  sandbox.view = {
    ox: cam.ox,
    oy: cam.oy,
    scale: cam.scale,
    width: drawnWidth,
    height: drawnHeight,
    screenWidth: width,
    screenHeight: height,
    region: cam.region,
    detail: detail || null,
    corners: { x: offsetX, y: offsetY, width: drawnWidth, height: drawnHeight },
    project: (x, y) => ({ x: mapX(x), y: mapY(y) }),
  };

  ctx.save();
  ctx.beginPath();
  ctx.rect(offsetX, offsetY, drawnWidth, drawnHeight);
  ctx.clip();

  if (mapMode) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(sandbox.terrain, offsetX, offsetY, drawnWidth, drawnHeight);
    const shade = ctx.createRadialGradient(
      width / 2, height / 2, Math.min(width, height) * 0.18,
      width / 2, height / 2, Math.max(width, height) * 0.72
    );
    shade.addColorStop(0, 'rgba(3, 8, 14, 0)');
    shade.addColorStop(1, 'rgba(3, 8, 14, 0.5)');
    ctx.fillStyle = shade;
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

    if (detail && sandbox.viewMode === 'full') {
      const crop = mapCrop(detail);
      const bx = cam.ox + crop.x * cam.scale;
      const by = cam.oy + crop.y * cam.scale;
      const bw = crop.width * cam.scale;
      const bh = crop.height * cam.scale;
      ctx.save();
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = 'rgba(56, 225, 255, 0.6)';
      ctx.lineWidth = 1.2;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(56, 225, 255, 0.92)';
      ctx.font = '10px "Microsoft YaHei", "Segoe UI", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('交战区域', bx + 6, by + 8);
      ctx.restore();
    }
  } else if (detail) {
    ctx.fillStyle = '#07111c';
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);
    ctx.save();
    ctx.fillStyle = 'rgba(160, 200, 230, 0.78)';
    ctx.font = '13px "Microsoft YaHei", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(sandbox.terrainError ? '地图加载失败，回退抽象渲染' : '地图加载中…', width / 2, height / 2);
    ctx.restore();
  } else {
    const mapWidth = 1000;
    const mapHeight = 1000;
    const mapGradient = ctx.createRadialGradient(
    offsetX + drawnWidth / 2, offsetY + drawnHeight / 2, 20,
    offsetX + drawnWidth / 2, offsetY + drawnHeight / 2, Math.max(drawnWidth, drawnHeight) * 0.8
  );
  mapGradient.addColorStop(0, '#0a1727');
  mapGradient.addColorStop(1, '#050b13');
  ctx.fillStyle = mapGradient;
  ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

  // 双方部署区氛围色
  const ownGlow = ctx.createRadialGradient(
    offsetX + drawnWidth * 0.1, offsetY + drawnHeight / 2, 8,
    offsetX + drawnWidth * 0.1, offsetY + drawnHeight / 2, drawnWidth * 0.55
  );
  ownGlow.addColorStop(0, 'rgba(77, 163, 255, 0.12)');
  ownGlow.addColorStop(1, 'rgba(77, 163, 255, 0)');
  ctx.fillStyle = ownGlow;
  ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);
  const enemyGlow = ctx.createRadialGradient(
    offsetX + drawnWidth * 0.9, offsetY + drawnHeight / 2, 8,
    offsetX + drawnWidth * 0.9, offsetY + drawnHeight / 2, drawnWidth * 0.55
  );
  enemyGlow.addColorStop(0, 'rgba(255, 95, 109, 0.1)');
  enemyGlow.addColorStop(1, 'rgba(255, 95, 109, 0)');
  ctx.fillStyle = enemyGlow;
  ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

  const gridStep = 100;
  for (let gridX = 0; gridX <= mapWidth; gridX += gridStep) {
    const screenX = mapX(gridX);
    ctx.strokeStyle = gridX % 500 === 0 ? SANDBOX_COLORS.gridStrong : SANDBOX_COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(screenX, offsetY);
    ctx.lineTo(screenX, offsetY + drawnHeight);
    ctx.stroke();
  }
  for (let gridY = 0; gridY <= mapHeight; gridY += gridStep) {
    const screenY = mapY(gridY);
    ctx.strokeStyle = gridY % 500 === 0 ? SANDBOX_COLORS.gridStrong : SANDBOX_COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(offsetX, screenY);
    ctx.lineTo(offsetX + drawnWidth, screenY);
    ctx.stroke();
  }
  // 中线
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.12)';
  ctx.setLineDash([6, 8]);
  ctx.beginPath();
  ctx.moveTo(offsetX + drawnWidth / 2, offsetY);
  ctx.lineTo(offsetX + drawnWidth / 2, offsetY + drawnHeight);
  ctx.stroke();
  ctx.setLineDash([]);
  // 部署区标注
  ctx.save();
  ctx.font = '10px "Microsoft YaHei", "Segoe UI", sans-serif';
  ctx.textBaseline = 'bottom';
  if (drawnWidth > 420) {
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(77, 163, 255, 0.45)';
    ctx.fillText('我方部署区', offsetX + 12, offsetY + drawnHeight - 10);
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(255, 95, 109, 0.42)';
    ctx.fillText('敌方部署区', offsetX + drawnWidth - 12, offsetY + drawnHeight - 10);
  }
  ctx.restore();
  }

  const frames = sandbox.frames;
  if (!frames.length) {
    ctx.restore();
    drawSandboxChrome(ctx, width, height, offsetX, offsetY, drawnWidth, drawnHeight);
    drawSandboxEmpty(ctx, width, height);
    return;
  }

  const maxIndex = frames.length - 1;
  const playback = clamp(sandbox.playback, 0, maxIndex);
  const index = Math.floor(playback);
  const nextIndex = Math.min(index + 1, maxIndex);
  const fraction = playback - index;
  const frame = frames[index];
  const nextFrame = frames[nextIndex];
  const units = interpolateUnits(frame, nextFrame, fraction);
  updateUnitAngles(frame, index > 0 ? frames[index - 1] : null);

  // 指令箭头（画在单位下层）
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, { x: unit.x, y: unit.y });
  for (const unit of units) {
    if (unit.target === null || unit.target === undefined) continue;
    const target = positions.get(unit.target);
    if (!target) continue;
    const selected = unit.id === sandbox.selectedId;
    const alpha = selected ? 0.5 : 0.12;
    ctx.save();
    ctx.strokeStyle = unit.team === 'own' ? 'rgba(77, 163, 255, ' + alpha + ')' : 'rgba(255, 95, 109, ' + alpha + ')';
    ctx.lineWidth = selected ? 1.6 : 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(mapX(unit.x), mapY(unit.y));
    ctx.lineTo(mapX(target.x), mapY(target.y));
    ctx.stroke();
    ctx.restore();
  }

  // 单位（按 y 排序，制造俯视层次）
  const sorted = units.slice().sort((left, right) => left.y - right.y);
  const leaders = new Set();
  const seenGroups = new Set();
  for (const unit of sorted) {
    if (unit.team !== 'own') continue;
    if (seenGroups.has(unit.group)) continue;
    seenGroups.add(unit.group);
    leaders.add(unit.id);
  }
  const time = performance.now() / 1000;
  const spriteSize = sandbox.viewMode === 'full' ? 18 : 32;
  for (const unit of sorted) {
    if (mapMode && spriteCache.tank) {
      drawSandboxSpriteUnit(ctx, unit, mapX(unit.x), mapY(unit.y), unit.id === sandbox.selectedId, leaders.has(unit.id), time, { size: spriteSize });
    } else {
      drawSandboxUnit(ctx, unit, mapX, mapY, unit.id === sandbox.selectedId, leaders.has(unit.id), time);
    }
  }

  // 当前帧事件（暂停时静态显示，播放时由实时特效呈现）
  if (!sandbox.playing) {
    drawFrameEvents(ctx, frame, mapX, mapY, 1);
  }

  // 实时特效（弹道 / 爆炸，随帧推进生成）
  const now = performance.now();
  for (const effect of sandbox.effects) {
    const age = (now - effect.born) / effect.life;
    if (age >= 1) continue;
    if (effect.kind === 'shot') {
      drawShotLine(ctx, mapX(effect.from.x), mapY(effect.from.y), mapX(effect.to.x), mapY(effect.to.y), (1 - age) * 0.95);
    } else if (effect.kind === 'kill') {
      drawKillBurst(ctx, mapX(effect.x), mapY(effect.y), age, 1);
    }
  }
  sandbox.effects = sandbox.effects.filter((effect) => now - effect.born < effect.life);

  ctx.restore();

  drawSandboxChrome(ctx, width, height, offsetX, offsetY, drawnWidth, drawnHeight);
  drawSandboxHud(ctx, sandbox, frame, width, height);
  if (sandbox.source === 'demo') {
    ctx.save();
    ctx.font = '11px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(255, 195, 92, 0.65)';
    ctx.fillText('演示数据 · DEMO', width - 14, height - 10);
    ctx.restore();
  }
}

function drawSandboxChrome(ctx, width, height, offsetX, offsetY, drawnWidth, drawnHeight) {
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.35)';
  ctx.lineWidth = 1;
  ctx.strokeRect(offsetX + 0.5, offsetY + 0.5, drawnWidth - 1, drawnHeight - 1);
  // 四角标识
  const corner = 16;
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.85)';
  ctx.lineWidth = 2;
  const corners = [
    [offsetX, offsetY, 1, 1],
    [offsetX + drawnWidth, offsetY, -1, 1],
    [offsetX, offsetY + drawnHeight, 1, -1],
    [offsetX + drawnWidth, offsetY + drawnHeight, -1, -1],
  ];
  for (const item of corners) {
    const cx = item[0];
    const cy = item[1];
    const dx = item[2];
    const dy = item[3];
    ctx.beginPath();
    ctx.moveTo(cx + dx * corner, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * corner);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSandboxEmpty(ctx, width, height) {
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(124, 147, 173, 0.9)';
  ctx.font = '15px "Segoe UI", "Microsoft YaHei", sans-serif';
  ctx.fillText('暂无对局数据', width / 2, height / 2 - 12);
  ctx.fillStyle = 'rgba(93, 115, 144, 0.85)';
  ctx.font = '12px "Segoe UI", "Microsoft YaHei", sans-serif';
  ctx.fillText(
    state.mode === 'demo' ? '演示模式：点击「开始推理」播放内置演示对局' : '在右侧选择 checkpoint 后点击「开始推理」，或从回放库加载一局',
    width / 2, height / 2 + 14
  );
  ctx.restore();
}

function drawSandboxHud(ctx, sandbox, frame, width, height) {
  ctx.save();
  ctx.font = '11px "Consolas", "Segoe UI", monospace';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(200, 228, 250, 0.85)';
  const info = frame
    ? '#' + (frame.episode + 1) + ' · tick ' + frame.tick + ' · t=' + fmtNum(frame.time, 1) + 's'
    : '等待推理数据';
  ctx.fillText(info, 14, 12);
  if (sandbox.mapDetail) {
    ctx.save();
    ctx.font = '10px "Microsoft YaHei", "Segoe UI", sans-serif';
    ctx.fillStyle = 'rgba(124, 160, 190, 0.9)';
    const viewLabel = sandbox.viewMode === 'full' ? '全场' : '战场 1:1';
    ctx.fillText((sandbox.mapDetail.name || '地图') + ' · ' + viewLabel + ' · 地形仅视觉', 14, 27);
    ctx.restore();
  }

  // 血量条
  const barWidth = 120;
  const hudLeft = width - 196;
  const own = frame ? frame.stats.own : 0;
  const enemy = frame ? frame.stats.enemy : 0;
  ctx.fillStyle = 'rgba(5, 12, 22, 0.75)';
  ctx.fillRect(hudLeft, 10, 186, 30);
  ctx.fillStyle = 'rgba(124, 147, 173, 0.95)';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('我方', hudLeft + 8, 15);
  ctx.fillText('敌方', hudLeft + 8, 26);
  ctx.fillStyle = 'rgba(77, 163, 255, 0.85)';
  ctx.fillRect(hudLeft + 48, 17, barWidth * clamp(own, 0, 1), 5);
  ctx.fillStyle = 'rgba(255, 95, 109, 0.85)';
  ctx.fillRect(hudLeft + 48, 28, barWidth * clamp(enemy, 0, 1), 5);

  // 图例
  ctx.textBaseline = 'bottom';
  ctx.fillStyle = 'rgba(124, 147, 173, 0.85)';
  ctx.textAlign = 'left';
  const legendY = height - 12;
  ctx.fillStyle = SANDBOX_COLORS.own;
  ctx.beginPath();
  ctx.arc(16, legendY - 4, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('我方', 25, legendY);
  ctx.fillStyle = SANDBOX_COLORS.enemy;
  ctx.beginPath();
  ctx.arc(70, legendY - 4, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('敌方', 79, legendY);
  ctx.fillStyle = '#ffe066';
  ctx.fillRect(124, legendY - 6, 14, 3);
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('射击', 144, legendY);
  ctx.fillStyle = '#ffb347';
  ctx.beginPath();
  ctx.arc(188, legendY - 4, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('击毁', 197, legendY);
  ctx.restore();
}

function drawSandboxUnit(ctx, unit, mapX, mapY, selected, leader, time) {
  const x = mapX(unit.x);
  const y = mapY(unit.y);
  const radius = 7.5;
  const own = unit.team === 'own';
  const color = own ? SANDBOX_COLORS.own : SANDBOX_COLORS.enemy;
  const hpFraction = clamp(unit.hp / Math.max(1, unit.maxHp), 0, 1);

  // 选中圈 / 编组圈
  if (selected || leader) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius + 6, 0, Math.PI * 2);
    ctx.strokeStyle = selected ? 'rgba(234, 252, 255, 0.95)' : 'rgba(56, 225, 255, 0.32)';
    ctx.lineWidth = selected ? 1.8 : 1;
    if (selected) {
      ctx.setLineDash([5, 4]);
      ctx.lineDashOffset = -time * 14;
    } else {
      ctx.setLineDash([3, 5]);
      ctx.lineDashOffset = time * 8;
    }
    ctx.stroke();
    ctx.restore();
  }

  // 光晕 + 本体
  ctx.save();
  ctx.shadowColor = own ? 'rgba(77, 163, 255, 0.9)' : 'rgba(255, 95, 109, 0.9)';
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(x, y, radius + 2.4, 0, Math.PI * 2);
  ctx.fillStyle = own ? 'rgba(77, 163, 255, 0.18)' : 'rgba(255, 95, 109, 0.18)';
  ctx.fill();
  ctx.restore();

  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = own ? '#2f6fbf' : '#b5303c';
  ctx.fill();
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = color;
  ctx.stroke();

  // 编号
  ctx.save();
  ctx.font = '8.5px "Consolas", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(240, 250, 255, 0.92)';
  ctx.fillText(String(unit.id), x, y + 0.5);
  ctx.restore();

  // 血条
  const barWidth = 21;
  const barHeight = 3;
  const barX = x - barWidth / 2;
  const barY = y - radius - 8;
  ctx.fillStyle = 'rgba(2, 6, 12, 0.8)';
  ctx.fillRect(barX - 1, barY - 1, barWidth + 2, barHeight + 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.fillRect(barX, barY, barWidth, barHeight);
  ctx.fillStyle = hpFraction > 0.6 ? '#39e08b' : (hpFraction > 0.3 ? '#ffc35c' : '#ff5f6d');
  ctx.fillRect(barX, barY, barWidth * hpFraction, barHeight);
}

function drawFrameEvents(ctx, frame, mapX, mapY, alpha) {
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, { x: unit.x, y: unit.y });
  const phase = (performance.now() % 900) / 900;
  for (const event of frame.events) {
    if (event.type === 'shot') {
      const from = positions.get(event.source);
      const to = positions.get(event.target);
      if (from && to) drawShotLine(ctx, mapX(from.x), mapY(from.y), mapX(to.x), mapY(to.y), alpha);
    } else if (event.type === 'kill') {
      const target = positions.get(event.target);
      const x = target ? target.x : event.x;
      const y = target ? target.y : event.y;
      if (x !== null && y !== null && x !== undefined && y !== undefined) {
        drawKillBurst(ctx, mapX(x), mapY(y), phase, alpha);
      }
    }
  }
}

function drawShotLine(ctx, x1, y1, x2, y2, alpha) {
  const strength = clamp(alpha, 0, 1);
  if (strength <= 0.01) return;
  ctx.save();
  ctx.globalAlpha = strength;
  ctx.strokeStyle = '#ffe066';
  ctx.lineWidth = 2;
  ctx.shadowColor = 'rgba(255, 200, 70, 0.95)';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
  // 枪口闪光
  ctx.save();
  ctx.globalAlpha = strength * 0.9;
  const muzzle = ctx.createRadialGradient(x1, y1, 0, x1, y1, 7);
  muzzle.addColorStop(0, 'rgba(255, 255, 235, 0.95)');
  muzzle.addColorStop(1, 'rgba(255, 190, 60, 0)');
  ctx.fillStyle = muzzle;
  ctx.beginPath();
  ctx.arc(x1, y1, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  // 命中点
  ctx.save();
  ctx.globalAlpha = strength * 0.8;
  ctx.fillStyle = '#fff5c9';
  ctx.beginPath();
  ctx.arc(x2, y2, 3.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawKillBurst(ctx, x, y, phase, alpha) {
  const progress = clamp(phase, 0, 1);
  const strength = (1 - progress) * clamp(alpha, 0, 1);
  if (strength <= 0.01) return;
  const radius = 10 + 26 * progress;
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, 'rgba(255, 244, 200, ' + 0.5 * strength + ')');
  gradient.addColorStop(0.45, 'rgba(255, 150, 60, ' + 0.36 * strength + ')');
  gradient.addColorStop(1, 'rgba(255, 120, 40, 0)');
  ctx.save();
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 205, 120, ' + 0.85 * strength + ')';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.arc(x, y, radius * 0.82, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function interpolateUnits(frame, nextFrame, fraction) {
  const merged = new Map();
  for (const unit of frame.units) merged.set(unit.id, { unit: unit, next: null });
  if (nextFrame && nextFrame !== frame) {
    for (const unit of nextFrame.units) {
      const hit = merged.get(unit.id);
      if (hit) hit.next = unit;
    }
  }
  const result = [];
  for (const entry of merged.values()) {
    const unit = entry.unit;
    const next = entry.next;
    const blend = clamp(fraction, 0, 1);
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
    });
  }
  return result;
}

function spawnFrameEffects(frame) {
  if (!frame) return;
  const sandbox = state.sandbox;
  const now = performance.now();
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, { x: unit.x, y: unit.y });
  for (const event of frame.events) {
    if (event.type === 'shot') {
      const from = positions.get(event.source);
      const to = positions.get(event.target);
      if (from && to) sandbox.effects.push({ kind: 'shot', from: from, to: to, born: now, life: 340 });
    } else if (event.type === 'kill') {
      const target = positions.get(event.target);
      const x = target ? target.x : event.x;
      const y = target ? target.y : event.y;
      if (x !== undefined && y !== undefined && x !== null && y !== null) {
        sandbox.effects.push({ kind: 'kill', x: x, y: y, born: now, life: 780 });
      }
    }
  }
  if (sandbox.effects.length > 160) sandbox.effects.splice(0, sandbox.effects.length - 160);
}

function advanceSandbox(dt) {
  const sandbox = state.sandbox;
  const frames = sandbox.frames;
  if (!frames.length) return;
  const maxIndex = frames.length - 1;
  if (!sandbox.playing) {
    sandbox.wasPlaying = false;
    return;
  }
  sandbox.wasPlaying = true;
  const previous = sandbox.playback;
  const next = Math.min(previous + dt * BASE_FPS * sandbox.speed, maxIndex);
  if (next > previous) {
    const previousIndex = Math.floor(previous);
    const nextIndex = Math.floor(next);
    if (nextIndex !== previousIndex) {
      if (nextIndex - previousIndex <= 3) {
        for (let index = previousIndex + 1; index <= nextIndex; index += 1) spawnFrameEffects(frames[index]);
      } else {
        spawnFrameEffects(frames[nextIndex]);
      }
    }
    sandbox.playback = next;
  }
  if (sandbox.playback >= maxIndex && !sandbox.streaming) {
    sandbox.playing = false;
    updatePlayerUi(true);
    return;
  }
  updatePlayerUi(false);
}

/* ---------------- 录制（canvas.captureStream + MediaRecorder） ---------------- */

function toggleRecording() {
  const recorder = state.sandbox.recorder;
  if (recorder && recorder.state !== 'inactive') stopRecording();
  else startRecording();
}

function startRecording() {
  const canvas = els.sandboxCanvas;
  if (!canvas) return;
  if (typeof window.MediaRecorder === 'undefined' || typeof canvas.captureStream !== 'function') {
    toast('当前浏览器不支持 canvas 录制（需要 MediaRecorder + captureStream）', 'error');
    return;
  }
  let stream = null;
  try {
    stream = canvas.captureStream(30);
  } catch (error) {
    toast('无法捕获画布视频流：' + error.message, 'error');
    return;
  }
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  let mimeType = '';
  for (const candidate of candidates) {
    if (typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported(candidate)) {
      mimeType = candidate;
      break;
    }
  }
  let recorder = null;
  try {
    recorder = mimeType
      ? new MediaRecorder(stream, { mimeType: mimeType, videoBitsPerSecond: 6000000 })
      : new MediaRecorder(stream, { videoBitsPerSecond: 6000000 });
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    toast('录制初始化失败：' + error.message, 'error');
    return;
  }
  const sandbox = state.sandbox;
  sandbox.recorder = recorder;
  sandbox.recStream = stream;
  sandbox.recChunks = [];
  recorder.addEventListener('dataavailable', (event) => {
    if (event.data && event.data.size) sandbox.recChunks.push(event.data);
  });
  recorder.addEventListener('stop', () => finalizeRecording(sandbox));
  recorder.start(250);
  if (els.recordBtn) {
    els.recordBtn.textContent = '■ 停止录制';
    els.recordBtn.style.color = '#ff8f9b';
  }
  if (els.recBadge) els.recBadge.hidden = false;
  toast('开始录制沙盘画面（canvas.captureStream 30fps → .webm）', 'ok');
}

function stopRecording() {
  const recorder = state.sandbox.recorder;
  if (recorder && recorder.state !== 'inactive') {
    state.sandbox.recCancel = false;
    recorder.stop();
  }
}

/** 重置沙盘时静默丢弃正在进行的录制（不触发下载） */
function cancelRecording() {
  const sandbox = state.sandbox;
  const recorder = sandbox.recorder;
  if (recorder && recorder.state !== 'inactive') {
    sandbox.recCancel = true;
    recorder.stop();
  }
}

function finalizeRecording(sandbox) {
  const chunks = sandbox.recChunks || [];
  const recorder = sandbox.recorder;
  const mimeType = (recorder && recorder.mimeType) || 'video/webm';
  const cancelled = !!sandbox.recCancel;
  if (sandbox.recStream) sandbox.recStream.getTracks().forEach((track) => track.stop());
  sandbox.recorder = null;
  sandbox.recStream = null;
  sandbox.recChunks = [];
  sandbox.recCancel = false;
  if (els.recordBtn) {
    els.recordBtn.textContent = '● 录制视频';
    els.recordBtn.style.color = '';
  }
  if (els.recBadge) els.recBadge.hidden = true;
  if (cancelled) return;
  if (!chunks.length) {
    toast('录制结束，但未捕获到视频数据', 'error');
    return;
  }
  const blob = new Blob(chunks, { type: mimeType.split(';')[0] || 'video/webm' });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = 'ironzero-sandbox-' + timestamp + '.webm';
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  toast('录制完成 · ' + filename + '（' + (blob.size / 1048576).toFixed(2) + ' MB）', 'ok');
}

/* ============================================================
 * Tab / 事件绑定 / 轮询 / 启动
 * ============================================================ */

function switchTab(name) {
  if (state.tab === 'sandbox' && name !== 'sandbox') stopRecording();
  state.tab = name;
  const buttons = document.querySelectorAll('.tab-btn');
  for (const button of buttons) {
    const active = button.getAttribute('data-tab') === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  const panels = document.querySelectorAll('.tab-panel');
  for (const panel of panels) panel.classList.toggle('active', panel.id === 'panel-' + name);
  if (window.location.hash !== '#' + name) {
    try {
      window.history.replaceState(null, '', '#' + name);
    } catch (error) {
      window.location.hash = name;
    }
  }
  if (name === 'train') {
    drawCharts();
  } else if (name === 'sandbox') {
    ensureSandboxContent();
    drawSandbox();
  } else if (name === 'replays') {
    loadReplays();
  }
}

function tabFromHash() {
  const hash = String(window.location.hash || '').replace('#', '');
  if (hash === 'sandbox' || hash === 'replays' || hash === 'train') return hash;
  return 'train';
}

function toast(message, kind) {
  const host = els.toastHost;
  if (!host) return;
  const node = document.createElement('div');
  node.className = 'toast' + (kind ? ' ' + kind : '');
  node.textContent = String(message);
  host.appendChild(node);
  setTimeout(() => {
    node.style.transition = 'opacity 0.25s';
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 280);
  }, 3800);
  while (host.childElementCount > 6) host.removeChild(host.firstElementChild);
}

function handleSandboxClick(event) {
  const sandbox = state.sandbox;
  const canvas = els.sandboxCanvas;
  const view = sandbox.view;
  if (!canvas || !view || !sandbox.frames.length) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  const frame = currentFrame();
  if (!frame) return;
  let best = null;
  let bestDistance = 22;
  for (const unit of frame.units) {
    const projected = view.project ? view.project(unit.x, unit.y) : { x: view.ox + unit.x * view.scale, y: view.oy + unit.y * view.scale };
    const distance = Math.hypot(projected.x - mouseX, projected.y - mouseY);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = unit;
    }
  }
  sandbox.selectedId = best ? best.id : null;
  if (best) {
    sandbox.eventLog.push({
      kind: 'info',
      time: frame.time,
      text: '选中 #' + best.id + '（' + (best.team === 'own' ? '我方' : '敌方') + '）'
        + ' HP ' + Math.round(best.hp) + '/' + Math.round(best.maxHp)
        + ' · order=' + (best.order || '—')
        + (best.target === null || best.target === undefined ? '' : ' · target #' + best.target),
    });
    if (sandbox.eventLog.length > 300) sandbox.eventLog.shift();
    sandbox.feedCount = -1;
    updatePlayerUi(true);
  }
}

function bindEvents() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  for (const button of tabButtons) {
    button.addEventListener('click', () => switchTab(button.getAttribute('data-tab') || 'train'));
  }
  if (els.startForm) els.startForm.addEventListener('submit', submitStartRun);
  if (els.stopBtn) els.stopBtn.addEventListener('click', stopCurrentRun);
  if (els.inferForm) els.inferForm.addEventListener('submit', submitInfer);
  if (els.replayRefresh) els.replayRefresh.addEventListener('click', () => { loadReplays(); });
  if (els.demoRetry) els.demoRetry.addEventListener('click', retryConnection);
  if (els.recordBtn) els.recordBtn.addEventListener('click', toggleRecording);
  if (els.mapSelect) {
    els.mapSelect.addEventListener('change', () => {
      selectMap(els.mapSelect.value || '');
    });
  }
  if (els.viewSelect) {
    els.viewSelect.addEventListener('change', () => {
      applyViewMode(els.viewSelect.value);
    });
  }

  if (els.playBtn) {
    els.playBtn.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.frames.length) {
        if (state.mode === 'demo') {
          loadDemoBattle();
        } else {
          toast('暂无可播放的对局，请先开始推理或从回放库加载', 'error');
          return;
        }
      }
      sandbox.playing = !sandbox.playing;
      if (sandbox.playing && sandbox.playback >= sandbox.frames.length - 1 && !sandbox.streaming) sandbox.playback = 0;
      updatePlayerUi(true);
    });
  }

  if (els.latestBtn) {
    els.latestBtn.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.frames.length) return;
      sandbox.playback = sandbox.frames.length - 1;
      sandbox.playing = !!sandbox.streaming;
      sandbox.follow = true;
      updatePlayerUi(true);
    });
  }

  if (els.speedSelect) {
    els.speedSelect.addEventListener('change', () => {
      state.sandbox.speed = clamp(num(els.speedSelect.value, 1), 0.5, 4);
    });
  }

  if (els.seek) {
    els.seek.addEventListener('input', () => {
      const sandbox = state.sandbox;
      if (!sandbox.frames.length) return;
      sandbox.playing = false;
      sandbox.follow = false;
      sandbox.playback = clampInt(els.seek.value, 0, sandbox.frames.length - 1, 0);
      sandbox.effects = [];
      updatePlayerUi(true);
      drawSandbox();
    });
  }

  if (els.sandboxCanvas) {
    els.sandboxCanvas.addEventListener('click', handleSandboxClick);
  }

  if (els.logPanel) {
    els.logPanel.addEventListener('scroll', () => {
      const panel = els.logPanel;
      state.autoScroll = panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 24;
    });
  }

  const chartPairs = [
    [els.chartWin, els.tipWin],
    [els.chartReturn, els.tipReturn],
  ];
  for (const pair of chartPairs) {
    const canvas = pair[0];
    const tip = pair[1];
    if (!canvas || !tip) continue;
    canvas.addEventListener('mousemove', (event) => handleChartHover(canvas, tip, event.clientX, event.clientY));
    canvas.addEventListener('mouseleave', () => { tip.hidden = true; });
  }

  window.addEventListener('resize', () => {
    if (state.tab === 'train') drawCharts();
    if (state.tab === 'sandbox') drawSandbox();
  });

  window.addEventListener('keydown', (event) => {
    if (state.tab !== 'sandbox') return;
    const target = event.target;
    if (target && ['INPUT', 'SELECT', 'TEXTAREA'].indexOf(target.tagName) >= 0) return;
    const sandbox = state.sandbox;
    if (event.code === 'Space' || event.key === ' ') {
      event.preventDefault();
      if (!sandbox.frames.length) return;
      sandbox.playing = !sandbox.playing;
      if (sandbox.playing && sandbox.playback >= sandbox.frames.length - 1 && !sandbox.streaming) sandbox.playback = 0;
      updatePlayerUi(true);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      sandbox.playing = false;
      sandbox.playback = clamp(sandbox.playback + 1, 0, Math.max(0, sandbox.frames.length - 1));
      updatePlayerUi(true);
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      sandbox.playing = false;
      sandbox.playback = clamp(sandbox.playback - 1, 0, Math.max(0, sandbox.frames.length - 1));
      updatePlayerUi(true);
    }
  });

  if (typeof ResizeObserver === 'function') {
    const stack = document.querySelector('.chart-stack');
    if (stack) {
      const observer = new ResizeObserver(() => {
        if (state.tab === 'train') drawCharts();
      });
      observer.observe(stack);
    }
  }
}

let lastAnimationTs = 0;

function animationLoop(timestamp) {
  requestAnimationFrame(animationLoop);
  const dt = lastAnimationTs ? Math.min(0.12, (timestamp - lastAnimationTs) / 1000) : 0;
  lastAnimationTs = timestamp;
  if (state.tab === 'sandbox') {
    advanceSandbox(dt);
    drawSandbox();
  }
}

async function tickMain() {
  if (state.mainBusy) return;
  state.mainBusy = true;
  try {
    await checkHealth();
    if (!state.connected) {
      if (state.mode !== 'demo') enterDemo();
      renderConn();
      return;
    }
    if (state.mode === 'demo') exitDemo();
    if (state.mode === 'boot') state.mode = 'live';
    await refreshRuns();
    if (state.currentRunId) await refreshCurrentRun();
    if (state.tab === 'replays') await loadReplays();
    if (!state.checkpoints.length) await loadCheckpoints();
    if (!state.maps.ready) await loadMaps();
    renderConn();
  } finally {
    state.mainBusy = false;
  }
}

async function boot() {
  cacheEls();
  bindEvents();
  startClock();
  state.mode = 'boot';
  switchTab(tabFromHash());
  requestAnimationFrame(animationLoop);
  renderConn();

  await checkHealth();
  if (state.connected) {
    state.mode = 'live';
    renderConn();
    await refreshRuns();
    if (state.runs.length && !state.currentRunId) await selectRun(state.runs[0].id);
    await loadCheckpoints();
    await loadMaps();
    await loadReplays();
  } else {
    enterDemo();
  }
  renderConn();
  updateKpis();
  drawCharts();
  updatePlayerUi(true);
  setInterval(tickMain, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { boot(); });
} else {
  boot();
}
