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
  if (explicit) return String(explicit).replace(/\/+$/, '');
  if (typeof window !== 'undefined' && window.location) {
    if (window.location.protocol === 'file:' || (window.location.port && window.location.port !== '8000')) {
      return 'http://127.0.0.1:8000';
    }
  }
  return String(API_DEFAULT_BASE).replace(/\/+$/, '');
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
  maps: () => api('/api/maps', { timeout: 35000 }),
  mapDetail: (id) => api('/api/maps/' + encodeURIComponent(id), { timeout: 35000 }),
  assetUrl: (relative) => apiBase() + '/api/assets/' + String(relative || '').split('/').map(encodeURIComponent).join('/'),
  rwppStatus: () => api('/api/rwpp/status', { timeout: 5000 }),
  rwppRecord: (config) => api('/api/rwpp/record', { method: 'POST', body: config }),
  rwppJob: (id) => api('/api/rwpp/jobs/' + encodeURIComponent(id), { timeout: 5000 }),
  rwppJobs: () => api('/api/rwpp/jobs', { timeout: 8000 }),
  rwppInfer: (config) => api('/api/rwpp/infer', { method: 'POST', body: config, timeout: 20000 }),
  rwppInferFrames: (id, after) => api('/api/rwpp/infer/' + encodeURIComponent(id) + '/frames?after=' + Math.max(0, Math.floor(after || 0)), { timeout: 10000 }),
  rwppTrain: (config) => api('/api/rwpp/train', { method: 'POST', body: config, timeout: 20000 }),
  rwppPpo: (config) => api('/api/rwpp/ppo', { method: 'POST', body: config, timeout: 30000 }),
  rwppPpoEvents: (id, after) => api('/api/rwpp/ppo/' + encodeURIComponent(id) + '/events?after=' + Math.max(0, Math.floor(after || 0)), { timeout: 10000 }),
  rwppPpoStop: (id) => api('/api/rwpp/ppo/' + encodeURIComponent(id) + '/stop', { method: 'POST', body: {}, timeout: 10000 }),
  rwppJobStop: (id) => api('/api/rwpp/jobs/' + encodeURIComponent(id) + '/stop', { method: 'POST', body: {}, timeout: 10000 }),
  rwppStopAll: () => api('/api/rwpp/stop_all', { method: 'POST', body: {}, timeout: 10000 }),
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

function parseBeijingDate(timeStr) {
  if (!timeStr && timeStr !== 0) return null;
  const s = String(timeStr).trim();
  let date;
  if (/^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(s)) {
    if (s.endsWith('Z') || /[+-]\d{2}:?\d{2}$/.test(s)) {
      date = new Date(s);
    } else {
      // 容器后端默认时区可能为 UTC，在此智能探测并自动纠正为北京时间 (UTC+8)
      const asUtc = new Date(s.replace(' ', 'T') + 'Z');
      const asLocal = new Date(s.replace(/-/g, '/'));
      const now = Date.now();
      if (Math.abs(now - asUtc.getTime()) < Math.abs(now - asLocal.getTime())) {
        date = asUtc;
      } else {
        date = asLocal;
      }
    }
  } else {
    date = new Date(s);
  }
  return isNaN(date.getTime()) ? null : date;
}

function fmtShortTime(value) {
  if (!value && value !== 0) return '';
  const d = parseBeijingDate(value);
  if (d) {
    return pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' + pad2(d.getHours()) + ':' + pad2(d.getMinutes());
  }
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
    const width = num(pickVal(raw, 'width_px', 'width', 'w'), 1000);
    const height = num(pickVal(raw, 'height_px', 'height', 'h'), 1000);
    return {
      width: Math.max(1, width),
      height: Math.max(1, height),
      id: raw.id || null,
      name: raw.name || null,
      tileWidth: num(pickVal(raw, 'tile_width', 'tileWidth'), 20),
      tileHeight: num(pickVal(raw, 'tile_height', 'tileHeight'), 20),
    };
  }
  return { width: 1000, height: 1000, id: null, name: null, tileWidth: 20, tileHeight: 20 };
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
    simMode: pickVal(item, 'sim_mode') === 'terrain' ? 'terrain' : 'arena',
    mapId: pickVal(item, 'map_id') || ((item.map && item.map.id) || null),
    mapName: pickVal(item, 'map_name') || ((item.map && item.map.name) || null),
    source: pickVal(item, 'source') || null,
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
  // "timeout" = the episode hit the tick cap without a decided battle (not a draw).
  if (text === 'timeout') return { label: '超时', cls: 'draw' };
  if (text === 'draw' || text === 'tie') return { label: '平局', cls: 'draw' };
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
    'map-select', 'view-select', 'map-status', 'fog-toggle', 'items-toggle',
    'play-btn', 'latest-btn', 'speed-select', 'seek', 'frame-label', 'frame-sub',
    'infer-form', 'ckpt-select', 'infer-sim', 'episodes-input', 'infer-opponent', 'infer-seed', 'infer-mode', 'infer-btn', 'infer-hint',
    'score-own', 'score-enemy', 'hp-own', 'hp-own-val', 'hp-enemy', 'hp-enemy-val',
    'sb-ckpt', 'sb-opponent', 'sb-episode', 'sb-frames', 'event-feed',
    'replay-card', 'replay-card-head', 'replay-toggle-btn', 'replay-toggle-icon', 'replay-toggle-text',
    'replay-content', 'replay-pagination', 'replay-page-info', 'replay-page-size', 'replay-prev-btn', 'replay-next-btn', 'replay-current-page',
    'replay-count', 'replay-refresh', 'replay-body', 'toast-host',
    'rwpp-status', 'rwpp-status-text', 'rwpp-record-btn',
    'rwpp-status-train', 'rwpp-status-train-text',
    'fe-policy', 'fe-ticks', 'fe-start-btn', 'fe-infer-status', 'fe-concurrency',
    'fe-kpi-concurrency', 'fe-kpi-slots', 'fe-kpi-running', 'fe-kpi-done', 'fe-kpi-frames',
    'fe-kpi-fps', 'fe-kpi-tick', 'fe-task-count', 'fe-task-summary', 'fe-task-cards',
    'fe-chart-fps', 'fe-chart-hp', 'fe-chart-alive',
    'fe-slots-max', 'fe-slots-free', 'fe-slots-running',
    'fe-train-map', 'fe-train-policy', 'fe-train-checkpoint-field', 'fe-train-checkpoint-select', 'fe-train-episodes', 'fe-train-ticks', 'fe-train-start', 'fe-train-stop',
    'fe-train-concurrency', 'fe-train-status', 'fe-train-progress', 'fe-train-episodes-label',
    'fe-train-current', 'fe-train-seconds', 'fe-train-summary', 'fe-train-log',
    'fe-train-log-status', 'fe-train-artifacts', 'fe-train-artifact-count', 'fe-train-refresh',
    'fe-train-kpi-progress', 'fe-train-kpi-progress-total', 'fe-train-kpi-seconds',
    'fe-train-kpi-threads', 'fe-train-kpi-avg', 'fe-train-kpi-frames', 'fe-train-kpi-ticks',
    'fe-train-chart-duration', 'fe-train-chart-frames', 'fe-train-chart-ticks',
    'fe-train-mode-note',
    'fe-ppo-kpi-progress', 'fe-ppo-kpi-progress-total', 'fe-ppo-kpi-seconds', 'fe-ppo-kpi-avg',
    'fe-ppo-kpi-dps', 'fe-ppo-kpi-return', 'fe-ppo-kpi-return-foot', 'fe-ppo-kpi-win', 'fe-ppo-kpi-win-foot', 'fe-ppo-kpi-entropy', 'fe-ppo-kpi-device',
    'fe-ppo-chart-return', 'fe-ppo-chart-win', 'fe-ppo-chart-hp', 'fe-ppo-chart-reward', 'fe-ppo-chart-entropy', 'fe-ppo-chart-duration',
    'fe-ppo-status', 'fe-ppo-progress', 'fe-ppo-progress-label', 'fe-ppo-current', 'fe-ppo-seconds',
    'fe-ppo-eval-hint', 'fe-ppo-summary-card', 'fe-ppo-summary', 'fe-ppo-summary-badge',
    'fe-ppo-checkpoint', 'fe-ppo-checkpoint-path', 'fe-ppo-log', 'fe-ppo-log-status',
    'fe-ppo-updates', 'fe-ppo-envs', 'fe-ppo-rollout', 'fe-ppo-ticks', 'fe-ppo-units', 'fe-ppo-device',
    'fe-ppo-map', 'fe-ppo-units-field', 'fe-ppo-map-hint', 'fe-ppo-init-mode', 'fe-ppo-checkpoint-field', 'fe-ppo-init-checkpoint', 'fe-ppo-init-hint',
    'fe-ppo-start', 'fe-ppo-stop', 'fe-ppo-hint', 'rwpp-status-ppo', 'rwpp-status-ppo-text',
    'fe-ppo-frame-ms', 'fe-ppo-vision', 'fe-ppo-kpi-device-foot',
    'fe-checkpoint-field', 'fe-checkpoint-select', 'fe-checkpoint-refresh', 'fe-model-note', 'fe-map-count',
    'fe-ppo-eval-every', 'fe-ppo-eval-episodes', 'fe-ppo-use-llm', 'fe-ppo-llm-note',
    'fe-ppo-reward', 'fe-ppo-reward-chip', 'fe-ppo-reward-note',
    'fe-ppo-strategy', 'fe-ppo-strategy-chip', 'fe-ppo-strategy-source',
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
    frameStep: 0,
    fetchCursor: 0,
    status: 'idle',
    inferDone: false,
    pollErrors: 0,
    polling: false,
    map: { width: 1000, height: 1000 },
    mapId: '',
    mapDetail: null,
    terrainView: null,
    terrainEntry: null,
    terrainLoading: false,
    terrainError: null,
    viewMode: 'battle',
    mapLoadToken: 0,
    simMode: 'arena',
    visionRadius: 260,
    fog: false,
    fogAvailable: false,
    fogEnabled: false,
    itemsEnabled: true,
    mapLocked: false,
    cameraRegion: null,
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
  replayPagination: { page: 1, pageSize: 8, collapsed: true },
  maps: { ready: false, available: false, error: null, list: [], units: {}, attempts: 0 },
  sandbox: createSandboxState(),
  mainBusy: false,
  inferTimer: null,
  rwpp: {
    available: false,
    checked: false,
    checking: false,
    statusAt: 0,
    service: null,
    serviceError: null,
    jobId: null,
    startedAt: 0,
    timer: null,
    polling: false,
    running: false,
    lastSeconds: 0,
    hostRunning: 0,
  },
  fe: {
    inferJobId: null,
    inferTimer: null,
    inferPolling: false,
    inferStatus: 'idle',
    inferFrames: 0,
    inferLastTick: 0,
    inferStartedAt: 0,
    trainJobId: null,
    trainTimer: null,
    trainPolling: false,
    trainStatus: 'idle',
    trainStartedAt: 0,
    trainProgress: { done: 0, total: 0, seconds: 0, current: 0 },
    trainArtifacts: [],
    mapsReady: false,
  },
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
  if (state.tab === 'sandbox' || state.tab === 'infer') ensureSandboxContent();
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

  const winValue = lastEval && numOrNull(lastEval.win_rate) !== null
    ? lastEval.win_rate
    : (numOrNull(latest.rolling_win_rate) !== null ? latest.rolling_win_rate : numOrNull(latest.win_rate));
  els.kpiWin.textContent = winValue === null || winValue === undefined ? '—' : fmtPct(winValue);
  if (els.kpiWinFoot) {
    els.kpiWinFoot.textContent = lastEval
      ? '评估 ' + state.evalCount + ' 次 · 每 ' + (numOrNull(config.eval_every) === null ? '—' : config.eval_every) + ' upd'
      : (numOrNull(latest.rolling_win_rate) !== null ? '训练最近完成局胜率' : '尚无评估数据');
  }

  const returnValue = lastEval && numOrNull(lastEval.mean_return) !== null
    ? lastEval.mean_return
    : (numOrNull(latest.rolling_return) !== null ? latest.rolling_return : numOrNull(latest.mean_return));
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

function updateReplayCollapseUi() {
  const p = state.replayPagination;
  if (els.replayCard) {
    els.replayCard.classList.toggle('collapsed', !!p.collapsed);
  }
  if (els.replayToggleIcon) {
    els.replayToggleIcon.textContent = p.collapsed ? '▶' : '▼';
  }
  if (els.replayToggleText) {
    els.replayToggleText.textContent = p.collapsed ? '展开' : '收起';
  }
}

function toggleReplayCollapse() {
  state.replayPagination.collapsed = !state.replayPagination.collapsed;
  updateReplayCollapseUi();
}

function renderReplays() {
  updateReplayCollapseUi();
  const body = els.replayBody;
  if (!body) return;
  body.innerHTML = '';
  const allItems = state.replays || [];
  const total = allItems.length;
  if (els.replayCount) els.replayCount.textContent = String(total);

  const p = state.replayPagination;
  const pageSize = Math.max(1, p.pageSize || 8);
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if (p.page > totalPages) p.page = totalPages;
  if (p.page < 1) p.page = 1;

  if (els.replayPageInfo) {
    els.replayPageInfo.textContent = '第 ' + p.page + ' / ' + totalPages + ' 页 · 共 ' + total + ' 条';
  }
  if (els.replayCurrentPage) {
    els.replayCurrentPage.textContent = String(p.page);
  }
  if (els.replayPrevBtn) {
    els.replayPrevBtn.disabled = (p.page <= 1);
  }
  if (els.replayNextBtn) {
    els.replayNextBtn.disabled = (p.page >= totalPages);
  }
  if (els.replayPageSize && String(els.replayPageSize.value) !== String(pageSize)) {
    els.replayPageSize.value = String(pageSize);
  }

  if (!total) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 10;
    cell.className = 'empty';
    cell.textContent = state.mode === 'demo' ? '暂无演示回放' : '尚无回放；运行一次推理后会自动入库';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }

  const startIdx = (p.page - 1) * pageSize;
  const items = allItems.slice(startIdx, startIdx + pageSize);

  for (const item of items) {
    const row = document.createElement('tr');
    row.className = 'data-row';
    const realReplay = isRealReplay(item);
    if (realReplay) row.classList.add('real-row');
    row.addEventListener('click', () => openReplay(item.id));

    const cellName = document.createElement('td');
    cellName.className = 'cell-name';
    cellName.textContent = item.name || basename(item.id);
    if (realReplay) {
      const realTag = document.createElement('span');
      realTag.className = 'real-tag';
      realTag.textContent = '真机';
      realTag.title = '由宿主机真实 Rusted Warfare 1.15 无头对局录制';
      cellName.appendChild(realTag);
    }
    cellName.title = item.id || item.name || '';

    const cellCheckpoint = document.createElement('td');
    cellCheckpoint.className = 'cell-path';
    cellCheckpoint.textContent = item.checkpoint ? basename(item.checkpoint) : '—';
    cellCheckpoint.title = item.checkpoint || '';

    const cellOpponent = document.createElement('td');
    cellOpponent.textContent = item.opponent || '—';

    const cellSim = document.createElement('td');
    const simTerrain = item.simMode === 'terrain';
    cellSim.textContent = (simTerrain ? '地形' : '竞技场') + (simTerrain && item.mapName ? ' · ' + item.mapName : '');
    cellSim.title = simTerrain ? ('实机地形模拟' + (item.mapName ? ' · ' + item.mapName : '')) : '抽象竞技场';
    cellSim.className = 'cell-sim' + (simTerrain ? ' terrain' : '');

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
    row.appendChild(cellSim);
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


/* ============================================================
 * 真机 RWPP 录制（宿主机真实 1.15 无头对局）
 * ============================================================ */

const RWPP_TICKS = 3600;
const RWPP_POLL_MS = 2000;
// Upper bound for following one real-game job in the UI. Long requests (up to 1 hour,
// 216000 ticks) can run for tens of minutes, so allow up to 2 hours of wall clock.
const RWPP_MAX_SECONDS = 7200;
const RWPP_STATUS_TTL_MS = 30000;

function isRealReplay(item) {
  if (!item) return false;
  return item.checkpoint === 'rwpp-live' || item.source === 'rwpp-bridge';
}

function renderRwppStatus() {
  const rwpp = state.rwpp;
  const stateName = rwpp.checked ? (rwpp.available ? 'ok' : 'down') : 'unknown';
  const label = !rwpp.checked
    ? '真机服务检测中…'
    : (rwpp.available ? '真机服务就绪' : '真机服务未启动');
  for (const badge of document.querySelectorAll('.rwpp-status')) {
    badge.setAttribute('data-state', stateName);
    if (rwpp.service) badge.title = '宿主机真机服务：' + rwpp.service;
  }
  for (const labelNode of document.querySelectorAll('.rwpp-status-text')) {
    labelNode.textContent = label;
  }
  if (els.rwppStatus) els.rwppStatus.setAttribute('data-state', stateName);
  if (els.rwppStatusText) els.rwppStatusText.textContent = label;
  if (els.rwppStatusTrain) els.rwppStatusTrain.setAttribute('data-state', stateName);
  if (els.rwppStatusTrainText) els.rwppStatusTrainText.textContent = label;
  if (els.rwppRecordBtn) {
    els.rwppRecordBtn.classList.toggle('unavailable', rwpp.checked && !rwpp.available);
  }
}

async function refreshRwppStatus(force) {
  const rwpp = state.rwpp;
  if (rwpp.checking) return;
  const effectiveTtl = rwpp.available ? RWPP_STATUS_TTL_MS : 3000;
  if (!force && rwpp.checked && Date.now() - rwpp.statusAt < effectiveTtl) return;
  if (state.mode === 'demo') {
    try {
      const ok = await checkHealth();
      if (ok) exitDemo();
    } catch (_) {}
    if (state.mode === 'demo') {
      rwpp.checked = true;
      rwpp.available = false;
      renderRwppStatus();
      return;
    }
  }
  rwpp.checking = true;
  try {
    const payload = await Api.rwppStatus();
    rwpp.available = !!(payload && payload.ok);
    rwpp.service = payload && payload.service ? String(payload.service) : null;
    rwpp.hostRunning = payload && numOrNull(payload.running) !== null ? num(payload.running, 0) : 0;
    rwpp.maxParallel = payload && numOrNull(payload.max_parallel) !== null ? num(payload.max_parallel, 4) : 4;
    rwpp.freeSlots = payload && numOrNull(payload.free_slots) !== null ? num(payload.free_slots, null) : null;
    rwpp.ppoRunning = payload && numOrNull(payload.ppo_running) !== null ? num(payload.ppo_running, 0) : 0;
    rwpp.serviceError = null;
  } catch (error) {
    rwpp.available = false;
    rwpp.serviceError = error && error.message ? error.message : String(error);
  } finally {
    rwpp.checking = false;
    rwpp.checked = true;
    rwpp.statusAt = Date.now();
    renderRwppStatus();
    feUpdateSlots();
  }
}

function setRwppRecording(running, seconds) {
  const rwpp = state.rwpp;
  rwpp.running = running;
  rwpp.lastSeconds = running ? Math.max(0, Math.round(num(seconds, rwpp.lastSeconds))) : 0;
  const button = els.rwppRecordBtn;
  if (!button) return;
  button.disabled = running;
  button.classList.toggle('running', running);
  if (running) {
    button.innerHTML = '<span class="rwpp-rec-dot"></span>真机录制中… ' + rwpp.lastSeconds + 's';
  } else {
    button.innerHTML = '<span class="rwpp-rec-dot"></span>真机录制';
  }
}

function handleRwppError(error) {
  const status = error && error.status;
  if (status === 503) {
    toast('真机服务未启动：在宿主机运行 python scripts/rwpp_real_service.py', 'error');
  } else if (status === 409) {
    toast('已有真机录制任务正在进行，请等待完成', 'error');
  } else {
    toast('真机录制失败：' + (error && error.message ? error.message : error), 'error');
  }
}

async function startRwppRecord() {
  const rwpp = state.rwpp;
  if (rwpp.running) return;
  if (state.mode === 'demo') {
    toast('演示模式下无法启动真机录制（后端未连接）', 'error');
    return;
  }
  if (els.rwppRecordBtn) els.rwppRecordBtn.disabled = true;
  try {
    const payload = await Api.rwppRecord({ ticks: RWPP_TICKS, policy: 'scripted' });
    const jobId = payload && (payload.job_id || payload.id);
    if (!jobId) throw new Error('后端未返回真机录制任务 id');
    rwpp.jobId = String(jobId);
    rwpp.startedAt = Date.now();
    rwpp.lastSeconds = 0;
    rwpp.available = true;
    rwpp.checked = true;
    rwpp.statusAt = Date.now();
    renderRwppStatus();
    setRwppRecording(true, 0);
    startRwppPolling();
    toast('真机录制已开始（ticks=' + RWPP_TICKS + ' ≈ ' + Math.round(RWPP_TICKS / 60) + 's 仿真时长；引擎分出胜负会提前结束）', 'ok');
  } catch (error) {
    setRwppRecording(false);
    handleRwppError(error);
    await refreshRwppStatus(true);
  }
}

function stopRwppPolling() {
  const rwpp = state.rwpp;
  if (rwpp.timer) {
    clearInterval(rwpp.timer);
    rwpp.timer = null;
  }
  rwpp.jobId = null;
  rwpp.startedAt = 0;
}

function startRwppPolling() {
  const rwpp = state.rwpp;
  if (rwpp.timer) clearInterval(rwpp.timer);
  rwpp.timer = setInterval(pollRwppJob, RWPP_POLL_MS);
  pollRwppJob();
}

function pickLatestRealReplay(replayPath) {
  const items = state.replays || [];
  if (!items.length) return null;
  if (replayPath) {
    const normalized = String(replayPath).replace(/\\/g, '/');
    const fileName = basename(normalized);
    const hit = items.find((item) => item.name === fileName || item.id.endsWith(fileName));
    if (hit) return hit;
  }
  return items.find(isRealReplay) || items[0];
}

async function pollRwppJob() {
  const rwpp = state.rwpp;
  if (!rwpp.jobId || rwpp.polling) return;
  rwpp.polling = true;
  try {
    const job = await Api.rwppJob(rwpp.jobId);
    if (!job) return;
    const seconds = numOrNull(job.seconds) !== null ? num(job.seconds, 0) : (Date.now() - rwpp.startedAt) / 1000;
    if (job.status === 'running') {
      setRwppRecording(true, seconds);
      if ((Date.now() - rwpp.startedAt) / 1000 > RWPP_MAX_SECONDS) {
        stopRwppPolling();
        setRwppRecording(false);
        toast('真机录制超时（超过 300s），请检查宿主机服务日志', 'error');
        await refreshRwppStatus(true);
      }
      return;
    }
    stopRwppPolling();
    setRwppRecording(false);
    if (job.status === 'done') {
      toast('真机录制完成（' + Math.round(seconds) + 's）', 'ok');
      await loadReplays();
      const target = pickLatestRealReplay(job.replay_path);
      if (target) {
        await openReplay(target.id);
        toast('已自动加载真机回放到沙盘', 'ok');
      }
      await refreshRwppStatus(true);
    } else {
      const detail = job.error || job.log || '未知错误';
      toast('真机录制失败：' + String(detail).slice(0, 160), 'error');
      await refreshRwppStatus(true);
    }
  } catch (error) {
    const elapsed = (Date.now() - rwpp.startedAt) / 1000;
    if (error && error.status === 503) {
      stopRwppPolling();
      setRwppRecording(false);
      handleRwppError(error);
      await refreshRwppStatus(true);
    } else if (elapsed > RWPP_MAX_SECONDS) {
      stopRwppPolling();
      setRwppRecording(false);
      toast('真机录制超时或连接中断', 'error');
    }
  } finally {
    rwpp.polling = false;
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
    const simMode = meta.sim_mode === 'terrain' ? 'terrain' : 'arena';
    const visionRadius = numOrNull(meta.vision_radius);
    sandbox.simMode = simMode;
    sandbox.visionRadius = visionRadius === null ? 260 : visionRadius;
    sandbox.fog = meta.fog === true;
    sandbox.fogAvailable = simMode === 'terrain' || sandbox.fog === true || visionRadius !== null;
    sandbox.fogEnabled = sandbox.fog;
    sandbox.itemsEnabled = true;
    sandbox.mapLocked = simMode === 'terrain';
    sandbox.cameraRegion = null;
    const replayMapId = meta.map_id || (meta.map && meta.map.id) || '';
    if (els.sandboxDemoChip) els.sandboxDemoChip.hidden = state.mode !== 'demo';
    if (replayMapId) {
      if (!state.maps.ready) await loadMaps();
      if (state.maps.available) {
        await selectMap(replayMapId, { silent: true });
      } else {
        syncMapSelect();
        syncMapStatus();
      }
    } else {
      syncMapSelect();
      syncMapStatus();
    }
    syncMapToggles();
    if (sandbox.simMode === 'terrain' && !sandbox.mapDetail) {
      toast('该地形回放缺少地图资源，暂按抽象网格回退渲染', 'error');
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
  const simMode = meta.sim_mode === 'terrain' ? 'terrain' : 'arena';
  const prevSim = sandbox.simMode;
  sandbox.simMode = simMode;
  const visionRadius = numOrNull(meta.vision_radius);
  if (visionRadius !== null) sandbox.visionRadius = visionRadius;
  sandbox.fog = meta.fog === true;
  sandbox.fogAvailable = simMode === 'terrain' || sandbox.fog === true || visionRadius !== null;
  if (!sandbox.fogAvailable) sandbox.fogEnabled = false;
  sandbox.mapLocked = simMode === 'terrain';
  sandbox.cameraRegion = null;
  const mapId = meta.map_id === undefined || meta.map_id === null ? null : String(meta.map_id);
  const applyMap = async () => {
    if (mapId && mapId !== sandbox.mapId) {
      if (!state.maps.ready) await loadMaps();
      if (state.maps.available) await selectMap(mapId, { silent: true });
    } else if (sandbox.mapDetail) {
      await refreshTerrainView();
    }
    syncMapToggles();
    if (prevSim !== simMode && sandbox.mapDetail) await refreshTerrainView();
  };
  void applyMap().catch(() => {});
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
  const simMode = (els.inferSim && els.inferSim.value) === 'terrain' ? 'terrain' : 'arena';
  const selectedMapId = state.sandbox.mapId || '';
  const mapId = simMode === 'terrain' ? selectedMapId : '';
  if (simMode === 'terrain' && !mapId) {
    toast('实机地形模拟需要先选择地图（沙盘顶部「地图」下拉）', 'error');
    return;
  }
  const payload = {
    checkpoint: checkpoint,
    episodes: clampInt(els.episodesInput && els.episodesInput.value, 1, 50, 1),
    opponent: (els.inferOpponent && els.inferOpponent.value) || 'scripted',
    seed: clampInt(els.inferSeed && els.inferSeed.value, 0, 999999, 11),
    mode: (els.inferMode && els.inferMode.value) || 'greedy',
    max_decisions: 400,
    map_id: mapId,
    sim_mode: simMode,
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
    sandbox.simMode = simMode;
    sandbox.mapLocked = simMode === 'terrain';
    sandbox.cameraRegion = null;
    if (simMode === 'terrain') {
      sandbox.fogAvailable = true;
      if (simMode === 'terrain' && !sandbox.mapDetail && mapId && state.maps.available) {
        await selectMap(mapId, { silent: true });
      } else if (sandbox.mapDetail) {
        await refreshTerrainView();
      }
    } else if (sandbox.mapDetail) {
      await refreshTerrainView();
    }
    syncMapToggles();
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
    const counts = {};
    for (const item of results) {
      if (!item || !item.result) continue;
      const label = resultMeta(item.result).label;
      counts[label] = (counts[label] || 0) + 1;
    }
    const resultText = Object.keys(counts).length
      ? ' · ' + Object.keys(counts).map((label) => label + ' ' + counts[label]).join(' / ')
      : '';
    toast('推理完成 · ' + results.length + ' 局' + resultText + '，回放已入库', 'ok');
    if (els.inferHint) els.inferHint.textContent = '推理完成：' + results.length + ' 局' + resultText + '，回放已写入回放库。';
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
    state.sandbox.terrainView = previous.terrainView;
    state.sandbox.viewMode = previous.viewMode;
    state.sandbox.terrainLoading = false;
    state.sandbox.simMode = previous.simMode;
    state.sandbox.visionRadius = previous.visionRadius;
    state.sandbox.fog = previous.fog;
    state.sandbox.fogAvailable = previous.fogAvailable;
    state.sandbox.fogEnabled = previous.fogEnabled;
    state.sandbox.itemsEnabled = previous.itemsEnabled;
    state.sandbox.mapLocked = previous.mapLocked;
    state.sandbox.cameraRegion = null;
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
      els.frameSub.textContent = 't=' + fmtNum(frame.time, 1) + 's · ' + sandboxSpeedLabel(sandbox)
        + ' · 第 ' + (frame.episode + 1) + '/' + Math.max(1, sandbox.episodes || 1) + ' 局'
        + (result ? ' · ' + result.label : '');
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

const spriteCache = {
  tank: null,
  turret: null,
  builder: null,
  extractor: null,
  factory: null,
  turret_base: null,
  turret_top: null,
  command_center: null,
  pending: null,
  failed: false,
};
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
    const entries = [
      ['tank', units.tank || 'units/tanks/tank.png'],
      ['turret', units.tank_turret || 'units/tanks/tank_turret.png'],
      ['builder', units.builder || 'units/original/builder.png'],
      ['extractor', units.extractor || 'units/original/extractor.png'],
      ['factory', units.factory || 'units/original/land_factory.png'],
      ['turret_base', units.turret_base || 'units/original/turret_base.png'],
      ['turret_top', units.turret_top || 'units/original/turret_top.png'],
      ['command_center', units.command_center || 'units/original/base.png'],
    ];
    await Promise.all(
      entries.map(async ([key, rel]) => {
        try {
          spriteCache[key] = await loadImage(Api.assetUrl(rel));
        } catch (err) {
          // 单个贴图缺失不打断流程
        }
      })
    );
    spriteCache.failed = !spriteCache.tank;
    return true;
  })();
  return spriteCache.pending;
}

async function loadMaps(force) {
  if (state.maps.ready && state.maps.available && !force) return;
  state.maps.attempts += 1;
  if (!force && state.maps.attempts > 10) return;
  try {
    const payload = await Api.maps();
    const list = payload && Array.isArray(payload.maps) ? payload.maps : [];
    state.maps.available = !!(payload && payload.available);
    state.maps.list = list.filter((item) => item && !item.broken);
    state.maps.units = (payload && payload.units) || {};
    state.maps.error = null;
    state.maps.ready = true;
  } catch (error) {
    state.maps.available = false;
    state.maps.list = [];
    state.maps.error = error.message || String(error);
  }
  renderMapOptions();
  renderFeMapControls();
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
  syncMapSelect();
  syncMapToggles();
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
    const hasError = !!state.maps.error;
    setMapStatus(
      hasError
        ? '正在连接并读取地图列表…（点击可重试）'
        : (state.maps.ready ? '未安装游戏资源缓存：请运行 web/scripts/prepare_assets.ps1' : '正在读取地图列表…'),
      hasError ? 'warn' : (state.maps.ready ? 'bad' : '')
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
  const modeLabel = (sandbox.source === 'infer' || sandbox.checkpoint === 'rwpp-live')
    ? '实机'
    : (sandbox.simMode === 'terrain' ? '地形模拟' : '竞技场');
  if (sandbox.mapDetail) {
    let viewLabel;
    if (sandbox.simMode === 'terrain') {
      viewLabel = sandbox.viewMode === 'full' ? '全场' : '跟随交战区域';
    } else {
      viewLabel = sandbox.viewMode === 'full' ? '全场' : '交战跟随';
    }
    const extras = [];
    if (sandbox.fogEnabled) extras.push('迷雾');
    if (sandbox.mapDetail && sandbox.itemsEnabled) extras.push('道具层');
    setMapStatus(
      modeLabel + ' · ' + (sandbox.mapDetail.name || '地图') + ' · ' + viewLabel
        + (extras.length ? ' · ' + extras.join(' · ') : '') + ' · 地形仅视觉',
      'ok'
    );
    return;
  }
  setMapStatus(modeLabel + ' · 抽象渲染');
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

function wholeMapRegion(detail) {
  return {
    x: 0,
    y: 0,
    width: Math.max(1, num(detail.width_px, 1000)),
    height: Math.max(1, num(detail.height_px, 1000)),
  };
}

function renderTerrain(detail, region, scale, images, options) {
  const opts = options || {};
  const layerNames = Array.isArray(opts.layers)
    ? opts.layers.map((name) => String(name).toLowerCase())
    : null;
  const tileW = num(detail.tile_width, 20);
  const tileH = num(detail.tile_height, 20);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.ceil(region.width * scale));
  canvas.height = Math.max(1, Math.ceil(region.height * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  ctx.imageSmoothingEnabled = scale < 1;
  if (opts.background !== false) {
    ctx.fillStyle = '#0a141f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  const tilesets = (detail.tilesets || []).slice().sort((left, right) => right.firstgid - left.firstgid);
  const columns = Math.max(1, num(detail.width, 1));
  const rows = Math.max(1, num(detail.height, 1));
  const tileX0 = Math.max(0, Math.floor(region.x / tileW));
  const tileY0 = Math.max(0, Math.floor(region.y / tileH));
  const tileX1 = Math.min(columns - 1, Math.ceil((region.x + region.width) / tileW));
  const tileY1 = Math.min(rows - 1, Math.ceil((region.y + region.height) / tileH));

  for (const layer of detail.layers || []) {
    if (layerNames) {
      const name = String((layer && layer.name) || '').toLowerCase();
      if (layerNames.indexOf(name) < 0) continue;
    }
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
  if (!entry.cropGround) {
    entry.cropGround = renderTerrain(detail, crop, 1, images, { layers: ['Ground'] });
  }
  if (!entry.cropItems) {
    entry.cropItems = renderTerrain(detail, crop, 1, images, { layers: ['Items', 'Units'], background: false });
  }
  return entry;
}

function ensureFullTerrain(entry, detail) {
  if (entry.fullGround && entry.fullItems) {
    return { ground: entry.fullGround, items: entry.fullItems, scale: entry.fullScale };
  }
  const width = Math.max(1, num(detail.width_px, 1000));
  const height = Math.max(1, num(detail.height_px, 1000));
  const scale = Math.min(1, 2048 / Math.max(width, height));
  entry.fullGround = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, { layers: ['Ground'] });
  entry.fullItems = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, {
    layers: ['Items', 'Units'],
    background: false,
  });
  entry.fullScale = scale;
  return { ground: entry.fullGround, items: entry.fullItems, scale: scale };
}

function ensureTerrainMap(entry, detail) {
  if (entry.mapGround && entry.mapItems) {
    return { ground: entry.mapGround, items: entry.mapItems, scale: entry.mapScale };
  }
  const width = Math.max(1, num(detail.width_px, 1000));
  const height = Math.max(1, num(detail.height_px, 1000));
  const scale = Math.min(1, 3000 / Math.max(width, height));
  entry.mapGround = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, { layers: ['Ground'] });
  entry.mapItems = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, {
    layers: ['Items', 'Units'],
    background: false,
  });
  entry.mapScale = scale;
  return { ground: entry.mapGround, items: entry.mapItems, scale: scale };
}

function updateTerrainView(sandbox) {
  const detail = sandbox.mapDetail;
  const entry = sandbox.terrainEntry;
  if (!detail || !entry) {
    sandbox.terrainView = null;
    return;
  }
  if (sandbox.simMode === 'terrain') {
    const map = ensureTerrainMap(entry, detail);
    sandbox.terrainView = {
      ground: map.ground,
      items: map.items,
      region: wholeMapRegion(detail),
      scale: map.scale,
    };
    return;
  }
  if (sandbox.viewMode === 'full') {
    const full = ensureFullTerrain(entry, detail);
    sandbox.terrainView = {
      ground: full.ground,
      items: full.items,
      region: wholeMapRegion(detail),
      scale: full.scale,
    };
    return;
  }
  const crop = mapCrop(detail);
  sandbox.terrainView = {
    ground: entry.cropGround,
    items: entry.cropItems,
    region: crop,
    scale: 1,
  };
}

async function refreshTerrainView() {
  const sandbox = state.sandbox;
  if (!sandbox.mapDetail || !sandbox.terrainEntry) {
    sandbox.terrainView = null;
    syncMapStatus();
    return;
  }
  const entry = sandbox.terrainEntry;
  const heavy = (sandbox.simMode === 'terrain' && !entry.mapGround)
    || (sandbox.simMode !== 'terrain' && sandbox.viewMode === 'full' && !entry.fullGround);
  if (heavy) {
    setMapStatus('地形渲染中…');
    await new Promise((resolve) => setTimeout(resolve, 16));
  }
  updateTerrainView(sandbox);
  sandbox.cameraRegion = null;
  syncMapStatus();
  syncMapSelect();
  syncMapToggles();
  drawSandbox();
}

function computeFollowRegion(sandbox, detail, units, screenW, screenH) {
  const mapWidth = Math.max(1, num(detail.width_px, 1000));
  const mapHeight = Math.max(1, num(detail.height_px, 1000));
  const canvasW = Math.max(1, num(screenW, 900));
  const canvasH = Math.max(1, num(screenH, 600));
  const aspect = canvasW / canvasH;
  const valid = (units || []).filter((unit) => Number.isFinite(unit.x) && Number.isFinite(unit.y));
  const own = valid.filter((unit) => unit.team === 'own');
  const boundsOf = (list) => {
    if (!list.length) return null;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const unit of list) {
      minX = Math.min(minX, unit.x);
      minY = Math.min(minY, unit.y);
      maxX = Math.max(maxX, unit.x);
      maxY = Math.max(maxY, unit.y);
    }
    return {
      minX: minX,
      minY: minY,
      maxX: maxX,
      maxY: maxY,
      width: Math.max(1, maxX - minX),
      height: Math.max(1, maxY - minY),
    };
  };
  // 单位可见优先：跟随双方单位包围盒；窗口宁可 letterbox 也绝不裁掉坦克。
  const focus = valid.length ? valid : own;
  const focusBounds = boundsOf(focus);
  let target;
  if (!focusBounds) {
    const fallbackW = Math.min(mapWidth, 1400);
    const fallbackH = Math.min(mapHeight, fallbackW / aspect);
    target = {
      x: (mapWidth - fallbackW) / 2,
      y: (mapHeight - fallbackH) / 2,
      width: fallbackW,
      height: fallbackH,
    };
  } else {
    const pad = 150;
    const margin = 300;
    let needW = Math.max(760, focusBounds.width + margin * 2, focusBounds.width + pad * 2);
    let needH = Math.max(760, focusBounds.height + margin * 2, focusBounds.height + pad * 2);
    // 按画布宽高比做“只扩不缩”的 cover 扩展（先保证单位装得下）
    let winW;
    let winH;
    if (needW / needH >= aspect) {
      winW = needW;
      winH = winW / aspect;
    } else {
      winH = needH;
      winW = winH * aspect;
    }
    // 窗口上限 1400，但绝不低于我方包围盒 + 留白：单位可见优先于填满画布
    const maxWinH = Math.max(1400, focusBounds.height + pad * 2);
    const maxWinW = Math.max(1400 * aspect, focusBounds.width + pad * 2);
    if (winH > maxWinH) {
      winH = maxWinH;
      winW = winH * aspect;
    }
    if (winW > maxWinW) {
      winW = maxWinW;
      winH = winW / aspect;
    }
    winW = Math.max(winW, focusBounds.width + pad * 2);
    winH = Math.max(winH, focusBounds.height + pad * 2);
    // 只有仍能装下单位时才缩到地图内；装不下则保留大窗口，由 mapCamera contain 绘制（允许留边）
    let regionW = winW;
    let regionH = winH;
    if (regionW > mapWidth || regionH > mapHeight) {
      const fit = Math.min(mapWidth / regionW, mapHeight / regionH);
      if (regionW * fit >= focusBounds.width + pad * 2 && regionH * fit >= focusBounds.height + pad * 2) {
        regionW *= fit;
        regionH *= fit;
      }
    }
    const centerX = (focusBounds.minX + focusBounds.maxX) / 2;
    const centerY = (focusBounds.minY + focusBounds.maxY) / 2;
    let x;
    let y;
    if (regionW <= mapWidth) {
      x = clamp(centerX - regionW / 2, focusBounds.maxX + pad - regionW, focusBounds.minX - pad);
      x = clamp(x, 0, mapWidth - regionW);
    } else {
      x = (mapWidth - regionW) / 2;
    }
    if (regionH <= mapHeight) {
      y = clamp(centerY - regionH / 2, focusBounds.maxY + pad - regionH, focusBounds.minY - pad);
      y = clamp(y, 0, mapHeight - regionH);
    } else {
      y = (mapHeight - regionH) / 2;
    }
    target = { x: x, y: y, width: regionW, height: regionH };
  }
  const previous = sandbox.cameraRegion;
  if (!previous) {
    sandbox.cameraRegion = { x: target.x, y: target.y, width: target.width, height: target.height };
    return sandbox.cameraRegion;
  }
  const smoothing = 0.2;
  const next = {
    x: previous.x + (target.x - previous.x) * smoothing,
    y: previous.y + (target.y - previous.y) * smoothing,
    width: previous.width + (target.width - previous.width) * smoothing,
    height: previous.height + (target.height - previous.height) * smoothing,
  };
  if (Math.abs(next.x - target.x) < 0.5 && Math.abs(next.y - target.y) < 0.5) {
    next.x = target.x;
    next.y = target.y;
  }
  if (Math.abs(next.width - target.width) < 0.5 && Math.abs(next.height - target.height) < 0.5) {
    next.width = target.width;
    next.height = target.height;
  }
  if (focusBounds) {
    const safePad = 20;
    const containsFocus =
      next.x <= focusBounds.minX - safePad &&
      next.x + next.width >= focusBounds.maxX + safePad &&
      next.y <= focusBounds.minY - safePad &&
      next.y + next.height >= focusBounds.maxY + safePad;
    if (!containsFocus) {
      sandbox.cameraRegion = { x: target.x, y: target.y, width: target.width, height: target.height };
      return sandbox.cameraRegion;
    }
  }
  sandbox.cameraRegion = next;
  return sandbox.cameraRegion;
}

function isUnitVisible(sandbox, unit, ownUnits) {
  if (!sandbox.fogEnabled) return true;
  if (unit.team === 'own') return true;
  const radius = num(sandbox.visionRadius, 260);
  for (const own of ownUnits) {
    if (Math.hypot(unit.x - own.x, unit.y - own.y) <= radius) return true;
  }
  return false;
}

async function selectMap(mapId, options) {
  const sandbox = state.sandbox;
  const opts = options || {};
  const nextId = mapId || '';
  if (!nextId) {
    sandbox.mapId = '';
    sandbox.mapDetail = null;
    sandbox.terrainView = null;
    sandbox.terrainEntry = null;
    sandbox.terrainLoading = false;
    sandbox.terrainError = null;
    sandbox.viewMode = 'battle';
    sandbox.cameraRegion = null;
    if (els.mapSelect) els.mapSelect.value = '';
    if (els.viewSelect) {
      els.viewSelect.value = 'battle';
      els.viewSelect.disabled = true;
    }
    syncMapStatus();
    syncMapToggles();
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
  sandbox.terrainView = null;
  sandbox.terrainEntry = null;
  sandbox.terrainError = null;
  sandbox.terrainLoading = true;
  sandbox.cameraRegion = null;
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
    sandbox.terrainLoading = false;
    await refreshTerrainView();
    if (!opts.silent) toast('已加载地图 · ' + (detail.name || nextId), 'ok');
  } catch (error) {
    if (sandbox.mapLoadToken !== token) return;
    sandbox.terrainLoading = false;
    sandbox.terrainError = error.message || String(error);
    sandbox.mapDetail = null;
    sandbox.mapId = '';
    sandbox.terrainEntry = null;
    sandbox.terrainView = null;
    if (els.mapSelect) els.mapSelect.value = '';
    if (els.viewSelect) els.viewSelect.disabled = true;
    syncMapStatus();
    syncMapToggles();
    drawSandbox();
    toast('地图加载失败，已回退抽象渲染：' + error.message, 'error');
  }
}

async function applyViewMode(mode) {
  const sandbox = state.sandbox;
  sandbox.viewMode = mode === 'full' ? 'full' : 'battle';
  if (els.viewSelect) els.viewSelect.value = sandbox.viewMode;
  sandbox.cameraRegion = null;
  if (sandbox.mapDetail) {
    await refreshTerrainView();
  } else {
    syncMapStatus();
    drawSandbox();
  }
}

function syncMapSelect() {
  if (els.mapSelect) {
    const id = state.sandbox.mapId || '';
    const exists = [...els.mapSelect.options].some((option) => option.value === id);
    els.mapSelect.value = exists ? id : '';
    els.mapSelect.disabled = !!state.sandbox.mapLocked;
    els.mapSelect.title = state.sandbox.mapLocked
      ? '该回放使用该地图（地形模拟固定地图）'
      : '选择用于渲染观战的实机地图（来自游戏资源缓存）';
  }
  if (els.viewSelect) {
    els.viewSelect.disabled = !state.sandbox.mapDetail;
    els.viewSelect.value = state.sandbox.viewMode || 'battle';
  }
  if (els.inferSim) {
    els.inferSim.value = state.sandbox.simMode === 'terrain' ? 'terrain' : 'arena';
  }
}

function syncMapToggles() {
  const sandbox = state.sandbox;
  if (els.fogToggle) {
    els.fogToggle.disabled = !sandbox.fogAvailable;
    els.fogToggle.setAttribute('aria-pressed', sandbox.fogEnabled ? 'true' : 'false');
    els.fogToggle.textContent = sandbox.fogEnabled ? '开' : '关';
  }
  if (els.itemsToggle) {
    els.itemsToggle.disabled = !sandbox.mapDetail;
    els.itemsToggle.setAttribute('aria-pressed', sandbox.itemsEnabled ? 'true' : 'false');
    els.itemsToggle.textContent = sandbox.itemsEnabled ? '开' : '关';
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
  const baseSize = Math.max(8, num(options && options.size, 30));
  const hpFraction = clamp(unit.hp / Math.max(1, unit.maxHp), 0, 1);
  const angles = unitAngleCache.get(unit.id) || { heading: -Math.PI / 2, turret: -Math.PI / 2 };

  // 实体类别推断
  const unitType = String(unit.unit_type || '').toLowerCase();
  const maxHp = num(unit.maxHp || unit.max_hp, 0);
  const isBuilding = !!(unit.is_building || unit.entity_type_id === 2 || unitType === 'building' || unitType === 'command_center' || unitType === 'factory' || unitType === 'extractor' || unitType === 'turret');
  const isBuilder = !!(unit.is_builder || unitType === 'builder' || unitType === 'combat_engineer');
  const isFactory = !!(unit.is_factory || unitType === 'factory' || unitType === 'mech_factory');
  const isExtractor = !!(unitType === 'extractor');
  const isCommandCenter = !!(unitType === 'command_center' || (isBuilding && maxHp >= 2500.0));
  const isTurret = !!(unitType === 'turret' || (isBuilding && !isCommandCenter && !isFactory && !isExtractor && maxHp <= 1000.0 && maxHp > 0));

  // 差异化尺寸
  let size = baseSize;
  if (isCommandCenter) size = baseSize * 2.2;
  else if (isFactory) size = baseSize * 1.8;
  else if (isExtractor) size = baseSize * 1.35;
  else if (isBuilding) size = baseSize * 1.5;
  else if (isBuilder) size = baseSize * 0.95;

  // 阵营底盘指示（建筑用圆角矩形，移动单位用椭圆）
  ctx.save();
  ctx.beginPath();
  if (isBuilding) {
    const bw = size * 0.85;
    const bh = size * 0.85;
    ctx.rect(x - bw / 2, y - bh / 2, bw, bh);
  } else {
    ctx.ellipse(x, y + size * 0.2, size * 0.42, size * 0.26, 0, 0, Math.PI * 2);
  }
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

  // --- 分类绘制贴图 ---
  let rendered = false;

  if (isCommandCenter && spriteCache.command_center) {
    ctx.save();
    ctx.translate(x, y);
    const img = spriteCache.command_center;
    // 原版指挥中心主基地：单帧 53x68 (整图 212x68 为 4 帧动图，切第 1 帧)
    if (img.width >= 200) {
      const frameW = 53;
      const frameH = 68;
      const drawW = size * (frameW / frameH);
      ctx.drawImage(img, 0, 0, frameW, frameH, -drawW / 2, -size / 2, drawW, size);
    } else {
      ctx.drawImage(img, -size / 2, -size / 2, size, size);
    }
    ctx.restore();
    rendered = true;
  } else if (isFactory && spriteCache.factory) {
    ctx.save();
    ctx.translate(x, y);
    const img = spriteCache.factory;
    const drawW = size * (img.width / img.height);
    ctx.drawImage(img, -drawW / 2, -size / 2, drawW, size);
    ctx.restore();
    rendered = true;
  } else if (isExtractor && spriteCache.extractor) {
    ctx.save();
    ctx.translate(x, y);
    const img = spriteCache.extractor;
    if (img.width >= 100) {
      const frameW = 37;
      const frameH = 59;
      const drawW = size * (frameW / frameH);
      ctx.drawImage(img, 0, 0, frameW, frameH, -drawW / 2, -size / 2, drawW, size);
    } else {
      ctx.drawImage(img, -size / 2, -size / 2, size, size);
    }
    ctx.restore();
    rendered = true;
  } else if (isTurret && spriteCache.turret_base) {
    ctx.save();
    ctx.translate(x, y);
    ctx.drawImage(spriteCache.turret_base, -size * 0.45, -size * 0.45, size * 0.9, size * 0.9);
    if (spriteCache.turret_top) {
      ctx.rotate(angles.turret + Math.PI / 2);
      ctx.drawImage(spriteCache.turret_top, -size * 0.35, -size * 0.35, size * 0.7, size * 0.7);
    }
    ctx.restore();
    rendered = true;
  } else if (isBuilder && spriteCache.builder) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angles.heading + Math.PI / 2);
    // 原版黄色建造车，20x20
    const bSize = size * 0.85;
    ctx.drawImage(spriteCache.builder, -bSize / 2, -bSize / 2, bSize, bSize);
    ctx.restore();
    rendered = true;
  } else if (!isBuilding && spriteCache.tank) {
    // 默认战斗坦克
    const bodyWidth = size * (UNIT_TILE_WIDTH / UNIT_TILE_HEIGHT);
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
    rendered = true;
  }

  // 保底矢量绘制（没有对应贴图时，用外观几何明确区分，绝不全画成坦克）
  if (!rendered) {
    ctx.save();
    if (isBuilding) {
      ctx.beginPath();
      ctx.rect(x - size * 0.4, y - size * 0.4, size * 0.8, size * 0.8);
      ctx.fillStyle = own ? '#1b4b8a' : '#8a1b24';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = color;
      ctx.stroke();
      // 内部科技标示
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.fillRect(x - size * 0.2, y - size * 0.2, size * 0.4, size * 0.4);
    } else if (isBuilder) {
      // 建造者：六边形工程车
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const rad = (i * Math.PI) / 3;
        const px = x + Math.cos(rad) * size * 0.38;
        const py = y + Math.sin(rad) * size * 0.38;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = own ? '#2f8fbf' : '#d97724';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(x, y, size * 0.32, 0, Math.PI * 2);
      ctx.fillStyle = own ? '#2f6fbf' : '#b5303c';
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
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
  const terrainSim = sandbox.simMode === 'terrain';
  const frames = sandbox.frames;

  let frame = null;
  let nextFrame = null;
  let units = [];
  if (frames.length) {
    const maxIndex = frames.length - 1;
    const playback = clamp(sandbox.playback, 0, maxIndex);
    const index = Math.floor(playback);
    const nextIndex = Math.min(index + 1, maxIndex);
    const fraction = playback - index;
    frame = frames[index];
    nextFrame = frames[nextIndex];
    units = interpolateUnits(frame, nextFrame, fraction);
    updateUnitAngles(frame, index > 0 ? frames[index - 1] : null);
  }

  let region;
  if (detail) {
    if (sandbox.viewMode === 'full') {
      region = wholeMapRegion(detail);
    } else if (terrainSim) {
      region = computeFollowRegion(sandbox, detail, units, width, height);
    } else {
      region = mapCrop(detail);
    }
  } else if (terrainSim && sandbox.map && sandbox.map.width > 1000) {
    region = {
      x: 0,
      y: 0,
      width: Math.max(1, num(sandbox.map.width, 1000)),
      height: Math.max(1, num(sandbox.map.height, 1000)),
    };
  } else {
    region = { x: 0, y: 0, width: 1000, height: 1000 };
  }
  const mapMode = !!(detail && sandbox.terrainView);
  const cam = mapCamera(width, height, region, mapMode ? 0 : 20);
  const offsetX = cam.ox + cam.region.x * cam.scale;
  const offsetY = cam.oy + cam.region.y * cam.scale;
  const drawnWidth = cam.region.width * cam.scale;
  const drawnHeight = cam.region.height * cam.scale;
  const mapX = (x) => {
    if (detail) {
      if (terrainSim) return cam.ox + num(x, 0) * cam.scale;
      return cam.ox + engineToMapPixel(x, 0, detail).x * cam.scale;
    }
    return cam.ox + num(x, 0) * cam.scale;
  };
  const mapY = (y) => {
    if (detail) {
      if (terrainSim) return cam.oy + num(y, 0) * cam.scale;
      return cam.oy + engineToMapPixel(0, y, detail).y * cam.scale;
    }
    return cam.oy + num(y, 0) * cam.scale;
  };
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
    simMode: sandbox.simMode,
    corners: { x: offsetX, y: offsetY, width: drawnWidth, height: drawnHeight },
    project: (x, y) => ({ x: mapX(x), y: mapY(y) }),
  };

  ctx.save();
  ctx.beginPath();
  ctx.rect(offsetX, offsetY, drawnWidth, drawnHeight);
  ctx.clip();

  if (mapMode) {
    const view = sandbox.terrainView;
    const sx = (cam.region.x - view.region.x) * view.scale;
    const sy = (cam.region.y - view.region.y) * view.scale;
    const sw = Math.max(1, cam.region.width * view.scale);
    const sh = Math.max(1, cam.region.height * view.scale);
    ctx.imageSmoothingEnabled = true;
    try {
      ctx.drawImage(view.ground, sx, sy, sw, sh, offsetX, offsetY, drawnWidth, drawnHeight);
      if (sandbox.itemsEnabled && view.items) {
        ctx.drawImage(view.items, sx, sy, sw, sh, offsetX, offsetY, drawnWidth, drawnHeight);
      }
    } catch (error) {
      // 画布绘制失败时保持暗色底，不阻塞单位渲染
    }

    const shade = ctx.createRadialGradient(
      width / 2, height / 2, Math.min(width, height) * 0.18,
      width / 2, height / 2, Math.max(width, height) * 0.72
    );
    shade.addColorStop(0, 'rgba(3, 8, 14, 0)');
    shade.addColorStop(1, 'rgba(3, 8, 14, 0.45)');
    ctx.fillStyle = shade;
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

    if (!terrainSim && detail && sandbox.viewMode === 'full') {
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
  } else if (detail || (terrainSim && sandbox.map && sandbox.map.width > 1000)) {
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
    const mapWidth = Math.max(1, region.width);
    const mapHeight = Math.max(1, region.height);
    const mapGradient = ctx.createRadialGradient(
      offsetX + drawnWidth / 2, offsetY + drawnHeight / 2, 20,
      offsetX + drawnWidth / 2, offsetY + drawnHeight / 2, Math.max(drawnWidth, drawnHeight) * 0.8
    );
    mapGradient.addColorStop(0, '#0a1727');
    mapGradient.addColorStop(1, '#050b13');
    ctx.fillStyle = mapGradient;
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

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

    const gridStep = Math.max(50, Math.round(Math.max(mapWidth, mapHeight) / 10 / 50) * 50);
    for (let gridX = 0; gridX <= mapWidth; gridX += gridStep) {
      const screenX = mapX(gridX);
      ctx.strokeStyle = gridX % (gridStep * 5) === 0 ? SANDBOX_COLORS.gridStrong : SANDBOX_COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(screenX, offsetY);
      ctx.lineTo(screenX, offsetY + drawnHeight);
      ctx.stroke();
    }
    for (let gridY = 0; gridY <= mapHeight; gridY += gridStep) {
      const screenY = mapY(gridY);
      ctx.strokeStyle = gridY % (gridStep * 5) === 0 ? SANDBOX_COLORS.gridStrong : SANDBOX_COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(offsetX, screenY);
      ctx.lineTo(offsetX + drawnWidth, screenY);
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.12)';
    ctx.setLineDash([6, 8]);
    ctx.beginPath();
    ctx.moveTo(offsetX + drawnWidth / 2, offsetY);
    ctx.lineTo(offsetX + drawnWidth / 2, offsetY + drawnHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    if (!terrainSim) {
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
  }

  if (!frames.length) {
    if (mapMode && sandbox.fogEnabled) {
      drawFogLayer(ctx, sandbox, units, mapX, mapY, cam, offsetX, offsetY, drawnWidth, drawnHeight, width, height);
    }
    ctx.restore();
    drawSandboxChrome(ctx, width, height, offsetX, offsetY, drawnWidth, drawnHeight);
    drawSandboxEmpty(ctx, width, height);
    return;
  }

  const ownUnits = units.filter((unit) => unit.team === 'own');
  const visibleUnits = units.filter((unit) => isUnitVisible(sandbox, unit, ownUnits));
  const hiddenEnemyIds = new Set(
    frame.units
      .filter((unit) => unit.team !== 'own' && !isUnitVisible(sandbox, unit, ownUnits))
      .map((unit) => unit.id)
  );

  if (mapMode && sandbox.fogEnabled) {
    drawFogLayer(ctx, sandbox, ownUnits, mapX, mapY, cam, offsetX, offsetY, drawnWidth, drawnHeight, width, height);
  }

  // 指令箭头（画在单位下层）
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, { x: unit.x, y: unit.y });
  for (const unit of visibleUnits) {
    if (unit.target === null || unit.target === undefined) continue;
    if (hiddenEnemyIds.has(unit.id) || hiddenEnemyIds.has(unit.target)) continue;
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
  const sorted = visibleUnits.slice().sort((left, right) => left.y - right.y);
  const leaders = new Set();
  const seenGroups = new Set();
  for (const unit of sorted) {
    if (unit.team !== 'own') continue;
    if (seenGroups.has(unit.group)) continue;
    seenGroups.add(unit.group);
    leaders.add(unit.id);
  }
  const time = performance.now() / 1000;
  // 铁锈战争真实世界尺寸：车身高 24px (30 * 0.8)，碰撞半径 11px
  // 随相机缩放 cam.scale 保持 1:1 物理比例，远景全场时保留保底尺寸确保可见
  const baseWorldSize = 24.0;
  const realSpriteSize = baseWorldSize * cam.scale;
  const minSpriteSize = sandbox.viewMode === 'full' ? 10 : 12;
  const spriteSize = Math.max(minSpriteSize, realSpriteSize);
  const baseRadius = 11.0;
  const unitRadius = Math.max(3.5, baseRadius * cam.scale);
  for (const unit of sorted) {
    if (mapMode && spriteCache.tank) {
      drawSandboxSpriteUnit(ctx, unit, mapX(unit.x), mapY(unit.y), unit.id === sandbox.selectedId, leaders.has(unit.id), time, { size: spriteSize });
    } else {
      drawSandboxUnit(ctx, unit, mapX, mapY, unit.id === sandbox.selectedId, leaders.has(unit.id), time, { radius: unitRadius });
    }
  }

  const eventVisible = sandbox.fogEnabled
    ? (event) => !hiddenEnemyIds.has(event.source) && !hiddenEnemyIds.has(event.target)
    : null;

  // 当前帧事件（暂停时静态显示，播放时由实时特效呈现）
  if (!sandbox.playing) {
    drawFrameEvents(ctx, frame, mapX, mapY, 1, eventVisible);
  }

  // 实时特效（弹道 / 爆炸，随帧推进生成）
  const now = performance.now();
  for (const effect of sandbox.effects) {
    const age = (now - effect.born) / effect.life;
    if (age >= 1) continue;
    if (sandbox.fogEnabled && (
      (effect.fromId !== undefined && hiddenEnemyIds.has(effect.fromId))
      || (effect.toId !== undefined && hiddenEnemyIds.has(effect.toId))
      || (effect.targetId !== undefined && hiddenEnemyIds.has(effect.targetId))
    )) continue;
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

const fogLayer = { canvas: null, ctx: null };

function ensureFogLayer(width, height) {
  if (!fogLayer.canvas) {
    fogLayer.canvas = document.createElement('canvas');
    fogLayer.ctx = fogLayer.canvas.getContext('2d');
  }
  const targetWidth = Math.max(1, Math.ceil(width));
  const targetHeight = Math.max(1, Math.ceil(height));
  if (fogLayer.canvas.width !== targetWidth || fogLayer.canvas.height !== targetHeight) {
    fogLayer.canvas.width = targetWidth;
    fogLayer.canvas.height = targetHeight;
  }
  return fogLayer;
}

function drawFogLayer(ctx, sandbox, ownUnits, mapX, mapY, cam, offsetX, offsetY, drawnWidth, drawnHeight, width, height) {
  const layer = ensureFogLayer(width, height);
  const fog = layer.ctx;
  if (!fog) return;
  fog.setTransform(1, 0, 0, 1, 0, 0);
  fog.clearRect(0, 0, width, height);
  fog.save();
  fog.beginPath();
  fog.rect(offsetX, offsetY, drawnWidth, drawnHeight);
  fog.clip();
  fog.fillStyle = 'rgba(2, 6, 12, 0.78)';
  fog.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

  const radius = Math.max(12, num(sandbox.visionRadius, 260) * cam.scale);
  fog.globalCompositeOperation = 'destination-out';
  for (const unit of ownUnits) {
    const x = mapX(unit.x);
    const y = mapY(unit.y);
    const gradient = fog.createRadialGradient(x, y, radius * 0.32, x, y, radius);
    gradient.addColorStop(0, 'rgba(0, 0, 0, 1)');
    gradient.addColorStop(0.62, 'rgba(0, 0, 0, 0.9)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    fog.fillStyle = gradient;
    fog.beginPath();
    fog.arc(x, y, radius, 0, Math.PI * 2);
    fog.fill();
  }
  fog.globalCompositeOperation = 'source-over';
  fog.restore();

  ctx.save();
  ctx.globalAlpha = 0.94;
  ctx.drawImage(layer.canvas, 0, 0, width, height);
  ctx.restore();
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
    state.mode === 'demo' ? '演示模式：点击「开始实机推理」播放内置演示对局' : '在右侧选择地图与时长，点击「开始实机推理」；或从下方回放库加载一局',
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
    ? '#' + (frame.episode + 1) + ' · tick ' + frame.tick + ' · t=' + fmtNum(frame.time, 1) + 's' + ' · ' + sandboxSpeedLabel(sandbox)
    : '等待推理数据';
  ctx.fillText(info, 14, 12);

  const realReplay = sandbox.checkpoint === 'rwpp-live';
  const modeLabel = (sandbox.source === 'infer' || realReplay)
    ? '实机'
    : (sandbox.simMode === 'terrain' ? '地形模拟' : '竞技场');
  const mapLabel = sandbox.mapDetail ? (sandbox.mapDetail.name || '地图') : '抽象竞技场';
  let viewLabel;
  if (sandbox.simMode === 'terrain') {
    viewLabel = sandbox.viewMode === 'full' ? '全场' : '跟随交战区域';
  } else {
    viewLabel = sandbox.viewMode === 'full' ? '全场' : '交战跟随';
  }
  ctx.save();
  ctx.font = '10px "Microsoft YaHei", "Segoe UI", sans-serif';
  ctx.fillStyle = sandbox.simMode === 'terrain' ? 'rgba(125, 227, 176, 0.95)' : 'rgba(124, 160, 190, 0.9)';
  const strategyLabel = sandboxStrategyLabel(sandbox);
  let metaLine = modeLabel + ' · ' + strategyLabel + mapLabel + ' · ' + viewLabel + ' · ' + (sandbox.frames ? sandbox.frames.length : 0) + ' 帧';
  if (sandbox.fogEnabled) metaLine += ' · 迷雾';
  ctx.fillText(metaLine, 14, 27);
  ctx.restore();

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

  // 右下角说明：道具层 / 迷雾
  ctx.textAlign = 'right';
  ctx.font = '9.5px "Microsoft YaHei", "Segoe UI", sans-serif';
  let noteY = height - 12;
  if (sandbox.mapDetail && sandbox.itemsEnabled) {
    ctx.fillStyle = 'rgba(150, 178, 205, 0.62)';
    ctx.fillText('建筑/资源点：地图道具层（静态，未模拟建造）', width - 14, noteY);
    noteY -= 13;
  }
  if (sandbox.fogEnabled) {
    ctx.fillStyle = 'rgba(150, 178, 205, 0.62)';
    ctx.fillText('迷雾：视觉层（按我方视野 ' + Math.round(num(sandbox.visionRadius, 260)) + 'px）', width - 14, noteY);
  }
  ctx.restore();
}

function drawSandboxUnit(ctx, unit, mapX, mapY, selected, leader, time, options) {
  const x = mapX(unit.x);
  const y = mapY(unit.y);
  const radius = Math.max(3.5, num(options && options.radius, 7.5));
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

function drawFrameEvents(ctx, frame, mapX, mapY, alpha, isVisible) {
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, { x: unit.x, y: unit.y });
  const phase = (performance.now() % 900) / 900;
  for (const event of frame.events) {
    if (isVisible && !isVisible(event)) continue;
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
      if (from && to) sandbox.effects.push({ kind: 'shot', from: from, to: to, fromId: event.source, toId: event.target, born: now, life: 340 });
    } else if (event.type === 'kill') {
      const target = positions.get(event.target);
      const x = target ? target.x : event.x;
      const y = target ? target.y : event.y;
      if (x !== undefined && y !== undefined && x !== null && y !== null) {
        sandbox.effects.push({ kind: 'kill', x: x, y: y, targetId: event.target, born: now, life: 780 });
      }
    }
  }
  if (sandbox.effects.length > 160) sandbox.effects.splice(0, sandbox.effects.length - 160);
}

function sandboxFrameStepSeconds(sandbox) {
  if (!sandbox) return 1 / BASE_FPS;
  const frames = sandbox.frames || [];
  const realGame = sandbox.source === 'infer' || sandbox.checkpoint === 'rwpp-live' || sandbox.simMode === 'terrain';
  if (!realGame) return 1 / BASE_FPS;
  if (frames.length >= 2) {
    const firstIndex = Math.max(0, frames.length - 12);
    const first = frames[firstIndex];
    const last = frames[frames.length - 1];
    const count = Math.max(1, frames.length - 1 - firstIndex);
    const tickDelta = num(last.tick, 0) - num(first.tick, 0);
    if (tickDelta > 0) {
      const step = (tickDelta / count) / 60;
      if (step > 0.001) {
        sandbox.frameStep = step;
        return step;
      }
    }
    const timeDelta = num(last.time, 0) - num(first.time, 0);
    if (timeDelta > 0) {
      const step = timeDelta / count;
      if (step > 0.001) {
        sandbox.frameStep = step;
        return step;
      }
    }
  }
  if (sandbox.source === 'infer' || sandbox.checkpoint === 'rwpp-live') {
    sandbox.frameStep = sandbox.frameStep || (1 / 6);
    return sandbox.frameStep;
  }
  return sandbox.frameStep || (1 / BASE_FPS);
}

function sandboxSpeedLabel(sandbox) {
  const speed = num(sandbox && sandbox.speed, 1);
  if (speed >= 30) return '最快';
  return (Math.round(speed * 10) / 10) + 'x';
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
  const stepSeconds = sandboxFrameStepSeconds(sandbox);
  const speedFactor = sandbox.speed >= 30 ? 30 : sandbox.speed;
  const framesPerSecond = speedFactor / stepSeconds;
  const next = Math.min(previous + dt * framesPerSecond, maxIndex);
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
  const resolved = name === 'train' ? 'train' : 'infer';
  if ((state.tab === 'sandbox' || state.tab === 'infer') && resolved !== 'infer') stopRecording();
  state.tab = resolved;
  const buttons = document.querySelectorAll('.tab-btn');
  for (const button of buttons) {
    const active = button.getAttribute('data-tab') === resolved;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  const panels = document.querySelectorAll('.tab-panel');
  for (const panel of panels) {
    const target = resolved === 'train' ? 'panel-train' : 'panel-infer';
    panel.classList.toggle('active', panel.id === target);
  }
  if (window.location.hash !== '#' + resolved) {
    try {
      window.history.replaceState(null, '', '#' + resolved);
    } catch (error) {
      window.location.hash = resolved;
    }
  }
  if (resolved === 'infer') {
    if (state.mode === 'demo') ensureSandboxContent();
    drawSandbox();
    refreshRwppStatus();
    loadReplays();
    void loadFeCheckpoints(true);
    feRenderInferDashboard();
    void feRestoreInferJobs();
  } else {
    refreshRwppStatus(true);
    feApplyTrainMode(state.fe.trainMode);
    feRenderPpoDashboard();
    void feRestorePpoJob();
    feRenderTrainDashboard(state.fe.trainJob);
    renderFeTrainArtifacts(state.fe.trainArtifacts);
    if (state.fe.trainStatus === 'idle' && state.rwpp.hostRunning > 0) {
      const hint = document.getElementById('fe-train-hint');
      if (hint) hint.textContent = '检测到宿主机有任务正在运行；若为训练任务，完成后回放会自动入库。';
    }
  }
}

function tabFromHash() {
  const hash = String(window.location.hash || '').replace('#', '');
  if (hash === 'train') return 'train';
  return 'infer';
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
  if (els.replayRefresh) {
    els.replayRefresh.addEventListener('click', () => {
      loadReplays();
      refreshRwppStatus(true);
      void loadFeCheckpoints(true);
    });
  }
  if (els.replayToggleBtn) {
    els.replayToggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleReplayCollapse();
    });
  }
  if (els.replayCardHead) {
    els.replayCardHead.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      toggleReplayCollapse();
    });
  }
  if (els.replayPrevBtn) {
    els.replayPrevBtn.addEventListener('click', () => {
      if (state.replayPagination.page > 1) {
        state.replayPagination.page -= 1;
        renderReplays();
      }
    });
  }
  if (els.replayNextBtn) {
    els.replayNextBtn.addEventListener('click', () => {
      const totalPages = Math.ceil((state.replays || []).length / (state.replayPagination.pageSize || 8));
      if (state.replayPagination.page < totalPages) {
        state.replayPagination.page += 1;
        renderReplays();
      }
    });
  }
  if (els.replayPageSize) {
    els.replayPageSize.addEventListener('change', () => {
      state.replayPagination.pageSize = Number(els.replayPageSize.value) || 8;
      state.replayPagination.page = 1;
      renderReplays();
    });
  }
  if (els.rwppRecordBtn) els.rwppRecordBtn.addEventListener('click', startRwppRecord);
  document.querySelectorAll('.rwpp-status').forEach((badge) => {
    badge.style.cursor = 'pointer';
    badge.addEventListener('click', async () => {
      toast('正在重新检测宿主机真机服务 (8300)…', 'info');
      await refreshRwppStatus(true);
      if (state.rwpp.available) {
        toast('真机服务就绪 (8300) · 空闲槽位: ' + (state.rwpp.freeSlots ?? '—'), 'ok');
      } else {
        toast('真机服务未就绪：请在宿主机终端执行 python scripts/rwpp_real_service.py', 'error');
      }
    });
  });
  if (els.feStartBtn) els.feStartBtn.addEventListener('click', feStartInfer);
  if (els.feTrainStart) els.feTrainStart.addEventListener('click', feStartTrain);
  if (els.feTrainStop) els.feTrainStop.addEventListener('click', feStopTrain);
  if (els.feTrainRefresh) {
    els.feTrainRefresh.addEventListener('click', () => {
      feRefreshTrainArtifacts();
      void loadFeCheckpoints(true);
    });
  }
  for (const button of document.querySelectorAll('[data-train-mode-btn]')) {
    button.addEventListener('click', () => feApplyTrainMode(button.getAttribute('data-train-mode-btn')));
  }
  if (els.fePpoStart) els.fePpoStart.addEventListener('click', feStartPpo);
  if (els.fePpoStop) els.fePpoStop.addEventListener('click', feStopPpo);
  if (els.fePolicy) els.fePolicy.addEventListener('change', feSyncInferPolicyUi);
  if (els.feCheckpointRefresh) {
    els.feCheckpointRefresh.addEventListener('click', async () => {
      toast('正在重新扫描 Checkpoint 权重目录…', 'info');
      await loadFeCheckpoints(true);
      const count = (state.fe.checkpointList || []).length;
      toast('已刷新 Checkpoint 列表，共发现 ' + count + ' 个模型', 'ok');
    });
  }
  if (els.feTrainPolicy) els.feTrainPolicy.addEventListener('change', feSyncTrainPolicyUi);
  if (els.fePpoMap) els.fePpoMap.addEventListener('change', feSyncPpoConfigUi);
  if (els.fePpoInitMode) els.fePpoInitMode.addEventListener('change', feSyncPpoConfigUi);
  if (els.fePpoUseLlm) {
    els.fePpoUseLlm.addEventListener('change', () => {
      if (els.fePpoLlmNote) els.fePpoLlmNote.hidden = !els.fePpoUseLlm.checked;
      feRenderPpoDashboard();
    });
  }
  if (els.demoRetry) els.demoRetry.addEventListener('click', retryConnection);
  if (els.mapStatus) {
    els.mapStatus.style.cursor = 'pointer';
    els.mapStatus.addEventListener('click', () => {
      toast('正在重新加载地图资源…', 'info');
      void loadMaps(true);
    });
  }
  if (els.recordBtn) els.recordBtn.addEventListener('click', toggleRecording);
  if (els.mapSelect) {
    els.mapSelect.addEventListener('change', () => {
      if (state.sandbox.simMode !== 'infer') {
        state.sandbox.simMode = 'terrain';
        state.sandbox.fogAvailable = true;
      }
      selectMap(els.mapSelect.value || '');
    });
  }
  if (els.fogToggle) {
    els.fogToggle.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.fogAvailable) return;
      sandbox.fogEnabled = !sandbox.fogEnabled;
      syncMapToggles();
      syncMapStatus();
      drawSandbox();
    });
  }
  if (els.itemsToggle) {
    els.itemsToggle.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.mapDetail) return;
      sandbox.itemsEnabled = !sandbox.itemsEnabled;
      syncMapToggles();
      syncMapStatus();
      drawSandbox();
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
      const raw = els.speedSelect.value;
      state.sandbox.speed = raw === 'max' ? 30 : clamp(num(raw, 1), 0.5, 4);
      updatePlayerUi(true);
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
    if (state.tab === 'infer' || state.tab === 'sandbox') {
      drawSandbox();
      feDrawInferCharts();
    } else if (state.tab === 'train') {
      feDrawTrainCharts();
    }
  });

  window.addEventListener('keydown', (event) => {
    if (state.tab !== 'sandbox' && state.tab !== 'infer') return;
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
        if (state.tab === 'infer' || state.tab === 'sandbox') drawSandbox();
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
  if (state.tab === 'sandbox' || state.tab === 'infer') {
    advanceSandbox(dt);
    drawSandbox();
  }
}


const FE_DEFAULT_MAP_NAME = '[p2]Lake (2p)';
const FE_INFER_POLL_MS = 700;
const FE_TRAIN_POLL_MS = 2000;


/* ============================================================
 * 指挥中心大屏：实机推理（并发）/ 实机训练（多线程）
 * ============================================================ */

const DASH_COLORS = {
  own: '#3fa9ff',
  enemy: '#ff5a5a',
  cyan: '#38e1ff',
  amber: '#ffc35c',
  green: '#39e08b',
  muted: '#7f9bb5',
};

Object.assign(state.fe, {
  concurrency: 1,
  inferTasks: [],
  watchJobId: null,
  inferTimer: null,
  inferPolling: false,
  inferAllDone: false,
  inferStartedAt: 0,
  fpsSeries: [],
  fpsAt: 0,
  jobsPollAt: 0,
  latestFps: 0,
  restored: false,
  trainJob: null,
  trainMetrics: [],
  trainEpisodes: 0,
  trainConcurrency: 2,
  trainStartedAt: 0,
  trainTimer: null,
  trainPolling: false,
  trainStatus: 'idle',
  trainArtifacts: [],
  mapsReady: false,
});

function feSetText(node, value) {
  if (node) node.textContent = value === null || value === undefined ? '—' : String(value);
}

function feSetStatusPill(node, label, tone) {
  if (!node) return;
  node.className = 'status-pill ' + (tone || 'idle');
  node.textContent = label;
}

/* ---------------- canvas 图表工具 ---------------- */

function dashSetup(canvas) {
  if (!canvas || !canvas.parentElement) return null;
  const box = canvas.parentElement;
  const width = Math.max(0, Math.floor(box.clientWidth));
  const height = Math.max(0, Math.floor(box.clientHeight));
  if (width < 20 || height < 20) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pw = Math.round(width * dpr);
  const ph = Math.round(height * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx: ctx, width: width, height: height };
}

function dashEmpty(ctx, width, height, message) {
  ctx.fillStyle = 'rgba(127, 155, 181, 0.65)';
  ctx.font = '10px "Consolas", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message || '暂无数据', width / 2, height / 2);
}

function dashGrid(ctx, padL, padT, plotW, plotH, yMin, yMax, yFormatter) {
  ctx.font = '9px "Consolas", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const count = 4;
  for (let i = 0; i <= count; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / count;
    const y = padT + plotH - (plotH * i) / count;
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.09)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(140, 170, 195, 0.75)';
    const label = yFormatter ? yFormatter(value) : String(Math.round(value * 100) / 100);
    ctx.fillText(label, padL - 6, y);
  }
}

function drawDashLine(canvas, config) {
  const setup = dashSetup(canvas);
  if (!setup) return;
  const ctx = setup.ctx;
  const width = setup.width;
  const height = setup.height;
  const cfg = config || {};
  const series = (cfg.series || []).filter((item) => item && Array.isArray(item.points) && item.points.length);
  ctx.clearRect(0, 0, width, height);
  if (!series.length) {
    dashEmpty(ctx, width, height, cfg.emptyText);
    return;
  }
  const padL = 42;
  const padR = 8;
  const padT = 8;
  const padB = 18;
  const plotW = Math.max(1, width - padL - padR);
  const plotH = Math.max(1, height - padT - padB);
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const item of series) {
    for (const point of item.points) {
      if (!isFinite(point.x) || !isFinite(point.y)) continue;
      xMin = Math.min(xMin, point.x);
      xMax = Math.max(xMax, point.x);
      yMin = Math.min(yMin, point.y);
      yMax = Math.max(yMax, point.y);
    }
  }
  if (!isFinite(xMin) || !isFinite(yMin)) {
    dashEmpty(ctx, width, height, cfg.emptyText);
    return;
  }
  if (cfg.xMin !== undefined) xMin = cfg.xMin;
  if (cfg.xMax !== undefined) xMax = cfg.xMax;
  if (cfg.yMin !== undefined) yMin = cfg.yMin;
  if (cfg.yMax !== undefined) yMax = cfg.yMax;
  if (xMax - xMin < 0.0001) xMax = xMin + 1;
  const pad = (yMax - yMin) * 0.12 || 1;
  if (cfg.yMax === undefined) yMax += pad;
  if (cfg.yMin === undefined) yMin = Math.max(0, yMin - pad * 0.15);
  if (yMax - yMin < 0.0001) yMax = yMin + 1;
  const sx = (value) => padL + ((value - xMin) / (xMax - xMin)) * plotW;
  const sy = (value) => padT + (1 - (value - yMin) / (yMax - yMin)) * plotH;
  dashGrid(ctx, padL, padT, plotW, plotH, yMin, yMax, cfg.yFormatter);
  if (cfg.xFormatter) {
    ctx.font = '9px "Consolas", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(140, 170, 195, 0.75)';
    const ticks = 3;
    for (let i = 0; i < ticks; i += 1) {
      const value = xMin + ((xMax - xMin) * i) / (ticks - 1 || 1);
      ctx.fillText(cfg.xFormatter(value), sx(value), padT + plotH + 4);
    }
  }
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.16)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
  for (const item of series) {
    const points = item.points.slice().sort((left, right) => left.x - right.x);
    ctx.save();
    ctx.strokeStyle = item.color || DASH_COLORS.cyan;
    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    ctx.shadowColor = item.color || DASH_COLORS.cyan;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = sx(point.x);
      const y = sy(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
}

function drawDashBars(canvas, config) {
  const setup = dashSetup(canvas);
  if (!setup) return;
  const ctx = setup.ctx;
  const width = setup.width;
  const height = setup.height;
  const cfg = config || {};
  const bars = Array.isArray(cfg.bars) ? cfg.bars.filter((bar) => bar && isFinite(bar.value)) : [];
  ctx.clearRect(0, 0, width, height);
  if (!bars.length) {
    dashEmpty(ctx, width, height, cfg.emptyText);
    return;
  }
  const padL = 42;
  const padR = 8;
  const padT = 10;
  const padB = 21;
  const plotW = Math.max(1, width - padL - padR);
  const plotH = Math.max(1, height - padT - padB);
  const maxValue = cfg.yMax !== undefined ? cfg.yMax : Math.max(1, ...bars.map((bar) => bar.value)) * 1.2;
  const minValue = cfg.yMin !== undefined ? cfg.yMin : 0;
  const span = Math.max(0.0001, maxValue - minValue);
  const sy = (value) => padT + (1 - (value - minValue) / span) * plotH;
  dashGrid(ctx, padL, padT, plotW, plotH, minValue, maxValue, cfg.yFormatter);
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.16)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
  const step = plotW / bars.length;
  const barW = Math.max(4, Math.min(26, step * 0.56));
  bars.forEach((bar, index) => {
    const x = padL + step * index + (step - barW) / 2;
    const y = sy(bar.value);
    const h = Math.max(1, padT + plotH - y);
    const gradient = ctx.createLinearGradient(0, y, 0, padT + plotH);
    const color = bar.color || cfg.color || DASH_COLORS.cyan;
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, 'rgba(10, 30, 52, 0.25)');
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, barW, h);
    ctx.strokeStyle = 'rgba(170, 230, 255, 0.25)';
    ctx.strokeRect(x, y, barW, h);
    ctx.font = '9px "Consolas", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(207, 232, 251, 0.9)';
    const valueLabel = cfg.valueFormatter ? cfg.valueFormatter(bar.value) : String(Math.round(bar.value * 10) / 10);
    ctx.fillText(valueLabel, x + barW / 2, y - 2);
    if (cfg.xFormatter) {
      ctx.fillStyle = 'rgba(140, 170, 195, 0.75)';
      ctx.textBaseline = 'top';
      ctx.fillText(cfg.xFormatter(bar.label, index), x + barW / 2, padT + plotH + 4);
    }
  });
}

/* ---------------- 实机推理：并发任务模型 ---------------- */

function feInferTask(jobId) {
  return (state.fe.inferTasks || []).find((task) => task.jobId === jobId) || null;
}

function feWatchTaskObject() {
  return feInferTask(state.fe.watchJobId);
}

function feRunningInferTasks() {
  return (state.fe.inferTasks || []).filter((task) => task.status === 'running');
}

function feNewInferTask(jobId, index, extra) {
  return Object.assign(
    {
      jobId: jobId,
      index: index,
      status: 'running',
      frames: [],
      fetchCursor: 0,
      frameTimes: [],
      statSamples: [],
      latestTick: 0,
      seconds: 0,
      port: null,
      error: null,
      result: null,
      inferDone: false,
      pendingDone: false,
      startedAt: Date.now(),
      playback: 0,
      eventLog: [],
      toastShown: false,
      policy: 'scripted',
      checkpointName: null,
    },
    extra || {}
  );
}

function feIngestTaskFrames(task, items) {
  if (!task || !Array.isArray(items) || !items.length) return;
  let added = 0;
  for (const item of items) {
    if (!item || typeof item !== 'object') continue;
    if (item.type === 'done') {
      task.inferDone = true;
      task.pendingDone = true;
      if (item.result) task.result = String(item.result);
      continue;
    }
    if (item.type === 'episode_end') continue;
    const isFrame = item.type === 'frame' || Array.isArray(item.units);
    if (!isFrame) continue;
    const frame = normalizeFrame(item);
    task.frames.push(frame);
    added += 1;
    task.latestTick = frame.tick;
    // The recorder marks the terminal frame with the real bridge result.
    if (frame.result) task.result = String(frame.result);
    task.frameTimes.push(Date.now());
    if (frame.stats) {
      task.statSamples.push({
        tick: frame.tick,
        ownHp: num(frame.stats.own, 0) * 100,
        enemyHp: num(frame.stats.enemy, 0) * 100,
        ownAlive: num(frame.stats.ownAlive, 0),
        enemyAlive: num(frame.stats.enemyAlive, 0),
      });
      if (task.statSamples.length > 30000) task.statSamples.splice(0, task.statSamples.length - 30000);
    }
    for (const event of frame.events || []) {
      task.eventLog.push(event);
    }
    if (task.eventLog.length > 400) task.eventLog.splice(0, task.eventLog.length - 400);
  }
  if (!added) return;
  if (state.fe.watchJobId === task.jobId) {
    const sandbox = state.sandbox;
    sandbox.total = Math.max(sandbox.total, task.frames.length);
    if (sandbox.follow && sandbox.playing && task.frames.length > 8) {
      sandbox.playback = Math.max(sandbox.playback, task.frames.length - 6);
    }
    updatePlayerUi(true);
  }
}

function feMarkTaskDone(task) {
  if (!task || task.status === 'done') return;
  task.status = 'done';
  void feFetchTaskFinal(task);
  task.pendingDone = false;
  if (task.seconds <= 0) task.seconds = Math.max(0, (Date.now() - task.startedAt) / 1000);
  if (task.result && !task.toastShown) {
    task.toastShown = true;
    toast('实例 #' + task.index + ' 对局结果：' + resultMeta(task.result).label
      + (task.result === 'timeout' ? '（到达每局 tick 上限）' : ''), 'ok');
  }
  if (state.fe.watchJobId === task.jobId) {
    const sandbox = state.sandbox;
    sandbox.status = 'done';
    sandbox.streaming = false;
    sandbox.playing = false;
    sandbox.inferDone = true;
    updatePlayerUi(true);
    drawSandbox();
  }
}

function feMarkTaskFailed(task, message) {
  if (!task || task.status === 'failed') return;
  task.status = 'failed';
  task.error = message || '宿主机任务失败';
  if (!task.toastShown) {
    task.toastShown = true;
    toast('实例 #' + task.index + ' 失败：' + String(task.error).slice(0, 120), 'error');
  }
  if (state.fe.watchJobId === task.jobId) {
    const sandbox = state.sandbox;
    sandbox.status = 'failed';
    sandbox.streaming = false;
    updatePlayerUi(true);
  }
}

function feHandleTaskError(task, error) {
  const status = error && error.status;
  if (status === 503) {
    feMarkTaskFailed(task, '真机服务未启动');
  } else if (status === 409) {
    feMarkTaskFailed(task, '并行槽位已满');
  }
}

async function feFetchTaskFrames(task) {
  if (!task || task.status === 'failed' || task.status === 'done') return;
  const payload = await Api.rwppInferFrames(task.jobId, task.fetchCursor || 0);
  if (!payload) return;
  const items = Array.isArray(payload.frames) ? payload.frames : [];
  const next = payload.next;
  if (items.length) {
    task.fetchCursor = numOrNull(next) !== null ? num(next, task.fetchCursor + items.length) : task.fetchCursor + items.length;
    feIngestTaskFrames(task, items);
  }
  const job = payload.job || {};
  if (numOrNull(job.seconds) !== null) task.seconds = num(job.seconds, task.seconds);
  const status = job.status || payload.status;
  if (status === 'failed') {
    feMarkTaskFailed(task, job.error || '宿主机任务失败');
    return;
  }
  if (status === 'done' || task.inferDone) task.pendingDone = true;
  if (task.pendingDone && !items.length) feMarkTaskDone(task);
}

async function feFetchTaskFinal(task, attempt) {
  try {
    const job = await Api.rwppJob(task.jobId);
    if (job) {
      if (job.port !== undefined && job.port !== null) task.port = job.port;
      if (job.replay_path) task.replayPath = job.replay_path;
      if (job.result) task.result = String(job.result);
      if (numOrNull(job.seconds) !== null) task.seconds = num(job.seconds, task.seconds);
    }
  } catch (error) {
    // final metadata is best effort
  }
  feRenderTaskCards();
  if ((task.port === null || task.port === undefined) && !attempt) {
    setTimeout(() => { void feFetchTaskFinal(task, 1); }, 900);
  }
}

async function feFetchTaskJob(task) {
  if (!task || task.status !== 'running') return;
  const job = await Api.rwppJob(task.jobId);
  if (!job) return;
  if (job.port !== undefined && job.port !== null) task.port = job.port;
  if (job.result) task.result = String(job.result);
  if (numOrNull(job.elapsed) !== null) task.seconds = num(job.elapsed, task.seconds);
  else if (numOrNull(job.seconds) !== null) task.seconds = num(job.seconds, task.seconds);
  if (job.status === 'done') {
    task.pendingDone = true;
    task.replayPath = job.replay_path || task.replayPath;
  } else if (job.status === 'failed') {
    feMarkTaskFailed(task, job.error || '宿主机任务失败');
  }
}

function feApplyWatchTask(jobId, options) {
  const task = feInferTask(jobId);
  if (!task) return;
  const previous = feInferTask(state.fe.watchJobId);
  if (previous && previous !== task && state.sandbox.frames === previous.frames) {
    previous.playback = state.sandbox.playback;
  }
  state.fe.watchJobId = jobId;
  const sandbox = state.sandbox;
  sandbox.frames = task.frames;
  sandbox.total = task.frames.length;
  const resetPlayback = !!(options && options.resetPlayback);
  sandbox.playback = resetPlayback
    ? 0
    : Math.min(Math.max(task.playback || 0, 0), Math.max(0, task.frames.length - 1));
  sandbox.source = 'infer';
  sandbox.inferId = jobId;
  sandbox.status = task.status === 'running' ? 'running' : (task.status === 'done' ? 'done' : (task.status === 'failed' ? 'failed' : 'idle'));
  sandbox.streaming = task.status === 'running';
  sandbox.playing = task.status === 'running';
  sandbox.follow = task.status === 'running';
  sandbox.simMode = 'terrain';
  sandbox.fogAvailable = true;
  sandbox.cameraRegion = null;
  sandbox.checkpoint = 'rwpp-live';
  sandbox.policy = task.policy || 'scripted';
  sandbox.modelCheckpoint = task.checkpointName || null;
  sandbox.opponent = 'ai';
  sandbox.seed = 0;
  sandbox.episodes = 1;
  sandbox.eventLog = (task.eventLog || []).slice(-300);
  sandbox.inferDone = task.status === 'done';
  sandbox.results = task.result
    ? [{ episode: 0, result: task.result, ret: null }]
    : [];
  sandbox.fetchCursor = task.fetchCursor;
  updatePlayerUi(true);
  drawSandbox();
  feRenderTaskCards();
  feDrawInferCharts();
}

async function feLoadTaskFramesOnce(task) {
  if (!task || task.frames.length) return;
  try {
    const payload = await Api.rwppInferFrames(task.jobId, 0);
    if (payload && Array.isArray(payload.frames) && payload.frames.length) {
      task.fetchCursor = numOrNull(payload.next) !== null ? num(payload.next, payload.frames.length) : payload.frames.length;
      feIngestTaskFrames(task, payload.frames);
      if (payload.job && numOrNull(payload.job.seconds) !== null) task.seconds = num(payload.job.seconds, task.seconds);
    }
  } catch (error) {
    // ignore restore-less frame loads
  }
}

async function feWatchTask(jobId) {
  const task = feInferTask(jobId);
  if (!task) return;
  feApplyWatchTask(jobId, { resetPlayback: false });
  if (!task.frames.length && task.status !== 'running') {
    await feLoadTaskFramesOnce(task);
    feApplyWatchTask(jobId, { resetPlayback: true });
  }
  feRenderInferDashboard();
}

function feUpdateFps() {
  const fe = state.fe;
  const now = Date.now();
  let recent = 0;
  for (const task of fe.inferTasks || []) {
    if (!Array.isArray(task.frameTimes)) continue;
    task.frameTimes = task.frameTimes.filter((time) => now - time <= 60000);
    recent += task.frameTimes.filter((time) => now - time <= 5000).length;
  }
  fe.latestFps = Math.round((recent / 5) * 10) / 10;
  if (fe.inferStartedAt && now - (fe.fpsAt || 0) >= 1000) {
    fe.fpsAt = now;
    fe.fpsSeries.push({ t: now, v: recent / 5 });
    const cutoff = now - 60000;
    fe.fpsSeries = fe.fpsSeries.filter((point) => point.t >= cutoff);
  }
}

function feDrawInferCharts() {
  const fe = state.fe;
  const watch = feWatchTaskObject();
  const fpsSeries = fe.fpsSeries || [];
  const windowEnd = Date.now();
  const windowStart = windowEnd - 60000;
  const recentFps = fpsSeries.filter((point) => point.t >= windowStart);
  if (recentFps.length) {
    drawDashLine(els.feChartFps, {
      series: [{
        label: 'fps',
        color: DASH_COLORS.cyan,
        points: recentFps.map((point) => ({ x: (point.t - windowStart) / 1000, y: point.v })),
      }],
      yMin: 0,
      xMin: 0,
      xMax: 60,
      yFormatter: (value) => String(Math.round(value * 10) / 10),
      xFormatter: (value) => Math.round(value - 60) + 's',
      emptyText: '等待收帧…',
    });
  } else {
    drawDashLine(els.feChartFps, { series: [], emptyText: '等待收帧…' });
  }
  const samples = watch && watch.statSamples ? watch.statSamples : [];
  if (samples.length) {
    drawDashLine(els.feChartHp, {
      series: [
        { label: 'own', color: DASH_COLORS.own, points: samples.map((point) => ({ x: point.tick, y: point.ownHp })) },
        { label: 'enemy', color: DASH_COLORS.enemy, points: samples.map((point) => ({ x: point.tick, y: point.enemyHp })) },
      ],
      yMin: 0,
      yMax: 100,
      yFormatter: (value) => Math.round(value) + '%',
      xFormatter: (value) => 't' + Math.round(value),
      emptyText: '等待对局数据…',
    });
    const maxAlive = Math.max(
      4,
      ...samples.map((point) => point.ownAlive + point.enemyAlive),
      ...samples.map((point) => Math.max(point.ownAlive, point.enemyAlive))
    );
    drawDashLine(els.feChartAlive, {
      series: [
        { label: 'own', color: DASH_COLORS.own, points: samples.map((point) => ({ x: point.tick, y: point.ownAlive })) },
        { label: 'enemy', color: DASH_COLORS.enemy, points: samples.map((point) => ({ x: point.tick, y: point.enemyAlive })) },
      ],
      yMin: 0,
      yMax: maxAlive + 1,
      yFormatter: (value) => String(Math.round(value)),
      xFormatter: (value) => 't' + Math.round(value),
      emptyText: '等待对局数据…',
    });
  } else {
    drawDashLine(els.feChartHp, { series: [], emptyText: '等待对局数据…' });
    drawDashLine(els.feChartAlive, { series: [], emptyText: '等待对局数据…' });
  }
}

function feDrawTrainCharts() {
  const metrics = (state.fe.trainMetrics || []).slice().sort((left, right) => num(left.episode, 0) - num(right.episode, 0));
  if (!metrics.length) {
    drawDashBars(els.feTrainChartDuration, { bars: [], emptyText: '完成一局后出现柱状图' });
    drawDashLine(els.feTrainChartFrames, { series: [], emptyText: '完成一局后出现曲线' });
    drawDashBars(els.feTrainChartTicks, { bars: [], emptyText: '完成一局后出现柱状图' });
    return;
  }
  drawDashBars(els.feTrainChartDuration, {
    bars: metrics.map((metric) => ({
      label: '#' + (num(metric.episode, 0) + 1),
      value: Math.max(0.1, num(metric.seconds, 0)),
      color: DASH_COLORS.cyan,
    })),
    yFormatter: (value) => Math.round(value) + 's',
    valueFormatter: (value) => Math.round(value * 10) / 10,
    xFormatter: (label) => String(label),
    emptyText: '完成一局后出现柱状图',
  });
  let runningFrames = 0;
  const framesPoints = metrics.map((metric) => {
    runningFrames += num(metric.frames, 0);
    return { x: num(metric.episode, 0) + 1, y: runningFrames };
  });
  drawDashLine(els.feTrainChartFrames, {
    series: [{ label: 'frames', color: DASH_COLORS.green, points: framesPoints }],
    yMin: 0,
    xMin: 0,
    xMax: Math.max(1, metrics.length),
    yFormatter: (value) => String(Math.round(value)),
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '完成一局后出现曲线',
  });
  drawDashBars(els.feTrainChartTicks, {
    bars: metrics.map((metric) => ({
      label: '#' + (num(metric.episode, 0) + 1),
      value: Math.max(1, numOrNull(metric.ticks) !== null ? num(metric.ticks, 0) : num(state.fe.trainJob && state.fe.trainJob.ticks, 0)),
      color: DASH_COLORS.amber,
    })),
    yFormatter: (value) => String(Math.round(value)),
    valueFormatter: (value) => String(Math.round(value)),
    xFormatter: (label) => String(label),
    emptyText: '完成一局后出现柱状图',
  });
}

function feRenderTaskCards() {
  const box = els.feTaskCards;
  if (!box) return;
  const tasks = state.fe.inferTasks || [];
  box.innerHTML = '';
  feSetText(els.feTaskCount, tasks.length);
  if (!tasks.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = '尚未启动并发实例';
    box.appendChild(empty);
    feSetText(els.feTaskSummary, '点击任务卡切换沙盘观看的实例');
    return;
  }
  for (const task of tasks) {
    const card = document.createElement('div');
    card.className = 'task-card' + (task.jobId === state.fe.watchJobId ? ' active' : '') + (task.status === 'failed' ? ' failed' : '');
    card.title = '点击切换沙盘观看实例 #' + task.index;
    card.addEventListener('click', () => { void feWatchTask(task.jobId); });
    const head = document.createElement('div');
    head.className = 'task-card-head';
    const title = document.createElement('b');
    title.textContent = '实例 #' + task.index;
    const pill = document.createElement('span');
    const tone = task.status === 'running' ? 'running' : (task.status === 'done' ? 'done' : (task.status === 'failed' ? 'failed' : 'idle'));
    const label = task.status === 'running' ? '运行中' : (task.status === 'done' ? '已完成' : (task.status === 'failed' ? '失败' : '空闲'));
    feSetStatusPill(pill, label, tone);
    head.appendChild(title);
    head.appendChild(pill);
    const meta = document.createElement('div');
    meta.className = 'task-card-meta';
    const items = [
      'port ' + (task.port === null || task.port === undefined ? '—' : task.port),
      '帧 ' + task.frames.length,
      'tick ' + Math.round(task.latestTick),
      Math.round(task.seconds) + 's',
    ];
    if (task.result) items.push('结果 ' + resultMeta(task.result).label);
    if (task.error) items.push('错误');
    for (const item of items) {
      const span = document.createElement('span');
      span.textContent = item;
      meta.appendChild(span);
    }
    card.appendChild(head);
    card.appendChild(meta);
    box.appendChild(card);
  }
  const running = tasks.filter((task) => task.status === 'running').length;
  const done = tasks.filter((task) => task.status === 'done').length;
  const failed = tasks.filter((task) => task.status === 'failed').length;
  feSetText(els.feTaskSummary, '运行中 ' + running + ' · 已完成 ' + done + (failed ? ' · 失败 ' + failed : '') + ' · 点击任务卡切换');
}

function feRenderInferKpis() {
  const fe = state.fe;
  const tasks = fe.inferTasks || [];
  const running = tasks.filter((task) => task.status === 'running').length;
  const done = tasks.filter((task) => task.status === 'done').length;
  const frames = tasks.reduce((sum, task) => sum + task.frames.length, 0);
  const watch = feWatchTaskObject();
  feSetText(els.feKpiConcurrency, tasks.length || fe.concurrency || 1);
  feSetText(els.feKpiRunning, running);
  feSetText(els.feKpiDone, done);
  feSetText(els.feKpiFrames, frames);
  feSetText(els.feKpiFps, (fe.latestFps || 0).toFixed(1));
  feSetText(els.feKpiTick, watch ? Math.round(watch.latestTick) : 0);
  if (els.feKpiSlots) {
    const maxParallel = state.rwpp.maxParallel || 4;
    const free = state.rwpp.freeSlots;
    feSetText(els.feKpiSlots, 'max ' + maxParallel + ' · 空闲 ' + (free === null || free === undefined ? '—' : free));
  }
  feSetText(els.feSlotsMax, state.rwpp.maxParallel || '—');
  feSetText(els.feSlotsFree, state.rwpp.freeSlots === null || state.rwpp.freeSlots === undefined ? '—' : state.rwpp.freeSlots);
  feSetText(els.feSlotsRunning, state.rwpp.hostRunning);
}

function feRenderInferDashboard() {
  feRenderInferKpis();
  feRenderTaskCards();
  feDrawInferCharts();
}

function feUpdateSlots() {
  const rwpp = state.rwpp;
  const maxParallel = rwpp.maxParallel || 4;
  const free = rwpp.freeSlots;
  if (els.feSlotsMax) els.feSlotsMax.textContent = String(maxParallel);
  if (els.feSlotsFree) els.feSlotsFree.textContent = free === null || free === undefined ? '—' : String(free);
  if (els.feSlotsRunning) els.feSlotsRunning.textContent = String(rwpp.hostRunning || 0);
  if (els.feKpiSlots) {
    els.feKpiSlots.textContent = 'max ' + maxParallel + ' · 空闲 ' + (free === null || free === undefined ? '—' : free);
  }
  const disableAbove = (select) => {
    if (!select) return;
    for (const option of select.options) {
      const value = num(option.value, 1);
      option.disabled = value > maxParallel;
    }
    if (num(select.value, 1) > maxParallel) select.value = String(Math.max(1, maxParallel));
  };
  disableAbove(els.feConcurrency);
  disableAbove(els.feTrainConcurrency);
}

function feSetInferStatus(text, tone) {
  if (!els.feInferStatus) return;
  els.feInferStatus.textContent = text;
  els.feInferStatus.className = 'hint-line' + (tone ? ' ' + tone : '');
}

function feSetInferRunning(running) {
  if (els.feStartBtn) {
    els.feStartBtn.disabled = running;
    els.feStartBtn.textContent = running ? '实机推理中…' : '开始实机推理';
  }
  if (els.mapSelect) els.mapSelect.disabled = running;
  if (els.feTicks) els.feTicks.disabled = running;
  if (els.feConcurrency) els.feConcurrency.disabled = running;
}

function handleFeError(error, label) {
  const status = error && error.status;
  const message = error && error.message ? String(error.message) : String(error || '');
  if (status === 409 || message.indexOf('slots are busy') >= 0) {
    toast('并行槽位已满：减少并发或等待其他实例结束', 'error');
  } else if (status === 503) {
    toast('真机服务未启动：在宿主机运行 python scripts/rwpp_real_service.py', 'error');
  } else {
    toast(label + '失败：' + message.slice(0, 160), 'error');
  }
}

async function feStartInfer() {
  const guard = state.fe;
  if (guard.inferStarting) return;
  guard.inferStarting = true;
  try {
    await feStartInferInner();
  } catch (error) {
    feSetInferRunning(false);
    guard.inferStatus = 'failed';
    handleFeError(error, '实机推理');
    feSetInferStatus('启动失败：' + (error && error.message ? error.message : error), 'bad');
    feRenderInferDashboard();
  } finally {
    guard.inferStarting = false;
  }
}

async function feStartInferInner() {
  const fe = state.fe;
  if (feRunningInferTasks().length) {
    toast('已有并发推理在运行，请等待完成', 'error');
    return;
  }
  if (state.mode === 'demo') {
    toast('演示模式下无法连接真机服务', 'error');
    return;
  }
  const selected = clampInt(els.feConcurrency && els.feConcurrency.value, 1, 4, 1);
  const maxParallel = state.rwpp.maxParallel || 4;
  const launchCount = Math.max(1, Math.min(selected, maxParallel));
  const mapItem = feMapItemById(els.mapSelect ? els.mapSelect.value : '') || feDefaultMap();
  if (!mapItem) {
    toast('请选择地图（需要游戏资源缓存）', 'error');
    return;
  }
  const ticks = clampInt(els.feTicks && els.feTicks.value, 120, 216000, 3600);
  const policy = (els.fePolicy && els.fePolicy.value) || 'scripted';
  let checkpointPath = '';
  let checkpointName = '';
  if (policy === 'model') {
    const checkpoints = state.fe.checkpointList || [];
    const selectedPath = els.feCheckpointSelect ? els.feCheckpointSelect.value : '';
    const item = checkpoints.find((entry) => entry && entry.path === selectedPath) || checkpoints[0];
    if (!item) {
      toast('没有可用 checkpoint：先到「实机训练」页跑一次 PPO', 'error');
      return;
    }
    checkpointPath = item.path;
    checkpointName = item.name || basename(String(item.path).replace(/\\/g, '/'));
  } else if (policy !== 'scripted') {
    toast('不支持的推理策略：' + policy, 'error');
    return;
  }
  await refreshRwppStatus(true);
  const free = state.rwpp.freeSlots;
  if (free !== null && free !== undefined && free < launchCount) {
    toast('服务端空闲槽位不足（空闲 ' + free + ' / 上限 ' + (state.rwpp.maxParallel || 4) + '），请减少并发或稍后重试', 'error');
    return;
  }
  fe.inferTasks = [];
  fe.watchJobId = null;
  fe.fpsSeries = [];
  fe.fpsAt = 0;
  fe.jobsPollAt = 0;
  fe.slotsAt = 0;
  fe.latestFps = 0;
  fe.inferStartedAt = Date.now();
  fe.inferAllDone = false;
  fe.concurrency = launchCount;
  if (els.feKpiConcurrency) els.feKpiConcurrency.textContent = String(launchCount);
  resetSandbox();
  const sandbox = state.sandbox;
  sandbox.simMode = 'terrain';
  sandbox.fogAvailable = true;
  sandbox.status = 'running';
  if (els.mapSelect) els.mapSelect.value = mapItem.id;
  await selectMap(mapItem.id, { silent: true });
  feSetInferRunning(true);
  const policyLabel = policy === 'model'
    ? '模型策略 · ' + checkpointName
    : '脚本策略';
  feSetInferStatus('启动 ' + launchCount + ' 路并发实例（' + policyLabel + '）…');
  feRenderInferDashboard();
  const inferConfig = {
    ticks: ticks,
    policy: policy,
    // 2 ms/tick ≈ 8x faster than real time. The policy decides every 10 ticks and
    // the scripted recorder samples every 4th observation, so real-time pacing
    // (16 ms) only makes the wait longer without changing what happens in game.
    frame_ms: 2,
    map_name: mapItem.name,
    map_path: mapItem.relative || '',
    map_width: num(mapItem.width_px, 0),
    map_height: num(mapItem.height_px, 0),
  };
  if (policy === 'model') {
    inferConfig.checkpoint = checkpointPath;
  }
  const results = await Promise.allSettled(
    Array.from({ length: launchCount }, () => Api.rwppInfer(inferConfig))
  );
  const jobIds = [];
  const failures = [];
  results.forEach((result) => {
    if (result.status === 'fulfilled' && result.value && result.value.job_id) jobIds.push(String(result.value.job_id));
    else failures.push(result.reason);
  });
  if (!jobIds.length) {
    feSetInferRunning(false);
    const first = failures[0];
    handleFeError(first, '实机推理');
    feSetInferStatus('启动失败：' + (first && first.message ? first.message : first), 'bad');
    feRenderInferDashboard();
    return;
  }
  jobIds.forEach((jobId, index) => {
    fe.inferTasks.push(feNewInferTask(jobId, index + 1, {
      policy: policy,
      checkpointName: checkpointName || null,
    }));
  });
  fe.taskSeq = jobIds.length;
  if (failures.length) {
    const first = failures[0];
    toast('仅启动 ' + jobIds.length + '/' + launchCount + ' 路：' + (first && first.message ? String(first.message).slice(0, 100) : '槽位或服务异常'), 'error');
  }
  fe.watchJobId = jobIds[0];
  feApplyWatchTask(jobIds[0], { resetPlayback: true });
  startFeInferPolling();
  feRenderInferDashboard();
  toast('已启动 ' + jobIds.length + ' 路并发实机推理（' + policyLabel + '）', 'ok');
}

function startFeInferPolling() {
  stopFeInferPolling();
  state.fe.inferTimer = setInterval(feInferTick, 700);
  void feInferTick();
}

function stopFeInferPolling() {
  const fe = state.fe;
  if (fe.inferTimer) {
    clearInterval(fe.inferTimer);
    fe.inferTimer = null;
  }
}

async function feInferTick() {
  const fe = state.fe;
  if (fe.inferPolling) return;
  const tasks = feRunningInferTasks();
  if (!tasks.length) return;
  fe.inferPolling = true;
  try {
    await Promise.all(
      tasks.map(async (task) => {
        try {
          await feFetchTaskFrames(task);
        } catch (error) {
          feHandleTaskError(task, error);
        }
      })
    );
    const now = Date.now();
    if (now - (fe.slotsAt || 0) >= 4000) {
      fe.slotsAt = now;
      try {
        await refreshRwppStatus(true);
      } catch (error) {
        // slots refresh is best effort
      }
    }
    if (now - (fe.jobsPollAt || 0) >= 2000) {
      fe.jobsPollAt = now;
      await Promise.all(
        tasks.map(async (task) => {
          try {
            await feFetchTaskJob(task);
          } catch (error) {
            // job lookups may transiently fail; frames polling continues
          }
        })
      );
    }
    feUpdateFps();
    feRenderInferDashboard();
    const watchTask = feWatchTaskObject();
    if (watchTask) {
      const watchRunning = watchTask.status === 'running';
      const watchSeconds = Math.round(watchTask.seconds || 0);
      feSetInferStatus(
        (watchRunning ? '运行中' : (watchTask.status === 'done' ? '已完成' : '失败'))
          + ' · ' + (watchTask.policy === 'model'
            ? '模型策略 · ' + (watchTask.checkpointName || '')
            : '脚本策略')
          + ' · 已收 ' + watchTask.frames.length + ' 帧 · tick=' + Math.round(watchTask.latestTick)
          + (watchRunning && watchSeconds ? ' · ' + watchSeconds + 's' : '')
          + (!watchRunning && watchTask.result ? ' · 结果 ' + resultMeta(watchTask.result).label : ''),
        watchRunning ? '' : (watchTask.status === 'done' ? 'ok' : 'bad')
      );
    }
    await feFinishInferIfDone();
  } finally {
    fe.inferPolling = false;
  }
}

async function feFinishInferIfDone() {
  const fe = state.fe;
  const tasks = fe.inferTasks || [];
  if (!tasks.length || tasks.some((task) => task.status === 'running')) return;
  if (fe.inferAllDone) return;
  fe.inferAllDone = true;
  stopFeInferPolling();
  feSetInferRunning(false);
  const done = tasks.filter((task) => task.status === 'done').length;
  const failed = tasks.length - done;
  const frames = tasks.reduce((sum, task) => sum + task.frames.length, 0);
  const counts = {};
  for (const task of tasks) {
    if (!task.result) continue;
    const label = resultMeta(task.result).label;
    counts[label] = (counts[label] || 0) + 1;
  }
  const resultText = Object.keys(counts).length
    ? ' · ' + Object.keys(counts).map((label) => label + ' ' + counts[label]).join(' / ')
    : '';
  feSetInferStatus('完成 · ' + done + ' 成功' + (failed ? ' / ' + failed + ' 失败' : '') + ' · 共 ' + frames + ' 帧' + resultText, failed ? 'warn' : 'ok');
  feRenderInferDashboard();
  await loadReplays();
  await refreshRwppStatus(true);
  toast('实机推理完成：' + done + '/' + tasks.length + ' 实例成功，共 ' + frames + ' 帧' + resultText, failed ? 'error' : 'ok');
}

async function feRestoreInferJobs() {
  const fe = state.fe;
  if (fe.restored || state.mode === 'demo') return;
  if (feRunningInferTasks().length) return;
  fe.restored = true;
  try {
    const payload = await Api.rwppJobs();
    const jobs = ((payload && payload.jobs) || []).filter((job) => job && job.kind === 'infer' && job.status === 'running');
    if (!jobs.length) return;
    const restored = jobs.slice(0, 4);
    fe.inferTasks = restored.map((job, index) =>
      feNewInferTask(String(job.id), index + 1, {
        status: 'running',
        seconds: num(job.elapsed, num(job.seconds, 0)),
        port: job.port || null,
        startedAt: Date.now() - Math.max(0, num(job.elapsed, 0)) * 1000,
      })
    );
    fe.inferStartedAt = Date.now();
    fe.watchJobId = fe.inferTasks[0].jobId;
    feSetInferRunning(true);
    feApplyWatchTask(fe.watchJobId, { resetPlayback: true });
    startFeInferPolling();
    feSetInferStatus('已接回 ' + fe.inferTasks.length + ' 个运行中的实机推理实例');
    toast('已接回 ' + fe.inferTasks.length + ' 个运行中的真机推理实例', 'ok');
  } catch (error) {
    // no restore
  }
}

/* ---------------- 实机训练：多线程采集大屏 ---------------- */

function feSetTrainRunning(running) {
  if (els.feTrainStart) {
    els.feTrainStart.disabled = running;
    els.feTrainStart.hidden = running;
    els.feTrainStart.textContent = running ? '实机训练中…' : '开始实机训练';
  }
  if (els.feTrainStop) {
    els.feTrainStop.hidden = !running;
    els.feTrainStop.disabled = !running;
  }
  if (els.feTrainEpisodes) els.feTrainEpisodes.disabled = running;
  if (els.feTrainTicks) els.feTrainTicks.disabled = running;
  if (els.feTrainMap) els.feTrainMap.disabled = running;
  if (els.feTrainConcurrency) els.feTrainConcurrency.disabled = running;
}

function feRenderTrainKpis(job) {
  const fe = state.fe;
  const metrics = (fe.trainMetrics || []).slice().sort((left, right) => num(left.episode, 0) - num(right.episode, 0));
  const total = job && numOrNull(job.episodes_total) !== null ? num(job.episodes_total, fe.trainEpisodes) : fe.trainEpisodes;
  const done = job && numOrNull(job.episodes_done) !== null ? num(job.episodes_done, metrics.length) : metrics.length;
  const elapsed = job && numOrNull(job.elapsed) !== null ? num(job.elapsed, 0) : (job && numOrNull(job.seconds) !== null ? num(job.seconds, 0) : (fe.trainStartedAt ? (Date.now() - fe.trainStartedAt) / 1000 : 0));
  const framesTotal = job && numOrNull(job.frames) !== null ? num(job.frames, 0) : metrics.reduce((sum, metric) => sum + num(metric.frames, 0), 0);
  const ticksTotal = job && numOrNull(job.ticks_total) !== null
    ? num(job.ticks_total, 0)
    : metrics.reduce((sum, metric) => sum + (numOrNull(metric.ticks) !== null ? num(metric.ticks, 0) : num(job && job.ticks, 0)), 0);
  const avg = metrics.length
    ? metrics.reduce((sum, metric) => sum + num(metric.seconds, 0), 0) / metrics.length
    : (done > 0 ? elapsed / done : 0);
  feSetText(els.feTrainKpiProgress, done);
  feSetText(els.feTrainKpiProgressTotal, '/ ' + total + ' 局');
  feSetText(els.feTrainKpiSeconds, Math.round(elapsed));
  feSetText(els.feTrainKpiThreads, job && numOrNull(job.concurrency) !== null ? num(job.concurrency, fe.trainConcurrency) : fe.trainConcurrency);
  feSetText(els.feTrainKpiAvg, avg.toFixed(1));
  feSetText(els.feTrainKpiFrames, framesTotal);
  feSetText(els.feTrainKpiTicks, ticksTotal);
  if (els.feTrainEpisodesLabel) els.feTrainEpisodesLabel.textContent = done + ' / ' + total + ' 局';
  if (els.feTrainSeconds) els.feTrainSeconds.textContent = Math.round(elapsed) + 's';
  if (els.feTrainCurrent) {
    const current = job && numOrNull(job.current_episode) !== null ? num(job.current_episode, 0) : 0;
    if (total > 0 && done >= total) els.feTrainCurrent.textContent = '已完成';
    else if (current > 0) els.feTrainCurrent.textContent = '当前第 ' + current + ' 局';
    else if (job && job.status === 'running') els.feTrainCurrent.textContent = '采集中…';
    else els.feTrainCurrent.textContent = '等待开始';
  }
  const percent = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  if (els.feTrainProgress) els.feTrainProgress.style.width = percent + '%';
  if (job) {
    const labels = { running: ['运行中', 'running'], done: ['已完成', 'done'], failed: ['失败', 'bad'] };
    const pair = labels[job.status] || ['空闲', 'idle'];
    feSetStatusPill(els.feTrainStatus, pair[0], pair[1]);
    feSetText(els.feTrainLogStatus, 'status=' + job.status + ' · concurrency=' + (job.concurrency || '—'));
  } else if (els.feTrainStatus) {
    feSetStatusPill(els.feTrainStatus, '空闲', 'idle');
  }
  if (els.feTrainSummary) {
    const resultCounts = {};
    for (const metric of metrics) {
      if (!metric || !metric.result) continue;
      const label = resultMeta(metric.result).label;
      resultCounts[label] = (resultCounts[label] || 0) + 1;
    }
    const resultText = Object.keys(resultCounts).length
      ? ' · ' + Object.keys(resultCounts).map((label) => label + ' ' + resultCounts[label]).join(' / ')
      : '';
    if (job && job.status === 'done') {
      els.feTrainSummary.textContent = '汇总：' + done + ' 局 · ' + framesTotal + ' 帧 · ' + ticksTotal + ' tick · ' + Math.round(elapsed) + 's · 平均 ' + avg.toFixed(1) + 's/局' + resultText;
    } else if (job && job.status === 'failed') {
      els.feTrainSummary.textContent = '失败：' + String(job.error || '未知错误').slice(0, 80);
    } else {
      els.feTrainSummary.textContent = '—';
    }
  }
}

function feRenderTrainLog(job) {
  const log = els.feTrainLog;
  if (!log) return;
  if (!job) {
    log.textContent = '等待训练任务…';
    return;
  }
  const lines = [];
  lines.push('status: ' + job.status);
  lines.push('episodes: ' + num(job.episodes_done, 0) + '/' + num(job.episodes_total, state.fe.trainEpisodes) + ' · concurrency: ' + (job.concurrency || '—'));
  const episodeResults = (Array.isArray(job.episodes_metrics) ? job.episodes_metrics : [])
    .map((metric) => metric && metric.result)
    .filter(Boolean);
  if (episodeResults.length) {
    const counts = {};
    for (const result of episodeResults) {
      const label = resultMeta(result).label;
      counts[label] = (counts[label] || 0) + 1;
    }
    lines.push('results: ' + Object.keys(counts).map((label) => label + ' ' + counts[label]).join(' · '));
  }
  lines.push('ticks: ' + num(job.ticks, 0) + ' · instances: ' + (job.instances || job.workers || '—'));
  lines.push('frames: ' + num(job.frames, 0) + ' · ticks_total: ' + num(job.ticks_total, 0));
  if (job.error) lines.push('');
  if (job.error) lines.push('ERROR: ' + job.error);
  const tail = job.last_log ? String(job.last_log).trim() : '';
  if (tail) {
    lines.push('');
    lines.push(tail.split('\n').slice(-10).join('\n'));
  }
  log.textContent = lines.join('\n');
}

function feRenderTrainDashboard(job) {
  const fe = state.fe;
  const jobData = job === undefined ? fe.trainJob : job;
  if (jobData) {
    fe.trainJob = jobData;
    if (Array.isArray(jobData.episodes_metrics)) {
      fe.trainMetrics = jobData.episodes_metrics.slice().sort((left, right) => num(left.episode, 0) - num(right.episode, 0));
    }
  }
  feRenderTrainKpis(jobData);
  feRenderTrainLog(jobData);
  feDrawTrainCharts();
}

async function feStartTrain() {
  const fe = state.fe;
  if (fe.trainStatus === 'running') return;
  if (state.mode === 'demo') {
    toast('演示模式下无法连接真机服务', 'error');
    return;
  }
  const mapName = els.feTrainMap ? els.feTrainMap.value : '';
  if (!mapName) {
    toast('请选择地图', 'error');
    return;
  }
  const episodes = clampInt(els.feTrainEpisodes && els.feTrainEpisodes.value, 1, 50, 3);
  const ticks = clampInt(els.feTrainTicks && els.feTrainTicks.value, 120, 216000, 1200);
  const concurrency = clampInt(els.feTrainConcurrency && els.feTrainConcurrency.value, 1, 4, 2);
  const policy = (els.feTrainPolicy && els.feTrainPolicy.value) || 'scripted';
  let checkpoint = null;
  if (policy === 'model') {
    checkpoint = (els.feTrainCheckpointSelect && els.feTrainCheckpointSelect.value) || null;
    if (!checkpoint) {
      toast('请选择用于采集对战的模型 Checkpoint', 'error');
      return;
    }
  }
  fe.trainMetrics = [];
  fe.trainEpisodes = episodes;
  fe.trainConcurrency = concurrency;
  fe.trainStatus = 'running';
  fe.trainStartedAt = Date.now();
  fe.trainArtifacts = [];
  feSetTrainRunning(true);
  feRenderTrainDashboard(null);
  feSetStatusPill(els.feTrainStatus, '启动中', 'running');
  feRenderTrainLog({ status: 'starting', episodes_done: 0, episodes_total: episodes, concurrency: concurrency, ticks: ticks, frames: 0, ticks_total: 0, last_log: '正在创建真机训练任务…' });
  feDrawTrainCharts();
  feSetText(els.feTrainSummary, '—');
  try {
    const response = await Api.rwppTrain({
      episodes: episodes,
      ticks: ticks,
      policy: policy,
      checkpoint: checkpoint,
      map_name: mapName,
      concurrency: concurrency,
    });
    const jobId = response && (response.job_id || response.id);
    if (!jobId) throw new Error('后端未返回训练任务 id');
    fe.trainJobId = String(jobId);
    fe.trainStatus = 'running';
    const hint = document.getElementById('fe-train-hint');
    if (hint) hint.textContent = '并行跑多个真实游戏进程，每线程独立采集一局；当前阶段为真机数据采集。';
    startFeTrainPolling();
    toast('实机训练已开始 · ' + episodes + ' 局 × ' + ticks + ' ticks · ' + concurrency + ' 路并发', 'ok');
  } catch (error) {
    feSetTrainRunning(false);
    fe.trainStatus = 'failed';
    if (error && error.status === 409) {
      toast('已有训练或实机任务正在运行，请先在控制台停止旧任务或稍候重试', 'error');
    } else {
      handleFeError(error, '实机训练');
    }
    feRenderTrainDashboard(null);
    feSetStatusPill(els.feTrainStatus, '失败', 'bad');
    feRenderTrainLog({ status: 'failed', episodes_done: 0, episodes_total: episodes, concurrency: concurrency, ticks: ticks, frames: 0, ticks_total: 0, error: error && error.message ? error.message : String(error) });
  }
}

async function feStopTrain() {
  const fe = state.fe;
  if (!fe.trainJobId || fe.trainStatus !== 'running') return;
  if (els.feTrainStop) els.feTrainStop.disabled = true;
  try {
    await Api.rwppJobStop(fe.trainJobId);
    toast('已请求停止实机数据采集任务', 'ok');
  } catch (error) {
    handleFeError(error, '停止训练');
  }
}

function startFeTrainPolling() {
  stopFeTrainPolling();
  state.fe.trainTimer = setInterval(fePollTrain, 2000);
  void fePollTrain();
}

function stopFeTrainPolling() {
  const fe = state.fe;
  if (fe.trainTimer) {
    clearInterval(fe.trainTimer);
    fe.trainTimer = null;
  }
}

async function fePollTrain() {
  const fe = state.fe;
  if (!fe.trainJobId || fe.trainPolling) return;
  fe.trainPolling = true;
  try {
    const job = await Api.rwppJob(fe.trainJobId);
    if (!job) return;
    const metrics = Array.isArray(job.episodes_metrics) ? job.episodes_metrics : [];
    const previous = fe.trainMetrics.length;
    fe.trainJob = job;
    fe.trainMetrics = metrics.slice().sort((left, right) => num(left.episode, 0) - num(right.episode, 0));
    feRenderTrainDashboard(job);
    if (fe.trainMetrics.length !== previous || job.status !== 'running') {
      await loadReplays();
      await renderFeTrainArtifactsFromJob(job);
    }
    if (job.status !== 'running') {
      stopFeTrainPolling();
      feSetTrainRunning(false);
      if (job.status === 'done') {
        fe.trainStatus = 'done';
        toast('实机训练完成（' + Math.round(num(job.elapsed, 0)) + 's · ' + num(job.episodes_done, fe.trainMetrics.length) + ' 局）', 'ok');
      } else {
        fe.trainStatus = 'failed';
        toast('实机训练失败：' + String(job.error || job.last_log || '未知错误').slice(0, 160), 'error');
      }
      await refreshRwppStatus(true);
      feRenderTrainDashboard(job);
    }
  } catch (error) {
    if (error && (error.status === 503 || error.status === 409)) {
      stopFeTrainPolling();
      feSetTrainRunning(false);
      fe.trainStatus = 'failed';
      handleFeError(error, '实机训练');
      feSetStatusPill(els.feTrainStatus, '失败', 'bad');
    }
  } finally {
    fe.trainPolling = false;
  }
}

async function renderFeTrainArtifactsFromJob(job) {
  const names = [];
  for (const path of job && Array.isArray(job.replays) ? job.replays : []) names.push(basename(String(path).replace(/\\/g, '/')));
  for (const metric of job && Array.isArray(job.episodes_metrics) ? job.episodes_metrics : []) {
    if (metric && metric.replay) names.push(basename(String(metric.replay).replace(/\\/g, '/')));
  }
  const items = state.replays || [];
  let picked = names
    .map((name) => items.find((item) => item && item.name === name))
    .filter(Boolean);
  const unique = [];
  const seen = new Set();
  for (const item of picked) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    unique.push(item);
  }
  if (!unique.length) {
    const total = job && numOrNull(job.episodes_done) !== null ? num(job.episodes_done, 1) : Math.max(1, names.length);
    picked = items.filter(isRealReplay).slice(0, Math.max(1, total));
  } else {
    picked = unique;
  }
  state.fe.trainArtifacts = picked;
  renderFeTrainArtifacts(picked);
}

function renderFeTrainArtifacts(items) {
  const body = els.feTrainArtifacts;
  if (!body) return;
  body.innerHTML = '';
  const list = Array.isArray(items) ? items : [];
  if (els.feTrainArtifactCount) els.feTrainArtifactCount.textContent = String(list.length);
  if (!list.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.className = 'empty';
    cell.textContent = '训练完成后显示本任务录制的回放';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const item of list) {
    const row = document.createElement('tr');
    row.className = 'data-row';
    const cellName = document.createElement('td');
    cellName.textContent = item.name || basename(item.id);
    if (isRealReplay(item)) {
      const tag = document.createElement('span');
      tag.className = 'real-tag';
      tag.textContent = '真机';
      cellName.appendChild(tag);
    }
    const cellSim = document.createElement('td');
    cellSim.textContent = (item.simMode === 'terrain' ? '地形' : '竞技场')
      + (item.simMode === 'terrain' && item.mapName ? ' · ' + item.mapName : '');
    const cellResult = document.createElement('td');
    const badge = document.createElement('span');
    const meta = resultMeta(item.result);
    badge.className = 'result-badge ' + meta.cls;
    badge.textContent = meta.label;
    cellResult.appendChild(badge);
    const cellFrames = document.createElement('td');
    cellFrames.textContent = item.frames === null || item.frames === undefined ? '—' : String(item.frames);
    const cellTime = document.createElement('td');
    cellTime.textContent = fmtShortTime(item.created_at) || '—';
    const cellActions = document.createElement('td');
    const playButton = document.createElement('button');
    playButton.type = 'button';
    playButton.className = 'btn btn-sm';
    playButton.textContent = '播放';
    playButton.addEventListener('click', (event) => {
      event.stopPropagation();
      void openFeReplay(item.id);
    });
    cellActions.appendChild(playButton);
    row.appendChild(cellName);
    row.appendChild(cellSim);
    row.appendChild(cellResult);
    row.appendChild(cellFrames);
    row.appendChild(cellTime);
    row.appendChild(cellActions);
    body.appendChild(row);
  }
}

async function openFeReplay(id) {
  switchTab('infer');
  await openReplay(id);
}

async function feRefreshTrainArtifacts() {
  await loadReplays();
  if (state.fe.trainJobId) {
    try {
      const job = await Api.rwppJob(state.fe.trainJobId);
      await renderFeTrainArtifactsFromJob(job);
      return;
    } catch (error) {
      // fall back to cached artifacts
    }
  }
  renderFeTrainArtifacts(state.fe.trainArtifacts);
}

/* ---------------- 地图选择 ---------------- */

function feMapList() {
  return (state.maps.list || []).filter((item) => item && !item.broken);
}

function feInferMapList() {
  const allowed = { skirmish: true, challenge: true, testing: true, ironzero: true };
  return feMapList().filter((item) => allowed[String(item.folder || '')]);
}

function feMapItemById(id) {
  return feMapList().find((item) => item.id === id) || null;
}

function feMapItemByName(name) {
  return feMapList().find((item) => item.name === name) || null;
}

function feDefaultMap() {
  return feMapItemByName(FE_DEFAULT_MAP_NAME) || feMapList()[0] || null;
}

function renderFeMapControls() {
  const allMaps = feMapList();
  const inferMaps = feInferMapList();
  const defaultItem = inferMaps.find((item) => item.name === FE_DEFAULT_MAP_NAME) || inferMaps[0] || feDefaultMap();
  if (els.mapSelect) {
    const previousId = els.mapSelect.value || (state.sandbox && state.sandbox.mapId) || '';
    els.mapSelect.innerHTML = '';
    for (const item of inferMaps) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.name + (item.players ? ' · ' + item.players + 'P' : '');
      els.mapSelect.appendChild(option);
    }
    const fallbackId = inferMaps.some((item) => item.id === previousId) ? previousId : (defaultItem ? defaultItem.id : '');
    els.mapSelect.value = fallbackId;
  }
  if (els.feMapCount) {
    els.feMapCount.textContent = '可用地图 ' + inferMaps.length + ' 张（已过滤战役/菜单图）';
  }
  if (els.feTrainMap) {
    const previousName = els.feTrainMap.value || (defaultItem ? defaultItem.name : '');
    els.feTrainMap.innerHTML = '';
    for (const item of allMaps) {
      const option = document.createElement('option');
      option.value = item.name;
      option.textContent = item.name + (item.players ? ' · ' + item.players + 'P' : '');
      els.feTrainMap.appendChild(option);
    }
    const fallbackName = allMaps.some((item) => item.name === previousName) ? previousName : (defaultItem ? defaultItem.name : '');
    els.feTrainMap.value = fallbackName;
  }
  if (els.fePpoMap) {
    const previousPpoMap = els.fePpoMap.value || 'IronZero25';
    els.fePpoMap.innerHTML = '';

    // Group 1: 纯坦克对称竞技场
    const arenaGroup = document.createElement('optgroup');
    arenaGroup.label = '对称竞技场（纯坦克训练 · 推荐）';
    const arenas = [
      { id: 'IronZero25', label: 'IronZero25（25v25 对称竞技场 · 纯坦克 · 推荐）' },
      { id: 'IronZero20', label: 'IronZero20（20v20 对称竞技场 · 纯坦克）' },
      { id: 'IronZero15', label: 'IronZero15（15v15 对称竞技场 · 纯坦克）' },
      { id: 'IronZero10', label: 'IronZero10（10v10 对称竞技场 · 纯坦克 · 快速验证）' },
      { id: 'IronZero5', label: 'IronZero5（5v5 对称竞技场 · 纯坦克 · 极速验证）' },
    ];
    for (const a of arenas) {
      const opt = document.createElement('option');
      opt.value = a.id;
      opt.textContent = a.label;
      arenaGroup.appendChild(opt);
    }
    els.fePpoMap.appendChild(arenaGroup);

    // Group 2: 原版实机地图（遭遇战 / 战役）
    if (inferMaps.length > 0) {
      const originalGroup = document.createElement('optgroup');
      originalGroup.label = '原版实机地图（遭遇战 / 战役挑战）';
      for (const item of inferMaps) {
        if (item.name && item.name.toLowerCase().startsWith('ironzero')) continue;
        const opt = document.createElement('option');
        opt.value = item.name;
        opt.textContent = item.name + (item.players ? ' · ' + item.players + 'P' : '') + (item.name === 'Artillery' ? '（22v25 带工厂建筑）' : '');
        originalGroup.appendChild(opt);
      }
      els.fePpoMap.appendChild(originalGroup);
    }

    const hasPpoOption = [...els.fePpoMap.options].some((o) => o.value === previousPpoMap);
    els.fePpoMap.value = hasPpoOption ? previousPpoMap : 'IronZero25';
  }
  if (!state.fe.mapsReady && defaultItem) {
    state.fe.mapsReady = true;
    const sandbox = state.sandbox;
    sandbox.simMode = 'terrain';
    sandbox.fogAvailable = true;
    if (els.mapSelect) els.mapSelect.value = defaultItem.id;
    void selectMap(defaultItem.id, { silent: true });
  }
  feUpdateSlots();
  feRenderInferDashboard();
}



/* ============================================================
 * 实机训练 · PPO 真机训练（strategy-in-the-loop）
 * ============================================================ */

const PPO_MAP = {
  name: 'Artillery',
  path: 'maps/challenge/Artillery.tmx',
  width: 2700,
  height: 2500,
};

Object.assign(state.fe, {
  trainMode: 'collect',
  ppo: {
    jobId: null,
    status: 'idle',
    events: [],
    cursor: 0,
    timer: null,
    polling: false,
    job: null,
    updatesTotal: 0,
    updatesDone: 0,
    latest: null,
    evaluations: [],
    donePayload: null,
    startedAt: 0,
    stopRequested: false,
    restoreTried: false,
    frameMs: 2,
    visionRadius: 260,
  },
});

function fePpoUpdateEvents() {
  const events = (state.fe.ppo.events || []).filter((event) => event && event.kind === 'update');
  return events
    .map((event) => event.payload || {})
    .sort((left, right) => num(left.update, 0) - num(right.update, 0));
}

function fePpoEvalEvents() {
  const ppo = state.fe.ppo;
  if (Array.isArray(ppo.evaluations) && ppo.evaluations.length) {
    return ppo.evaluations.slice().sort((a, b) => num(a.update, 0) - num(b.update, 0));
  }
  const events = (ppo.events || []).filter((event) => event && event.kind === 'eval');
  return events
    .map((event) => event.payload || {})
    .sort((left, right) => num(left.update, 0) - num(right.update, 0));
}

function fePpoDurations(updates) {
  const durations = [];
  let previous = 0;
  for (const update of updates) {
    const elapsed = num(update.elapsed_s, 0);
    durations.push(Math.max(0, elapsed - previous));
    previous = elapsed;
  }
  return durations;
}

function fePpoSeriesRange(seriesList) {
  let min = Infinity;
  let max = -Infinity;
  for (const points of seriesList) {
    for (const point of points) {
      if (!isFinite(point.y)) continue;
      min = Math.min(min, point.y);
      max = Math.max(max, point.y);
    }
  }
  if (!isFinite(min)) return { yMin: 0, yMax: 1 };
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.14 || 0.5;
  return { yMin: min - pad, yMax: max + pad };
}

function fePpoEventLine(event) {
  const payload = (event && event.payload) || {};
  if (event.kind === 'start') {
    const config = payload.config || {};
    return '[start] device=' + (payload.device || '—')
      + ' · updates=' + num(config.total_updates, 0)
      + ' · envs=' + num(config.num_envs, 0)
      + ' · rollout=' + num(config.rollout_steps, 0);
  }
  if (event.kind === 'update') {
    return '[update ' + num(payload.update, 0) + ']'
      + ' return=' + num(payload.mean_return, 0).toFixed(3)
      + ' win=' + (num(payload.win_rate, 0) * 100).toFixed(0) + '%'
      + ' entropy=' + num(payload.entropy, 0).toFixed(2)
      + ' pi=' + num(payload.policy_loss, 0).toFixed(3)
      + ' t=' + num(payload.elapsed_s, 0).toFixed(1) + 's';
  }
  if (event.kind === 'eval') {
    return '[eval ' + num(payload.update, 0) + ']'
      + ' win=' + (num(payload.win_rate, 0) * 100).toFixed(0) + '%'
      + ' loss=' + (num(payload.loss_rate, 0) * 100).toFixed(0) + '%'
      + ' timeout=' + (num(payload.timeout_rate, 0) * 100).toFixed(0) + '%'
      + ' return=' + num(payload.mean_return, 0).toFixed(3);
  }
  if (event.kind === 'done') {
    const finalEval = payload.final_eval || {};
    return '[done] total=' + num(payload.total_seconds, 0).toFixed(1) + 's'
      + ' win=' + (num(finalEval.win_rate, 0) * 100).toFixed(0) + '%'
      + ' checkpoint=' + (payload.checkpoint ? basename(String(payload.checkpoint).replace(/\\/g, '/')) : '—');
  }
  return '[' + (event.kind || 'event') + ']';
}

function feSetPpoRunning(running) {
  const ppo = state.fe.ppo;
  if (els.fePpoStart) {
    els.fePpoStart.disabled = running;
    els.fePpoStart.textContent = running ? 'PPO 训练中…' : '开始 PPO 真机训练';
  }
  if (els.fePpoStop) {
    els.fePpoStop.hidden = !running;
    els.fePpoStop.disabled = !running || !!ppo.stopRequested;
  }
  for (const key of ['fePpoUpdates', 'fePpoEnvs', 'fePpoRollout', 'fePpoTicks', 'fePpoDevice', 'fePpoFrameMs', 'fePpoVision', 'fePpoEvalEvery', 'fePpoEvalEpisodes', 'fePpoUseLlm']) {
    if (els[key]) els[key].disabled = running;
  }
}

function feSetPpoStatus(label, tone) {
  feSetStatusPill(els.fePpoStatus, label, tone);
}

function fePpoFrameMsLabel(frameMs) {
  const value = num(frameMs, 2);
  if (value === 16) return '真实时间 16ms（1×）';
  if (value === 2) return '快速 2ms（约 8×）';
  if (value === 1) return '极速 1ms（约 16×）';
  return value + 'ms';
}

function feRenderPpoLog() {
  const ppo = state.fe.ppo;
  const log = els.fePpoLog;
  if (!log) return;
  const lines = [];
  for (const event of (ppo.events || []).slice(-16)) lines.push(fePpoEventLine(event));
  if (ppo.job && ppo.job.last_log) {
    const tail = String(ppo.job.last_log).trim().split('\n').slice(-8).join('\n');
    if (tail) {
      lines.push('');
      lines.push(tail);
    }
  }
  if (ppo.job && ppo.job.error) {
    lines.push('');
    lines.push('ERROR: ' + ppo.job.error);
  }
  log.textContent = lines.length ? lines.join('\n') : '等待 PPO 训练任务…';
  log.classList.toggle('has-error', !!(ppo.job && ppo.job.error));
  if (els.fePpoLogStatus) {
    els.fePpoLogStatus.textContent = 'status=' + (ppo.status || 'idle')
      + ' · frame_ms=' + ppo.frameMs + ' · fog=' + (ppo.visionRadius > 0 ? ppo.visionRadius + 'px' : 'off')
      + ' · llm=' + (ppo.useLlm ? 'on' : 'off')
      + ' · eval=' + (ppo.evalEvery > 0 ? '每' + ppo.evalEvery + '轮' : '最后') + '/' + ppo.evalEpisodes
      + ' · events ' + (ppo.events || []).length;
  }
}

function feRenderPpoSummary() {
  const ppo = state.fe.ppo;
  const card = els.fePpoSummaryCard;
  if (!card) return;
  const done = ppo.donePayload;
  if (!done) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const finalEval = (done.final_eval) || (ppo.job && ppo.job.summary && ppo.job.summary.final_eval) || {};
  const totalSeconds = num(done.total_seconds, num(ppo.job && ppo.job.elapsed, 0));
  const updates = num(ppo.job && ppo.job.updates_total, ppo.updatesTotal);
  const envs = num(ppo.job && ppo.job.envs, 2);
  const pct = (value) => (num(value, 0) * 100).toFixed(0) + '%';
  const rows = [
    ['总耗时', totalSeconds.toFixed(1) + ' s'],
    ['更新轮数', updates],
    ['每轮均秒', updates > 0 ? (totalSeconds / updates).toFixed(1) + ' s' : '—'],
    ['Win / Loss / Timeout', pct(finalEval.win_rate) + ' / ' + pct(finalEval.loss_rate) + ' / ' + pct(finalEval.timeout_rate)],
    ['Final return', num(finalEval.mean_return, 0).toFixed(3)],
    ['Eval 局数', num(finalEval.episodes, 0)],
    ['我方 / 敌方 HP', pct(finalEval.mean_own_hp_frac) + ' / ' + pct(finalEval.mean_enemy_hp_frac)],
    ['平均决策数', num(finalEval.mean_decisions, 0).toFixed(1)],
    ['吞吐', envs >= 2
      ? envs + ' 实例并行 · ' + (totalSeconds > 0 ? (updates / totalSeconds).toFixed(2) : '0.00') + ' 轮/s'
      : '单实例'],
    ['仿真速度', fePpoFrameMsLabel(ppo.frameMs)],
    ['观测迷雾', ppo.visionRadius > 0 ? ppo.visionRadius + ' px' : '关闭'],
    ['评估设置', (ppo.evalEvery > 0 ? '每 ' + ppo.evalEvery + ' 轮' : '仅最后') + ' · ' + ppo.evalEpisodes + ' 局'],
    ['战略层', ppo.useLlm ? 'LLM 启用（缺 key 自动退回规则）' : '关闭'],
  ];
  if (els.fePpoSummary) {
    els.fePpoSummary.innerHTML = '';
    for (const pair of rows) {
      const row = document.createElement('div');
      row.className = 'kv';
      const label = document.createElement('span');
      label.textContent = pair[0];
      const value = document.createElement('b');
      value.textContent = String(pair[1]);
      row.appendChild(label);
      row.appendChild(value);
      els.fePpoSummary.appendChild(row);
    }
  }
  const checkpoint = done.checkpoint
    || ((ppo.job && ppo.job.summary && ppo.job.summary.config && ppo.job.summary.config.out_dir)
      ? String(ppo.job.summary.config.out_dir).replace(/\\/g, '/') + '/checkpoint.pt'
      : '');
  if (els.fePpoCheckpoint && els.fePpoCheckpointPath) {
    if (checkpoint) {
      els.fePpoCheckpoint.hidden = false;
      els.fePpoCheckpointPath.textContent = checkpoint;
      els.fePpoCheckpointPath.title = checkpoint;
    } else {
      els.fePpoCheckpoint.hidden = true;
    }
  }
  feSetText(els.fePpoSummaryBadge, ppo.stopRequested ? 'stopped' : 'final');
}

function feDrawPpoCharts() {
  const ppo = state.fe.ppo;
  const updates = fePpoUpdateEvents();
  const evals = fePpoEvalEvents();

  // 1. 对战得分趋势（训练得分 vs 实战盲测得分）
  const trainReturns = updates.map((update) => {
    const val = numOrNull(update.rolling_return) !== null
      ? num(update.rolling_return, 0)
      : num(update.mean_return, 0);
    return { x: num(update.update, 0), y: val };
  });
  const evalReturns = evals
    .filter((ev) => numOrNull(ev.mean_return) !== null)
    .map((ev) => ({ x: num(ev.update, 0), y: num(ev.mean_return, 0) }));

  const returnSeries = [{ label: '训练得分', color: DASH_COLORS.cyan, points: trainReturns }];
  if (evalReturns.length) {
    returnSeries.push({ label: '实战盲测', color: DASH_COLORS.green, points: evalReturns });
  }
  const returnRange = fePpoSeriesRange([trainReturns, evalReturns]);
  drawDashLine(els.fePpoChartReturn, {
    series: returnSeries,
    yMin: returnRange.yMin,
    yMax: returnRange.yMax,
    yFormatter: (value) => (value >= 0 ? '+' : '') + value.toFixed(2),
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待 update / eval 事件…',
  });

  // 2. 实战盲测与训练胜率
  const trainWins = updates
    .filter((u) => num(u.episodes, 0) > 0 || num(u.total_episodes, 0) > 0 || !!u.has_completed)
    .map((update) => ({
      x: num(update.update, 0),
      y: num(numOrNull(update.rolling_win_rate) !== null ? update.rolling_win_rate : update.win_rate, 0) * 100,
    }));
  const evalWins = evals.map((ev) => ({ x: num(ev.update, 0), y: num(ev.win_rate, 0) * 100 }));

  const winSeries = [];
  if (evalWins.length) {
    winSeries.push({ label: '盲测胜率', color: DASH_COLORS.green, points: evalWins });
  }
  if (trainWins.length) {
    winSeries.push({ label: '训练胜率', color: DASH_COLORS.cyan, points: trainWins });
  }
  drawDashLine(els.fePpoChartWin, {
    series: winSeries.length ? winSeries : [{ color: DASH_COLORS.green, points: evalWins }],
    yMin: 0,
    yMax: 100,
    yFormatter: (value) => Math.round(value) + '%',
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待对战胜率数据…',
  });

  // 3. 双方剩余血量比（战力走势）
  const ownHp = evals
    .filter((ev) => numOrNull(ev.mean_own_hp_frac) !== null)
    .map((ev) => ({ x: num(ev.update, 0), y: num(ev.mean_own_hp_frac, 0) * 100 }));
  const enemyHp = evals
    .filter((ev) => numOrNull(ev.mean_enemy_hp_frac) !== null)
    .map((ev) => ({ x: num(ev.update, 0), y: num(ev.mean_enemy_hp_frac, 0) * 100 }));
  drawDashLine(els.fePpoChartHp, {
    series: [
      { label: '我方HP%', color: DASH_COLORS.own, points: ownHp },
      { label: '敌方HP%', color: DASH_COLORS.enemy, points: enemyHp },
    ],
    yMin: 0,
    yMax: 100,
    yFormatter: (value) => Math.round(value) + '%',
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待评估对局产出战力血量比…',
  });

  // 4. 得分构成分解（Reward Breakdown）
  const killsPoints = updates
    .filter((u) => u.reward_components && numOrNull(u.reward_components.kills) !== null)
    .map((u) => ({ x: num(u.update, 0), y: num(u.reward_components.kills, 0) }));
  const engPoints = updates
    .filter((u) => u.reward_components && numOrNull(u.reward_components.engagement) !== null)
    .map((u) => ({ x: num(u.update, 0), y: num(u.reward_components.engagement, 0) }));
  const lossPoints = updates
    .filter((u) => u.reward_components && numOrNull(u.reward_components.losses) !== null)
    .map((u) => ({ x: num(u.update, 0), y: num(u.reward_components.losses, 0) }));
  const rewardRange = fePpoSeriesRange([killsPoints, engPoints, lossPoints]);
  drawDashLine(els.fePpoChartReward, {
    series: [
      { label: '击杀得分', color: DASH_COLORS.green, points: killsPoints },
      { label: '推进交火', color: DASH_COLORS.cyan, points: engPoints },
      { label: '战损扣分', color: DASH_COLORS.enemy, points: lossPoints },
    ],
    yMin: rewardRange.yMin,
    yMax: rewardRange.yMax,
    yFormatter: (value) => (value >= 0 ? '+' : '') + value.toFixed(2),
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待奖励分项统计…',
  });

  // 5. 策略探索与损失
  const entropy = updates.map((update) => ({ x: num(update.update, 0), y: num(update.entropy, 0) }));
  const policyLoss = updates.map((update) => ({ x: num(update.update, 0), y: num(update.policy_loss, 0) }));
  const entropyRange = fePpoSeriesRange([entropy, policyLoss]);
  drawDashLine(els.fePpoChartEntropy, {
    series: [
      { color: DASH_COLORS.cyan, points: entropy },
      { color: DASH_COLORS.amber, points: policyLoss },
    ],
    yMin: entropyRange.yMin,
    yMax: entropyRange.yMax,
    yFormatter: (value) => value.toFixed(2),
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待 update 事件…',
  });

  // 6. 每轮耗时柱状图
  const durations = fePpoDurations(updates);
  drawDashBars(els.fePpoChartDuration, {
    bars: updates.map((update, index) => ({
      label: '#' + num(update.update, 0),
      value: Math.max(0.01, durations[index] || 0),
      color: DASH_COLORS.cyan,
    })),
    yFormatter: (value) => value.toFixed(1) + 's',
    valueFormatter: (value) => value.toFixed(1),
    xFormatter: (label) => String(label),
    emptyText: '等待 update 事件…',
  });

  const evaluations = (ppo.job && Array.isArray(ppo.job.evaluations)) ? ppo.job.evaluations : [];
  const lastEval = evaluations.length ? evaluations[evaluations.length - 1] : null;
  if (els.fePpoEvalHint) {
    els.fePpoEvalHint.textContent = lastEval
      ? 'eval #' + num(lastEval.update, 0) + ' · win ' + (num(lastEval.win_rate, 0) * 100).toFixed(0)
        + '% · timeout ' + (num(lastEval.timeout_rate, 0) * 100).toFixed(0)
        + '% · return ' + num(lastEval.mean_return, 0).toFixed(2)
      : '—';
  }
}

function feRenderPpoKpis() {
  const ppo = state.fe.ppo;
  const job = ppo.job || {};
  const updates = fePpoUpdateEvents();
  const latest = job.latest_update || (updates.length ? updates[updates.length - 1] : {});
  const total = numOrNull(job.updates_total) !== null ? num(job.updates_total, ppo.updatesTotal) : ppo.updatesTotal;
  const done = numOrNull(job.updates_done) !== null ? num(job.updates_done, ppo.updatesDone) : ppo.updatesDone;
  const elapsed = numOrNull(job.elapsed) !== null ? num(job.elapsed, 0) : (ppo.startedAt ? (Date.now() - ppo.startedAt) / 1000 : 0);
  const avg = done > 0 ? elapsed / done : 0;
  const durations = fePpoDurations(updates);
  const lastDuration = durations.length ? durations[durations.length - 1] : avg;
  const envs = num(job.envs, num(els.fePpoEnvs && els.fePpoEnvs.value, 2));
  const rollout = num(job.rollout_steps, num(els.fePpoRollout && els.fePpoRollout.value, 20));
  const dps = lastDuration > 0 ? (envs * rollout) / lastDuration : 0;
  feSetText(els.fePpoKpiProgress, done);
  feSetText(els.fePpoKpiProgressTotal, '/ ' + total + ' 轮');
  feSetText(els.fePpoKpiSeconds, Math.round(elapsed));
  feSetText(els.fePpoKpiAvg, avg.toFixed(1));
  feSetText(els.fePpoKpiDps, dps.toFixed(1));
  const anyCompleted = updates.some((u) => num(u.episodes, 0) > 0 || num(u.total_episodes, 0) > 0 || !!u.has_completed);
  const bestReturn = numOrNull(latest.rolling_return) !== null
    ? latest.rolling_return
    : numOrNull(latest.mean_return);

  if (bestReturn !== null && bestReturn !== undefined && (anyCompleted || bestReturn !== 0)) {
    feSetText(els.fePpoKpiReturn, Number(bestReturn).toFixed(2));
    if (els.fePpoKpiReturnFoot) {
      els.fePpoKpiReturnFoot.textContent = num(latest.episodes, 0) > 0
        ? '本轮结算 ' + latest.episodes + ' 局 · mean ' + Number(latest.mean_return).toFixed(2)
        : (anyCompleted ? '最近结算滑动均值 (rolling)' : '对战采样进行中 · 待结算');
    }
  } else {
    feSetText(els.fePpoKpiReturn, '0.00');
    if (els.fePpoKpiReturnFoot) {
      els.fePpoKpiReturnFoot.textContent = done > 0 ? '对战采样进行中 · 待结算' : 'latest_update.mean_return';
    }
  }

  const evalsList = fePpoEvalEvents();
  const lastEv = evalsList.length ? evalsList[evalsList.length - 1] : null;
  let displayWinRate = null;
  let winSource = '';
  if (lastEv && numOrNull(lastEv.win_rate) !== null) {
    displayWinRate = num(lastEv.win_rate, 0);
    winSource = 'eval #' + num(lastEv.update, 0) + ' 盲测胜率';
  } else if (anyCompleted) {
    displayWinRate = numOrNull(latest.rolling_win_rate) !== null
      ? num(latest.rolling_win_rate, 0)
      : (numOrNull(latest.win_rate) !== null ? num(latest.win_rate, 0) : 0);
    winSource = '训练采样胜率（最近完成局）';
  }

  if (displayWinRate !== null) {
    feSetText(els.fePpoKpiWin, (displayWinRate * 100).toFixed(1));
    if (els.fePpoKpiWinFoot) {
      els.fePpoKpiWinFoot.textContent = winSource || '实战评估胜率';
    }
  } else {
    feSetText(els.fePpoKpiWin, '0.0');
    if (els.fePpoKpiWinFoot) {
      els.fePpoKpiWinFoot.textContent = done > 0 ? '等待对局结束或首轮评估' : '真机盲测评估胜率';
    }
  }
  feSetText(els.fePpoKpiEntropy, num(latest.entropy, 0).toFixed(3));
  const deviceLabel = String(job.device || (els.fePpoDevice && els.fePpoDevice.value) || 'auto').toLowerCase();
  feSetText(els.fePpoKpiDevice, deviceLabel.toUpperCase());
  if (els.fePpoKpiDeviceFoot) {
    els.fePpoKpiDeviceFoot.textContent = 'frame_ms=' + ppo.frameMs
      + ' · 雾=' + (ppo.visionRadius > 0 ? ppo.visionRadius + 'px' : '关闭')
      + ' · LLM=' + (ppo.useLlm ? 'on' : 'off');
  }
  const percent = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  if (els.fePpoProgress) els.fePpoProgress.style.width = percent + '%';
  feSetText(els.fePpoProgressLabel, done + ' / ' + total + ' 轮');
  if (els.fePpoSeconds) els.fePpoSeconds.textContent = Math.round(elapsed) + 's';
  if (els.fePpoCurrent) {
    if (ppo.status === 'running') els.fePpoCurrent.textContent = '更新 ' + Math.min(done + 1, total || done + 1) + ' / ' + (total || '—');
    else if (ppo.status === 'done') els.fePpoCurrent.textContent = '已完成';
    else if (ppo.status === 'failed') els.fePpoCurrent.textContent = ppo.stopRequested ? '已停止' : '失败';
    else els.fePpoCurrent.textContent = '等待开始';
  }
  if (ppo.status === 'running') feSetPpoStatus('运行中', 'running');
  else if (ppo.status === 'done') feSetPpoStatus('已完成', 'done');
  else if (ppo.status === 'failed') feSetPpoStatus(ppo.stopRequested ? '已停止' : '失败', ppo.stopRequested ? 'warn' : 'bad');
  else feSetPpoStatus('空闲', 'idle');
}

function feRenderPpoDashboard() {
  feRenderPpoKpis();
  feRenderPpoLog();
  feRenderPpoSummary();
  feRenderPpoReward();
  feRenderPpoStrategy();
  feDrawPpoCharts();
}

function feApplyTrainMode(mode) {
  const target = mode === 'ppo' ? 'ppo' : 'collect';
  state.fe.trainMode = target;
  for (const button of document.querySelectorAll('[data-train-mode-btn]')) {
    const active = button.getAttribute('data-train-mode-btn') === target;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  for (const block of document.querySelectorAll('[data-train-mode]')) {
    block.hidden = block.getAttribute('data-train-mode') !== target;
  }
  if (els.feTrainModeNote) {
    els.feTrainModeNote.textContent = target === 'ppo'
      ? 'PPO 真机训练：Strategy-in-the-loop · Artillery 挑战图（22 vs 25 AI）'
      : '数据采集：多线程录制真机对局（默认）';
  }
  if (target === 'ppo') {
    feRenderPpoDashboard();
    setTimeout(() => feDrawPpoCharts(), 40);
  } else {
    feRenderTrainDashboard(state.fe.trainJob);
    renderFeTrainArtifacts(state.fe.trainArtifacts);
    setTimeout(() => feDrawTrainCharts(), 40);
  }
}

function startFePpoPolling() {
  stopFePpoPolling();
  state.fe.ppo.timer = setInterval(fePollPpo, 1500);
  void fePollPpo();
}

function stopFePpoPolling() {
  const ppo = state.fe.ppo;
  if (ppo.timer) {
    clearInterval(ppo.timer);
    ppo.timer = null;
  }
}

async function fePollPpo() {
  const ppo = state.fe.ppo;
  if (!ppo.jobId || ppo.polling) return;
  ppo.polling = true;
  try {
    const payload = await Api.rwppPpoEvents(ppo.jobId, ppo.cursor || 0);
    const events = Array.isArray(payload && payload.events) ? payload.events : [];
    for (const event of events) {
      if (!event || numOrNull(event.seq) === null) continue;
      const seq = num(event.seq, 0);
      ppo.cursor = Math.max(ppo.cursor || 0, seq);
      if (!ppo.events.some((item) => item && item.seq === seq)) ppo.events.push(event);
    }
    ppo.events.sort((left, right) => num(left.seq, 0) - num(right.seq, 0));
    const job = payload && payload.job ? payload.job : null;
    if (job) {
      ppo.job = job;
      if (numOrNull(job.updates_total) !== null) ppo.updatesTotal = num(job.updates_total, ppo.updatesTotal);
      if (numOrNull(job.updates_done) !== null) ppo.updatesDone = num(job.updates_done, ppo.updatesDone);
      if (job.latest_update) ppo.latest = job.latest_update;
      if (Array.isArray(job.evaluations)) ppo.evaluations = job.evaluations;
      if (numOrNull(job.frame_ms) !== null) ppo.frameMs = num(job.frame_ms, ppo.frameMs);
      if (numOrNull(job.vision_radius) !== null) ppo.visionRadius = num(job.vision_radius, ppo.visionRadius);
      if (numOrNull(job.eval_every) !== null) ppo.evalEvery = num(job.eval_every, ppo.evalEvery);
      if (numOrNull(job.eval_episodes) !== null) ppo.evalEpisodes = num(job.eval_episodes, ppo.evalEpisodes);
      if (job.use_llm !== undefined && job.use_llm !== null) ppo.useLlm = !!job.use_llm;
    }
    const doneEvent = ppo.events.filter((item) => item.kind === 'done').pop();
    if (doneEvent) ppo.donePayload = doneEvent.payload || {};
    const status = (job && job.status) || (payload && payload.status) || 'running';
    ppo.status = status;
    feRenderPpoDashboard();
    if (status !== 'running') {
      stopFePpoPolling();
      feSetPpoRunning(false);
      if (status === 'done') {
        const seconds = num(ppo.donePayload && ppo.donePayload.total_seconds, num(ppo.job && ppo.job.elapsed, 0));
        toast('PPO 训练完成（' + Math.round(seconds) + 's）', 'ok');
      } else {
        const stopped = ppo.stopRequested;
        const message = String((ppo.job && ppo.job.error) || '未知错误').slice(0, 140);
        toast(stopped ? 'PPO 训练已停止' : 'PPO 训练失败：' + message, stopped ? 'ok' : 'error');
      }
      await refreshRwppStatus(true);
      feRenderPpoDashboard();
    }
  } catch (error) {
    if (error && error.status === 404) {
      stopFePpoPolling();
      ppo.status = 'failed';
      feSetPpoRunning(false);
      toast('PPO 任务不存在或已结束', 'error');
      feRenderPpoDashboard();
    } else if (error && error.status === 503) {
      stopFePpoPolling();
      ppo.status = 'failed';
      feSetPpoRunning(false);
      handleFeError(error, 'PPO 训练');
      feRenderPpoDashboard();
    }
  } finally {
    ppo.polling = false;
  }
}

async function feStartPpo() {
  const ppo = state.fe.ppo;
  if (ppo.status === 'running') {
    toast('PPO 训练正在进行中，请等待或先停止', 'error');
    return;
  }
  if (state.mode === 'demo') {
    toast('演示模式下无法连接真机服务', 'error');
    return;
  }
  const updates = clampInt(els.fePpoUpdates && els.fePpoUpdates.value, 1, 500, 20);
  const envs = clampInt(els.fePpoEnvs && els.fePpoEnvs.value, 1, 6, 2);
  const rollout = clampInt(els.fePpoRollout && els.fePpoRollout.value, 1, 512, 20);
  const episodeTicks = clampInt(els.fePpoTicks && els.fePpoTicks.value, 120, 216000, 3600);
  const device = (els.fePpoDevice && els.fePpoDevice.value) || 'auto';
  const frameMs = clampInt(els.fePpoFrameMs && els.fePpoFrameMs.value, 0, 16, 2);
  const visionRadius = clampInt(els.fePpoVision && els.fePpoVision.value, 0, 2000, 260);
  const evalEvery = clampInt(els.fePpoEvalEvery && els.fePpoEvalEvery.value, 0, 500, 0);
  const evalEpisodes = clampInt(els.fePpoEvalEpisodes && els.fePpoEvalEpisodes.value, 1, 16, 1);
  const useLlm = !!(els.fePpoUseLlm && els.fePpoUseLlm.checked);
  const selectedMap = (els.fePpoMap && els.fePpoMap.value) || 'IronZero25';
  let ppoMapName = selectedMap;
  let ppoMapPath = `maps/ironzero/${selectedMap}.tmx`;
  let unitsPerTeam = clampInt(els.fePpoUnits && els.fePpoUnits.value, 1, 25, 25);
  let mapWidth = 2700.0;
  let mapHeight = 2500.0;

  const originalMap = feMapList().find((m) => m.name === selectedMap || m.id === selectedMap);
  const isIronZero = selectedMap.toLowerCase().startsWith('ironzero');

  if (isIronZero) {
    ppoMapName = `IronZero${unitsPerTeam}`;
    ppoMapPath = `maps/ironzero/${ppoMapName}.tmx`;
    mapWidth = 2700.0;
    mapHeight = 2500.0;
  } else if (originalMap) {
    ppoMapName = originalMap.name;
    ppoMapPath = originalMap.relative || '';
    mapWidth = num(originalMap.width_px, 2700.0);
    mapHeight = num(originalMap.height_px, 2500.0);
    if (selectedMap === 'Artillery' && els.fePpoUnits && els.fePpoUnits.value === '25') {
      unitsPerTeam = 22;
    }
  } else if (selectedMap === 'Artillery') {
    ppoMapName = 'Artillery';
    ppoMapPath = 'maps/challenge/Artillery.tmx';
    mapWidth = 2700.0;
    mapHeight = 2500.0;
    unitsPerTeam = 22;
  }

  const initMode = (els.fePpoInitMode && els.fePpoInitMode.value) || 'warmstart';
  let initCheckpoint = null;
  if (initMode === 'checkpoint') {
    initCheckpoint = (els.fePpoInitCheckpoint && els.fePpoInitCheckpoint.value) || null;
  } else if (initMode === 'warmstart') {
    initCheckpoint = 'runs/bc_ironzero25/checkpoint.pt';
  }

  ppo.jobId = null;
  ppo.job = null;
  ppo.events = [];
  ppo.cursor = 0;
  ppo.latest = null;
  ppo.evaluations = [];
  ppo.donePayload = null;
  ppo.status = 'running';
  ppo.stopRequested = false;
  ppo.restoreTried = true;
  ppo.startedAt = Date.now();
  ppo.updatesTotal = updates;
  ppo.updatesDone = 0;
  ppo.frameMs = frameMs;
  ppo.visionRadius = visionRadius;
  ppo.evalEvery = evalEvery;
  ppo.evalEpisodes = evalEpisodes;
  ppo.useLlm = useLlm;
  feSetPpoRunning(true);
  feSetPpoStatus('启动中', 'running');
  feRenderPpoDashboard();
  try {
    const response = await Api.rwppPpo({
      updates: updates,
      envs: envs,
      rollout_steps: rollout,
      episode_ticks: episodeTicks,
      device: device,
      map_name: ppoMapName,
      map_path: ppoMapPath,
      map_width: mapWidth,
      map_height: mapHeight,
      frame_ms: frameMs,
      vision_radius: visionRadius,
      eval_every: evalEvery,
      eval_episodes: evalEpisodes,
      use_llm: useLlm,
      units_per_team: unitsPerTeam,
      init_checkpoint: initCheckpoint,
      init_mode: initMode,
    });
    const jobId = response && (response.job_id || response.id);
    if (!jobId) throw new Error('后端未返回 PPO 任务 id');
    ppo.jobId = String(jobId);
    ppo.status = 'running';
    startFePpoPolling();
    const initDesc = initMode === 'warmstart' ? '预热策略(BC)' : (initMode === 'checkpoint' ? '选定Checkpoint' : '从零随机');
    toast('PPO 真机训练已开始 · ' + ppoMapName + ' · ' + unitsPerTeam + 'v' + unitsPerTeam + ' 坦克 · 起点: ' + initDesc
      + ' · ' + updates + ' 轮 × ' + envs + ' 实例 · frame_ms=' + frameMs
      + ' · fog=' + (visionRadius || 'off')
      + (useLlm ? ' · LLM 战略层' : ''), 'ok');
  } catch (error) {
    ppo.status = 'failed';
    feSetPpoRunning(false);
    if (error && error.status === 409) {
      toast('已有训练或实机任务正在运行中，请先在控制台点击【停止训练】或等待完成', 'error');
      void feRestorePpoJob(true);
    } else {
      handleFeError(error, 'PPO 训练');
    }
    feRenderPpoDashboard();
  }
}

async function feStopPpo() {
  const ppo = state.fe.ppo;
  if (!ppo.jobId || ppo.status !== 'running') return;
  ppo.stopRequested = true;
  if (els.fePpoStop) els.fePpoStop.disabled = true;
  try {
    await Api.rwppPpoStop(ppo.jobId);
    toast('已请求停止 PPO 训练', 'ok');
  } catch (error) {
    handleFeError(error, '停止 PPO 训练');
  }
}

async function feRestorePpoJob(force) {
  const ppo = state.fe.ppo;
  if (ppo.jobId && ppo.status === 'running') return;
  if (ppo.restoreTried && !force) return;
  ppo.restoreTried = true;
  try {
    const payload = await Api.rwppJobs();
    const jobs = ((payload && payload.jobs) || []).filter((job) => job && job.kind === 'ppo');
    const running = jobs.find((job) => job.status === 'running');
    if (!running) {
      if (state.rwpp.ppoRunning > 0 && els.fePpoHint) {
        els.fePpoHint.textContent = '检测到宿主机有 PPO 任务正在运行，请稍后刷新或重新进入本页查看。';
      }
      return;
    }
    ppo.jobId = String(running.id);
    ppo.status = 'running';
    ppo.job = running;
    ppo.stopRequested = false;
    ppo.updatesTotal = num(running.updates_total, 0);
    ppo.updatesDone = num(running.updates_done, 0);
    ppo.startedAt = Date.now() - num(running.elapsed, 0) * 1000;
    if (els.fePpoUpdates && running.updates_total) els.fePpoUpdates.value = String(running.updates_total);
    if (els.fePpoEnvs && running.envs) els.fePpoEnvs.value = String(running.envs);
    if (els.fePpoRollout && running.rollout_steps) els.fePpoRollout.value = String(running.rollout_steps);
    if (els.fePpoTicks && running.episode_ticks) els.fePpoTicks.value = String(running.episode_ticks);
    if (els.fePpoDevice && running.device) els.fePpoDevice.value = String(running.device);
    if (numOrNull(running.frame_ms) !== null) {
      ppo.frameMs = num(running.frame_ms, 2);
      if (els.fePpoFrameMs) els.fePpoFrameMs.value = String(ppo.frameMs);
    }
    if (numOrNull(running.vision_radius) !== null) {
      ppo.visionRadius = num(running.vision_radius, 260);
      if (els.fePpoVision) els.fePpoVision.value = String(ppo.visionRadius);
    }
    if (numOrNull(running.eval_every) !== null) {
      ppo.evalEvery = num(running.eval_every, 0);
      if (els.fePpoEvalEvery) els.fePpoEvalEvery.value = String(ppo.evalEvery);
    }
    if (numOrNull(running.eval_episodes) !== null) {
      ppo.evalEpisodes = num(running.eval_episodes, 1);
      if (els.fePpoEvalEpisodes) els.fePpoEvalEpisodes.value = String(ppo.evalEpisodes);
    }
    if (running.use_llm !== undefined && running.use_llm !== null) {
      ppo.useLlm = !!running.use_llm;
      if (els.fePpoUseLlm) els.fePpoUseLlm.checked = ppo.useLlm;
      if (els.fePpoLlmNote) els.fePpoLlmNote.hidden = !ppo.useLlm;
    }
    feSetPpoRunning(true);
    startFePpoPolling();
    if (state.tab === 'train') {
      feApplyTrainMode('ppo');
      toast('已恢复运行中的 PPO 训练任务', 'ok');
    }
  } catch (error) {
    // restore is best effort
  }
}


/* ============================================================
 * task-11：模型推理 checkpoint / PPO 评估与 LLM / 奖励分解 / 战略指挥
 * ============================================================ */

Object.assign(state.fe, {
  checkpointList: [],
  checkpointsLoaded: false,
  checkpointsLoading: false,
});

Object.assign(state.fe.ppo, {
  evalEvery: 0,
  evalEpisodes: 1,
  useLlm: false,
});

function feFormatCheckpointTime(mtimeStr) {
  const d = parseBeijingDate(mtimeStr);
  if (!d) return String(mtimeStr || '').slice(5, 16);
  const now = new Date();
  const isToday = d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
  const timePart = pad2(d.getHours()) + ':' + pad2(d.getMinutes());
  if (isToday) return '今天 ' + timePart;
  const yesterday = new Date(now.getTime() - 86400000);
  const isYesterday = d.getFullYear() === yesterday.getFullYear() && d.getMonth() === yesterday.getMonth() && d.getDate() === yesterday.getDate();
  if (isYesterday) return '昨天 ' + timePart;
  return pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' + timePart;
}

function feInferMapFromPath(path) {
  const s = String(path || '').toLowerCase();
  if (s.includes('king of the middle')) return 'King of the Middle (3p)';
  if (s.includes('ironzero25')) return 'IronZero25';
  if (s.includes('ironzero20')) return 'IronZero20';
  if (s.includes('ironzero15')) return 'IronZero15';
  if (s.includes('ironzero10')) return 'IronZero10';
  if (s.includes('ironzero5')) return 'IronZero5';
  if (s.includes('artillery')) return 'Artillery';
  if (s.includes('lake')) return 'Lake (2p)';
  if (s.includes('hills')) return 'Hills (2p)';
  return '';
}

function feCheckpointDisplayName(item, index) {
  if (!item) return '';
  const parts = [];
  if (index === 0) {
    parts.push('⭐ [最新]');
  }
  const mapName = item.map || feInferMapFromPath(item.path || item.name);
  if (mapName) {
    parts.push('[' + mapName.replace(/\[p\d+\]/, '').trim() + ']');
  }
  const timeStr = feFormatCheckpointTime(item.mtime);
  if (timeStr) parts.push(timeStr);

  if (numOrNull(item.update) !== null) {
    parts.push('upd ' + num(item.update, 0));
  }
  if (item.eval && numOrNull(item.eval.win_rate) !== null) {
    parts.push('胜率 ' + Math.round(item.eval.win_rate * 100) + '%');
  }

  const rawPath = String(item.path || item.name || '').replace(/\\/g, '/');
  const fileName = basename(rawPath);
  const parentName = basename(rawPath.replace(/[/\\][^/\\]+$/, ''));
  if (fileName.includes('best')) {
    parts.push('最优(best)');
  }
  const shortId = parentName ? parentName.replace('rwpp-ppo-', '') : '';
  if (shortId && shortId !== parentName) {
    parts.push('(' + shortId + ')');
  } else if (parentName) {
    parts.push('(' + parentName + ')');
  }
  return parts.length ? parts.join(' · ') : (item.name || rawPath);
}

async function loadFeCheckpoints(force) {
  const fe = state.fe;
  if (fe.checkpointsLoading) return;
  if (fe.checkpointsLoaded && !force) return;
  fe.checkpointsLoading = true;
  if (state.mode === 'demo') {
    const list = (typeof DEMO !== 'undefined' && Array.isArray(DEMO.checkpoints)) ? DEMO.checkpoints : [];
    fe.checkpointList = list.slice();
    fe.checkpointsLoaded = true;
    fe.checkpointsLoading = false;
    renderFeCheckpointOptions();
    return;
  }
  try {
    const payload = await Api.checkpoints();
    const list = Array.isArray(payload) ? payload : ((payload && payload.checkpoints) || []);
    fe.checkpointList = list.slice().sort((left, right) => String(right.mtime || '').localeCompare(String(left.mtime || '')));
    fe.checkpointsLoaded = true;
  } catch (error) {
    fe.checkpointList = [];
    fe.checkpointsLoaded = false;
  } finally {
    fe.checkpointsLoading = false;
    renderFeCheckpointOptions();
  }
}

function renderFeCheckpointOptions() {
  const list = state.fe.checkpointList || [];
  const select = els.feCheckpointSelect;
  if (select) {
    const previous = select.value;
    select.innerHTML = '';
    if (!list.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '（暂无检测到 Checkpoint，请点击上方刷新）';
      select.appendChild(option);
      select.disabled = true;
    } else {
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        const option = document.createElement('option');
        option.value = item.path;
        option.textContent = feCheckpointDisplayName(item, i)
          + (item.size_mb ? ' · ' + item.size_mb + 'MB' : '');
        select.appendChild(option);
      }
      select.disabled = false;
      select.value = list.some((item) => item.path === previous) ? previous : list[0].path;
    }
  }
  if (els.fePolicy) {
    const modelOption = Array.prototype.find.call(els.fePolicy.options, (option) => option.value === 'model');
    if (modelOption) {
      modelOption.disabled = false; // 用户随时可以自由选择 PPO 模型策略进行推理
    }
  }
  // 同步填充数据采集面板的模型 Checkpoint 下拉框
  const trainSelect = els.feTrainCheckpointSelect;
  if (trainSelect) {
    const previous = trainSelect.value;
    trainSelect.innerHTML = '';
    if (!list.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '（无可用模型）';
      trainSelect.appendChild(option);
      trainSelect.disabled = true;
    } else {
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        const option = document.createElement('option');
        option.value = item.path;
        option.textContent = feCheckpointDisplayName(item, i)
          + (item.size_mb ? ' · ' + item.size_mb + 'MB' : '');
        trainSelect.appendChild(option);
      }
      trainSelect.disabled = false;
      trainSelect.value = list.some((item) => item.path === previous) ? previous : list[0].path;
    }
  }
  // 同步填充 PPO 训练面板的初始 Checkpoint 下拉框
  const ppoSelect = els.fePpoInitCheckpoint;
  if (ppoSelect) {
    const previous = ppoSelect.value;
    ppoSelect.innerHTML = '';
    if (!list.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '（无可用权重）';
      ppoSelect.appendChild(option);
      ppoSelect.disabled = true;
    } else {
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        const option = document.createElement('option');
        option.value = item.path;
        option.textContent = feCheckpointDisplayName(item, i)
          + (item.size_mb ? ' · ' + item.size_mb + 'MB' : '');
        ppoSelect.appendChild(option);
      }
      ppoSelect.disabled = false;
      const bestDefault = list.find((item) => /bc|best|ft/i.test(item.name || item.path)) || list[0];
      ppoSelect.value = list.some((item) => item.path === previous) ? previous : (bestDefault ? bestDefault.path : list[0].path);
    }
  }
  feSyncInferPolicyUi();
  feSyncTrainPolicyUi();
  feSyncPpoConfigUi();
}

function feSyncTrainPolicyUi() {
  const isModel = !!(els.feTrainPolicy && els.feTrainPolicy.value === 'model');
  if (els.feTrainCheckpointField) els.feTrainCheckpointField.hidden = !isModel;
}

function feSyncPpoConfigUi() {
  const mapChoice = (els.fePpoMap && els.fePpoMap.value) || 'IronZero25';
  const isIronZero = mapChoice.toLowerCase().startsWith('ironzero');
  const isArtillery = mapChoice.toLowerCase().includes('artillery');

  if (els.fePpoUnitsField) {
    els.fePpoUnitsField.hidden = !isIronZero;
  }
  if (els.fePpoMapHint) {
    if (isIronZero) {
      els.fePpoMapHint.textContent = '竞技场纯微操：开局双方自动生成指定数量的纯坦克部队，直接进行集火与拉扯对决。';
    } else if (isArtillery) {
      els.fePpoMapHint.textContent = '原版战役挑战：开局双方预置 22 辆火炮部队与阵地建筑，直接爆发大兵团炮火决战。';
    } else {
      els.fePpoMapHint.textContent = '原版遭遇战规则：初始部队严格遵循原版地图设定（双方各 1 个主基地 + 1 辆建造车），无预置坦克部队。当前微操模型不包含采矿造工厂等宏观运营指令，双方开局不会产生部队交火。若要训练坦克走位集火，强烈建议选用上方【对称竞技场】或原版【Artillery】！';
    }
  }

  if (els.fePpoUnits) {
    if (mapChoice === 'IronZero25') els.fePpoUnits.value = '25';
    else if (mapChoice === 'IronZero20') els.fePpoUnits.value = '20';
    else if (mapChoice === 'IronZero15') els.fePpoUnits.value = '15';
    else if (mapChoice === 'IronZero10') els.fePpoUnits.value = '10';
    else if (mapChoice === 'IronZero5') els.fePpoUnits.value = '5';
    else if (isArtillery) els.fePpoUnits.value = '22';
    else {
      els.fePpoUnits.value = '2';
    }
  }
  const initMode = (els.fePpoInitMode && els.fePpoInitMode.value) || 'warmstart';
  if (els.fePpoCheckpointField) {
    els.fePpoCheckpointField.hidden = (initMode !== 'checkpoint');
  }
  if (els.fePpoInitHint) {
    if (initMode === 'warmstart') {
      els.fePpoInitHint.textContent = '预热策略基于真实对战规则蒸馏（BC），坦克开局会自动集火与编队推进，可避免从零探索在前几百轮毫无击杀的问题。';
    } else if (initMode === 'checkpoint') {
      els.fePpoInitHint.textContent = '在上方选定的 Checkpoint 权重基础上继续进行 PPO 在环对抗强化学习。';
    } else {
      els.fePpoInitHint.textContent = '从完全随机参数的神经网络从零探索。RTS 微操动作空间庞大，初始几十轮胜率提升较慢。';
    }
  }
}

function feSyncInferPolicyUi() {
  const isModel = !!(els.fePolicy && els.fePolicy.value === 'model');
  if (els.feCheckpointField) els.feCheckpointField.hidden = !isModel;
  if (els.feModelNote) els.feModelNote.hidden = !isModel;
  if (isModel && !(state.fe.checkpointList || []).length && els.feInferStatus) {
    els.feInferStatus.textContent = '没有可用 checkpoint：先到「实机训练」页跑一次 PPO';
    els.feInferStatus.className = 'hint-line bad';
  }
}

function sandboxStrategyLabel(sandbox) {
  const infer = sandbox.source === 'infer';
  const realReplay = sandbox.checkpoint === 'rwpp-live';
  if (!infer && !realReplay) return '';
  if (infer && sandbox.policy === 'model') {
    const name = sandbox.modelCheckpoint ? String(sandbox.modelCheckpoint).replace(/\\/g, '/') : '';
    return '模型策略' + (name ? ' · ' + name : '') + ' · ';
  }
  return '脚本策略 · ';
}

const REWARD_COMPONENT_LABELS = {
  potential: '势能 potential',
  engagement: '交战 engagement',
  strategy: '战略 strategy',
  kills: '击杀 kills',
  losses: '损失 losses',
  economy: '采矿经济 economy',
  factories: '军工建造 factories',
  production: '战车出厂 production',
  building_loss: '建筑折损 building_loss',
  invalid: '非法 invalid',
  idle: '空闲 idle',
  terminal: '终局 terminal',
  timeout: '超时 timeout',
};

function feRenderPpoReward() {
  const box = els.fePpoReward;
  if (!box) return;
  const updates = fePpoUpdateEvents();
  const latest = updates.length ? updates[updates.length - 1] : null;
  const components = latest && latest.reward_components && typeof latest.reward_components === 'object'
    ? latest.reward_components
    : null;
  const keys = components ? Object.keys(components) : [];
  if (!components || !keys.length) {
    box.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = '完成一局后出现';
    box.appendChild(empty);
    feSetText(els.fePpoRewardChip, '—');
    return;
  }
  const ordered = [];
  for (const key of Object.keys(REWARD_COMPONENT_LABELS)) {
    if (Object.prototype.hasOwnProperty.call(components, key)) ordered.push(key);
  }
  for (const key of keys) {
    if (ordered.indexOf(key) < 0) ordered.push(key);
  }
  let total = 0;
  let maxAbs = 0.0001;
  for (const key of ordered) {
    const value = num(components[key], 0);
    total += value;
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  box.innerHTML = '';
  for (const key of ordered) {
    const value = num(components[key], 0);
    const row = document.createElement('div');
    row.className = 'reward-row';
    const label = document.createElement('span');
    label.className = 'reward-label';
    label.textContent = REWARD_COMPONENT_LABELS[key] || key;
    const bar = document.createElement('span');
    bar.className = 'reward-bar';
    const fill = document.createElement('i');
    fill.className = value >= 0 ? 'pos' : 'neg';
    fill.style.width = Math.max(2, Math.round((Math.abs(value) / maxAbs) * 100)) + '%';
    bar.appendChild(fill);
    const val = document.createElement('span');
    val.className = 'reward-value ' + (value >= 0 ? 'pos' : 'neg');
    val.textContent = (value >= 0 ? '+' : '') + value.toFixed(3);
    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(val);
    box.appendChild(row);
  }
  feSetText(els.fePpoRewardChip, (total >= 0 ? '+' : '') + total.toFixed(3));
}

function feFormatStrategyValue(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value)) return value.map((item) => feFormatStrategyValue(item)).join(', ');
  if (typeof value === 'object') {
    return Object.keys(value).map((key) => key + '=' + feFormatStrategyValue(value[key])).join(', ');
  }
  return String(value);
}

function feRenderPpoStrategy() {
  const box = els.fePpoStrategy;
  if (!box) return;
  const ppo = state.fe.ppo;
  const events = (ppo.events || []).filter((event) => event && event.kind === 'strategy');
  const latest = events.length ? events[events.length - 1] : null;
  if (!latest) {
    box.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = ppo.useLlm
      ? '已启用 LLM 战略层，等待第一条 directive…'
      : '未启用战略层（在 PPO 表单勾选「启用 LLM 战略层」）';
    box.appendChild(empty);
    feSetText(els.fePpoStrategyChip, ppo.useLlm ? '等待中' : '未启用');
    feSetText(els.fePpoStrategySource, '—');
    return;
  }
  const payload = latest.payload || {};
  const source = String(payload.source || 'rule');
  feSetText(els.fePpoStrategyChip, source === 'llm' ? 'LLM' : '规则');
  feSetText(els.fePpoStrategySource, String(payload.created_at || '') + ' · directive #' + num(latest.seq, 0));
  const rows = [
    ['intent', payload.intent],
    ['attack_region', payload.attack_region],
    ['rally_point', payload.rally_point],
    ['retreat_hp_threshold', payload.retreat_hp_threshold],
    ['priority_target', payload.priority_target],
    ['notes', payload.notes],
  ];
  box.innerHTML = '';
  for (const pair of rows) {
    const row = document.createElement('div');
    row.className = 'strategy-kv';
    const label = document.createElement('span');
    label.textContent = pair[0];
    const value = document.createElement('b');
    value.textContent = feFormatStrategyValue(pair[1]);
    row.appendChild(label);
    row.appendChild(value);
    box.appendChild(row);
  }
  if (ppo.useLlm && source !== 'llm') {
    const warn = document.createElement('div');
    warn.className = 'strategy-warn';
    warn.textContent = 'LLM 未配置（缺 IRONZERO_LLM_API_KEY）或调用失败，已退回规则策略。';
    box.appendChild(warn);
  }
  const logLines = events.slice(-5).map((event) => {
    const data = event.payload || {};
    return '#' + num(event.seq, 0) + ' ' + String(data.created_at || '')
      + ' · ' + String(data.intent || '—')
      + ' · attack=' + feFormatStrategyValue(data.attack_region)
      + ' · source=' + String(data.source || 'rule');
  });
  if (logLines.length) {
    const log = document.createElement('div');
    log.className = 'strategy-log';
    log.textContent = logLines.join('\n');
    box.appendChild(log);
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
    if (!state.maps.ready) await loadMaps();
    await refreshRwppStatus();
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

  // 避免网络挂起导致右上角一直停留在 boot 连接中
  try {
    const healthPromise = checkHealth();
    const timeoutPromise = new Promise((resolve) => setTimeout(() => resolve(false), 4000));
    await Promise.race([healthPromise, timeoutPromise]);
  } catch (e) {
    state.connected = false;
  }
  if (state.connected) {
    state.mode = 'live';
    renderConn();
    await loadMaps();
    await loadReplays();
    await refreshRwppStatus(true);
    await loadFeCheckpoints();
    feRenderInferDashboard();
    feApplyTrainMode(state.fe.trainMode);
    feRenderTrainDashboard(null);
    feRenderPpoDashboard();
    void feRestoreInferJobs();
    void feRestorePpoJob();
  } else {
    enterDemo();
  }
  renderConn();
  updatePlayerUi(true);
  feRenderInferDashboard();
  feRenderTrainDashboard(null);
  setTimeout(() => {
    feDrawInferCharts();
    feDrawTrainCharts();
    feDrawPpoCharts();
  }, 120);
  setInterval(tickMain, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { boot(); });
} else {
  boot();
}
