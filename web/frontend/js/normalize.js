/**
 * normalize.js — 后端字段归一化：map/team/unit/event/frame/result/replay，以及状态与结果文案映射。
 */

import { basename, clamp, num, numOrNull, pickVal } from './util.js';

export function normalizeMap(raw) {
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
export function normalizeTeam(value) {
  if (value === undefined || value === null) return 'enemy';
  if (typeof value === 'number') return value === 0 ? 'own' : 'enemy';
  const text = String(value).toLowerCase();
  if (text === '0' || text === 'own' || text === 'ally' || text === 'blue' || text === 'player' || text === 'ours') return 'own';
  return 'enemy';
}

export function normalizeUnit(raw) {
  const unit = raw && typeof raw === 'object' ? raw : {};
  const maxHp = Math.max(1, num(pickVal(unit, 'max_hp', 'maxHp', 'hp_max'), 100));
  const hp = clamp(num(pickVal(unit, 'hp', 'health'), maxHp), 0, maxHp);
  const unitType = pickVal(unit, 'unit_type', 'unitType', 'type');
  const isBuilding = pickVal(unit, 'is_building', 'isBuilding');
  const isBuilder = pickVal(unit, 'is_builder', 'isBuilder');
  const isFactory = pickVal(unit, 'is_factory', 'isFactory');
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
    // 贴图解析所需字段（见 docs/沙盘贴图契约.md 2.1）：一律驼峰，原值大小写敏感
    unitType: unitType === undefined || unitType === null ? null : String(unitType),
    isBuilding: !!isBuilding,
    isBuilder: !!isBuilder,
    isFactory: !!isFactory,
  };
}

export function normalizeEvent(raw) {
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

export function normalizeFrame(raw) {
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

export function normalizeResult(raw) {
  const item = raw && typeof raw === 'object' ? raw : {};
  return {
    episode: Math.round(num(item.episode, 0)),
    result: pickVal(item, 'result', 'outcome') || null,
    ret: numOrNull(pickVal(item, 'return', 'ret')),
    decisions: numOrNull(pickVal(item, 'decisions', 'frames')),
    replayPath: pickVal(item, 'replay_path', 'path') || null,
  };
}

export function normalizeReplay(raw) {
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

export const STATUS_META = {
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

export function statusMeta(status) {
  const key = String(status || '').toLowerCase();
  return STATUS_META[key] || { label: status ? String(status) : '未知', cls: 'idle' };
}

export function resultMeta(result) {
  const text = String(result || '').toLowerCase();
  if (text === 'victory' || text === 'win' || text === 'won' || text === 'success') return { label: '胜利', cls: 'win' };
  if (text === 'defeat' || text === 'loss' || text === 'lost' || text === 'failure') return { label: '失败', cls: 'loss' };
  // "timeout" = the episode hit the tick cap without a decided battle (not a draw).
  if (text === 'timeout') return { label: '超时', cls: 'draw' };
  if (text === 'draw' || text === 'tie') return { label: '平局', cls: 'draw' };
  return { label: result ? String(result) : '—', cls: 'unknown' };
}
