/**
 * playback.js — 对局沙盘 · 播放：沙盘重置、帧步进/倍速、单位插值、事件特效、播放 UI。
 */

import { demoReplayPayload, initDemoData } from './demo.js';
import { normalizeMap, resultMeta } from './normalize.js';
import { cancelRecording } from './recorder.js';
import { drawSandbox } from './render.js';
import { collectFrameEvents } from './rwpp.js';
import { createSandboxState, els, state } from './state.js';
import { syncMapStatus } from './terrain.js';
import { BASE_FPS, basename, clamp, fmtNum, num } from './util.js';
import { syncMapSelect } from './view.js';

export function playerStatus(sandbox) {
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

export function resetSandbox(options) {
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

export function loadDemoBattle() {
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

export function ensureSandboxContent() {
  const sandbox = state.sandbox;
  if (sandbox.frames.length) return;
  if (state.mode === 'demo') loadDemoBattle();
}

export function pickDemoFrameIndex(frames) {
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

export function currentFrame() {
  const frames = state.sandbox.frames;
  if (!frames.length) return null;
  return frames[clamp(Math.round(state.sandbox.playback), 0, frames.length - 1)];
}

export function updatePlayerUi(force) {
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

export function renderEventFeed() {
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

export function interpolateUnits(frame, nextFrame, fraction) {
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
      // 贴图相关字段逐字段透传（类型不会中途变化，取左帧值）
      unitType: unit.unitType,
      isBuilding: unit.isBuilding,
      isBuilder: unit.isBuilder,
      isFactory: unit.isFactory,
    });
  }
  return result;
}

export function spawnFrameEffects(frame) {
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

export function sandboxFrameStepSeconds(sandbox) {
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

export function sandboxSpeedLabel(sandbox) {
  const speed = num(sandbox && sandbox.speed, 1);
  if (speed >= 30) return '最快';
  return (Math.round(speed * 10) / 10) + 'x';
}

export function advanceSandbox(dt) {
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
