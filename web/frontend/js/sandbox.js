/**
 * sandbox.js — 对局沙盘 · 推理任务：提交推理、帧流分批接收、轮询与结束处理。
 */

import { Api } from './api.js';
import { switchTab, toast } from './dashboard.js';
import { demoReplayPayload, initDemoData } from './demo.js';
import { normalizeFrame, normalizeMap, normalizeResult, resultMeta } from './normalize.js';
import { resetSandbox, updatePlayerUi } from './playback.js';
import { loadReplays } from './replays.js';
import { collectFrameEvents } from './rwpp.js';
import { els, state } from './state.js';
import { loadMaps } from './terrain.js';
import { clampInt, num, numOrNull } from './util.js';
import { refreshTerrainView, selectMap, syncMapToggles } from './view.js';

export function applyInferStart(meta) {
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

export function applyInferBatch(batch, next) {
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

export function pushFrameEvents(frame) {
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

export async function submitInfer(event) {
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

export function startDemoInference() {
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

export function startInferPolling() {
  stopInferPolling();
  state.inferTimer = setInterval(pollInfer, 700);
  pollInfer();
}

export function stopInferPolling() {
  if (state.inferTimer) {
    clearInterval(state.inferTimer);
    state.inferTimer = null;
  }
}

export async function pollInfer() {
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

export async function handleInferFinished() {
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
