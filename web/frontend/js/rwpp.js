/**
 * rwpp.js — 真机 RWPP 录制：宿主机服务状态、录制任务启动与轮询、回放打开/删除。
 */

import { Api } from './api.js';
import { checkHealth, exitDemo } from './connection.js';
import { switchTab, toast } from './dashboard.js';
import { DEMO, demoReplayPayload } from './demo.js';
import { feUpdateSlots } from './feInferUi.js';
import { normalizeFrame, normalizeMap } from './normalize.js';
import { resetSandbox, updatePlayerUi } from './playback.js';
import { loadReplays, renderReplays } from './replays.js';
import { els, state } from './state.js';
import { loadMaps, syncMapStatus } from './terrain.js';
import { basename, num, numOrNull } from './util.js';
import { selectMap, syncMapSelect, syncMapToggles } from './view.js';

export const RWPP_TICKS = 3600;

export const RWPP_POLL_MS = 2000;

// Upper bound for following one real-game job in the UI. Long requests (up to 1 hour,
// 216000 ticks) can run for tens of minutes, so allow up to 2 hours of wall clock.
export const RWPP_MAX_SECONDS = 7200;

export const RWPP_STATUS_TTL_MS = 30000;

export function isRealReplay(item) {
  if (!item) return false;
  return item.checkpoint === 'rwpp-live' || item.source === 'rwpp-bridge';
}

export function renderRwppStatus() {
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

export async function refreshRwppStatus(force) {
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

export function setRwppRecording(running, seconds) {
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

export function handleRwppError(error) {
  const status = error && error.status;
  if (status === 503) {
    toast('真机服务未启动：在宿主机运行 python scripts/rwpp_real_service.py', 'error');
  } else if (status === 409) {
    toast('已有真机录制任务正在进行，请等待完成', 'error');
  } else {
    toast('真机录制失败：' + (error && error.message ? error.message : error), 'error');
  }
}

export async function startRwppRecord() {
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

export function stopRwppPolling() {
  const rwpp = state.rwpp;
  if (rwpp.timer) {
    clearInterval(rwpp.timer);
    rwpp.timer = null;
  }
  rwpp.jobId = null;
  rwpp.startedAt = 0;
}

export function startRwppPolling() {
  const rwpp = state.rwpp;
  if (rwpp.timer) clearInterval(rwpp.timer);
  rwpp.timer = setInterval(pollRwppJob, RWPP_POLL_MS);
  pollRwppJob();
}

export function pickLatestRealReplay(replayPath) {
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

export async function pollRwppJob() {
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

export async function openReplay(id) {
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

export function collectFrameEvents(frames) {
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

export async function removeReplay(id) {
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
