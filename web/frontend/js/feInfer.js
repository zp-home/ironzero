/**
 * feInfer.js — 指挥中心大屏 · 实机推理任务对象：创建、帧摄入、标记完成/失败、观看同步。
 */

import { Api } from './api.js';
import { toast } from './dashboard.js';
import { feDrawInferCharts, feRenderInferDashboard, feRenderTaskCards } from './feInferUi.js';
import { normalizeFrame, resultMeta } from './normalize.js';
import { updatePlayerUi } from './playback.js';
import { drawSandbox } from './render.js';
import { state } from './state.js';
import { num, numOrNull } from './util.js';

/* ---------------- 实机推理：并发任务模型 ---------------- */
export function feInferTask(jobId) {
  return (state.fe.inferTasks || []).find((task) => task.jobId === jobId) || null;
}

export function feWatchTaskObject() {
  return feInferTask(state.fe.watchJobId);
}

export function feRunningInferTasks() {
  return (state.fe.inferTasks || []).filter((task) => task.status === 'running');
}

export function feNewInferTask(jobId, index, extra) {
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

export function feIngestTaskFrames(task, items) {
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

export function feMarkTaskDone(task) {
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

export function feMarkTaskFailed(task, message) {
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

/**
 * Mark a task the user stopped. Distinct from "failed": the episode was killed on
 * request, so there is no replay and no result, but nothing went wrong. Keeping a
 * separate状态 lets the card show 已停止 instead of 失败 (the host service reports
 * `stopped` after scripts/rwpp_service/proc.py::stop_job).
 */
export function feMarkTaskStopped(task) {
  if (!task || task.status === 'stopped' || task.status === 'done') return;
  task.status = 'stopped';
  task.error = null;
  task.streaming = false;
  if (!task.toastShown) {
    task.toastShown = true;
    toast('实例 #' + task.index + ' 已停止', 'ok');
  }
  if (state.fe.watchJobId === task.jobId) {
    const sandbox = state.sandbox;
    sandbox.status = 'stopped';
    sandbox.streaming = false;
    sandbox.playing = false;
    updatePlayerUi(true);
    drawSandbox();
  }
  feRenderTaskCards();
}

export function feHandleTaskError(task, error) {
  const status = error && error.status;
  if (status === 503) {
    feMarkTaskFailed(task, '真机服务未启动');
  } else if (status === 409) {
    feMarkTaskFailed(task, '并行槽位已满');
  }
}

export async function feFetchTaskFrames(task) {
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
  if (status === 'stopped') {
    feMarkTaskStopped(task);
    return;
  }
  if (status === 'failed') {
    feMarkTaskFailed(task, job.error || '宿主机任务失败');
    return;
  }
  if (status === 'done' || task.inferDone) task.pendingDone = true;
  if (task.pendingDone && !items.length) feMarkTaskDone(task);
}

export async function feFetchTaskFinal(task, attempt) {
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

export async function feFetchTaskJob(task) {
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

export function feApplyWatchTask(jobId, options) {
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

export async function feLoadTaskFramesOnce(task) {
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

export async function feWatchTask(jobId) {
  const task = feInferTask(jobId);
  if (!task) return;
  feApplyWatchTask(jobId, { resetPlayback: false });
  if (!task.frames.length && task.status !== 'running') {
    await feLoadTaskFramesOnce(task);
    feApplyWatchTask(jobId, { resetPlayback: true });
  }
  feRenderInferDashboard();
}
