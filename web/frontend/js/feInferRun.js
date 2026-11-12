/**
 * feInferRun.js — 指挥中心大屏 · 实机推理调度：启动/停止、轮询、收尾、任务恢复。
 */

import { Api } from './api.js';
import { toast } from './dashboard.js';
import { feApplyWatchTask, feFetchTaskFrames, feFetchTaskJob } from './feInfer.js';
import { feHandleTaskError, feNewInferTask, feRunningInferTasks } from './feInfer.js';
import { feWatchTaskObject } from './feInfer.js';
import { feRenderInferDashboard, feSetInferRunning, feSetInferStatus } from './feInferUi.js';
import { feUpdateFps, handleFeError } from './feInferUi.js';
import { feDefaultMap, feMapItemById } from './feMaps.js';
import { resultMeta } from './normalize.js';
import { resetSandbox } from './playback.js';
import { loadReplays } from './replays.js';
import { refreshRwppStatus } from './rwpp.js';
import { els, state } from './state.js';
import { basename, clampInt, num } from './util.js';
import { selectMap } from './view.js';

export async function feStartInfer() {
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

export async function feStartInferInner() {
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
  const frameMs = els.feFrameMs ? parseInt(els.feFrameMs.value, 10) : 2;
  const inferConfig = {
    ticks: ticks,
    policy: policy,
    frame_ms: frameMs,
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

export function startFeInferPolling() {
  stopFeInferPolling();
  state.fe.inferTimer = setInterval(feInferTick, 700);
  void feInferTick();
}

export function stopFeInferPolling() {
  const fe = state.fe;
  if (fe.inferTimer) {
    clearInterval(fe.inferTimer);
    fe.inferTimer = null;
  }
}

export async function feInferTick() {
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

export async function feFinishInferIfDone() {
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

export async function feRestoreInferJobs() {
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
