/**
 * ppoRun.js — 实机 PPO 训练 · 调度：启动/停止、轮询、任务恢复。
 */

import { Api } from './api.js';
import { toast } from './dashboard.js';
import { handleFeError } from './feInferUi.js';
import { feMapList } from './feMaps.js';
import { feApplyTrainMode, feRenderPpoDashboard, feSetPpoRunning, feSetPpoStatus } from './ppo.js';
import { refreshRwppStatus } from './rwpp.js';
import { els, state } from './state.js';
import { clampInt, num, numOrNull } from './util.js';

export function startFePpoPolling() {
  stopFePpoPolling();
  state.fe.ppo.timer = setInterval(fePollPpo, 1500);
  void fePollPpo();
}

export function stopFePpoPolling() {
  const ppo = state.fe.ppo;
  if (ppo.timer) {
    clearInterval(ppo.timer);
    ppo.timer = null;
  }
}

export async function fePollPpo() {
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

export async function feStartPpo() {
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
  const envs = clampInt(els.fePpoEnvs && els.fePpoEnvs.value, 1, 10, 4);
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

export async function feStopPpo() {
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

export async function feRestorePpoJob(force) {
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
