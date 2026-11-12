/**
 * ppo.js — 实机 PPO 训练 · 常量与展示层：PPO_MAP、事件解析、日志/摘要/KPI、训练模式。
 */

import { feSetStatusPill, feSetText } from './control.js';
import { feDrawTrainCharts } from './feInferUi.js';
import { feRenderTrainDashboard, renderFeTrainArtifacts } from './feTrain.js';
import { feRenderPpoReward, feRenderPpoStrategy } from './model.js';
import { feDrawPpoCharts } from './ppoCharts.js';
import { els, state } from './state.js';
import { basename, num, numOrNull } from './util.js';

export const PPO_MAP = {
  name: 'Artillery',
  path: 'maps/challenge/Artillery.tmx',
  width: 2700,
  height: 2500,
};

export function fePpoUpdateEvents() {
  const events = (state.fe.ppo.events || []).filter((event) => event && event.kind === 'update');
  return events
    .map((event) => event.payload || {})
    .sort((left, right) => num(left.update, 0) - num(right.update, 0));
}

export function fePpoEvalEvents() {
  const ppo = state.fe.ppo;
  if (Array.isArray(ppo.evaluations) && ppo.evaluations.length) {
    return ppo.evaluations.slice().sort((a, b) => num(a.update, 0) - num(b.update, 0));
  }
  const events = (ppo.events || []).filter((event) => event && event.kind === 'eval');
  return events
    .map((event) => event.payload || {})
    .sort((left, right) => num(left.update, 0) - num(right.update, 0));
}

export function fePpoDurations(updates) {
  const durations = [];
  let previous = 0;
  for (const update of updates) {
    const elapsed = num(update.elapsed_s, 0);
    durations.push(Math.max(0, elapsed - previous));
    previous = elapsed;
  }
  return durations;
}

export function fePpoSeriesRange(seriesList) {
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

export function fePpoEventLine(event) {
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

export function feSetPpoRunning(running) {
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

export function feSetPpoStatus(label, tone) {
  feSetStatusPill(els.fePpoStatus, label, tone);
}

export function fePpoFrameMsLabel(frameMs) {
  const value = num(frameMs, 2);
  if (value === 16) return '真实时间 16ms（1×）';
  if (value === 2) return '快速 2ms（约 8×）';
  if (value === 1) return '极速 1ms（约 16×）';
  return value + 'ms';
}

export function feRenderPpoLog() {
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

export function feRenderPpoSummary() {
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

export function feRenderPpoKpis() {
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

export function feRenderPpoDashboard() {
  feRenderPpoKpis();
  feRenderPpoLog();
  feRenderPpoSummary();
  feRenderPpoReward();
  feRenderPpoStrategy();
  feDrawPpoCharts();
}

export function feApplyTrainMode(mode) {
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

/**
 * state.fe 的默认字段（原为顶层 Object.assign 语句）。
 * ES Module 存在循环依赖，顶层求值顺序不保证与原文件一致，故由 main.js 在
 * 所有模块求值完成后按原顺序调用本函数。语句内容逐字保留。
 */
export function initFePpoState() {
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
}
