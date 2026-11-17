/**
 * ppo.js — 实机 PPO 训练 · 常量与展示层：PPO_MAP、事件解析、日志/摘要/KPI、训练模式。
 */

import { feSetStatusPill, feSetText } from './control.js';
import { difficultyLabel } from './feMapCatalog.js';
import { feDrawTrainCharts } from './feInferUi.js';
import { feRenderTrainDashboard, renderFeTrainArtifacts } from './feTrain.js';
import { feRenderPpoReward, feRenderPpoStartHint, feRenderPpoStrategy } from './model.js';
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
  for (const key of ['fePpoUpdates', 'fePpoEnvs', 'fePpoRollout', 'fePpoTicks', 'fePpoDevice', 'fePpoFrameMs', 'fePpoFastForward', 'fePpoVision', 'fePpoEvalEvery', 'fePpoEvalEpisodes', 'fePpoUseLlm', 'fePpoMap', 'fePpoEnemyAiDifficulty']) {
    if (els[key]) els[key].disabled = running;
  }
}

export function feSetPpoStatus(label, tone) {
  feSetStatusPill(els.fePpoStatus, label, tone);
}

export function fePpoFrameMsLabel(frameMs) {
  // Measured on 6000 real ticks with the delta pinned at physics_ms=16:
  // 16 ms = 1x (real time), 2 ms = 6.5x, 1 ms = 10.7x, 0 ms = 26.5x. The old
  // "约 16×" for 1 ms never existed on hardware -- 68% of a 1 ms budget is
  // Thread.sleep wake-up overhead, so 0 ms is only 2.5x faster than 1 ms.
  const value = num(frameMs, 0);
  if (value <= 0) return '不限速 0ms（约 26×）';
  if (value === 16) return '真实时间 16ms（1×）';
  if (value === 2) return '快速 2ms（约 6.5×）';
  if (value === 1) return '极速 1ms（约 10×）';
  return value + 'ms';
}

/**
 * 引擎快进倍数的说明文案（fast_forward=1 表示关闭）。
 *
 * 与 frame_ms 是两回事：frame_ms 只决定墙钟节奏（physics_ms 钉死步长时不影响
 * 仿真内容），fast_forward 走引擎自带 gameSpeed，每 tick 推进 N 倍游戏时间，
 * 一局所需 tick 同比减少 —— 它会改变保真度。
 */
export function fePpoFastForwardLabel(fastForward) {
  const value = num(fastForward, 1);
  if (value <= 1) return '关闭（1× 原速）';
  return value + '× 引擎快进（每 tick 推进 ' + value + ' 倍游戏时间）';
}

/**
 * 从表单控件读一个整数；缺失或非法时用 fallback。
 *
 * 这里刻意不读 `state.fe.ppo`：任务还没启动时那些字段只是"上次用过的默认值"，
 * 和表单当前显示的值可能完全不同（日志行曾因此显示 `eval=最后/1`，而表单上是
 * 「每 10 轮 / 4 局」）。
 */
function formInt(key, fallback) {
  const element = els[key];
  if (!element) return fallback;
  return numOrNull(element.value) === null ? fallback : Math.round(Number(element.value));
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
    // Before a job exists these fields hold the last app defaults, not what the
    // form is set to. Reading the live form keeps this line truthful ("每 10 轮 /
    // 4 局" instead of the stale "最后 / 1") and it switches to the job's own
    // values once one is running.
    const running = !!ppo.jobId;
    const evalEvery = running ? num(ppo.evalEvery, 0) : formInt('fePpoEvalEvery', 10);
    const evalEpisodes = running ? num(ppo.evalEpisodes, 1) : formInt('fePpoEvalEpisodes', 4);
    const frameMs = running ? num(ppo.frameMs, 0) : formInt('fePpoFrameMs', 0);
    const fastForward = running ? num(ppo.fastForward, 1) : formInt('fePpoFastForward', 1);
    const vision = running ? num(ppo.visionRadius, 0) : formInt('fePpoVision', 0);
    const useLlm = running ? !!ppo.useLlm : !!(els.fePpoUseLlm && els.fePpoUseLlm.checked);
    // Scenario: the *resolved* map geometry and difficulty from the host job once
    // it exists, the live form before that. A run whose geometry was wrong does
    // not fail — it just trains on skewed observations — so this line is the only
    // place an operator can notice.
    const mapLabel = running
      ? String(ppo.mapName || '—')
        + (num(ppo.mapWidth, 0) > 0 ? ' · ' + Math.round(num(ppo.mapWidth, 0)) + '×' + Math.round(num(ppo.mapHeight, 0)) + 'px' : '')
      : String((els.fePpoMap && els.fePpoMap.value) || '—');
    const difficulty = running ? num(ppo.enemyAiDifficulty, 0) : formInt('fePpoEnemyAiDifficulty', 1);
    els.fePpoLogStatus.textContent = 'status=' + (ppo.status || 'idle')
      + ' · map=' + mapLabel
      + ' · enemy_ai_diff=' + difficulty
      + ' · frame_ms=' + frameMs + (fastForward > 1 ? ' · ff=' + fastForward + '×' : '')
      + ' · fog=' + (vision > 0 ? vision + 'px' : 'off')
      + ' · llm=' + (useLlm ? 'on' : 'off')
      + ' · eval=' + (evalEvery > 0 ? '每' + evalEvery + '轮' : '最后') + '/' + evalEpisodes
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
    // Scenario first: which map (and at which pixel size) and which opponent
    // strength this run actually used. The geometry is the observation's
    // normalisation denominator — a wrong value never raises, it just makes every
    // position mean something else.
    ['地图 / 尺寸', String(ppo.mapName || (ppo.job && ppo.job.map_name) || '—')
      + (num(ppo.mapWidth, 0) > 0
        ? ' · ' + Math.round(num(ppo.mapWidth, 0)) + '×' + Math.round(num(ppo.mapHeight, 0)) + ' px'
        : '')],
    ['对手难度', numOrNull(ppo.enemyAiDifficulty) === null
      ? '—'
      : num(ppo.enemyAiDifficulty, 0) + '（' + difficultyLabel(num(ppo.enemyAiDifficulty, 0)) + '）'],
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
    ['引擎快进', fePpoFastForwardLabel(ppo.fastForward)],
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
      + (num(ppo.fastForward, 1) > 1 ? ' · ff=' + ppo.fastForward + '×' : '')
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
  // Reflect the checkpoint training actually started from; "warmstart" resolves
  // to a different BC file per tank count and the form cannot know which one.
  feRenderPpoStartHint();
  feDrawPpoCharts();
}

/**
 * 当前表单选中的场景描述，例如「IronZero25 · 25v25 纯坦克」。
 *
 * 这行常驻在训练页右上角；写死成 Artillery 时会和正下方的场景下拉框自相矛盾，
 * 用户没法判断到底会跑哪张图。
 */
export function fePpoScenarioLabel() {
  const map = (els.fePpoMap && els.fePpoMap.value) || 'Lake';
  const units = num(els.fePpoUnits && els.fePpoUnits.value, 25);
  if (map.toLowerCase().startsWith('ironzero')) {
    return map + ' · ' + units + 'v' + units + ' 纯坦克';
  }
  if (map.toLowerCase().includes('artillery')) {
    return 'Artillery · 原版挑战图';
  }
  return map;
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
    // This banner used to hard-code "Artillery 挑战图（22 vs 25 AI）" regardless of
    // the map actually selected, while the form above it said IronZero25.
    els.feTrainModeNote.textContent = target === 'ppo'
      ? 'PPO 真机训练：Strategy-in-the-loop · ' + fePpoScenarioLabel()
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
      // These must equal the `selected`/`value=` in index.html. They used to
      // disagree (frame_ms 2 vs 1, vision 260 vs 0), so the KPI line could
      // report a pacing/fog setting the run never actually used. frame_ms 0 is
      // the recommended "unpaced" option (~26.5x with physics_ms pinned);
      // fast_forward 1 means off (the engine's own gameSpeed).
      frameMs: 0,
      fastForward: 1,
      visionRadius: 0,
      // Enemy AI difficulty the form starts on. Must equal the `selected` option
      // in index.html (1 = 困难, x1.40 income). ``None``/0 would mean "the engine
      // global default", i.e. the easiest real opponent while the form showed a
      // difficulty setting — that mismatch is what this pins down. Guarded by
      // tests/test_service_map_catalog.py.
      enemyAiDifficulty: 1,
      // Resolved by the host service once a job exists (see fePollPpo): which
      // checkpoint this run actually started from, and in which mode.
      initCheckpoint: null,
      initMode: null,
    },
  });
}
