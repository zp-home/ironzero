/**
 * feInferUi.js — 指挥中心大屏 · 实机推理展示层：FPS、图表、任务卡、KPI、状态灯。
 */

import { DASH_COLORS, drawDashBars, drawDashLine, feSetStatusPill, feSetText } from './control.js';
import { toast } from './dashboard.js';
import { feWatchTask, feWatchTaskObject } from './feInfer.js';
import { resultMeta } from './normalize.js';
import { els, state } from './state.js';
import { num, numOrNull } from './util.js';

export function feUpdateFps() {
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

export function feDrawInferCharts() {
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

export function feDrawTrainCharts() {
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

export function feRenderTaskCards() {
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

export function feRenderInferKpis() {
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

export function feRenderInferDashboard() {
  feRenderInferKpis();
  feRenderTaskCards();
  feDrawInferCharts();
}

export function feUpdateSlots() {
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

export function feSetInferStatus(text, tone) {
  if (!els.feInferStatus) return;
  els.feInferStatus.textContent = text;
  els.feInferStatus.className = 'hint-line' + (tone ? ' ' + tone : '');
}

export function feSetInferRunning(running) {
  if (els.feStartBtn) {
    els.feStartBtn.disabled = running;
    els.feStartBtn.textContent = running ? '实机推理中…' : '开始实机推理';
  }
  if (els.mapSelect) els.mapSelect.disabled = running;
  if (els.feTicks) els.feTicks.disabled = running;
  if (els.feConcurrency) els.feConcurrency.disabled = running;
}

export function handleFeError(error, label) {
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
