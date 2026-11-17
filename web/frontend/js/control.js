/**
 * control.js — 指挥中心大屏 · 公共件：常量与配色、图表基元（dash*）、状态文案、state.fe 首段初始化。
 */

import { state } from './state.js';
import { num } from './util.js';

export const FE_DEFAULT_MAP_NAME = '[p2]Lake (2p)';

export const FE_INFER_POLL_MS = 700;

export const FE_TRAIN_POLL_MS = 2000;

export const DASH_COLORS = {
  own: '#3fa9ff',
  enemy: '#ff5a5a',
  cyan: '#38e1ff',
  amber: '#ffc35c',
  green: '#39e08b',
  muted: '#7f9bb5',
};

export function feSetText(node, value) {
  if (node) node.textContent = value === null || value === undefined ? '—' : String(value);
}

export function feSetStatusPill(node, label, tone) {
  if (!node) return;
  node.className = 'status-pill ' + (tone || 'idle');
  node.textContent = label;
}

/* ---------------- canvas 图表工具 ---------------- */
export function dashSetup(canvas) {
  if (!canvas || !canvas.parentElement) return null;
  const box = canvas.parentElement;
  const width = Math.max(0, Math.floor(box.clientWidth));
  const height = Math.max(0, Math.floor(box.clientHeight));
  if (width < 20 || height < 20) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pw = Math.round(width * dpr);
  const ph = Math.round(height * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx: ctx, width: width, height: height };
}

export function dashEmpty(ctx, width, height, message) {
  ctx.fillStyle = 'rgba(127, 155, 181, 0.65)';
  ctx.font = '10px "Consolas", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message || '暂无数据', width / 2, height / 2);
}

export function dashGrid(ctx, padL, padT, plotW, plotH, yMin, yMax, yFormatter) {
  ctx.font = '9px "Consolas", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const count = 4;
  for (let i = 0; i <= count; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / count;
    const y = padT + plotH - (plotH * i) / count;
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.09)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(140, 170, 195, 0.75)';
    const label = yFormatter ? yFormatter(value) : String(Math.round(value * 100) / 100);
    ctx.fillText(label, padL - 6, y);
  }
}

/**
 * 折线图单条曲线允许送去描点的最大点数。
 *
 * 画布宽度只有几百像素，超过这个数量的点在物理上画不出区别，却要为每个点
 * 走一次 `lineTo`。实机推理的 `statSamples` 会累积到 3 万条，每 700ms 全量
 * 描点（外加逐点建对象）是沙盘卡顿的次要来源之一。
 */
export const DASH_MAX_POINTS = 480;

/**
 * 把点集等间隔抽稀到 ``maxPoints`` 以内（保留最后一个点，让曲线右端如实）。
 *
 * 抽稀而不是「取前 N 个」：图表横轴是时间，只取头部会丢掉最新数据。
 * 输入已就绪（数组），不修改入参。
 */
export function thinPoints(points, maxPoints) {
  const list = Array.isArray(points) ? points : [];
  const cap = Math.max(2, Math.floor(num(maxPoints, DASH_MAX_POINTS)));
  if (list.length <= cap) return list;
  const step = (list.length - 1) / (cap - 1);
  const out = [];
  for (let index = 0; index < cap; index += 1) {
    out.push(list[Math.round(index * step)]);
  }
  // 浮点步长可能让最后一个索引落在倒数第二点，显式补上最新点。
  const last = list[list.length - 1];
  if (out[out.length - 1] !== last) out[out.length - 1] = last;
  return out;
}

/**
 * 折线图。
 *
 * 性能约定（见 ``DASH_MAX_POINTS``）：每条曲线描点前先抽稀，且**不做逐帧排序**。
 * 原先这里对 `item.points` 做 `slice().sort()`，而所有调用方传的都是按横轴
 * 递增构造的数组 —— 每 700ms 对 3 万个点做一次多余的全量排序，纯属浪费。
 * 需要乱序横轴时，调用方应显式传 ``sortPoints: true``。
 */
export function drawDashLine(canvas, config) {
  const setup = dashSetup(canvas);
  if (!setup) return;
  const ctx = setup.ctx;
  const width = setup.width;
  const height = setup.height;
  const cfg = config || {};
  const series = (cfg.series || []).filter((item) => item && Array.isArray(item.points) && item.points.length);
  ctx.clearRect(0, 0, width, height);
  if (!series.length) {
    dashEmpty(ctx, width, height, cfg.emptyText);
    return;
  }
  // Prepare each series once: optional sort, then downsample to what can be seen.
  const prepared = series.map((item) => {
    const raw = cfg.sortPoints ? item.points.slice().sort((left, right) => left.x - right.x) : item.points;
    return { item: item, points: thinPoints(raw, cfg.maxPoints) };
  });
  const padL = 42;
  const padR = 8;
  const padT = 8;
  const padB = 18;
  const plotW = Math.max(1, width - padL - padR);
  const plotH = Math.max(1, height - padT - padB);
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  // Bounds come from the thinned points: they already span the same x range
  // (first and last are preserved), so this is the same frame with less work.
  for (const entry of prepared) {
    for (const point of entry.points) {
      if (!isFinite(point.x) || !isFinite(point.y)) continue;
      xMin = Math.min(xMin, point.x);
      xMax = Math.max(xMax, point.x);
      yMin = Math.min(yMin, point.y);
      yMax = Math.max(yMax, point.y);
    }
  }
  if (!isFinite(xMin) || !isFinite(yMin)) {
    dashEmpty(ctx, width, height, cfg.emptyText);
    return;
  }
  if (cfg.xMin !== undefined) xMin = cfg.xMin;
  if (cfg.xMax !== undefined) xMax = cfg.xMax;
  if (cfg.yMin !== undefined) yMin = cfg.yMin;
  if (cfg.yMax !== undefined) yMax = cfg.yMax;
  if (xMax - xMin < 0.0001) xMax = xMin + 1;
  const pad = (yMax - yMin) * 0.12 || 1;
  if (cfg.yMax === undefined) yMax += pad;
  if (cfg.yMin === undefined) yMin = Math.max(0, yMin - pad * 0.15);
  if (yMax - yMin < 0.0001) yMax = yMin + 1;
  const sx = (value) => padL + ((value - xMin) / (xMax - xMin)) * plotW;
  const sy = (value) => padT + (1 - (value - yMin) / (yMax - yMin)) * plotH;
  dashGrid(ctx, padL, padT, plotW, plotH, yMin, yMax, cfg.yFormatter);
  if (cfg.xFormatter) {
    ctx.font = '9px "Consolas", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(140, 170, 195, 0.75)';
    const ticks = 3;
    for (let i = 0; i < ticks; i += 1) {
      const value = xMin + ((xMax - xMin) * i) / (ticks - 1 || 1);
      ctx.fillText(cfg.xFormatter(value), sx(value), padT + plotH + 4);
    }
  }
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.16)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
  for (const entry of prepared) {
    const item = entry.item;
    const points = entry.points;
    ctx.save();
    ctx.strokeStyle = item.color || DASH_COLORS.cyan;
    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    ctx.shadowColor = item.color || DASH_COLORS.cyan;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = sx(point.x);
      const y = sy(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
}

export function drawDashBars(canvas, config) {
  const setup = dashSetup(canvas);
  if (!setup) return;
  const ctx = setup.ctx;
  const width = setup.width;
  const height = setup.height;
  const cfg = config || {};
  const bars = Array.isArray(cfg.bars) ? cfg.bars.filter((bar) => bar && isFinite(bar.value)) : [];
  ctx.clearRect(0, 0, width, height);
  if (!bars.length) {
    dashEmpty(ctx, width, height, cfg.emptyText);
    return;
  }
  const padL = 42;
  const padR = 8;
  const padT = 10;
  const padB = 21;
  const plotW = Math.max(1, width - padL - padR);
  const plotH = Math.max(1, height - padT - padB);
  const maxValue = cfg.yMax !== undefined ? cfg.yMax : Math.max(1, ...bars.map((bar) => bar.value)) * 1.2;
  const minValue = cfg.yMin !== undefined ? cfg.yMin : 0;
  const span = Math.max(0.0001, maxValue - minValue);
  const sy = (value) => padT + (1 - (value - minValue) / span) * plotH;
  dashGrid(ctx, padL, padT, plotW, plotH, minValue, maxValue, cfg.yFormatter);
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.16)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
  const step = plotW / bars.length;
  const barW = Math.max(4, Math.min(26, step * 0.56));
  bars.forEach((bar, index) => {
    const x = padL + step * index + (step - barW) / 2;
    const y = sy(bar.value);
    const h = Math.max(1, padT + plotH - y);
    const gradient = ctx.createLinearGradient(0, y, 0, padT + plotH);
    const color = bar.color || cfg.color || DASH_COLORS.cyan;
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, 'rgba(10, 30, 52, 0.25)');
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, barW, h);
    ctx.strokeStyle = 'rgba(170, 230, 255, 0.25)';
    ctx.strokeRect(x, y, barW, h);
    ctx.font = '9px "Consolas", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(207, 232, 251, 0.9)';
    const valueLabel = cfg.valueFormatter ? cfg.valueFormatter(bar.value) : String(Math.round(bar.value * 10) / 10);
    ctx.fillText(valueLabel, x + barW / 2, y - 2);
    if (cfg.xFormatter) {
      ctx.fillStyle = 'rgba(140, 170, 195, 0.75)';
      ctx.textBaseline = 'top';
      ctx.fillText(cfg.xFormatter(bar.label, index), x + barW / 2, padT + plotH + 4);
    }
  });
}

/**
 * state.fe 的默认字段（原为顶层 Object.assign 语句）。
 * ES Module 存在循环依赖，顶层求值顺序不保证与原文件一致，故由 main.js 在
 * 所有模块求值完成后按原顺序调用本函数。语句内容逐字保留。
 */
export function initFeControlState() {
  Object.assign(state.fe, {
    concurrency: 1,
    inferTasks: [],
    watchJobId: null,
    inferTimer: null,
    inferPolling: false,
    inferAllDone: false,
    inferStartedAt: 0,
    fpsSeries: [],
    fpsAt: 0,
    jobsPollAt: 0,
    latestFps: 0,
    restored: false,
    trainJob: null,
    trainMetrics: [],
    trainEpisodes: 0,
    trainConcurrency: 2,
    trainStartedAt: 0,
    trainTimer: null,
    trainPolling: false,
    trainStatus: 'idle',
    trainArtifacts: [],
    mapsReady: false,
  });
}
