/**
 * charts.js — 图表（canvas 自绘）：刻度、折线、图例、悬停提示。
 */

import { els, state } from './state.js';
import { clamp, fmtNum, fmtPct } from './util.js';

export function niceTicks(minValue, maxValue, targetCount) {
  let min = minValue;
  let max = maxValue;
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 1;
  }
  if (max - min < 1e-9) {
    max = min + 1;
  }
  const rough = (max - min) / Math.max(1, targetCount || 4);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const normalized = rough / magnitude;
  let step;
  if (normalized <= 1) step = 1;
  else if (normalized <= 2) step = 2;
  else if (normalized <= 2.5) step = 2.5;
  else if (normalized <= 5) step = 5;
  else step = 10;
  step *= magnitude;
  const low = Math.floor(min / step) * step;
  const high = Math.ceil(max / step) * step;
  const ticks = [];
  for (let value = low; value <= high + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return { min: low, max: Math.max(high, low + step), step: step, ticks: ticks };
}

export function drawLineChart(canvas, tip, config) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const padLeft = 46;
  const padRight = 18;
  const padTop = 14;
  const padBottom = 26;
  const x0 = padLeft;
  const x1 = Math.max(padLeft + 20, width - padRight);
  const y0 = padTop;
  const y1 = Math.max(padTop + 20, height - padBottom);

  const series = (config.series || []).filter((item) => item && Array.isArray(item.points));
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = config.yMin !== undefined ? config.yMin : Infinity;
  let yMax = config.yMax !== undefined ? config.yMax : -Infinity;
  let totalPoints = 0;
  for (const item of series) {
    for (const point of item.points) {
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
      totalPoints += 1;
      xMin = Math.min(xMin, point.x);
      xMax = Math.max(xMax, point.x);
      if (config.yMin === undefined) yMin = Math.min(yMin, point.y);
      if (config.yMax === undefined) yMax = Math.max(yMax, point.y);
    }
  }
  if (!totalPoints) {
    xMin = 0;
    xMax = 1;
    if (config.yMin === undefined) yMin = 0;
    if (config.yMax === undefined) yMax = 1;
  }
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) {
    xMin = 0;
    xMax = 1;
  }
  if (xMax - xMin < 1e-9) {
    xMin = Math.max(0, xMin - 1);
    xMax = xMax + 1;
  }
  const xPad = (xMax - xMin) * 0.03;
  xMin -= xPad;
  xMax += xPad;
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = 0;
    yMax = 1;
  }
  if (config.yMin === undefined) {
    const span = Math.max(1e-6, yMax - yMin);
    yMin -= span * 0.12;
    if (yMax > 0 && yMin < 0 && config.includeZero !== false) yMin = Math.min(0, yMin);
  }
  if (config.yMax === undefined) {
    const span = Math.max(1e-6, yMax - yMin);
    yMax += span * 0.1;
  }
  if (config.xFloor !== undefined && config.xFloor !== null) xMin = Math.max(xMin, config.xFloor);
  const yTicks = niceTicks(yMin, yMax, 4);
  const xTicks = niceTicks(xMin, xMax, 4);

  const mapX = (value) => x0 + ((value - xTicks.min) / (xTicks.max - xTicks.min)) * (x1 - x0);
  const mapY = (value) => y1 - ((value - yTicks.min) / (yTicks.max - yTicks.min)) * (y1 - y0);

  // 绘图区背景
  ctx.fillStyle = 'rgba(4, 10, 18, 0.55)';
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);

  // 网格
  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = '10px Consolas, "Segoe UI", monospace';
  ctx.textBaseline = 'middle';
  for (const value of yTicks.ticks) {
    const y = mapY(value);
    if (y < y0 - 1 || y > y1 + 1) continue;
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.08)';
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    ctx.fillStyle = 'rgba(124, 147, 173, 0.95)';
    ctx.textAlign = 'right';
    ctx.fillText(config.yFormat ? config.yFormat(value) : String(value), x0 - 8, y);
  }
  ctx.textBaseline = 'top';
  for (const value of xTicks.ticks) {
    const x = mapX(value);
    if (x < x0 - 1 || x > x1 + 1) continue;
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.06)';
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
    if (value >= xTicks.min && value <= xTicks.max) {
      ctx.fillStyle = 'rgba(124, 147, 173, 0.95)';
      ctx.textAlign = 'center';
      ctx.fillText(config.xFormat ? config.xFormat(value) : String(value), clamp(x, x0 + 10, x1 - 10), y1 + 8);
    }
  }
  ctx.strokeStyle = 'rgba(31, 58, 89, 0.9)';
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.restore();

  if (!totalPoints) {
    ctx.save();
    ctx.fillStyle = 'rgba(93, 115, 144, 0.85)';
    ctx.font = '12px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('暂无数据', (x0 + x1) / 2, (y0 + y1) / 2);
    ctx.restore();
    canvas.__chart = { x0: x0, x1: x1, y0: y0, y1: y1, xMin: xTicks.min, xMax: xTicks.max, series: series, cfg: config };
    return;
  }

  // 参考线
  if (config.refLine !== undefined && config.refLine !== null) {
    const y = mapY(config.refLine);
    if (y >= y0 && y <= y1) {
      ctx.save();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = 'rgba(255, 195, 92, 0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      ctx.setLineDash([]);
      if (config.refLabel) {
        ctx.fillStyle = 'rgba(255, 195, 92, 0.75)';
        ctx.font = '10px Consolas, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(config.refLabel, x0 + 6, y - 3);
      }
      ctx.restore();
    }
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, x1 - x0, y1 - y0);
  ctx.clip();

  for (const item of series) {
    const points = item.points;
    if (!points.length) continue;
    const color = item.color || '#38e1ff';
    // 面积填充
    if (points.length > 1) {
      const gradient = ctx.createLinearGradient(0, y0, 0, y1);
      gradient.addColorStop(0, hexToRgba(color, 0.22));
      gradient.addColorStop(1, hexToRgba(color, 0.01));
      ctx.beginPath();
      ctx.moveTo(mapX(points[0].x), y1);
      for (const point of points) ctx.lineTo(mapX(point.x), mapY(point.y));
      ctx.lineTo(mapX(points[points.length - 1].x), y1);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();
    }
    // 折线
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.shadowColor = hexToRgba(color, 0.75);
    ctx.shadowBlur = 8;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = mapX(point.x);
      const y = mapY(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();

    // 评估点 / 稀疏点标记
    if (points.length <= 80) {
      for (const point of points) {
        ctx.beginPath();
        ctx.arc(mapX(point.x), mapY(point.y), 2.2, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
    // 末端高亮
    const lastPoint = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(mapX(lastPoint.x), mapY(lastPoint.y), 3.4, 0, Math.PI * 2);
    ctx.fillStyle = '#eafcff';
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.restore();

  canvas.__chart = {
    x0: x0, x1: x1, y0: y0, y1: y1,
    xMin: xTicks.min, xMax: xTicks.max,
    series: series, cfg: config,
  };
  if (tip) tip.hidden = true;
}

export function hexToRgba(hex, alpha) {
  const text = String(hex || '#38e1ff').replace('#', '');
  const full = text.length === 3 ? text.split('').map((chr) => chr + chr).join('') : text;
  const value = parseInt(full, 16);
  if (!Number.isFinite(value)) return 'rgba(56, 225, 255, ' + alpha + ')';
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
}

export function chartSeriesConfigs() {
  const winPoints = state.series.win.length ? state.series.win : state.series.trainWin;
  const winName = state.series.win.length ? '评估胜率' : '训练胜率';
  return {
    win: {
      series: [{ name: winName, color: '#38e1ff', points: winPoints, format: (value) => fmtPct(value) }],
      yMin: 0,
      yMax: 1,
      refLine: 0.5,
      refLabel: '50% 基线',
      yFormat: (value) => Math.round(value * 100) + '%',
      xFormat: (value) => 'u' + Math.round(value),
      xFloor: 0,
      includeZero: true,
    },
    ret: {
      series: [
        { name: '训练回报', color: '#ffc35c', points: state.series.retTrain, format: (value) => fmtNum(value, 3) },
        { name: '评估回报', color: '#39e08b', points: state.series.retEval, format: (value) => fmtNum(value, 3) },
      ],
      yFormat: (value) => fmtNum(value, 2),
      xFormat: (value) => 'u' + Math.round(value),
      xFloor: 0,
      includeZero: true,
    },
  };
}

export function renderLegend(host, series) {
  if (!host) return;
  host.innerHTML = '';
  for (const item of series) {
    const span = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.style.background = item.color;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(item.name));
    const value = document.createElement('b');
    const points = item.points || [];
    value.textContent = points.length
      ? (item.format ? item.format(points[points.length - 1].y) : fmtNum(points[points.length - 1].y, 3))
      : '—';
    span.appendChild(value);
    host.appendChild(span);
  }
}

export function drawCharts() {
  if (!els.chartWin || !els.chartReturn) return;
  const configs = chartSeriesConfigs();
  drawLineChart(els.chartWin, els.tipWin, configs.win);
  drawLineChart(els.chartReturn, els.tipReturn, configs.ret);
  renderLegend(els.legendWin, configs.win.series);
  renderLegend(els.legendReturn, configs.ret.series);
  state.dirtyCharts = false;
}

export function nearestPoint(points, targetX) {
  if (!Array.isArray(points) || !points.length) return null;
  let best = points[0];
  let bestDistance = Math.abs(best.x - targetX);
  for (const point of points) {
    const distance = Math.abs(point.x - targetX);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = point;
    }
  }
  return best;
}

export function handleChartHover(canvas, tip, clientX, clientY) {
  if (!canvas || !tip) return;
  const chart = canvas.__chart;
  if (!chart) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = clientX - rect.left;
  if (mouseX < chart.x0 || mouseX > chart.x1) {
    tip.hidden = true;
    return;
  }
  const primary = chart.series.length ? chart.series[0] : null;
  if (!primary || !primary.points.length) {
    tip.hidden = true;
    return;
  }
  const targetX = chart.xMin + ((mouseX - chart.x0) / Math.max(1, chart.x1 - chart.x0)) * (chart.xMax - chart.xMin);
  const anchor = nearestPoint(primary.points, targetX);
  if (!anchor) {
    tip.hidden = true;
    return;
  }
  tip.innerHTML = '';
  const header = document.createElement('div');
  header.textContent = (chart.cfg.xName || 'update') + ' ' + Math.round(anchor.x);
  header.style.color = '#8ce6ff';
  header.style.marginBottom = '2px';
  tip.appendChild(header);
  for (const item of chart.series) {
    const point = nearestPoint(item.points, anchor.x);
    if (!point) continue;
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '6px';
    const swatch = document.createElement('i');
    swatch.style.display = 'inline-block';
    swatch.style.width = '10px';
    swatch.style.height = '3px';
    swatch.style.borderRadius = '2px';
    swatch.style.background = item.color;
    const label = document.createElement('span');
    label.textContent = item.name + '：';
    const value = document.createElement('b');
    value.textContent = item.format ? item.format(point.y) : fmtNum(point.y, 3);
    value.style.color = '#eafcff';
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(value);
    tip.appendChild(row);
  }
  tip.hidden = false;
  tip.style.left = clamp(mouseX, 60, rect.width - 60) + 'px';
  tip.style.top = clamp(clientY - rect.top - 8, 28, rect.height) + 'px';
}
