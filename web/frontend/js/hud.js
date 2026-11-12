/**
 * hud.js — 沙盘 canvas 渲染 · 叠加层：HUD 文本、矢量单位、帧事件、射线与击杀特效。
 */

import { sandboxStrategyLabel } from './model.js';
import { sandboxSpeedLabel } from './playback.js';
import { SANDBOX_COLORS } from './terrain.js';
import { clamp, fmtNum, num } from './util.js';

export function drawSandboxHud(ctx, sandbox, frame, width, height) {
  ctx.save();
  ctx.font = '11px "Consolas", "Segoe UI", monospace';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(200, 228, 250, 0.85)';
  const info = frame
    ? '#' + (frame.episode + 1) + ' · tick ' + frame.tick + ' · t=' + fmtNum(frame.time, 1) + 's' + ' · ' + sandboxSpeedLabel(sandbox)
    : '等待推理数据';
  ctx.fillText(info, 14, 12);

  const realReplay = sandbox.checkpoint === 'rwpp-live';
  const modeLabel = (sandbox.source === 'infer' || realReplay)
    ? '实机'
    : (sandbox.simMode === 'terrain' ? '地形模拟' : '竞技场');
  const mapLabel = sandbox.mapDetail ? (sandbox.mapDetail.name || '地图') : '抽象竞技场';
  let viewLabel;
  if (sandbox.simMode === 'terrain') {
    viewLabel = sandbox.viewMode === 'full' ? '全场' : '跟随交战区域';
  } else {
    viewLabel = sandbox.viewMode === 'full' ? '全场' : '交战跟随';
  }
  ctx.save();
  ctx.font = '10px "Microsoft YaHei", "Segoe UI", sans-serif';
  ctx.fillStyle = sandbox.simMode === 'terrain' ? 'rgba(125, 227, 176, 0.95)' : 'rgba(124, 160, 190, 0.9)';
  const strategyLabel = sandboxStrategyLabel(sandbox);
  let metaLine = modeLabel + ' · ' + strategyLabel + mapLabel + ' · ' + viewLabel + ' · ' + (sandbox.frames ? sandbox.frames.length : 0) + ' 帧';
  if (sandbox.fogEnabled) metaLine += ' · 迷雾';
  ctx.fillText(metaLine, 14, 27);
  ctx.restore();

  // 血量条
  const barWidth = 120;
  const hudLeft = width - 196;
  const own = frame ? frame.stats.own : 0;
  const enemy = frame ? frame.stats.enemy : 0;
  ctx.fillStyle = 'rgba(5, 12, 22, 0.75)';
  ctx.fillRect(hudLeft, 10, 186, 30);
  ctx.fillStyle = 'rgba(124, 147, 173, 0.95)';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('我方', hudLeft + 8, 15);
  ctx.fillText('敌方', hudLeft + 8, 26);
  ctx.fillStyle = 'rgba(77, 163, 255, 0.85)';
  ctx.fillRect(hudLeft + 48, 17, barWidth * clamp(own, 0, 1), 5);
  ctx.fillStyle = 'rgba(255, 95, 109, 0.85)';
  ctx.fillRect(hudLeft + 48, 28, barWidth * clamp(enemy, 0, 1), 5);

  // 图例
  ctx.textBaseline = 'bottom';
  ctx.fillStyle = 'rgba(124, 147, 173, 0.85)';
  ctx.textAlign = 'left';
  const legendY = height - 12;
  ctx.fillStyle = SANDBOX_COLORS.own;
  ctx.beginPath();
  ctx.arc(16, legendY - 4, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('我方', 25, legendY);
  ctx.fillStyle = SANDBOX_COLORS.enemy;
  ctx.beginPath();
  ctx.arc(70, legendY - 4, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('敌方', 79, legendY);
  ctx.fillStyle = '#ffe066';
  ctx.fillRect(124, legendY - 6, 14, 3);
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('射击', 144, legendY);
  ctx.fillStyle = '#ffb347';
  ctx.beginPath();
  ctx.arc(188, legendY - 4, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(200, 228, 250, 0.8)';
  ctx.fillText('击毁', 197, legendY);

  // 右下角说明：道具层 / 迷雾
  ctx.textAlign = 'right';
  ctx.font = '9.5px "Microsoft YaHei", "Segoe UI", sans-serif';
  let noteY = height - 12;
  if (sandbox.mapDetail && sandbox.itemsEnabled) {
    ctx.fillStyle = 'rgba(150, 178, 205, 0.62)';
    ctx.fillText('建筑/资源点：地图道具层（静态，未模拟建造）', width - 14, noteY);
    noteY -= 13;
  }
  if (sandbox.fogEnabled) {
    ctx.fillStyle = 'rgba(150, 178, 205, 0.62)';
    ctx.fillText('迷雾：视觉层（按我方视野 ' + Math.round(num(sandbox.visionRadius, 260)) + 'px）', width - 14, noteY);
  }
  ctx.restore();
}

export function drawSandboxUnit(ctx, unit, mapX, mapY, selected, leader, time, options) {
  const x = mapX(unit.x);
  const y = mapY(unit.y);
  const radius = Math.max(3.5, num(options && options.radius, 7.5));
  const own = unit.team === 'own';
  const color = own ? SANDBOX_COLORS.own : SANDBOX_COLORS.enemy;
  const hpFraction = clamp(unit.hp / Math.max(1, unit.maxHp), 0, 1);

  // 选中圈 / 编组圈
  if (selected || leader) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius + 6, 0, Math.PI * 2);
    ctx.strokeStyle = selected ? 'rgba(234, 252, 255, 0.95)' : 'rgba(56, 225, 255, 0.32)';
    ctx.lineWidth = selected ? 1.8 : 1;
    if (selected) {
      ctx.setLineDash([5, 4]);
      ctx.lineDashOffset = -time * 14;
    } else {
      ctx.setLineDash([3, 5]);
      ctx.lineDashOffset = time * 8;
    }
    ctx.stroke();
    ctx.restore();
  }

  // 光晕 + 本体
  ctx.save();
  ctx.shadowColor = own ? 'rgba(77, 163, 255, 0.9)' : 'rgba(255, 95, 109, 0.9)';
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(x, y, radius + 2.4, 0, Math.PI * 2);
  ctx.fillStyle = own ? 'rgba(77, 163, 255, 0.18)' : 'rgba(255, 95, 109, 0.18)';
  ctx.fill();
  ctx.restore();

  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = own ? '#2f6fbf' : '#b5303c';
  ctx.fill();
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = color;
  ctx.stroke();

  // 编号
  ctx.save();
  ctx.font = '8.5px "Consolas", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(240, 250, 255, 0.92)';
  ctx.fillText(String(unit.id), x, y + 0.5);
  ctx.restore();

  // 血条
  const barWidth = 21;
  const barHeight = 3;
  const barX = x - barWidth / 2;
  const barY = y - radius - 8;
  ctx.fillStyle = 'rgba(2, 6, 12, 0.8)';
  ctx.fillRect(barX - 1, barY - 1, barWidth + 2, barHeight + 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.fillRect(barX, barY, barWidth, barHeight);
  ctx.fillStyle = hpFraction > 0.6 ? '#39e08b' : (hpFraction > 0.3 ? '#ffc35c' : '#ff5f6d');
  ctx.fillRect(barX, barY, barWidth * hpFraction, barHeight);
}

export function drawFrameEvents(ctx, frame, mapX, mapY, alpha, isVisible) {
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, { x: unit.x, y: unit.y });
  const phase = (performance.now() % 900) / 900;
  for (const event of frame.events) {
    if (isVisible && !isVisible(event)) continue;
    if (event.type === 'shot') {
      const from = positions.get(event.source);
      const to = positions.get(event.target);
      if (from && to) drawShotLine(ctx, mapX(from.x), mapY(from.y), mapX(to.x), mapY(to.y), alpha);
    } else if (event.type === 'kill') {
      const target = positions.get(event.target);
      const x = target ? target.x : event.x;
      const y = target ? target.y : event.y;
      if (x !== null && y !== null && x !== undefined && y !== undefined) {
        drawKillBurst(ctx, mapX(x), mapY(y), phase, alpha);
      }
    }
  }
}

export function drawShotLine(ctx, x1, y1, x2, y2, alpha) {
  const strength = clamp(alpha, 0, 1);
  if (strength <= 0.01) return;
  ctx.save();
  ctx.globalAlpha = strength;
  ctx.strokeStyle = '#ffe066';
  ctx.lineWidth = 2;
  ctx.shadowColor = 'rgba(255, 200, 70, 0.95)';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
  // 枪口闪光
  ctx.save();
  ctx.globalAlpha = strength * 0.9;
  const muzzle = ctx.createRadialGradient(x1, y1, 0, x1, y1, 7);
  muzzle.addColorStop(0, 'rgba(255, 255, 235, 0.95)');
  muzzle.addColorStop(1, 'rgba(255, 190, 60, 0)');
  ctx.fillStyle = muzzle;
  ctx.beginPath();
  ctx.arc(x1, y1, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  // 命中点
  ctx.save();
  ctx.globalAlpha = strength * 0.8;
  ctx.fillStyle = '#fff5c9';
  ctx.beginPath();
  ctx.arc(x2, y2, 3.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

export function drawKillBurst(ctx, x, y, phase, alpha) {
  const progress = clamp(phase, 0, 1);
  const strength = (1 - progress) * clamp(alpha, 0, 1);
  if (strength <= 0.01) return;
  const radius = 10 + 26 * progress;
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, 'rgba(255, 244, 200, ' + 0.5 * strength + ')');
  gradient.addColorStop(0.45, 'rgba(255, 150, 60, ' + 0.36 * strength + ')');
  gradient.addColorStop(1, 'rgba(255, 120, 40, 0)');
  ctx.save();
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 205, 120, ' + 0.85 * strength + ')';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.arc(x, y, radius * 0.82, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}
