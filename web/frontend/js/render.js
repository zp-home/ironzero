/**
 * render.js — 沙盘 canvas 渲染 · 主流程：drawSandbox 主循环、迷雾层、边框、空态。
 */

import { drawFrameEvents, drawKillBurst, drawSandboxHud, drawSandboxUnit } from './hud.js';
import { drawShotLine } from './hud.js';
import { interpolateUnits } from './playback.js';
import { drawSandboxSpriteUnit, updateUnitAngles } from './sprites.js';
import { els, state } from './state.js';
import { SANDBOX_COLORS, engineToMapPixel, mapCamera, mapCrop, wholeMapRegion } from './terrain.js';
import { clamp, num } from './util.js';
import { computeFollowRegion, isUnitVisible } from './view.js';

export function drawSandbox() {
  const canvas = els.sandboxCanvas;
  const wrap = els.stageWrap;
  if (!canvas || !wrap) return;
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;
  if (width < 8 || height < 8) return;
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

  const background = ctx.createLinearGradient(0, 0, 0, height);
  background.addColorStop(0, '#060d17');
  background.addColorStop(1, '#03070d');
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  const sandbox = state.sandbox;
  const detail = sandbox.mapDetail;
  const terrainSim = sandbox.simMode === 'terrain';
  const frames = sandbox.frames;
  const customScale = sandbox.customScale || 1.0;

  let frame = null;
  let nextFrame = null;
  let units = [];
  if (frames.length) {
    const maxIndex = frames.length - 1;
    const playback = clamp(sandbox.playback, 0, maxIndex);
    const index = Math.floor(playback);
    const nextIndex = Math.min(index + 1, maxIndex);
    const fraction = playback - index;
    frame = frames[index];
    nextFrame = frames[nextIndex];
    units = interpolateUnits(frame, nextFrame, fraction);
    updateUnitAngles(frame, index > 0 ? frames[index - 1] : null);
  }

  let region;
  if (detail) {
    if (sandbox.viewMode === 'full') {
      region = wholeMapRegion(detail);
    } else if (terrainSim) {
      region = computeFollowRegion(sandbox, detail, units, width, height);
    } else {
      region = mapCrop(detail);
    }
  } else if (terrainSim && sandbox.map && sandbox.map.width > 1000) {
    region = {
      x: 0,
      y: 0,
      width: Math.max(1, num(sandbox.map.width, 1000)),
      height: Math.max(1, num(sandbox.map.height, 1000)),
    };
  } else {
    region = { x: 0, y: 0, width: 1000, height: 1000 };
  }
  const mapMode = !!(detail && sandbox.terrainView);
  const cam = mapCamera(width, height, region, mapMode ? 0 : 20);
  cam.scale *= customScale;
  cam.ox -= (cam.region.x * (cam.scale - (cam.scale / customScale)));
  cam.oy -= (cam.region.y * (cam.scale - (cam.scale / customScale)));
  const offsetX = cam.ox + cam.region.x * cam.scale;
  const offsetY = cam.oy + cam.region.y * cam.scale;
  const drawnWidth = cam.region.width * cam.scale;
  const drawnHeight = cam.region.height * cam.scale;
  const mapX = (x) => {
    if (detail) {
      if (terrainSim) return cam.ox + num(x, 0) * cam.scale;
      return cam.ox + engineToMapPixel(x, 0, detail).x * cam.scale;
    }
    return cam.ox + num(x, 0) * cam.scale;
  };
  const mapY = (y) => {
    if (detail) {
      if (terrainSim) return cam.oy + num(y, 0) * cam.scale;
      return cam.oy + engineToMapPixel(0, y, detail).y * cam.scale;
    }
    return cam.oy + num(y, 0) * cam.scale;
  };
  sandbox.view = {
    ox: cam.ox,
    oy: cam.oy,
    scale: cam.scale,
    width: drawnWidth,
    height: drawnHeight,
    screenWidth: width,
    screenHeight: height,
    region: cam.region,
    detail: detail || null,
    simMode: sandbox.simMode,
    corners: { x: offsetX, y: offsetY, width: drawnWidth, height: drawnHeight },
    project: (x, y) => ({ x: mapX(x), y: mapY(y) }),
  };

  ctx.save();
  ctx.beginPath();
  ctx.rect(offsetX, offsetY, drawnWidth, drawnHeight);
  ctx.clip();

  if (mapMode) {
    const view = sandbox.terrainView;
    const sx = (cam.region.x - view.region.x) * view.scale;
    const sy = (cam.region.y - view.region.y) * view.scale;
    const sw = Math.max(1, cam.region.width * view.scale);
    const sh = Math.max(1, cam.region.height * view.scale);
    ctx.imageSmoothingEnabled = true;
    try {
      ctx.drawImage(view.ground, sx, sy, sw, sh, offsetX, offsetY, drawnWidth, drawnHeight);
      if (sandbox.itemsEnabled && view.items) {
        ctx.drawImage(view.items, sx, sy, sw, sh, offsetX, offsetY, drawnWidth, drawnHeight);
      }
    } catch (error) {
      // 画布绘制失败时保持暗色底，不阻塞单位渲染
    }

    const shade = ctx.createRadialGradient(
      width / 2, height / 2, Math.min(width, height) * 0.18,
      width / 2, height / 2, Math.max(width, height) * 0.72
    );
    shade.addColorStop(0, 'rgba(3, 8, 14, 0)');
    shade.addColorStop(1, 'rgba(3, 8, 14, 0.45)');
    ctx.fillStyle = shade;
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

    if (!terrainSim && detail && sandbox.viewMode === 'full') {
      const crop = mapCrop(detail);
      const bx = cam.ox + crop.x * cam.scale;
      const by = cam.oy + crop.y * cam.scale;
      const bw = crop.width * cam.scale;
      const bh = crop.height * cam.scale;
      ctx.save();
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = 'rgba(56, 225, 255, 0.6)';
      ctx.lineWidth = 1.2;
      ctx.strokeRect(bx, by, bw, bh);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(56, 225, 255, 0.92)';
      ctx.font = '10px "Microsoft YaHei", "Segoe UI", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('交战区域', bx + 6, by + 8);
      ctx.restore();
    }
  } else if (detail || (terrainSim && sandbox.map && sandbox.map.width > 1000)) {
    ctx.fillStyle = '#07111c';
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);
    ctx.save();
    ctx.fillStyle = 'rgba(160, 200, 230, 0.78)';
    ctx.font = '13px "Microsoft YaHei", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(sandbox.terrainError ? '地图加载失败，回退抽象渲染' : '地图加载中…', width / 2, height / 2);
    ctx.restore();
  } else {
    const mapWidth = Math.max(1, region.width);
    const mapHeight = Math.max(1, region.height);
    const mapGradient = ctx.createRadialGradient(
      offsetX + drawnWidth / 2, offsetY + drawnHeight / 2, 20,
      offsetX + drawnWidth / 2, offsetY + drawnHeight / 2, Math.max(drawnWidth, drawnHeight) * 0.8
    );
    mapGradient.addColorStop(0, '#0a1727');
    mapGradient.addColorStop(1, '#050b13');
    ctx.fillStyle = mapGradient;
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

    const ownGlow = ctx.createRadialGradient(
      offsetX + drawnWidth * 0.1, offsetY + drawnHeight / 2, 8,
      offsetX + drawnWidth * 0.1, offsetY + drawnHeight / 2, drawnWidth * 0.55
    );
    ownGlow.addColorStop(0, 'rgba(77, 163, 255, 0.12)');
    ownGlow.addColorStop(1, 'rgba(77, 163, 255, 0)');
    ctx.fillStyle = ownGlow;
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);
    const enemyGlow = ctx.createRadialGradient(
      offsetX + drawnWidth * 0.9, offsetY + drawnHeight / 2, 8,
      offsetX + drawnWidth * 0.9, offsetY + drawnHeight / 2, drawnWidth * 0.55
    );
    enemyGlow.addColorStop(0, 'rgba(255, 95, 109, 0.1)');
    enemyGlow.addColorStop(1, 'rgba(255, 95, 109, 0)');
    ctx.fillStyle = enemyGlow;
    ctx.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

    const gridStep = Math.max(50, Math.round(Math.max(mapWidth, mapHeight) / 10 / 50) * 50);
    for (let gridX = 0; gridX <= mapWidth; gridX += gridStep) {
      const screenX = mapX(gridX);
      ctx.strokeStyle = gridX % (gridStep * 5) === 0 ? SANDBOX_COLORS.gridStrong : SANDBOX_COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(screenX, offsetY);
      ctx.lineTo(screenX, offsetY + drawnHeight);
      ctx.stroke();
    }
    for (let gridY = 0; gridY <= mapHeight; gridY += gridStep) {
      const screenY = mapY(gridY);
      ctx.strokeStyle = gridY % (gridStep * 5) === 0 ? SANDBOX_COLORS.gridStrong : SANDBOX_COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(offsetX, screenY);
      ctx.lineTo(offsetX + drawnWidth, screenY);
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(56, 225, 255, 0.12)';
    ctx.setLineDash([6, 8]);
    ctx.beginPath();
    ctx.moveTo(offsetX + drawnWidth / 2, offsetY);
    ctx.lineTo(offsetX + drawnWidth / 2, offsetY + drawnHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    if (!terrainSim) {
      ctx.save();
      ctx.font = '10px "Microsoft YaHei", "Segoe UI", sans-serif';
      ctx.textBaseline = 'bottom';
      if (drawnWidth > 420) {
        ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(77, 163, 255, 0.45)';
        ctx.fillText('我方部署区', offsetX + 12, offsetY + drawnHeight - 10);
        ctx.textAlign = 'right';
        ctx.fillStyle = 'rgba(255, 95, 109, 0.42)';
        ctx.fillText('敌方部署区', offsetX + drawnWidth - 12, offsetY + drawnHeight - 10);
      }
      ctx.restore();
    }
  }

  if (!frames.length) {
    if (mapMode && sandbox.fogEnabled) {
      drawFogLayer(ctx, sandbox, units, mapX, mapY, cam, offsetX, offsetY, drawnWidth, drawnHeight, width, height);
    }
    ctx.restore();
    drawSandboxChrome(ctx, width, height, offsetX, offsetY, drawnWidth, drawnHeight);
    drawSandboxEmpty(ctx, width, height);
    return;
  }

  const ownUnits = units.filter((unit) => unit.team === 'own');
  const visibleUnits = units.filter((unit) => isUnitVisible(sandbox, unit, ownUnits));
  const hiddenEnemyIds = new Set(
    frame.units
      .filter((unit) => unit.team !== 'own' && !isUnitVisible(sandbox, unit, ownUnits))
      .map((unit) => unit.id)
  );

  if (mapMode && sandbox.fogEnabled) {
    drawFogLayer(ctx, sandbox, ownUnits, mapX, mapY, cam, offsetX, offsetY, drawnWidth, drawnHeight, width, height);
  }

  // 指令箭头（画在单位下层）
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, { x: unit.x, y: unit.y });
  for (const unit of visibleUnits) {
    if (unit.target === null || unit.target === undefined) continue;
    if (hiddenEnemyIds.has(unit.id) || hiddenEnemyIds.has(unit.target)) continue;
    const target = positions.get(unit.target);
    if (!target) continue;
    const selected = unit.id === sandbox.selectedId;
    const alpha = selected ? 0.5 : 0.12;
    ctx.save();
    ctx.strokeStyle = unit.team === 'own' ? 'rgba(77, 163, 255, ' + alpha + ')' : 'rgba(255, 95, 109, ' + alpha + ')';
    ctx.lineWidth = selected ? 1.6 : 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(mapX(unit.x), mapY(unit.y));
    ctx.lineTo(mapX(target.x), mapY(target.y));
    ctx.stroke();
    ctx.restore();
  }

  // 单位（按 y 排序，制造俯视层次）
  const sorted = visibleUnits.slice().sort((left, right) => left.y - right.y);
  const leaders = new Set();
  const seenGroups = new Set();
  for (const unit of sorted) {
    if (unit.team !== 'own') continue;
    if (seenGroups.has(unit.group)) continue;
    seenGroups.add(unit.group);
    leaders.add(unit.id);
  }
  const time = performance.now() / 1000;
  // 贴图尺寸由 drawSandboxSpriteUnit 按契约 2.3 自行计算（frameW/frameH * scale * cam.scale），
  // 这里只传相机缩放；抽象模式（无实机地图）仍用几何渲染，尺寸按世界碰撞半径换算。
  const baseRadius = 11.0;
  const unitRadius = Math.max(3.5, baseRadius * cam.scale);
  for (const unit of sorted) {
    if (mapMode) {
      drawSandboxSpriteUnit(ctx, unit, mapX(unit.x), mapY(unit.y), unit.id === sandbox.selectedId, leaders.has(unit.id), time, { scale: cam.scale });
    } else {
      drawSandboxUnit(ctx, unit, mapX, mapY, unit.id === sandbox.selectedId, leaders.has(unit.id), time, { radius: unitRadius });
    }
  }

  const eventVisible = sandbox.fogEnabled
    ? (event) => !hiddenEnemyIds.has(event.source) && !hiddenEnemyIds.has(event.target)
    : null;

  // 当前帧事件（暂停时静态显示，播放时由实时特效呈现）
  if (!sandbox.playing) {
    drawFrameEvents(ctx, frame, mapX, mapY, 1, eventVisible);
  }

  // 实时特效（弹道 / 爆炸，随帧推进生成）
  const now = performance.now();
  for (const effect of sandbox.effects) {
    const age = (now - effect.born) / effect.life;
    if (age >= 1) continue;
    if (sandbox.fogEnabled && (
      (effect.fromId !== undefined && hiddenEnemyIds.has(effect.fromId))
      || (effect.toId !== undefined && hiddenEnemyIds.has(effect.toId))
      || (effect.targetId !== undefined && hiddenEnemyIds.has(effect.targetId))
    )) continue;
    if (effect.kind === 'shot') {
      drawShotLine(ctx, mapX(effect.from.x), mapY(effect.from.y), mapX(effect.to.x), mapY(effect.to.y), (1 - age) * 0.95);
    } else if (effect.kind === 'kill') {
      drawKillBurst(ctx, mapX(effect.x), mapY(effect.y), age, 1);
    }
  }
  sandbox.effects = sandbox.effects.filter((effect) => now - effect.born < effect.life);

  ctx.restore();

  drawSandboxChrome(ctx, width, height, offsetX, offsetY, drawnWidth, drawnHeight);
  drawSandboxHud(ctx, sandbox, frame, width, height);
  if (sandbox.source === 'demo') {
    ctx.save();
    ctx.font = '11px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(255, 195, 92, 0.65)';
    ctx.fillText('演示数据 · DEMO', width - 14, height - 10);
    ctx.restore();
  }
}

export const fogLayer = { canvas: null, ctx: null };

export function ensureFogLayer(width, height) {
  if (!fogLayer.canvas) {
    fogLayer.canvas = document.createElement('canvas');
    fogLayer.ctx = fogLayer.canvas.getContext('2d');
  }
  const targetWidth = Math.max(1, Math.ceil(width));
  const targetHeight = Math.max(1, Math.ceil(height));
  if (fogLayer.canvas.width !== targetWidth || fogLayer.canvas.height !== targetHeight) {
    fogLayer.canvas.width = targetWidth;
    fogLayer.canvas.height = targetHeight;
  }
  return fogLayer;
}

export function drawFogLayer(ctx, sandbox, ownUnits, mapX, mapY, cam, offsetX, offsetY, drawnWidth, drawnHeight, width, height) {
  const layer = ensureFogLayer(width, height);
  const fog = layer.ctx;
  if (!fog) return;
  fog.setTransform(1, 0, 0, 1, 0, 0);
  fog.clearRect(0, 0, width, height);
  fog.save();
  fog.beginPath();
  fog.rect(offsetX, offsetY, drawnWidth, drawnHeight);
  fog.clip();
  fog.fillStyle = 'rgba(2, 6, 12, 0.78)';
  fog.fillRect(offsetX, offsetY, drawnWidth, drawnHeight);

  const radius = Math.max(12, num(sandbox.visionRadius, 260) * cam.scale);
  fog.globalCompositeOperation = 'destination-out';
  for (const unit of ownUnits) {
    const x = mapX(unit.x);
    const y = mapY(unit.y);
    const gradient = fog.createRadialGradient(x, y, radius * 0.32, x, y, radius);
    gradient.addColorStop(0, 'rgba(0, 0, 0, 1)');
    gradient.addColorStop(0.62, 'rgba(0, 0, 0, 0.9)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    fog.fillStyle = gradient;
    fog.beginPath();
    fog.arc(x, y, radius, 0, Math.PI * 2);
    fog.fill();
  }
  fog.globalCompositeOperation = 'source-over';
  fog.restore();

  ctx.save();
  ctx.globalAlpha = 0.94;
  ctx.drawImage(layer.canvas, 0, 0, width, height);
  ctx.restore();
}

export function drawSandboxChrome(ctx, width, height, offsetX, offsetY, drawnWidth, drawnHeight) {
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.35)';
  ctx.lineWidth = 1;
  ctx.strokeRect(offsetX + 0.5, offsetY + 0.5, drawnWidth - 1, drawnHeight - 1);
  // 四角标识
  const corner = 16;
  ctx.strokeStyle = 'rgba(56, 225, 255, 0.85)';
  ctx.lineWidth = 2;
  const corners = [
    [offsetX, offsetY, 1, 1],
    [offsetX + drawnWidth, offsetY, -1, 1],
    [offsetX, offsetY + drawnHeight, 1, -1],
    [offsetX + drawnWidth, offsetY + drawnHeight, -1, -1],
  ];
  for (const item of corners) {
    const cx = item[0];
    const cy = item[1];
    const dx = item[2];
    const dy = item[3];
    ctx.beginPath();
    ctx.moveTo(cx + dx * corner, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * corner);
    ctx.stroke();
  }
  ctx.restore();
}

export function drawSandboxEmpty(ctx, width, height) {
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(124, 147, 173, 0.9)';
  ctx.font = '15px "Segoe UI", "Microsoft YaHei", sans-serif';
  ctx.fillText('暂无对局数据', width / 2, height / 2 - 12);
  ctx.fillStyle = 'rgba(93, 115, 144, 0.85)';
  ctx.font = '12px "Segoe UI", "Microsoft YaHei", sans-serif';
  ctx.fillText(
    state.mode === 'demo' ? '演示模式：点击「开始实机推理」播放内置演示对局' : '在右侧选择地图与时长，点击「开始实机推理」；或从下方回放库加载一局',
    width / 2, height / 2 + 14
  );
  ctx.restore();
}
