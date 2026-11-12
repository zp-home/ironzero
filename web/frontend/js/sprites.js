/**
 * sprites.js — 单位贴图：贴图加载/缓存/帧动画、朝向角、贴图单位绘制。
 */

import { Api } from './api.js';
import { state } from './state.js';
import { SANDBOX_COLORS } from './terrain.js';
import { clamp, num } from './util.js';

/** 移动单位贴图帧率（docs/沙盘贴图契约.md 2.4） */
export const UNIT_SPRITE_FPS = 6;

/** 单位最小可见高度（屏幕像素），过小的贴图等比放大到该高度 */
export const UNIT_SPRITE_MIN_H = 10;

/** 进入实机渲染前等待贴图加载的最长时间（毫秒），超时先渲染、后续帧自然补齐 */
export const UNIT_SPRITE_LOAD_TIMEOUT_MS = 4000;

/** rel -> Image，跨类型共享；加载失败只 console.warn 一次 */
export const unitSpriteImages = new Map();

/** rel -> Promise，避免同一 rel 并发重复请求 */
export const unitSpritePending = new Map();

/** 已警告过的键，保证同一问题只 warn 一次 */
export const unitSpriteWarned = new Set();

export const unitAngleCache = new Map();

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('图片加载失败：' + url));
    image.src = url;
  });
}

export function warnSpriteOnce(key, message) {
  if (unitSpriteWarned.has(key)) return;
  unitSpriteWarned.add(key);
  console.warn('[沙盘贴图] ' + message);
}

/**
 * 按 rel 取已加载的贴图；未加载则发起加载并返回 null（下一帧起可用）。
 * 加载失败写入 null 并 warn 一次，之后不再重试（has() 判断，避免 null 被当成未缓存而每帧重试）。
 */
export function unitSpriteImage(rel) {
  if (!rel) return null;
  if (unitSpriteImages.has(rel)) return unitSpriteImages.get(rel);
  if (!unitSpritePending.has(rel)) {
    const job = loadImage(Api.assetUrl(rel))
      .then((image) => {
        unitSpriteImages.set(rel, image);
        return image;
      })
      .catch(() => {
        unitSpriteImages.set(rel, null);
        warnSpriteOnce('rel:' + rel, '贴图加载失败，该单位回退矢量绘制：' + rel);
        return null;
      })
      .finally(() => {
        unitSpritePending.delete(rel);
      });
    unitSpritePending.set(rel, job);
  }
  return null;
}

/** 预加载清单内全部贴图，返回加载 Promise 数组（不阻塞调用方；缺失的按契约 warn 一次并回退矢量）。 */
export function preloadUnitSprites(manifest) {
  if (!manifest || typeof manifest !== 'object') return [];
  const rels = new Set();
  const push = (entry) => {
    if (!entry || typeof entry !== 'object') return;
    if (entry.image) rels.add(entry.image);
    if (entry.turretImage) rels.add(entry.turretImage);
  };
  for (const key of Object.keys(manifest.byType || {})) push(manifest.byType[key]);
  for (const key of Object.keys(manifest.fallback || {})) push(manifest.fallback[key]);
  const jobs = [];
  for (const rel of rels) {
    unitSpriteImage(rel); // 触发请求
    const job = unitSpritePending.get(rel);
    if (job) jobs.push(job);
  }
  return jobs;
}

/**
 * 解析单位贴图描述，顺序：
 *   unitType → byType[aliases[unitType]] → fallback[builder|building|mobile]
 * 返回清单里的原始条目，或 null（无清单 / 无可用条目）。
 */
export function resolveUnitSprite(unit) {
  const manifest = state.maps && state.maps.unitSprites;
  if (!manifest || typeof manifest !== 'object') {
    warnSpriteOnce('manifest', '后端未下发 unitSprites，单位回退矢量绘制');
    return null;
  }
  const byType = manifest.byType && typeof manifest.byType === 'object' ? manifest.byType : {};
  const aliases = manifest.aliases && typeof manifest.aliases === 'object' ? manifest.aliases : {};
  const fallback = manifest.fallback && typeof manifest.fallback === 'object' ? manifest.fallback : {};
  const type = unit.unitType ? String(unit.unitType) : '';
  let entry = null;
  if (type && byType[type]) {
    entry = byType[type];
  } else if (type && aliases[type] && byType[aliases[type]]) {
    entry = byType[aliases[type]];
  } else if (unit.isBuilder && fallback.builder) {
    entry = fallback.builder;
  } else if (unit.isBuilding && fallback.building) {
    entry = fallback.building;
  } else {
    entry = fallback.mobile || fallback.building || null;
  }
  if (!entry || typeof entry !== 'object') return null;
  return entry;
}

/**
 * 预加载全部单位贴图再进入实机渲染，避免“切到地图后先闪一帧矢量图形”。
 * 单张失败已在 unitSpriteImage 里 warn 并回退，这里只等最长时间，不让坏图卡住整图加载。
 */
export async function ensureUnitSprites() {
  const jobs = preloadUnitSprites(state.maps && state.maps.unitSprites);
  if (!jobs.length) return true;
  await Promise.race([
    Promise.all(jobs),
    new Promise((resolve) => setTimeout(resolve, UNIT_SPRITE_LOAD_TIMEOUT_MS)),
  ]);
  return true;
}

export function updateUnitAngles(frame, previousFrame) {
  const previous = new Map();
  if (previousFrame) {
    for (const unit of previousFrame.units) previous.set(unit.id, unit);
  }
  const positions = new Map();
  for (const unit of frame.units) positions.set(unit.id, unit);
  for (const unit of frame.units) {
    const record = unitAngleCache.get(unit.id) || { heading: -Math.PI / 2, turret: -Math.PI / 2 };
    const before = previous.get(unit.id);
    if (before) {
      const dx = unit.x - before.x;
      const dy = unit.y - before.y;
      if (Math.hypot(dx, dy) > 0.6) record.heading = Math.atan2(dy, dx);
    }
    let target = null;
    if (unit.target !== null && unit.target !== undefined) target = positions.get(unit.target) || null;
    if (!target) {
      let bestDistance = Infinity;
      for (const item of frame.units) {
        if (item.team === unit.team) continue;
        const distance = Math.hypot(item.x - unit.x, item.y - unit.y);
        if (distance < bestDistance) {
          bestDistance = distance;
          target = item;
        }
      }
    }
    if (target) record.turret = Math.atan2(target.y - unit.y, target.x - unit.x);
    unitAngleCache.set(unit.id, record);
  }
}

/** 清单缺失时按契约 1 节的 fallback 尺寸兜底（只用于矢量保底的几何大小） */
export const UNIT_FALLBACK_SPRITE = {
  mobile: { frameW: 16, frameH: 30, scale: 1, kind: 'mobile' },
  builder: { frameW: 20, frameH: 20, scale: 1, kind: 'mobile' },
  building: { frameW: 35, frameH: 44, scale: 1, kind: 'building' },
};

/**
 * 从清单条目解析出分帧盒；清单缺字段时用图片真实尺寸反推（契约 1 节约束 2）。
 * 图片还没加载完（image 为空）且清单没给 frameW/frameH 时，用 kind 的契约兜底尺寸，
 * 避免图片到位前先按 1px 画一帧导致闪跳。
 */
export function unitSpriteFrameBox(sprite, image, kind) {
  const frames = Math.max(1, Math.round(num(sprite.frames, 1)));
  const fallback = UNIT_FALLBACK_SPRITE[kind] || UNIT_FALLBACK_SPRITE.mobile;
  let frameW = num(sprite.frameW, 0);
  let frameH = num(sprite.frameH, 0);
  if (frameW <= 0) frameW = image ? Math.floor(image.width / frames) : fallback.frameW;
  if (frameH <= 0) frameH = image ? image.height : fallback.frameH;
  // 契约要求 frameH == 图片高；不符时以图片为准，避免把整图压扁
  if (image && Math.abs(frameH - image.height) > 0.5) frameH = image.height;
  return {
    frames: frames,
    frameW: Math.max(1, frameW),
    frameH: Math.max(1, frameH),
    scale: num(sprite.scale, 1) || 1,
  };
}

/** 契约 2.3：1:1 世界像素，drawH 不足 UNIT_SPRITE_MIN_H 时整体等比放大。 */
export function unitSpriteDrawSize(box, camScale) {
  const factor = (box.scale || 1) * camScale;
  let drawW = box.frameW * factor;
  let drawH = box.frameH * factor;
  if (!(drawH > 0)) {
    drawH = UNIT_SPRITE_MIN_H;
    drawW = Math.max(1, box.frameW * factor);
  }
  if (drawH < UNIT_SPRITE_MIN_H) {
    const boost = UNIT_SPRITE_MIN_H / drawH;
    drawW *= boost;
    drawH = UNIT_SPRITE_MIN_H;
  }
  return { drawW: drawW, drawH: drawH };
}

/** 契约 2.4：移动单位按时间取帧，建筑固定第 0 帧。 */
export function unitSpriteFrameIndex(frames, isBuilding, time) {
  if (isBuilding || frames <= 1) return 0;
  const raw = Math.floor(Math.max(0, num(time, 0)) * UNIT_SPRITE_FPS);
  return ((raw % frames) + frames) % frames;
}

/**
 * 沙盘单位绘制（docs/沙盘贴图契约.md 第 2 节）：贴图优先，矢量保底。
 * options.scale 为相机缩放 cam.scale；贴图尺寸由本函数按契约 2.3 自行计算。
 * 保留阵营底盘指示、选中/精英环、血条；单位 x,y 为贴图中心。
 */
export function drawSandboxSpriteUnit(ctx, unit, x, y, selected, leader, time, options) {
  const own = unit.team === 'own';
  const color = own ? SANDBOX_COLORS.own : SANDBOX_COLORS.enemy;
  const camScale = Math.max(0.0001, num(options && options.scale, 1));
  const hpFraction = clamp(unit.hp / Math.max(1, unit.maxHp), 0, 1);
  const angles = unitAngleCache.get(unit.id) || { heading: -Math.PI / 2, turret: -Math.PI / 2 };
  const isBuilding = !!unit.isBuilding;
  const isBuilder = !!unit.isBuilder;

  const sprite = resolveUnitSprite(unit);
  const image = sprite ? unitSpriteImage(sprite.image) : null;
  const turretImage = sprite && sprite.turretImage ? unitSpriteImage(sprite.turretImage) : null;

  // 尺寸：有清单条目就用条目（图片未到位也保持一致，避免闪跳），否则用契约 fallback 尺寸
  const fallbackKind = isBuilding ? 'building' : (isBuilder ? 'builder' : 'mobile');
  const box = sprite ? unitSpriteFrameBox(sprite, image, fallbackKind) : UNIT_FALLBACK_SPRITE[fallbackKind];
  const size = unitSpriteDrawSize(box, camScale);
  const drawW = size.drawW;
  const drawH = size.drawH;
  const minSpan = Math.min(drawW, drawH);

  // 阵营底盘指示（建筑圆形矩形，移动单位椭圆）
  ctx.save();
  ctx.beginPath();
  if (isBuilding) {
    const bw = Math.max(6, drawW * 0.86);
    const bh = Math.max(6, drawH * 0.86);
    ctx.rect(x - bw / 2, y - bh / 2, bw, bh);
  } else {
    ctx.ellipse(x, y + drawH * 0.18, Math.max(4, drawW * 0.42), Math.max(3, drawH * 0.24), 0, 0, Math.PI * 2);
  }
  ctx.strokeStyle = own ? 'rgba(63, 169, 255, 0.9)' : 'rgba(255, 90, 90, 0.9)';
  ctx.lineWidth = Math.max(1, minSpan * 0.06);
  ctx.shadowColor = color;
  ctx.shadowBlur = Math.min(18, Math.max(4, minSpan * 0.3));
  ctx.stroke();
  ctx.restore();

  if (selected || leader) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, Math.max(7, Math.max(drawW, drawH) * 0.6), 0, Math.PI * 2);
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

  // 贴图本体
  let rendered = false;
  if (image) {
    const frames = Math.max(1, Math.min(box.frames, Math.max(1, Math.floor(image.width / box.frameW))));
    const frame = unitSpriteFrameIndex(frames, isBuilding, time);
    const frameH = Math.max(1, Math.min(box.frameH, image.height));
    ctx.save();
    ctx.translate(x, y);
    if (!isBuilding) ctx.rotate(angles.heading + Math.PI / 2);
    ctx.drawImage(
      image,
      frame * box.frameW, 0, box.frameW, frameH,
      -drawW / 2, -drawH / 2, drawW, drawH
    );
    ctx.restore();
    rendered = true;

    // 炮塔单独贴图：围绕单位中心按炮塔角度叠加（建筑不随航向旋转，但炮塔要转）
    if (turretImage) {
      const turretBox = unitSpriteFrameBox(
        {
          frames: num(sprite.turretFrames, 1),
          frameW: num(sprite.turretFrameW, 0),
          frameH: num(sprite.turretFrameH, 0),
          scale: num(sprite.turretScale, box.scale),
        },
        turretImage,
        'mobile'
      );
      const turretSize = unitSpriteDrawSize(turretBox, camScale);
      const turretFrames = Math.max(
        1,
        Math.min(turretBox.frames, Math.max(1, Math.floor(turretImage.width / turretBox.frameW)))
      );
      const turretFrame = unitSpriteFrameIndex(turretFrames, false, time);
      const turretFrameH = Math.max(1, Math.min(turretBox.frameH, turretImage.height));
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angles.turret + Math.PI / 2);
      ctx.drawImage(
        turretImage,
        turretFrame * turretBox.frameW, 0, turretBox.frameW, turretFrameH,
        -turretSize.drawW / 2, -turretSize.drawH / 2, turretSize.drawW, turretSize.drawH
      );
      ctx.restore();
    }
  }

  // 矢量保底：只在清单缺失或图片加载失败时出现（resolveUnitSprite / unitSpriteImage 已各 warn 一次）
  if (!rendered) {
    ctx.save();
    if (isBuilding) {
      ctx.beginPath();
      ctx.rect(x - drawW * 0.4, y - drawH * 0.4, drawW * 0.8, drawH * 0.8);
      ctx.fillStyle = own ? '#1b4b8a' : '#8a1b24';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.fillRect(x - minSpan * 0.2, y - minSpan * 0.2, minSpan * 0.4, minSpan * 0.4);
    } else if (isBuilder) {
      const radius = Math.max(4, minSpan * 0.45);
      ctx.beginPath();
      for (let i = 0; i < 6; i += 1) {
        const rad = (i * Math.PI) / 3;
        const px = x + Math.cos(rad) * radius;
        const py = y + Math.sin(rad) * radius;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = own ? '#2f8fbf' : '#d97724';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(x, y, Math.max(4, minSpan * 0.45), 0, Math.PI * 2);
      ctx.fillStyle = own ? '#2f6fbf' : '#b5303c';
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
    ctx.restore();
  }

  // 血条：按绘制高度定位到贴图上方，避免大建筑血条压在贴图上
  const barWidth = Math.max(14, drawW * 0.78);
  const barHeight = Math.max(2.5, Math.min(6, drawH * 0.1));
  const barX = x - barWidth / 2;
  const barY = y - drawH / 2 - barHeight - 2;
  ctx.fillStyle = 'rgba(2, 6, 12, 0.8)';
  ctx.fillRect(barX - 1, barY - 1, barWidth + 2, barHeight + 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.fillRect(barX, barY, barWidth, barHeight);
  ctx.fillStyle = hpFraction > 0.6 ? '#39e08b' : (hpFraction > 0.3 ? '#ffc35c' : '#ff5f6d');
  ctx.fillRect(barX, barY, barWidth * hpFraction, barHeight);
}
