/**
 * camera.js — 沙盘相机：以光标为锚点的缩放、拖动平移与边界钳制（纯函数，无 DOM）。
 *
 * 坐标模型
 * --------
 * 沙盘的绘制矩形 rect 完全由三个量决定：
 *
 *   zoom  相对「刚好装满画布」的倍数，1.0 = 全览（fit），因此恒 ≥ 1，永远不会露出画布外的空白
 *   panX  相对画布中心、在屏幕上向右平移的像素数（地图坐标无关，纯屏幕量）
 *   panY  同上，向下为正
 *
 * 由此反解出世界→屏幕的仿射参数（scale/ox/oy），并保证
 * ``ox = rect.x - region.x * scale``，也就是 mapX(x) = ox + x*scale 与
 * rect 严格自洽 —— 这是修复「贴图地形与单位对不齐」的关键约束。
 *
 * zoom = 1 且 pan = 0 时本模块的输出必须与旧实现 mapCamera() 逐像素一致。
 */

import { clamp, num } from './util.js';

/** 最小缩放：1.0 即「刚好装满」，不允许再缩小（否则画布四周会露出空白） */
export const MIN_ZOOM = 1;

/** 最大缩放：相对 fit 的 8 倍，足以让单个坦克占满半屏 */
export const MAX_ZOOM = 8;

/** 滚轮缩放的每格灵敏度：deltaY 100 一格 ≈ 1.36× / 0.73× */
export const WHEEL_ZOOM_STEP = 320;

/**
 * 计算基准几何：不改动之前 mapCamera() 的 contain 拟合结果。
 *
 * @param {number} width   画布 CSS 宽度
 * @param {number} height  画布 CSS 高度
 * @param {{x:number,y:number,width:number,height:number}} region 可视区域（世界坐标，单位随模式而定）
 * @param {number} padding 四周留边（地图模式为 0，抽象模式为 20）
 * @returns {{width:number,height:number,region:object,scale:number,ox:number,oy:number,rect:object}}
 */
export function baseGeometry(width, height, region, padding) {
  const pad = num(padding, 0);
  const rw = Math.max(1, num(region && region.width, 1));
  const rh = Math.max(1, num(region && region.height, 1));
  const geoRegion = {
    x: num(region && region.x, 0),
    y: num(region && region.y, 0),
    width: rw,
    height: rh,
  };
  const scale = Math.max(0.0001, Math.min((width - pad * 2) / rw, (height - pad * 2) / rh));
  const ox = (width - rw * scale) / 2 - geoRegion.x * scale;
  const oy = (height - rh * scale) / 2 - geoRegion.y * scale;
  return {
    width: width,
    height: height,
    region: geoRegion,
    scale: scale,
    ox: ox,
    oy: oy,
    // 基准绘制矩形：与 mapCamera() 的 offset 计算等价
    rect: {
      x: ox + geoRegion.x * scale,
      y: oy + geoRegion.y * scale,
      width: rw * scale,
      height: rh * scale,
    },
  };
}

/** 取（并就地修正）沙盘的相机状态；缺失或脏值时回落到全览 */
export function ensureCamera(sandbox) {
  const target = sandbox;
  const raw = target && target.camera;
  const zoom = raw ? num(raw.zoom, 1) : 1;
  const panX = raw ? num(raw.panX, 0) : 0;
  const panY = raw ? num(raw.panY, 0) : 0;
  const camera = {
    zoom: clamp(zoom, MIN_ZOOM, MAX_ZOOM),
    panX: Number.isFinite(panX) ? panX : 0,
    panY: Number.isFinite(panY) ? panY : 0,
  };
  if (target && raw !== camera) target.camera = camera;
  return camera;
}

/** 复位到全览（切换地图 / 切换视野 / 双击画布时调用） */
export function resetCamera(sandbox) {
  if (!sandbox) return { zoom: 1, panX: 0, panY: 0 };
  const camera = { zoom: 1, panX: 0, panY: 0 };
  sandbox.camera = camera;
  return camera;
}

/** 是否处于全览状态（UI 用它决定「复位」按钮是否可点） */
export function isDefaultCamera(camera) {
  if (!camera) return true;
  return Math.abs(num(camera.zoom, 1) - 1) < 0.001
    && Math.abs(num(camera.panX, 0)) < 0.5
    && Math.abs(num(camera.panY, 0)) < 0.5;
}

/**
 * 单轴钳制。
 *
 * - 内容宽于视口：限制在「内容始终盖满视口」的区间内，拖到边界即停，不会露空白。
 * - 内容窄于视口：全览（zoom=1）时严格居中（贴合、干净，也等价于旧实现）；
 *   一旦放大，允许在「内容仍留在视口内」的范围内微调。
 *
 * 后一条不能用「窄于视口就强制居中」代替 —— 否则低缩放区间的光标锚点直接失效
 * （放大时视觉上仍像从中心/角落扩），横向平移也要等内容宽于视口才生效，手感很别扭。
 */
function clampAxisCenter(center, size, viewport, zoom) {
  if (size <= viewport) {
    if (zoom <= 1 + 1e-9) return viewport / 2;
    const half = size / 2;
    return clamp(center, half, viewport - half);
  }
  return clamp(center, viewport - size / 2, size / 2);
}

/**
 * 把相机状态解算成实际渲染参数。
 * 返回的 panX/panY 是钳制后的「有效值」，调用方应写回沙盘，避免拖出边界后累积。
 *
 * @returns {{zoom:number,panX:number,panY:number,scale:number,ox:number,oy:number,
 *            rect:{x:number,y:number,width:number,height:number}}}
 */
export function applyCamera(geom, camera) {
  const zoom = clamp(num(camera && camera.zoom, 1), MIN_ZOOM, MAX_ZOOM);
  const width = geom.region.width * geom.scale * zoom;
  const height = geom.region.height * geom.scale * zoom;
  const centerX = clampAxisCenter(geom.width / 2 + num(camera && camera.panX, 0), width, geom.width, zoom);
  const centerY = clampAxisCenter(geom.height / 2 + num(camera && camera.panY, 0), height, geom.height, zoom);
  const scale = geom.scale * zoom;
  const rect = {
    x: centerX - width / 2,
    y: centerY - height / 2,
    width: width,
    height: height,
  };
  return {
    zoom: zoom,
    panX: centerX - geom.width / 2,
    panY: centerY - geom.height / 2,
    scale: scale,
    ox: rect.x - geom.region.x * scale,
    oy: rect.y - geom.region.y * scale,
    rect: rect,
  };
}

/**
 * 以画布上的 (screenX, screenY) 为不动点缩放。
 * 该点下的地图内容缩放前后保持在同一像素位置 —— 这就是「滚轮从左上角放大」的修复点。
 */
export function zoomAt(geom, camera, screenX, screenY, targetZoom) {
  const current = applyCamera(geom, camera);
  const zoom = clamp(num(targetZoom, current.zoom), MIN_ZOOM, MAX_ZOOM);
  if (Math.abs(zoom - current.zoom) < 1e-6) return current;
  const fractionX = (num(screenX, 0) - current.rect.x) / Math.max(1, current.rect.width);
  const fractionY = (num(screenY, 0) - current.rect.y) / Math.max(1, current.rect.height);
  const width = geom.region.width * geom.scale * zoom;
  const height = geom.region.height * geom.scale * zoom;
  const centerX = num(screenX, 0) - fractionX * width + width / 2;
  const centerY = num(screenY, 0) - fractionY * height + height / 2;
  return applyCamera(geom, {
    zoom: zoom,
    panX: centerX - geom.width / 2,
    panY: centerY - geom.height / 2,
  });
}

/** 按屏幕像素平移（拖动时传鼠标位移的相反数） */
export function panBy(geom, camera, dx, dy) {
  const current = applyCamera(geom, camera);
  return applyCamera(geom, {
    zoom: current.zoom,
    panX: current.panX + num(dx, 0),
    panY: current.panY + num(dy, 0),
  });
}

/**
 * 滚轮增量 → 缩放倍率。
 * deltaMode：0 = 像素，1 = 行，2 = 页；统一折算成像素后走指数曲线，
 * 保证任意鼠标/触控板都能得到平滑且方向正确的放大缩小。
 */
export function zoomFactorFromWheel(deltaY, deltaMode) {
  const mode = num(deltaMode, 0);
  const unit = mode === 1 ? 16 : (mode === 2 ? 100 : 1);
  const delta = clamp(num(deltaY, 0) * unit, -600, 600);
  if (!delta) return 1;
  return Math.exp(-delta / WHEEL_ZOOM_STEP);
}

/** 把相机状态写回沙盘（只写有效值，保持 sandbox.camera 始终合法） */
export function commitCamera(sandbox, solved) {
  if (!sandbox) return null;
  const camera = {
    zoom: solved.zoom,
    panX: solved.panX,
    panY: solved.panY,
  };
  sandbox.camera = camera;
  return camera;
}

/** 渲染层用的便捷封装：从沙盘取相机，返回解算结果 */
export function resolveCamera(geom, sandbox) {
  return applyCamera(geom, ensureCamera(sandbox));
}
