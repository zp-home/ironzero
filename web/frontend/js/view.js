/**
 * view.js — 实机地图 · 相机与视图：跟随区域计算、地图选择、视图模式、控件同步。
 */

import { Api } from './api.js';
import { toast } from './dashboard.js';
import { drawSandbox } from './render.js';
import { ensureUnitSprites } from './sprites.js';
import { els, state } from './state.js';
import { buildTerrainEntry, ensureFullTerrain, ensureTerrainMap, mapCrop } from './terrain.js';
import { setMapStatus, syncMapStatus, wholeMapRegion } from './terrain.js';
import { clamp, num } from './util.js';

export function updateTerrainView(sandbox) {
  const detail = sandbox.mapDetail;
  const entry = sandbox.terrainEntry;
  if (!detail || !entry) {
    sandbox.terrainView = null;
    return;
  }
  if (sandbox.simMode === 'terrain') {
    const map = ensureTerrainMap(entry, detail);
    sandbox.terrainView = {
      ground: map.ground,
      items: map.items,
      region: wholeMapRegion(detail),
      scale: map.scale,
    };
    return;
  }
  if (sandbox.viewMode === 'full') {
    const full = ensureFullTerrain(entry, detail);
    sandbox.terrainView = {
      ground: full.ground,
      items: full.items,
      region: wholeMapRegion(detail),
      scale: full.scale,
    };
    return;
  }
  const crop = mapCrop(detail);
  sandbox.terrainView = {
    ground: entry.cropGround,
    items: entry.cropItems,
    region: crop,
    scale: 1,
  };
}

export async function refreshTerrainView() {
  const sandbox = state.sandbox;
  if (!sandbox.mapDetail || !sandbox.terrainEntry) {
    sandbox.terrainView = null;
    syncMapStatus();
    return;
  }
  const entry = sandbox.terrainEntry;
  const heavy = (sandbox.simMode === 'terrain' && !entry.mapGround)
    || (sandbox.simMode !== 'terrain' && sandbox.viewMode === 'full' && !entry.fullGround);
  if (heavy) {
    setMapStatus('地形渲染中…');
    await new Promise((resolve) => setTimeout(resolve, 16));
  }
  updateTerrainView(sandbox);
  sandbox.cameraRegion = null;
  syncMapStatus();
  syncMapSelect();
  syncMapToggles();
  drawSandbox();
}

export function computeFollowRegion(sandbox, detail, units, screenW, screenH) {
  const mapWidth = Math.max(1, num(detail.width_px, 1000));
  const mapHeight = Math.max(1, num(detail.height_px, 1000));
  const canvasW = Math.max(1, num(screenW, 900));
  const canvasH = Math.max(1, num(screenH, 600));
  const aspect = canvasW / canvasH;
  const valid = (units || []).filter((unit) => Number.isFinite(unit.x) && Number.isFinite(unit.y));
  const own = valid.filter((unit) => unit.team === 'own');
  const boundsOf = (list) => {
    if (!list.length) return null;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const unit of list) {
      minX = Math.min(minX, unit.x);
      minY = Math.min(minY, unit.y);
      maxX = Math.max(maxX, unit.x);
      maxY = Math.max(maxY, unit.y);
    }
    return {
      minX: minX,
      minY: minY,
      maxX: maxX,
      maxY: maxY,
      width: Math.max(1, maxX - minX),
      height: Math.max(1, maxY - minY),
    };
  };
  // 单位可见优先：跟随双方单位包围盒；窗口宁可 letterbox 也绝不裁掉坦克。
  const focus = valid.length ? valid : own;
  const focusBounds = boundsOf(focus);
  let target;
  if (!focusBounds) {
    const fallbackW = Math.min(mapWidth, 1400);
    const fallbackH = Math.min(mapHeight, fallbackW / aspect);
    target = {
      x: (mapWidth - fallbackW) / 2,
      y: (mapHeight - fallbackH) / 2,
      width: fallbackW,
      height: fallbackH,
    };
  } else {
    const pad = 150;
    const margin = 300;
    let needW = Math.max(760, focusBounds.width + margin * 2, focusBounds.width + pad * 2);
    let needH = Math.max(760, focusBounds.height + margin * 2, focusBounds.height + pad * 2);
    // 按画布宽高比做“只扩不缩”的 cover 扩展（先保证单位装得下）
    let winW;
    let winH;
    if (needW / needH >= aspect) {
      winW = needW;
      winH = winW / aspect;
    } else {
      winH = needH;
      winW = winH * aspect;
    }
    // 窗口上限 1400，但绝不低于我方包围盒 + 留白：单位可见优先于填满画布
    const maxWinH = Math.max(1400, focusBounds.height + pad * 2);
    const maxWinW = Math.max(1400 * aspect, focusBounds.width + pad * 2);
    if (winH > maxWinH) {
      winH = maxWinH;
      winW = winH * aspect;
    }
    if (winW > maxWinW) {
      winW = maxWinW;
      winH = winW / aspect;
    }
    winW = Math.max(winW, focusBounds.width + pad * 2);
    winH = Math.max(winH, focusBounds.height + pad * 2);
    // 只有仍能装下单位时才缩到地图内；装不下则保留大窗口，由 mapCamera contain 绘制（允许留边）
    let regionW = winW;
    let regionH = winH;
    if (regionW > mapWidth || regionH > mapHeight) {
      const fit = Math.min(mapWidth / regionW, mapHeight / regionH);
      if (regionW * fit >= focusBounds.width + pad * 2 && regionH * fit >= focusBounds.height + pad * 2) {
        regionW *= fit;
        regionH *= fit;
      }
    }
    const centerX = (focusBounds.minX + focusBounds.maxX) / 2;
    const centerY = (focusBounds.minY + focusBounds.maxY) / 2;
    let x;
    let y;
    if (regionW <= mapWidth) {
      x = clamp(centerX - regionW / 2, focusBounds.maxX + pad - regionW, focusBounds.minX - pad);
      x = clamp(x, 0, mapWidth - regionW);
    } else {
      x = (mapWidth - regionW) / 2;
    }
    if (regionH <= mapHeight) {
      y = clamp(centerY - regionH / 2, focusBounds.maxY + pad - regionH, focusBounds.minY - pad);
      y = clamp(y, 0, mapHeight - regionH);
    } else {
      y = (mapHeight - regionH) / 2;
    }
    target = { x: x, y: y, width: regionW, height: regionH };
  }
  const previous = sandbox.cameraRegion;
  if (!previous) {
    sandbox.cameraRegion = { x: target.x, y: target.y, width: target.width, height: target.height };
    return sandbox.cameraRegion;
  }
  const smoothing = 0.2;
  const next = {
    x: previous.x + (target.x - previous.x) * smoothing,
    y: previous.y + (target.y - previous.y) * smoothing,
    width: previous.width + (target.width - previous.width) * smoothing,
    height: previous.height + (target.height - previous.height) * smoothing,
  };
  if (Math.abs(next.x - target.x) < 0.5 && Math.abs(next.y - target.y) < 0.5) {
    next.x = target.x;
    next.y = target.y;
  }
  if (Math.abs(next.width - target.width) < 0.5 && Math.abs(next.height - target.height) < 0.5) {
    next.width = target.width;
    next.height = target.height;
  }
  if (focusBounds) {
    const safePad = 20;
    const containsFocus =
      next.x <= focusBounds.minX - safePad &&
      next.x + next.width >= focusBounds.maxX + safePad &&
      next.y <= focusBounds.minY - safePad &&
      next.y + next.height >= focusBounds.maxY + safePad;
    if (!containsFocus) {
      sandbox.cameraRegion = { x: target.x, y: target.y, width: target.width, height: target.height };
      return sandbox.cameraRegion;
    }
  }
  sandbox.cameraRegion = next;
  return sandbox.cameraRegion;
}

export function isUnitVisible(sandbox, unit, ownUnits) {
  if (!sandbox.fogEnabled) return true;
  if (unit.team === 'own') return true;
  const radius = num(sandbox.visionRadius, 260);
  for (const own of ownUnits) {
    if (Math.hypot(unit.x - own.x, unit.y - own.y) <= radius) return true;
  }
  return false;
}

export async function selectMap(mapId, options) {
  const sandbox = state.sandbox;
  const opts = options || {};
  const nextId = mapId || '';
  if (!nextId) {
    sandbox.mapId = '';
    sandbox.mapDetail = null;
    sandbox.terrainView = null;
    sandbox.terrainEntry = null;
    sandbox.terrainLoading = false;
    sandbox.terrainError = null;
    sandbox.viewMode = 'battle';
    sandbox.cameraRegion = null;
    if (els.mapSelect) els.mapSelect.value = '';
    if (els.viewSelect) {
      els.viewSelect.value = 'battle';
      els.viewSelect.disabled = true;
    }
    syncMapStatus();
    syncMapToggles();
    drawSandbox();
    return;
  }
  if (!state.maps.available) {
    toast('地图资源不可用：请先准备游戏资源缓存', 'error');
    if (els.mapSelect) els.mapSelect.value = '';
    return;
  }
  const token = (sandbox.mapLoadToken || 0) + 1;
  sandbox.mapLoadToken = token;
  sandbox.mapId = nextId;
  sandbox.mapDetail = null;
  sandbox.terrainView = null;
  sandbox.terrainEntry = null;
  sandbox.terrainError = null;
  sandbox.terrainLoading = true;
  sandbox.cameraRegion = null;
  if (els.mapSelect) els.mapSelect.value = nextId;
  if (els.viewSelect) els.viewSelect.disabled = false;
  syncMapStatus();
  drawSandbox();
  try {
    const detail = await Api.mapDetail(nextId);
    if (sandbox.mapLoadToken !== token) return;
    const entry = await buildTerrainEntry(detail);
    if (sandbox.mapLoadToken !== token) return;
    await ensureUnitSprites();
    if (sandbox.mapLoadToken !== token) return;
    sandbox.mapDetail = detail;
    sandbox.terrainEntry = entry;
    sandbox.terrainLoading = false;
    await refreshTerrainView();
    if (!opts.silent) toast('已加载地图 · ' + (detail.name || nextId), 'ok');
  } catch (error) {
    if (sandbox.mapLoadToken !== token) return;
    sandbox.terrainLoading = false;
    sandbox.terrainError = error.message || String(error);
    sandbox.mapDetail = null;
    sandbox.mapId = '';
    sandbox.terrainEntry = null;
    sandbox.terrainView = null;
    if (els.mapSelect) els.mapSelect.value = '';
    if (els.viewSelect) els.viewSelect.disabled = true;
    syncMapStatus();
    syncMapToggles();
    drawSandbox();
    toast('地图加载失败，已回退抽象渲染：' + error.message, 'error');
  }
}

export async function applyViewMode(mode) {
  const sandbox = state.sandbox;
  sandbox.viewMode = mode === 'full' ? 'full' : 'battle';
  if (els.viewSelect) els.viewSelect.value = sandbox.viewMode;
  sandbox.cameraRegion = null;
  if (sandbox.mapDetail) {
    await refreshTerrainView();
  } else {
    syncMapStatus();
    drawSandbox();
  }
}

export function syncMapSelect() {
  if (els.mapSelect) {
    const id = state.sandbox.mapId || '';
    const exists = [...els.mapSelect.options].some((option) => option.value === id);
    els.mapSelect.value = exists ? id : '';
    els.mapSelect.disabled = !!state.sandbox.mapLocked;
    els.mapSelect.title = state.sandbox.mapLocked
      ? '该回放使用该地图（地形模拟固定地图）'
      : '选择用于渲染观战的实机地图（来自游戏资源缓存）';
  }
  if (els.viewSelect) {
    els.viewSelect.disabled = !state.sandbox.mapDetail;
    els.viewSelect.value = state.sandbox.viewMode || 'battle';
  }
  if (els.inferSim) {
    els.inferSim.value = state.sandbox.simMode === 'terrain' ? 'terrain' : 'arena';
  }
}

export function syncMapToggles() {
  const sandbox = state.sandbox;
  if (els.fogToggle) {
    els.fogToggle.disabled = !sandbox.fogAvailable;
    els.fogToggle.setAttribute('aria-pressed', sandbox.fogEnabled ? 'true' : 'false');
    els.fogToggle.textContent = sandbox.fogEnabled ? '开' : '关';
  }
  if (els.itemsToggle) {
    els.itemsToggle.disabled = !sandbox.mapDetail;
    els.itemsToggle.setAttribute('aria-pressed', sandbox.itemsEnabled ? 'true' : 'false');
    els.itemsToggle.textContent = sandbox.itemsEnabled ? '开' : '关';
  }
}
