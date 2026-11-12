/**
 * terrain.js — 实机地图 · 目录与地形预渲染：地图清单、瓦片集、地形缓存、局部裁剪。
 */

import { Api } from './api.js';
import { renderFeMapControls } from './feMaps.js';
import { loadImage, preloadUnitSprites } from './sprites.js';
import { els, state } from './state.js';
import { clamp, num } from './util.js';
import { syncMapSelect, syncMapToggles } from './view.js';

/* ---------------- 沙盘渲染 ---------------- */
export const SANDBOX_COLORS = {
  own: '#4da3ff',
  enemy: '#ff5f6d',
  grid: 'rgba(56, 225, 255, 0.07)',
  gridStrong: 'rgba(56, 225, 255, 0.15)',
};

export const terrainCache = new Map();

export async function loadMaps(force) {
  if (state.maps.ready && state.maps.available && !force) return;
  state.maps.attempts += 1;
  if (!force && state.maps.attempts > 10) return;
  try {
    const payload = await Api.maps();
    const list = payload && Array.isArray(payload.maps) ? payload.maps : [];
    state.maps.available = !!(payload && payload.available);
    state.maps.list = list.filter((item) => item && !item.broken);
    state.maps.units = (payload && payload.units) || {};
    state.maps.unitSprites = (payload && payload.unitSprites) || null;
    state.maps.error = null;
    state.maps.ready = true;
    preloadUnitSprites(state.maps.unitSprites);
  } catch (error) {
    state.maps.available = false;
    state.maps.list = [];
    state.maps.error = error.message || String(error);
  }
  renderMapOptions();
  renderFeMapControls();
  return;
}

export function renderMapOptions() {
  const select = els.mapSelect;
  if (!select) return;
  const current = state.sandbox.mapId || '';
  while (select.options.length > 1) select.remove(1);
  if (state.maps.available) {
    const folderRank = { skirmish: 0, normal: 1, survival: 2, challenge: 3, testing: 4, menu_background: 5 };
    const sorted = state.maps.list.slice().sort((left, right) => {
      const rankLeft = folderRank[left.folder] === undefined ? 9 : folderRank[left.folder];
      const rankRight = folderRank[right.folder] === undefined ? 9 : folderRank[right.folder];
      if (rankLeft !== rankRight) return rankLeft - rankRight;
      const playersLeft = num(left.players, 0);
      const playersRight = num(right.players, 0);
      if (playersLeft !== playersRight) return playersRight - playersLeft;
      return String(left.name || '').localeCompare(String(right.name || ''));
    });
    for (const item of sorted) {
      const option = document.createElement('option');
      option.value = item.id;
      const people = item.players ? item.players + 'P · ' : '';
      option.textContent = item.name + ' · ' + people + (item.folder || 'map');
      option.title = (item.relative || '') + ' · ' + item.width_px + '×' + item.height_px;
      select.appendChild(option);
    }
  }
  const exists = [...select.options].some((option) => option.value === current);
  select.value = exists ? current : '';
  syncMapSelect();
  syncMapToggles();
  syncMapStatus();
}

export function setMapStatus(text, tone) {
  if (!els.mapStatus) return;
  els.mapStatus.textContent = text;
  els.mapStatus.className = 'map-status' + (tone ? ' ' + tone : '');
}

export function syncMapStatus() {
  const sandbox = state.sandbox;
  if (state.mode === 'demo') {
    setMapStatus('演示模式 · 抽象沙盘');
    return;
  }
  if (!state.maps.available) {
    const hasError = !!state.maps.error;
    setMapStatus(
      hasError
        ? '正在连接并读取地图列表…（点击可重试）'
        : (state.maps.ready ? '未安装游戏资源缓存：请运行 web/scripts/prepare_assets.ps1' : '正在读取地图列表…'),
      hasError ? 'warn' : (state.maps.ready ? 'bad' : '')
    );
    return;
  }
  if (sandbox.terrainLoading) {
    setMapStatus('地图加载中…');
    return;
  }
  if (sandbox.terrainError) {
    setMapStatus('地图加载失败，已回退抽象渲染 · ' + sandbox.terrainError, 'bad');
    return;
  }
  const modeLabel = (sandbox.source === 'infer' || sandbox.checkpoint === 'rwpp-live')
    ? '实机'
    : (sandbox.simMode === 'terrain' ? '地形模拟' : '竞技场');
  if (sandbox.mapDetail) {
    let viewLabel;
    if (sandbox.simMode === 'terrain') {
      viewLabel = sandbox.viewMode === 'full' ? '全场' : '跟随交战区域';
    } else {
      viewLabel = sandbox.viewMode === 'full' ? '全场' : '交战跟随';
    }
    const extras = [];
    if (sandbox.fogEnabled) extras.push('迷雾');
    if (sandbox.mapDetail && sandbox.itemsEnabled) extras.push('道具层');
    setMapStatus(
      modeLabel + ' · ' + (sandbox.mapDetail.name || '地图') + ' · ' + viewLabel
        + (extras.length ? ' · ' + extras.join(' · ') : '') + ' · 地形仅视觉',
      'ok'
    );
    return;
  }
  setMapStatus(modeLabel + ' · 抽象渲染');
}

export function mapCrop(detail) {
  if (detail.crop && detail.crop.width && detail.crop.height) {
    return {
      x: num(detail.crop.x, 0),
      y: num(detail.crop.y, 0),
      width: num(detail.crop.width, 1200),
      height: num(detail.crop.height, 1200),
    };
  }
  const tileW = num(detail.tile_width, 20);
  const tileH = num(detail.tile_height, 20);
  return {
    x: 0,
    y: 0,
    width: Math.max(1, num(detail.width_px, num(detail.width, 1000) * tileW)),
    height: Math.max(1, num(detail.height_px, num(detail.height, 1000) * tileH)),
  };
}

export function engineToMapPixel(x, y, detail) {
  const crop = mapCrop(detail);
  return {
    x: crop.x + (clamp(num(x, 0), 0, 1000) / 1000) * crop.width,
    y: crop.y + (clamp(num(y, 0), 0, 1000) / 1000) * crop.height,
  };
}

export function mapCamera(width, height, region, padding) {
  const pad = num(padding, 0);
  const rw = Math.max(1, region.width);
  const rh = Math.max(1, region.height);
  const scale = Math.max(0.0001, Math.min((width - pad * 2) / rw, (height - pad * 2) / rh));
  return {
    region: { x: region.x, y: region.y, width: rw, height: rh },
    scale: scale,
    ox: (width - rw * scale) / 2 - region.x * scale,
    oy: (height - rh * scale) / 2 - region.y * scale,
    width: width,
    height: height,
  };
}

export function findTileset(tilesets, gid) {
  for (const set of tilesets) {
    if (gid >= set.firstgid) return set;
  }
  return null;
}

export function wholeMapRegion(detail) {
  return {
    x: 0,
    y: 0,
    width: Math.max(1, num(detail.width_px, 1000)),
    height: Math.max(1, num(detail.height_px, 1000)),
  };
}

export function renderTerrain(detail, region, scale, images, options) {
  const opts = options || {};
  const layerNames = Array.isArray(opts.layers)
    ? opts.layers.map((name) => String(name).toLowerCase())
    : null;
  const tileW = num(detail.tile_width, 20);
  const tileH = num(detail.tile_height, 20);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.ceil(region.width * scale));
  canvas.height = Math.max(1, Math.ceil(region.height * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  ctx.imageSmoothingEnabled = scale < 1;
  if (opts.background !== false) {
    ctx.fillStyle = '#0a141f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  const tilesets = (detail.tilesets || []).slice().sort((left, right) => right.firstgid - left.firstgid);
  const columns = Math.max(1, num(detail.width, 1));
  const rows = Math.max(1, num(detail.height, 1));
  const tileX0 = Math.max(0, Math.floor(region.x / tileW));
  const tileY0 = Math.max(0, Math.floor(region.y / tileH));
  const tileX1 = Math.min(columns - 1, Math.ceil((region.x + region.width) / tileW));
  const tileY1 = Math.min(rows - 1, Math.ceil((region.y + region.height) / tileH));

  for (const layer of detail.layers || []) {
    if (layerNames) {
      const name = String((layer && layer.name) || '').toLowerCase();
      if (layerNames.indexOf(name) < 0) continue;
    }
    const data = layer && layer.data;
    if (!Array.isArray(data)) continue;
    for (let ty = tileY0; ty <= tileY1; ty += 1) {
      const rowOffset = ty * columns;
      for (let tx = tileX0; tx <= tileX1; tx += 1) {
        const raw = data[rowOffset + tx];
        if (!raw) continue;
        const gid = raw & 0x1fffffff;
        if (!gid) continue;
        const set = findTileset(tilesets, gid);
        if (!set) continue;
        const image = images.get(set.firstgid);
        if (!image) continue;
        const local = gid - set.firstgid;
        const col = local % Math.max(1, set.columns);
        const row = Math.floor(local / Math.max(1, set.columns));
        if (row >= Math.max(1, set.rows) || col >= Math.max(1, set.columns)) continue;
        const sw = num(set.tile_width, tileW);
        const sh = num(set.tile_height, tileH);
        const dx = (tx * tileW - region.x) * scale;
        const dy = (ty * tileH - region.y) * scale;
        ctx.drawImage(image, col * sw, row * sh, sw, sh, dx, dy, tileW * scale, tileH * scale);
      }
    }
  }
  return canvas;
}

/**
 * 地形道具层版本。改动道具层内容（layers 组合）时必须 +1，
 * 否则 terrainCache 里长命的旧 entry 会继续用旧画布渲染。
 * v2：不再把 `Units` 层烘进底图 —— 该层是编辑器里的开局单位占位，
 *     运行时单位位置由回放帧决定，烘进去会让每个开局单位被画两遍（幽灵残影）。
 */
export const TERRAIN_ITEMS_LAYERS = ['Items'];

export const TERRAIN_ITEMS_LAYERS_VERSION = 2;

export async function buildTerrainEntry(detail) {
  const cacheKey = detail.id || String(detail.relative || detail.name);
  const cached = terrainCache.get(cacheKey);
  const images = new Map();
  await Promise.all(
    (detail.tilesets || []).map(async (set) => {
      if (!set || !set.image) return;
      try {
        images.set(set.firstgid, await loadImage(Api.assetUrl(set.image)));
      } catch (error) {
        // 单个 tileset 失败不阻塞整图渲染，缺失区域保留底色
      }
    })
  );
  const crop = mapCrop(detail);
  const entry = cached || {};
  entry.detail = detail;
  entry.images = images;
  // 道具层版本不一致 → 丢弃旧画布重建（旧版 entry 没有该字段，等于版本 0）
  if (entry.itemsLayersVersion !== TERRAIN_ITEMS_LAYERS_VERSION) {
    entry.itemsLayersVersion = TERRAIN_ITEMS_LAYERS_VERSION;
    entry.cropItems = null;
    entry.fullItems = null;
    entry.mapItems = null;
  }
  if (!entry.cropGround) {
    entry.cropGround = renderTerrain(detail, crop, 1, images, { layers: ['Ground'] });
  }
  if (!entry.cropItems) {
    entry.cropItems = renderTerrain(detail, crop, 1, images, {
      layers: TERRAIN_ITEMS_LAYERS,
      background: false,
    });
  }
  return entry;
}

export function ensureFullTerrain(entry, detail) {
  const itemsStale = entry.itemsLayersVersion !== TERRAIN_ITEMS_LAYERS_VERSION;
  if (entry.fullGround && entry.fullItems && !itemsStale) {
    return { ground: entry.fullGround, items: entry.fullItems, scale: entry.fullScale };
  }
  const width = Math.max(1, num(detail.width_px, 1000));
  const height = Math.max(1, num(detail.height_px, 1000));
  const scale = Math.min(1, 2048 / Math.max(width, height));
  entry.fullGround = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, { layers: ['Ground'] });
  entry.fullItems = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, {
    layers: TERRAIN_ITEMS_LAYERS,
    background: false,
  });
  entry.itemsLayersVersion = TERRAIN_ITEMS_LAYERS_VERSION;
  entry.fullScale = scale;
  return { ground: entry.fullGround, items: entry.fullItems, scale: scale };
}

export function ensureTerrainMap(entry, detail) {
  const itemsStale = entry.itemsLayersVersion !== TERRAIN_ITEMS_LAYERS_VERSION;
  if (entry.mapGround && entry.mapItems && !itemsStale) {
    return { ground: entry.mapGround, items: entry.mapItems, scale: entry.mapScale };
  }
  const width = Math.max(1, num(detail.width_px, 1000));
  const height = Math.max(1, num(detail.height_px, 1000));
  const scale = Math.min(1, 3000 / Math.max(width, height));
  entry.mapGround = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, { layers: ['Ground'] });
  entry.mapItems = renderTerrain(detail, wholeMapRegion(detail), scale, entry.images, {
    layers: TERRAIN_ITEMS_LAYERS,
    background: false,
  });
  entry.itemsLayersVersion = TERRAIN_ITEMS_LAYERS_VERSION;
  entry.mapScale = scale;
  return { ground: entry.mapGround, items: entry.mapItems, scale: scale };
}
