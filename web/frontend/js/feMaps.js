/**
 * feMaps.js — 指挥中心大屏 · 地图下拉控制：可选地图筛选、默认地图、控件渲染。
 */

import { FE_DEFAULT_MAP_NAME } from './control.js';
import { feCatalogMaps, feMapCatalog } from './feMapCatalog.js';
import { feRenderInferDashboard, feUpdateSlots } from './feInferUi.js';
import { els, state } from './state.js';
import { selectMap } from './view.js';

/* ---------------- 地图选择 ---------------- */
export function feMapList() {
  return (state.maps.list || []).filter((item) => item && !item.broken);
}

export function feInferMapList() {
  const allowed = { skirmish: true, challenge: true, testing: true, ironzero: true };
  return feMapList().filter((item) => allowed[String(item.folder || '')]);
}

export function feMapItemById(id) {
  return feMapList().find((item) => item.id === id) || null;
}

export function feMapItemByName(name) {
  return feMapList().find((item) => item.name === name) || null;
}

export function feDefaultMap() {
  return feMapItemByName(FE_DEFAULT_MAP_NAME) || feMapList()[0] || null;
}

export function renderFeMapControls() {
  const allMaps = feMapList();
  const inferMaps = feInferMapList();
  const defaultItem = inferMaps.find((item) => item.name === FE_DEFAULT_MAP_NAME) || inferMaps[0] || feDefaultMap();
  if (els.mapSelect) {
    const previousId = els.mapSelect.value || (state.sandbox && state.sandbox.mapId) || '';
    els.mapSelect.innerHTML = '';
    for (const item of inferMaps) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.name + (item.players ? ' · ' + item.players + 'P' : '');
      els.mapSelect.appendChild(option);
    }
    const fallbackId = inferMaps.some((item) => item.id === previousId) ? previousId : (defaultItem ? defaultItem.id : '');
    els.mapSelect.value = fallbackId;
  }
  if (els.feMapCount) {
    els.feMapCount.textContent = '可用地图 ' + inferMaps.length + ' 张（已过滤战役/菜单图）';
  }
  if (els.feTrainMap) {
    // The collection form trains on the nine stock 2p maps, so it is driven by
    // the training catalog (`GET /maps`) rather than by "every .tmx the sandbox
    // can render". Its value is the map *name*, which is what `/train` accepts.
    const catalog = feCatalogMaps();
    const previousName = els.feTrainMap.value || (defaultItem ? defaultItem.name : '');
    els.feTrainMap.innerHTML = '';
    if (catalog.length) {
      for (const item of catalog) {
        const option = document.createElement('option');
        option.value = item.name;
        option.textContent = item.name + ' · ' + Math.round(item.map_width) + '×' + Math.round(item.map_height);
        option.title = item.map_path || '';
        els.feTrainMap.appendChild(option);
      }
      const catalogName = catalog.some((item) => item.name === previousName) ? previousName : catalog[0].name;
      els.feTrainMap.value = catalogName;
    } else {
      // Catalog unavailable (host service down): fall back to the asset list so
      // the form still works, just without the pixel sizes.
      for (const item of allMaps) {
        const option = document.createElement('option');
        option.value = item.name;
        option.textContent = item.name + (item.players ? ' · ' + item.players + 'P' : '');
        els.feTrainMap.appendChild(option);
      }
      const fallbackName = allMaps.some((item) => item.name === previousName) ? previousName : (defaultItem ? defaultItem.name : '');
      els.feTrainMap.value = fallbackName;
    }
  }
  if (els.fePpoMap) {
    const previousPpoMap = els.fePpoMap.value || 'Lake';
    const catalog = feCatalogMaps();
    if (catalog.length) {
      // Rebuild from the catalog so the options (and therefore the pixel sizes
      // the form submits) come from the same source the host resolves against.
      // The static options in index.html mirror this list and cover the case
      // where the catalog request has not finished yet.
      els.fePpoMap.innerHTML = '';
      const originalGroup = document.createElement('optgroup');
      originalGroup.label = '原版 2p 地图（真实尺寸来自 GET /maps）';
      for (const item of catalog) {
        const opt = document.createElement('option');
        opt.value = item.name;
        opt.textContent = item.name + ' · ' + Math.round(item.map_width) + '×' + Math.round(item.map_height)
          + (item.water_ratio ? ' · 水 ' + Math.round(item.water_ratio * 100) + '%' : '');
        opt.title = item.map_path || '';
        originalGroup.appendChild(opt);
      }
      els.fePpoMap.appendChild(originalGroup);

      const arenaGroup = document.createElement('optgroup');
      arenaGroup.label = '对称竞技场（纯坦克训练）';
      for (const a of [
        { id: 'IronZero25', label: 'IronZero25（25v25 对称竞技场）' },
        { id: 'IronZero20', label: 'IronZero20（20v20 对称竞技场）' },
        { id: 'IronZero15', label: 'IronZero15（15v15 对称竞技场）' },
        { id: 'IronZero10', label: 'IronZero10（10v10 对称竞技场）' },
        { id: 'IronZero5', label: 'IronZero5（5v5 对称竞技场）' },
      ]) {
        const opt = document.createElement('option');
        opt.value = a.id;
        opt.textContent = a.label;
        arenaGroup.appendChild(opt);
      }
      els.fePpoMap.appendChild(arenaGroup);

      const otherGroup = document.createElement('optgroup');
      otherGroup.label = '其他原版图';
      const artillery = document.createElement('option');
      artillery.value = 'Artillery';
      artillery.textContent = 'Artillery（原版战役挑战图 · 22v25 带工厂建筑）';
      otherGroup.appendChild(artillery);
      els.fePpoMap.appendChild(otherGroup);
    } else {
      // Catalog not loaded yet: keep whatever the static markup already offers
      // instead of emptying the control (an empty select would make
      // `feStartPpo` fall back to a map the user never picked).
      if (!els.fePpoMap.options.length) {
        const opt = document.createElement('option');
        opt.value = 'Lake';
        opt.textContent = 'Lake';
        els.fePpoMap.appendChild(opt);
      }
    }
    const hasPpoOption = [...els.fePpoMap.options].some((o) => o.value === previousPpoMap);
    els.fePpoMap.value = hasPpoOption ? previousPpoMap : (catalog.length ? catalog[0].name : 'Lake');
  }
  if (!state.fe.mapsReady && defaultItem) {
    state.fe.mapsReady = true;
    const sandbox = state.sandbox;
    sandbox.simMode = 'terrain';
    sandbox.fogAvailable = true;
    if (els.mapSelect) els.mapSelect.value = defaultItem.id;
    void selectMap(defaultItem.id, { silent: true });
  }
  feUpdateSlots();
  feRenderInferDashboard(true);
}
