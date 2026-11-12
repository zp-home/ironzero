/**
 * feMaps.js — 指挥中心大屏 · 地图下拉控制：可选地图筛选、默认地图、控件渲染。
 */

import { FE_DEFAULT_MAP_NAME } from './control.js';
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
    const previousName = els.feTrainMap.value || (defaultItem ? defaultItem.name : '');
    els.feTrainMap.innerHTML = '';
    for (const item of allMaps) {
      const option = document.createElement('option');
      option.value = item.name;
      option.textContent = item.name + (item.players ? ' · ' + item.players + 'P' : '');
      els.feTrainMap.appendChild(option);
    }
    const fallbackName = allMaps.some((item) => item.name === previousName) ? previousName : (defaultItem ? defaultItem.name : '');
    els.feTrainMap.value = fallbackName;
  }
  if (els.fePpoMap) {
    const previousPpoMap = els.fePpoMap.value || 'IronZero25';
    els.fePpoMap.innerHTML = '';

    // Group 1: 纯坦克对称竞技场
    const arenaGroup = document.createElement('optgroup');
    arenaGroup.label = '对称竞技场（纯坦克训练 · 推荐）';
    const arenas = [
      { id: 'IronZero25', label: 'IronZero25（25v25 对称竞技场 · 纯坦克 · 推荐）' },
      { id: 'IronZero20', label: 'IronZero20（20v20 对称竞技场 · 纯坦克）' },
      { id: 'IronZero15', label: 'IronZero15（15v15 对称竞技场 · 纯坦克）' },
      { id: 'IronZero10', label: 'IronZero10（10v10 对称竞技场 · 纯坦克 · 快速验证）' },
      { id: 'IronZero5', label: 'IronZero5（5v5 对称竞技场 · 纯坦克 · 极速验证）' },
    ];
    for (const a of arenas) {
      const opt = document.createElement('option');
      opt.value = a.id;
      opt.textContent = a.label;
      arenaGroup.appendChild(opt);
    }
    els.fePpoMap.appendChild(arenaGroup);

    // Group 2: 原版实机地图（遭遇战 / 战役）
    if (inferMaps.length > 0) {
      const originalGroup = document.createElement('optgroup');
      originalGroup.label = '原版实机地图（遭遇战 / 战役挑战）';
      for (const item of inferMaps) {
        if (item.name && item.name.toLowerCase().startsWith('ironzero')) continue;
        const opt = document.createElement('option');
        opt.value = item.name;
        opt.textContent = item.name + (item.players ? ' · ' + item.players + 'P' : '') + (item.name === 'Artillery' ? '（22v25 带工厂建筑）' : '');
        originalGroup.appendChild(opt);
      }
      els.fePpoMap.appendChild(originalGroup);
    }

    const hasPpoOption = [...els.fePpoMap.options].some((o) => o.value === previousPpoMap);
    els.fePpoMap.value = hasPpoOption ? previousPpoMap : 'IronZero25';
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
  feRenderInferDashboard();
}
