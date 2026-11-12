/**
 * checkpoint.js — 模型推理 · checkpoint 选择：列表加载、排序、显示名、下拉渲染。
 */

import { Api } from './api.js';
import { DEMO } from './demo.js';
import { feSyncInferPolicyUi, feSyncPpoConfigUi, feSyncTrainPolicyUi } from './model.js';
import { els, state } from './state.js';
import { basename, num, numOrNull, pad2, parseBeijingDate } from './util.js';

export function feFormatCheckpointTime(mtimeStr) {
  const d = parseBeijingDate(mtimeStr);
  if (!d) return String(mtimeStr || '').slice(5, 16);
  const now = new Date();
  const isToday = d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
  const timePart = pad2(d.getHours()) + ':' + pad2(d.getMinutes());
  if (isToday) return '今天 ' + timePart;
  const yesterday = new Date(now.getTime() - 86400000);
  const isYesterday = d.getFullYear() === yesterday.getFullYear() && d.getMonth() === yesterday.getMonth() && d.getDate() === yesterday.getDate();
  if (isYesterday) return '昨天 ' + timePart;
  return pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' + timePart;
}

export function feInferMapFromPath(path) {
  const s = String(path || '').toLowerCase();
  if (s.includes('king of the middle')) return 'King of the Middle (3p)';
  if (s.includes('ironzero25')) return 'IronZero25';
  if (s.includes('ironzero20')) return 'IronZero20';
  if (s.includes('ironzero15')) return 'IronZero15';
  if (s.includes('ironzero10')) return 'IronZero10';
  if (s.includes('ironzero5')) return 'IronZero5';
  if (s.includes('artillery')) return 'Artillery';
  if (s.includes('lake')) return 'Lake (2p)';
  if (s.includes('hills')) return 'Hills (2p)';
  return '';
}

export function sortCheckpoints(list, sortBy) {
  const arr = (list || []).slice();
  if (sortBy === 'win_rate') {
    return arr.sort((a, b) => {
      const wrA = (a.eval && typeof a.eval.win_rate === 'number') ? a.eval.win_rate : -1;
      const wrB = (b.eval && typeof b.eval.win_rate === 'number') ? b.eval.win_rate : -1;
      if (wrB !== wrA) return wrB - wrA;
      return String(b.mtime || '').localeCompare(String(a.mtime || ''));
    });
  }
  if (sortBy === 'update') {
    return arr.sort((a, b) => {
      const upA = typeof a.update === 'number' ? a.update : -1;
      const upB = typeof b.update === 'number' ? b.update : -1;
      if (upB !== upA) return upB - upA;
      return String(b.mtime || '').localeCompare(String(a.mtime || ''));
    });
  }
  if (sortBy === 'name') {
    return arr.sort((a, b) => String(a.name || a.path || '').localeCompare(String(b.name || b.path || '')));
  }
  // 默认按时间倒序
  return arr.sort((a, b) => String(b.mtime || '').localeCompare(String(a.mtime || '')));
}

export function feCheckpointDisplayName(item, index) {
  if (!item) return '';
  const rawPath = String(item.path || item.name || '').replace(/\\/g, '/');
  const fileName = basename(rawPath);
  const parentName = basename(rawPath.replace(/[/\\][^/\\]+$/, ''));

  const parts = [];
  if (index === 0) {
    parts.push('⭐ [最新/首选]');
  }
  if (parentName.startsWith('econ_')) {
    parts.push('🏗️ [经济/建造]');
  }
  const mapName = item.map || feInferMapFromPath(item.path || item.name);
  if (mapName) {
    parts.push('[' + mapName.replace(/\[p\d+\]/, '').trim() + ']');
  }
  const timeStr = feFormatCheckpointTime(item.mtime);
  if (timeStr) parts.push(timeStr);

  if (numOrNull(item.update) !== null) {
    parts.push('upd ' + num(item.update, 0));
  }
  if (item.eval && numOrNull(item.eval.win_rate) !== null) {
    parts.push('胜率 ' + Math.round(item.eval.win_rate * 100) + '%');
  }

  if (fileName.includes('best')) {
    parts.push('最优(best)');
  }
  const shortId = parentName ? parentName.replace('rwpp-ppo-', '') : '';
  if (shortId && shortId !== parentName) {
    parts.push('(' + shortId + ')');
  } else if (parentName) {
    parts.push('(' + parentName + ')');
  }
  return parts.length ? parts.join(' · ') : (item.name || rawPath);
}

export async function loadFeCheckpoints(force) {
  const fe = state.fe;
  if (fe.checkpointsLoading) return;
  if (fe.checkpointsLoaded && !force) {
      if (fe.checkpointList && fe.checkpointList.length > 0) return;
  }
  fe.checkpointsLoading = true;
  if (state.mode === 'demo') {
    const list = (typeof DEMO !== 'undefined' && Array.isArray(DEMO.checkpoints)) ? DEMO.checkpoints : [];
    fe.checkpointList = list.slice();
    fe.checkpointsLoaded = true;
    fe.checkpointsLoading = false;
    renderFeCheckpointOptions();
    return;
  }
  try {
    const payload = await Api.checkpoints();
    const list = Array.isArray(payload) ? payload : ((payload && payload.checkpoints) || []);
    // 即使列表为空，也不要轻易置空，防止逻辑覆盖
    if (list.length > 0) {
        fe.checkpointList = list.slice().sort((left, right) => String(right.mtime || '').localeCompare(String(left.mtime || '')));
        fe.checkpointsLoaded = true;
    } else {
        fe.checkpointsLoaded = false;
    }
  } catch (error) {
    fe.checkpointsLoaded = false;
  } finally {
    fe.checkpointsLoading = false;
    renderFeCheckpointOptions();
  }
}

export function renderFeCheckpointOptions() {
  const rawList = state.fe.checkpointList || [];
  const inferSortBy = (els.feCheckpointSort && els.feCheckpointSort.value) || 'time';
  const ppoSortBy = (els.fePpoCheckpointSort && els.fePpoCheckpointSort.value) || 'time';

  const inferList = sortCheckpoints(rawList, inferSortBy);
  const ppoList = sortCheckpoints(rawList, ppoSortBy);

  const select = els.feCheckpointSelect;
  if (select) {
    const previous = select.value;
    select.innerHTML = '';
    if (!inferList.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '（暂无检测到 Checkpoint，请点击上方刷新）';
      select.appendChild(option);
      select.disabled = true;
    } else {
      for (let i = 0; i < inferList.length; i += 1) {
        const item = inferList[i];
        const option = document.createElement('option');
        option.value = item.path;
        option.textContent = feCheckpointDisplayName(item, i)
          + (item.size_mb ? ' · ' + item.size_mb + 'MB' : '');
        select.appendChild(option);
      }
      select.disabled = false;
      select.value = inferList.some((item) => item.path === previous) ? previous : inferList[0].path;
    }
  }
  if (els.fePolicy) {
    const modelOption = Array.prototype.find.call(els.fePolicy.options, (option) => option.value === 'model');
    if (modelOption) {
      modelOption.disabled = false;
    }
  }
  // 同步填充数据采集面板的模型 Checkpoint 下拉框
  const trainSelect = els.feTrainCheckpointSelect;
  if (trainSelect) {
    const previous = trainSelect.value;
    trainSelect.innerHTML = '';
    if (!inferList.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '（无可用模型）';
      trainSelect.appendChild(option);
      trainSelect.disabled = true;
    } else {
      for (let i = 0; i < inferList.length; i += 1) {
        const item = inferList[i];
        const option = document.createElement('option');
        option.value = item.path;
        option.textContent = feCheckpointDisplayName(item, i)
          + (item.size_mb ? ' · ' + item.size_mb + 'MB' : '');
        trainSelect.appendChild(option);
      }
      trainSelect.disabled = false;
      trainSelect.value = inferList.some((item) => item.path === previous) ? previous : inferList[0].path;
    }
  }
  // 同步填充 PPO 训练面板的初始 Checkpoint 下拉框
  const ppoSelect = els.fePpoInitCheckpoint;
  if (ppoSelect) {
    const previous = ppoSelect.value;
    ppoSelect.innerHTML = '';
    if (!ppoList.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '（无可用权重）';
      ppoSelect.appendChild(option);
      ppoSelect.disabled = true;
    } else {
      for (let i = 0; i < ppoList.length; i += 1) {
        const item = ppoList[i];
        const option = document.createElement('option');
        option.value = item.path;
        option.textContent = feCheckpointDisplayName(item, i)
          + (item.size_mb ? ' · ' + item.size_mb + 'MB' : '');
        ppoSelect.appendChild(option);
      }
      ppoSelect.disabled = false;
      const bestDefault = ppoList.find((item) => /bc|best|ft/i.test(item.name || item.path)) || ppoList[0];
      ppoSelect.value = ppoList.some((item) => item.path === previous) ? previous : (bestDefault ? bestDefault.path : ppoList[0].path);
    }
  }
  feSyncInferPolicyUi();
  feSyncTrainPolicyUi();
  feSyncPpoConfigUi();
}
