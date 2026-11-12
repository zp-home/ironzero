/**
 * replays.js — Checkpoint 列表 / 回放库：分页、折叠、渲染与加载。
 */

import { Api } from './api.js';
import { DEMO } from './demo.js';
import { normalizeReplay, resultMeta } from './normalize.js';
import { isRealReplay, openReplay, removeReplay } from './rwpp.js';
import { els, state } from './state.js';
import { basename, fmtNum, fmtPct, fmtShortTime, numOrNull, pickVal } from './util.js';

export function renderCheckpoints(list) {
  const select = els.ckptSelect;
  if (!select) return;
  const items = Array.isArray(list) ? list : [];
  const previous = select.value;
  select.innerHTML = '';
  const randomOption = document.createElement('option');
  randomOption.value = 'random';
  randomOption.textContent = items.length
    ? 'random · 随机策略（无 checkpoint）'
    : '暂无 checkpoint · 使用随机策略';
  select.appendChild(randomOption);
  for (const item of items) {
    if (!item) continue;
    const path = pickVal(item, 'path', 'file') || item.name || '';
    if (!path) continue;
    const option = document.createElement('option');
    option.value = path;
    const update = numOrNull(item.update);
    const winRate = item.eval && numOrNull(item.eval.win_rate);
    option.textContent = (item.name || basename(path))
      + (update === null ? '' : ' · u' + update)
      + (winRate === null ? '' : ' · win ' + fmtPct(winRate));
    select.appendChild(option);
  }
  if (previous) {
    const matched = Array.prototype.some.call(select.options, (option) => option.value === previous);
    if (matched) select.value = previous;
  }
}

export function updateReplayCollapseUi() {
  const p = state.replayPagination;
  if (els.replayCard) {
    els.replayCard.classList.toggle('collapsed', !!p.collapsed);
  }
  if (els.replayToggleIcon) {
    els.replayToggleIcon.textContent = p.collapsed ? '▶' : '▼';
  }
  if (els.replayToggleText) {
    els.replayToggleText.textContent = p.collapsed ? '展开' : '收起';
  }
}

export function toggleReplayCollapse() {
  state.replayPagination.collapsed = !state.replayPagination.collapsed;
  updateReplayCollapseUi();
}

export function renderReplays() {
  updateReplayCollapseUi();
  const body = els.replayBody;
  if (!body) return;
  body.innerHTML = '';
  const allItems = state.replays || [];
  const total = allItems.length;
  if (els.replayCount) els.replayCount.textContent = String(total);

  const p = state.replayPagination;
  const pageSize = Math.max(1, p.pageSize || 8);
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if (p.page > totalPages) p.page = totalPages;
  if (p.page < 1) p.page = 1;

  if (els.replayPageInfo) {
    els.replayPageInfo.textContent = '第 ' + p.page + ' / ' + totalPages + ' 页 · 共 ' + total + ' 条';
  }
  if (els.replayCurrentPage) {
    els.replayCurrentPage.textContent = String(p.page);
  }
  if (els.replayPrevBtn) {
    els.replayPrevBtn.disabled = (p.page <= 1);
  }
  if (els.replayNextBtn) {
    els.replayNextBtn.disabled = (p.page >= totalPages);
  }
  if (els.replayPageSize && String(els.replayPageSize.value) !== String(pageSize)) {
    els.replayPageSize.value = String(pageSize);
  }

  if (!total) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 10;
    cell.className = 'empty';
    cell.textContent = state.mode === 'demo' ? '暂无演示回放' : '尚无回放；运行一次推理后会自动入库';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }

  const startIdx = (p.page - 1) * pageSize;
  const items = allItems.slice(startIdx, startIdx + pageSize);

  for (const item of items) {
    const row = document.createElement('tr');
    row.className = 'data-row';
    const realReplay = isRealReplay(item);
    if (realReplay) row.classList.add('real-row');
    row.addEventListener('click', () => openReplay(item.id));

    const cellName = document.createElement('td');
    cellName.className = 'cell-name';
    cellName.textContent = item.name || basename(item.id);
    if (realReplay) {
      const realTag = document.createElement('span');
      realTag.className = 'real-tag';
      realTag.textContent = '真机';
      realTag.title = '由宿主机真实 Rusted Warfare 1.15 无头对局录制';
      cellName.appendChild(realTag);
    }
    cellName.title = item.id || item.name || '';

    const cellCheckpoint = document.createElement('td');
    cellCheckpoint.className = 'cell-path';
    cellCheckpoint.textContent = item.checkpoint ? basename(item.checkpoint) : '—';
    cellCheckpoint.title = item.checkpoint || '';

    const cellOpponent = document.createElement('td');
    cellOpponent.textContent = item.opponent || '—';

    const cellSim = document.createElement('td');
    const simTerrain = item.simMode === 'terrain';
    cellSim.textContent = (simTerrain ? '地形' : '竞技场') + (simTerrain && item.mapName ? ' · ' + item.mapName : '');
    cellSim.title = simTerrain ? ('实机地形模拟' + (item.mapName ? ' · ' + item.mapName : '')) : '抽象竞技场';
    cellSim.className = 'cell-sim' + (simTerrain ? ' terrain' : '');

    const cellResult = document.createElement('td');
    const badge = document.createElement('span');
    const meta = resultMeta(item.result);
    badge.className = 'result-badge ' + meta.cls;
    badge.textContent = meta.label;
    cellResult.appendChild(badge);

    const cellReturn = document.createElement('td');
    cellReturn.textContent = item.ret === null || item.ret === undefined ? '—' : fmtNum(item.ret, 3);

    const cellEpisodes = document.createElement('td');
    cellEpisodes.textContent = item.episodes === null || item.episodes === undefined ? '—' : String(item.episodes);

    const cellFrames = document.createElement('td');
    cellFrames.textContent = item.frames === null || item.frames === undefined ? '—' : String(item.frames);

    const cellTime = document.createElement('td');
    cellTime.textContent = fmtShortTime(item.created_at) || '—';

    const cellActions = document.createElement('td');
    const actions = document.createElement('div');
    actions.className = 'row-actions';
    const playButton = document.createElement('button');
    playButton.type = 'button';
    playButton.className = 'btn btn-sm';
    playButton.textContent = '播放';
    playButton.addEventListener('click', (event) => {
      event.stopPropagation();
      openReplay(item.id);
    });
    const deleteButton = document.createElement('button');
    deleteButton.type = 'button';
    deleteButton.className = 'btn btn-danger btn-sm';
    deleteButton.textContent = '删除';
    deleteButton.addEventListener('click', (event) => {
      event.stopPropagation();
      removeReplay(item.id);
    });
    actions.appendChild(playButton);
    actions.appendChild(deleteButton);
    cellActions.appendChild(actions);

    row.appendChild(cellName);
    row.appendChild(cellCheckpoint);
    row.appendChild(cellOpponent);
    row.appendChild(cellSim);
    row.appendChild(cellResult);
    row.appendChild(cellReturn);
    row.appendChild(cellEpisodes);
    row.appendChild(cellFrames);
    row.appendChild(cellTime);
    row.appendChild(cellActions);
    body.appendChild(row);
  }
}

export async function loadReplays() {
  if (state.mode === 'demo') {
    state.replays = DEMO.replays.slice();
    renderReplays();
    return;
  }
  try {
    const data = await Api.replays();
    const list = Array.isArray(data) ? data : (data && Array.isArray(data.replays) ? data.replays : []);
    state.replays = list.map(normalizeReplay);
    renderReplays();
  } catch (error) {
    // 保留上一次列表
  }
}
