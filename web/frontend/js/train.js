/**
 * train.js — 训练监控 · 事件流：指标点序列、日志面板、事件应用、KPI 与当前 run 详情。
 */

import { drawCharts } from './charts.js';
import { statusMeta } from './normalize.js';
import { els, state } from './state.js';
import { clockString, fmtNum, fmtPct, fmtShortTime, num, numOrNull, pickVal } from './util.js';
import { safeStringify } from './util.js';

export function upsertPoint(series, x, y) {
  const valueX = numOrNull(x);
  const valueY = numOrNull(y);
  if (valueX === null || valueY === null) return;
  const last = series[series.length - 1];
  if (last && last.x === valueX) {
    last.y = valueY;
    return;
  }
  if (last && valueX < last.x) {
    let low = 0;
    let high = series.length;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (series[mid].x < valueX) low = mid + 1;
      else high = mid;
    }
    if (series[low] && series[low].x === valueX) series[low].y = valueY;
    else series.splice(low, 0, { x: valueX, y: valueY });
    return;
  }
  series.push({ x: valueX, y: valueY });
  if (series.length > 4000) series.splice(0, series.length - 4000);
}

export function resetRunData(clearSelection) {
  state.series = { win: [], retTrain: [], retEval: [], entropy: [], trainWin: [] };
  state.latest = {};
  state.lastEval = null;
  state.evalCount = 0;
  state.eventCursor = 0;
  state.totalUpdates = null;
  state.logs = [];
  state.logTailLoaded = false;
  state.dirtyCharts = true;
  if (clearSelection) {
    state.currentRunId = null;
    state.runDetail = null;
  }
  if (els.logPanel) els.logPanel.innerHTML = '<div class="log-empty">等待日志…</div>';
  if (els.logCount) els.logCount.textContent = '0';
  updateKpis();
  updateRunHeader();
  drawCharts();
}

export function makeLogNode(entry) {
  const line = document.createElement('div');
  line.className = 'log-line kind-' + (entry.kind || 'log');
  if (entry.time) {
    const ts = document.createElement('span');
    ts.className = 'ts';
    ts.textContent = '[' + entry.time + ']';
    line.appendChild(ts);
  }
  line.appendChild(document.createTextNode(entry.text || ''));
  return line;
}

export function appendLog(entry) {
  if (!entry) return;
  const item = {
    text: String(entry.text === undefined ? '' : entry.text),
    kind: entry.kind || 'log',
    time: entry.time || clockString(),
  };
  state.logs.push(item);
  if (state.logs.length > 800) state.logs.splice(0, state.logs.length - 800);
  const panel = els.logPanel;
  if (!panel) return;
  const empty = panel.querySelector('.log-empty');
  if (empty) empty.remove();
  panel.appendChild(makeLogNode(item));
  while (panel.childElementCount > 800) panel.removeChild(panel.firstElementChild);
  if (state.autoScroll) panel.scrollTop = panel.scrollHeight;
  if (els.logCount) els.logCount.textContent = String(state.logs.length);
}

export function syncLogTail(lines) {
  if (!Array.isArray(lines) || !lines.length) return;
  if (state.logTailLoaded) return;
  state.logTailLoaded = true;
  const tail = lines.slice(-80);
  for (const line of tail) {
    const text = String(line);
    if (!text.trim()) continue;
    appendLog({ text: text, kind: 'stdout' });
  }
}

export function handleStartEvent(payload) {
  const config = (payload && payload.config) || {};
  const totalUpdates = numOrNull(config.total_updates) !== null ? config.total_updates : numOrNull(config.updates);
  if (totalUpdates !== null) state.totalUpdates = totalUpdates;
  const parts = [];
  if (totalUpdates !== null) parts.push('updates=' + totalUpdates);
  if (numOrNull(config.num_envs) !== null) parts.push('envs=' + config.num_envs);
  if (numOrNull(config.rollout_steps) !== null) parts.push('rollout=' + config.rollout_steps);
  if (config.opponent) parts.push('opponent=' + config.opponent);
  if (numOrNull(config.seed) !== null) parts.push('seed=' + config.seed);
  const baseline = payload && payload.random_baseline;
  if (baseline && numOrNull(baseline.win_rate) !== null) parts.push('随机基线胜率=' + fmtPct(baseline.win_rate));
  appendLog({ text: '训练启动 · ' + (parts.join(' · ') || '配置已载入'), kind: 'status' });
  state.dirtyCharts = true;
}

export function handleUpdateEvent(payload) {
  if (!payload) return;
  const update = numOrNull(payload.update);
  state.latest = Object.assign({}, state.latest, payload);
  if (update !== null) {
    if (numOrNull(payload.mean_return) !== null) upsertPoint(state.series.retTrain, update, payload.mean_return);
    if (numOrNull(payload.win_rate) !== null) upsertPoint(state.series.trainWin, update, payload.win_rate);
    if (numOrNull(payload.entropy) !== null) upsertPoint(state.series.entropy, update, payload.entropy);
  }
  state.dirtyCharts = true;
}

export function handleEvalEvent(payload) {
  if (!payload) return;
  const update = numOrNull(payload.update);
  const isNewEval = update === null || !state.series.win.some((point) => point.x === update);
  state.lastEval = Object.assign({}, payload);
  if (isNewEval) state.evalCount += 1;
  if (update !== null) {
    if (numOrNull(payload.win_rate) !== null) upsertPoint(state.series.win, update, payload.win_rate);
    if (numOrNull(payload.mean_return) !== null) upsertPoint(state.series.retEval, update, payload.mean_return);
  }
  appendLog({
    text: '评估 update=' + (update === null ? '—' : update)
      + ' · 胜率 ' + fmtPct(payload.win_rate)
      + ' · 回报 ' + fmtNum(payload.mean_return, 3)
      + ' · 对局 ' + (numOrNull(payload.episodes) === null ? '—' : payload.episodes),
    kind: 'eval',
  });
  state.dirtyCharts = true;
}

export function handleDoneEvent(payload) {
  const finalEval = payload && payload.final_eval;
  appendLog({
    text: '训练完成' + (finalEval ? ' · final win_rate=' + fmtPct(finalEval.win_rate) + ' · return=' + fmtNum(finalEval.mean_return, 3) : '')
      + (payload && numOrNull(payload.total_seconds) !== null ? ' · 用时 ' + fmtNum(payload.total_seconds, 1) + 's' : ''),
    kind: 'done',
  });
  if (finalEval) {
    // 不重复打日志：final_eval 与最后一次 eval 事件内容一致
    state.lastEval = Object.assign({}, finalEval);
    const update = numOrNull(finalEval.update);
    if (update !== null) {
      if (numOrNull(finalEval.win_rate) !== null) upsertPoint(state.series.win, update, finalEval.win_rate);
      if (numOrNull(finalEval.mean_return) !== null) upsertPoint(state.series.retEval, update, finalEval.mean_return);
    }
  }
  state.dirtyCharts = true;
}

export function applyEvent(event) {
  if (!event || typeof event !== 'object') return;
  const kind = String(pickVal(event, 'kind', 'type', 'event') || '').toLowerCase();
  const payload = event.payload && typeof event.payload === 'object' ? event.payload : event;
  const seq = numOrNull(pickVal(event, 'seq', 'sequence'));
  if (seq !== null) state.eventCursor = Math.max(state.eventCursor, seq);

  if (kind === 'start') {
    handleStartEvent(payload);
    return;
  }
  if (kind === 'update' || kind === 'metrics' || kind === 'train') {
    handleUpdateEvent(payload);
    return;
  }
  if (kind === 'eval' || kind === 'evaluation' || kind === 'test') {
    handleEvalEvent(payload);
    return;
  }
  if (kind === 'done' || kind === 'finish' || kind === 'finished') {
    handleDoneEvent(payload);
    return;
  }
  if (kind === 'error') {
    const message = pickVal(payload, 'message', 'error', 'detail');
    appendLog({ text: '错误：' + (message ? String(message) : safeStringify(payload)), kind: 'error' });
    return;
  }
  if (kind === 'log' || kind === 'info' || kind === 'stdout') {
    const text = pickVal(payload, 'line', 'message', 'text');
    appendLog({ text: text ? String(text) : safeStringify(payload), kind: 'log' });
    if (numOrNull(payload.update) !== null) handleUpdateEvent(payload);
    return;
  }
  // 未知 kind：带指标按指标处理，否则作为普通日志
  const hasMetrics = numOrNull(payload.update) !== null
    && (numOrNull(payload.win_rate) !== null || numOrNull(payload.mean_return) !== null || numOrNull(payload.entropy) !== null);
  if (hasMetrics) {
    handleUpdateEvent(payload);
    return;
  }
  const text = pickVal(payload, 'line', 'message', 'text');
  if (text) appendLog({ text: String(text), kind: 'log' });
  else if (kind) appendLog({ text: '[' + kind + '] ' + safeStringify(payload), kind: 'log' });
}

export function applyEvents(events, next) {
  if (Array.isArray(events)) {
    for (const event of events) applyEvent(event);
  }
  if (next !== undefined && next !== null) state.eventCursor = Math.max(state.eventCursor, num(next, state.eventCursor));
  updateKpis();
  updateRunHeader();
  if (state.dirtyCharts) drawCharts();
}

export function applyDetailExtras(detail) {
  if (!detail || typeof detail !== 'object') return;
  const config = detail.config && typeof detail.config === 'object' ? detail.config : {};
  const totalUpdates = numOrNull(config.total_updates) !== null ? config.total_updates : numOrNull(config.updates);
  if (totalUpdates !== null) state.totalUpdates = totalUpdates;
  const latest = detail.latest;
  if (latest && typeof latest === 'object' && Object.keys(latest).length) {
    state.latest = Object.assign({}, state.latest, latest);
    const update = numOrNull(latest.update);
    if (update !== null) {
      if (numOrNull(latest.mean_return) !== null) upsertPoint(state.series.retTrain, update, latest.mean_return);
      if (numOrNull(latest.win_rate) !== null) upsertPoint(state.series.trainWin, update, latest.win_rate);
    }
  }
  const evals = Array.isArray(detail.evals) ? detail.evals : [];
  if (evals.length) {
    state.evalCount = Math.max(state.evalCount, evals.length);
    for (const item of evals) {
      if (!item || typeof item !== 'object') continue;
      const update = numOrNull(item.update);
      if (update === null) continue;
      if (numOrNull(item.win_rate) !== null) upsertPoint(state.series.win, update, item.win_rate);
      if (numOrNull(item.mean_return) !== null) upsertPoint(state.series.retEval, update, item.mean_return);
    }
    state.lastEval = Object.assign({}, evals[evals.length - 1]);
  }
  state.dirtyCharts = true;
}

export function describeConfig(config) {
  const parts = [];
  if (numOrNull(config.updates) !== null) parts.push(config.updates + ' upd');
  if (numOrNull(config.envs) !== null) parts.push(config.envs + ' envs');
  if (numOrNull(config.rollout_steps) !== null) parts.push('rollout ' + config.rollout_steps);
  if (config.opponent) parts.push('vs ' + config.opponent);
  if (numOrNull(config.seed) !== null) parts.push('seed ' + config.seed);
  return parts.length ? parts.join(' · ') : '—';
}

export function describeEvalPolicy(config) {
  const parts = [];
  if (numOrNull(config.eval_every) !== null) parts.push('每 ' + config.eval_every + ' upd');
  if (numOrNull(config.eval_episodes) !== null) parts.push(config.eval_episodes + ' 局');
  if (numOrNull(config.lr) !== null) parts.push('lr ' + config.lr);
  return parts.length ? parts.join(' · ') : '—';
}

export function updateRunHeader() {
  const detail = state.runDetail || {};
  const listRun = state.runs.find((run) => run.id === state.currentRunId);
  const status = detail.status || (listRun && listRun.status);
  const meta = statusMeta(status);
  if (els.runStatus) {
    els.runStatus.className = 'status-pill ' + meta.cls;
    els.runStatus.textContent = state.currentRunId ? meta.label : '未选择';
  }
  if (els.stopBtn) {
    const active = status && ['running', 'queued', 'pending'].indexOf(String(status).toLowerCase()) >= 0;
    els.stopBtn.disabled = !active;
  }
  if (!els.metaRun) return;
  if (!state.currentRunId) {
    els.metaRun.textContent = '—';
    els.metaKind.textContent = '—';
    els.metaCreated.textContent = '—';
    els.metaConfig.textContent = '—';
    els.metaEval.textContent = '—';
    return;
  }
  const config = detail.config && typeof detail.config === 'object' ? detail.config : {};
  els.metaRun.textContent = detail.name || detail.id || state.currentRunId;
  els.metaKind.textContent = detail.kind || 'train';
  els.metaCreated.textContent = fmtShortTime(detail.created_at) || '—';
  els.metaConfig.textContent = describeConfig(config);
  els.metaEval.textContent = describeEvalPolicy(config);
}

export function updateKpis() {
  if (!els.kpiUpdate) return;
  const latest = state.latest || {};
  const lastEval = state.lastEval;
  const config = (state.runDetail && state.runDetail.config) || {};
  const totalUpdates = numOrNull(state.totalUpdates) !== null
    ? state.totalUpdates
    : (numOrNull(config.total_updates) !== null ? config.total_updates : numOrNull(config.updates));
  const updateValue = numOrNull(latest.update) !== null
    ? numOrNull(latest.update)
    : (state.series.win.length ? state.series.win[state.series.win.length - 1].x : null);

  els.kpiUpdate.textContent = updateValue === null ? '—' : String(Math.round(updateValue));
  if (els.kpiUpdateFoot) {
    els.kpiUpdateFoot.textContent = totalUpdates !== null
      ? '共 ' + Math.round(totalUpdates) + ' updates · 评估 ' + state.evalCount + ' 次'
      : (state.evalCount ? '评估 ' + state.evalCount + ' 次' : '等待首个 update');
  }

  const winValue = lastEval && numOrNull(lastEval.win_rate) !== null
    ? lastEval.win_rate
    : (numOrNull(latest.rolling_win_rate) !== null ? latest.rolling_win_rate : numOrNull(latest.win_rate));
  els.kpiWin.textContent = winValue === null || winValue === undefined ? '—' : fmtPct(winValue);
  if (els.kpiWinFoot) {
    els.kpiWinFoot.textContent = lastEval
      ? '评估 ' + state.evalCount + ' 次 · 每 ' + (numOrNull(config.eval_every) === null ? '—' : config.eval_every) + ' upd'
      : (numOrNull(latest.rolling_win_rate) !== null ? '训练最近完成局胜率' : '尚无评估数据');
  }

  const returnValue = lastEval && numOrNull(lastEval.mean_return) !== null
    ? lastEval.mean_return
    : (numOrNull(latest.rolling_return) !== null ? latest.rolling_return : numOrNull(latest.mean_return));
  els.kpiReturn.textContent = returnValue === null || returnValue === undefined ? '—' : fmtNum(returnValue, 3);
  if (els.kpiReturnFoot) {
    els.kpiReturnFoot.textContent = numOrNull(latest.policy_loss) === null
      ? 'policy_loss —'
      : 'policy_loss ' + fmtNum(latest.policy_loss, 4);
  }

  els.kpiEntropy.textContent = numOrNull(latest.entropy) === null ? '—' : fmtNum(latest.entropy, 3);
  if (els.kpiEntropyFoot) {
    els.kpiEntropyFoot.textContent = numOrNull(latest.value_loss) === null
      ? 'value_loss —'
      : 'value_loss ' + fmtNum(latest.value_loss, 4);
  }
}
