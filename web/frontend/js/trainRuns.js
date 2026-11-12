/**
 * trainRuns.js — 训练监控 · run 列表：列表渲染、选中、刷新、启动/停止。
 */

import { Api } from './api.js';
import { drawCharts } from './charts.js';
import { toast } from './dashboard.js';
import { DEMO } from './demo.js';
import { statusMeta } from './normalize.js';
import { els, state } from './state.js';
import { applyDetailExtras, applyEvents, resetRunData, syncLogTail, updateKpis } from './train.js';
import { updateRunHeader } from './train.js';
import { clampInt, fmtShortTime, numOrNull } from './util.js';

export function renderRunList() {
  const box = els.runList;
  if (!box) return;
  box.innerHTML = '';
  const runs = state.runs || [];
  if (els.runsCount) els.runsCount.textContent = String(runs.length);
  if (!runs.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = state.mode === 'demo' ? '暂无演示运行' : '暂无训练运行，填写左侧表单启动一次训练';
    box.appendChild(empty);
    return;
  }
  for (const run of runs) {
    const item = document.createElement('div');
    item.className = 'run-item' + (run.id === state.currentRunId ? ' active' : '');
    item.tabIndex = 0;
    item.setAttribute('role', 'button');
    const top = document.createElement('div');
    top.className = 'r-top';
    const name = document.createElement('span');
    name.className = 'run-name';
    name.textContent = run.name || run.id || '未命名';
    name.title = run.name || run.id || '';
    const pill = document.createElement('span');
    const meta = statusMeta(run.status);
    pill.className = 'status-pill ' + meta.cls;
    pill.textContent = meta.label;
    top.appendChild(name);
    top.appendChild(pill);

    const sub = document.createElement('div');
    sub.className = 'run-sub';
    const update = run.latest && numOrNull(run.latest.update) !== null ? run.latest.update : null;
    sub.textContent = 'update ' + (update === null ? '—' : update)
      + ' · 评估 ' + (numOrNull(run.eval_count) === null ? (Array.isArray(run.evals) ? run.evals.length : '—') : run.eval_count)
      + ' · ' + (fmtShortTime(run.created_at) || '—');

    item.appendChild(top);
    item.appendChild(sub);
    item.addEventListener('click', () => { selectRun(run.id); });
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectRun(run.id);
      }
    });
    box.appendChild(item);
  }
}

export async function refreshRuns() {
  if (state.mode === 'demo') return;
  try {
    const data = await Api.runs();
    const list = Array.isArray(data) ? data : (data && Array.isArray(data.runs) ? data.runs : []);
    state.runs = list.filter((item) => item && typeof item === 'object');
    state.runErrors = 0;
    renderRunList();
    if (!state.currentRunId && state.runs.length) await selectRun(state.runs[0].id);
    else if (state.currentRunId && !state.runs.some((run) => run.id === state.currentRunId) && state.runs.length) {
      await selectRun(state.runs[0].id);
    }
  } catch (error) {
    state.runErrors += 1;
    if (state.runErrors === 2) toast('读取训练运行列表失败：' + error.message, 'error');
  }
}

export async function selectRun(id) {
  if (!id) return;
  if (state.mode === 'demo') {
    selectDemoRun(id);
    return;
  }
  state.currentRunId = id;
  resetRunData(false);
  renderRunList();
  updateRunHeader();
  try {
    const detail = await Api.run(id);
    if (state.currentRunId !== id) return;
    state.runDetail = detail || {};
    syncLogTail(detail && detail.log_tail ? detail.log_tail : []);
    applyDetailExtras(detail);
    updateRunHeader();
    updateKpis();
    const eventsResponse = await Api.runEvents(id, 0);
    if (state.currentRunId !== id) return;
    applyEvents(eventsResponse && eventsResponse.events, eventsResponse && eventsResponse.next);
    drawCharts();
  } catch (error) {
    toast('加载运行详情失败：' + error.message, 'error');
  }
}

export function selectDemoRun(id) {
  const run = DEMO.runs.find((item) => item.id === id) || DEMO.runs[0];
  if (!run) return;
  state.currentRunId = run.id;
  state.runDetail = DEMO.details[run.id] || null;
  resetRunData(false);
  renderRunList();
  updateRunHeader();
  const detail = state.runDetail;
  syncLogTail(detail && detail.log_tail ? detail.log_tail : []);
  applyDetailExtras(detail);
  const events = DEMO.events[run.id] || [];
  const lastSeq = events.length ? events[events.length - 1].seq : 0;
  applyEvents(events, lastSeq);
  drawCharts();
}

export async function refreshCurrentRun() {
  const id = state.currentRunId;
  if (!id || state.mode !== 'live') return;
  try {
    const detail = await Api.run(id);
    if (state.currentRunId !== id) return;
    state.runDetail = detail || state.runDetail;
    syncLogTail(detail && detail.log_tail ? detail.log_tail : []);
    applyDetailExtras(detail);
    updateRunHeader();
    const eventsResponse = await Api.runEvents(id, state.eventCursor);
    if (state.currentRunId !== id) return;
    applyEvents(eventsResponse && eventsResponse.events, eventsResponse && eventsResponse.next);
    state.runErrors = 0;
  } catch (error) {
    state.runErrors += 1;
    if (state.runErrors === 3) toast('读取运行数据失败：' + error.message, 'error');
  }
}

export async function stopCurrentRun() {
  const id = state.currentRunId;
  if (!id) return;
  if (state.mode === 'demo') {
    toast('演示数据：后端未连接，无法停止真实训练', 'error');
    return;
  }
  if (els.stopBtn) els.stopBtn.disabled = true;
  try {
    await Api.stopRun(id);
    toast('已发送停止指令', 'ok');
    await refreshRuns();
    await refreshCurrentRun();
  } catch (error) {
    toast('停止训练失败：' + error.message, 'error');
  } finally {
    updateRunHeader();
  }
}

export async function submitStartRun(event) {
  event.preventDefault();
  if (state.mode === 'demo') {
    toast('后端未连接：无法启动真实训练（当前为演示数据）', 'error');
    return;
  }
  const form = els.startForm;
  if (!form) return;
  const data = new FormData(form);
  const getValue = (name) => (data.get(name) === null ? '' : String(data.get(name)));
  const payload = {
    name: getValue('name').trim(),
    updates: clampInt(getValue('updates'), 1, 5000, 200),
    envs: clampInt(getValue('envs'), 1, 64, 8),
    rollout_steps: clampInt(getValue('rollout_steps'), 8, 512, 64),
    opponent: getValue('opponent') || 'scripted',
    seed: clampInt(getValue('seed'), 0, 999999, 0),
    eval_every: clampInt(getValue('eval_every'), 1, 1000, 25),
    eval_episodes: clampInt(getValue('eval_episodes'), 1, 128, 16),
  };
  if (!payload.name) delete payload.name;
  if (els.startBtn) {
    els.startBtn.disabled = true;
    els.startBtn.textContent = '启动中…';
  }
  try {
    const response = await Api.startRun(payload);
    const id = response && (response.id || response.run_id);
    toast('训练任务已启动' + (id ? ' · ' + id : ''), 'ok');
    await refreshRuns();
    if (id) await selectRun(id);
  } catch (error) {
    toast('启动训练失败：' + error.message, 'error');
  } finally {
    if (els.startBtn) {
      els.startBtn.disabled = false;
      els.startBtn.textContent = '启动训练';
    }
  }
}
