/**
 * connection.js — 连接状态与降级：时钟、连接徽标渲染、健康检查、重连、进入/退出演示模式。
 */

import { Api } from './api.js';
import { tickMain, toast } from './dashboard.js';
import { DEMO, initDemoData } from './demo.js';
import { ensureSandboxContent, resetSandbox } from './playback.js';
import { renderCheckpoints, renderReplays } from './replays.js';
import { els, state } from './state.js';
import { syncMapStatus } from './terrain.js';
import { resetRunData } from './train.js';
import { renderRunList, selectDemoRun } from './trainRuns.js';
import { clockString } from './util.js';

export let clockTimer = null;

export function startClock() {
  if (clockTimer) clearInterval(clockTimer);
  const tick = () => {
    if (els.clock) els.clock.textContent = clockString();
  };
  tick();
  clockTimer = setInterval(tick, 1000);
}

export function renderConn() {
  const badge = els.connBadge;
  if (!badge) return;
  let modeName = 'down';
  let label = '后端未连接';
  let sub = '等待 /api/health';
  if (state.mode === 'boot') {
    modeName = 'connecting';
    label = '连接中…';
    sub = '检测 /api/health';
  } else if (state.connected) {
    modeName = 'ok';
    label = '已连接';
    sub = 'v' + ((state.health && state.health.version) || '—')
      + ' · jobs ' + ((state.health && state.health.jobs_running) !== undefined ? state.health.jobs_running : '—');
  } else if (state.mode === 'demo') {
    modeName = 'demo';
    label = '后端未连接';
    sub = '演示数据 · 仅界面预览';
  } else {
    modeName = 'down';
    label = '连接中断';
    sub = '显示最后一次数据';
  }
  badge.dataset.state = modeName;
  if (els.connLabel) els.connLabel.textContent = label;
  if (els.connSub) els.connSub.textContent = sub;

  const banner = els.demoBanner;
  if (!banner) return;
  const showBanner = state.mode === 'demo' || (state.mode === 'live' && !state.connected);
  banner.hidden = !showBanner;
  if (showBanner && els.demoBannerText) {
    els.demoBannerText.textContent = state.mode === 'demo'
      ? '后端未连接，当前使用内置演示数据渲染界面，仅用于视觉预览，不代表真实训练结果。'
      : '后端连接中断，页面显示最后一次获取的数据；恢复后会自动继续刷新。';
  }
}

export function showDemoChips(visible) {
  const chips = document.querySelectorAll('.demo-chip');
  for (const chip of chips) chip.hidden = !visible;
}

export async function checkHealth() {
  try {
    const health = await Api.health();
    state.connected = !!health && health.ok !== false;
    state.health = health || {};
  } catch (err) {
    state.connected = false;
    state.health = null;
  }
  return state.connected;
}

export async function retryConnection() {
  const ok = await checkHealth();
  renderConn();
  if (ok) {
    toast('后端已连接，正在加载真实数据…', 'ok');
    tickMain();
  } else {
    toast('仍然无法连接后端（/api/health）', 'error');
  }
}

export function enterDemo() {
  const firstTime = state.mode !== 'demo';
  state.mode = 'demo';
  initDemoData();
  showDemoChips(true);
  if (!firstTime) return;
  state.runs = DEMO.runs.slice();
  renderRunList();
  state.replays = DEMO.replays.slice();
  renderReplays();
  renderCheckpoints(DEMO.checkpoints);
  if (els.inferHint) els.inferHint.textContent = '后端未连接：可播放内置演示对局（演示数据，仅用于界面预览）。';
  if (DEMO.runs.length) selectDemoRun(DEMO.runs[0].id);
  syncMapStatus();
  if (state.tab === 'sandbox' || state.tab === 'infer') ensureSandboxContent();
}

export function exitDemo() {
  state.mode = 'live';
  showDemoChips(false);
  state.currentRunId = null;
  state.runDetail = null;
  resetRunData(true);
  state.runs = [];
  renderRunList();
  state.checkpoints = [];
  renderCheckpoints([]);
  state.replays = [];
  renderReplays();
  resetSandbox();
  if (els.inferHint) els.inferHint.textContent = '选择 checkpoint 后开始推理，沙盘将实时回放决策帧。';
  renderConn();
}
