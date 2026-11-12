/**
 * main.js — 启动入口：只有 boot() 与 DOMContentLoaded 调度；state.fe 分段初始化也在这里按原顺序执行。
 */

import { bindEvents } from './bindings.js';
import { loadFeCheckpoints } from './checkpoint.js';
import { checkHealth, enterDemo, renderConn, startClock } from './connection.js';
import { animationLoop, switchTab, tabFromHash, tickMain } from './dashboard.js';
import { feRestoreInferJobs } from './feInferRun.js';
import { feDrawInferCharts, feDrawTrainCharts, feRenderInferDashboard } from './feInferUi.js';
import { feRenderTrainDashboard } from './feTrain.js';
import { updatePlayerUi } from './playback.js';
import { feApplyTrainMode, feRenderPpoDashboard } from './ppo.js';
import { feDrawPpoCharts } from './ppoCharts.js';
import { feRestorePpoJob } from './ppoRun.js';
import { loadReplays } from './replays.js';
import { refreshRwppStatus } from './rwpp.js';
import { cacheEls, state } from './state.js';
import { loadMaps } from './terrain.js';
import { initFeControlState } from './control.js';
import { initFePpoState } from './ppo.js';
import { initFeModelState } from './model.js';

export async function boot() {
  initFeControlState();
  initFePpoState();
  initFeModelState();
  cacheEls();
  bindEvents();
  startClock();
  state.mode = 'boot';
  switchTab(tabFromHash());
  requestAnimationFrame(animationLoop);
  renderConn();

  // 避免网络挂起导致右上角一直停留在 boot 连接中
  try {
    const healthPromise = checkHealth();
    const timeoutPromise = new Promise((resolve) => setTimeout(() => resolve(false), 4000));
    await Promise.race([healthPromise, timeoutPromise]);
  } catch (e) {
    state.connected = false;
  }
  if (state.connected) {
    state.mode = 'live';
    renderConn();
    await loadMaps();
    await loadReplays();
    await refreshRwppStatus(true);
    await loadFeCheckpoints();
    feRenderInferDashboard();
    feApplyTrainMode(state.fe.trainMode);
    feRenderTrainDashboard(null);
    feRenderPpoDashboard();
    void feRestoreInferJobs();
    void feRestorePpoJob();
  } else {
    enterDemo();
  }
  renderConn();
  updatePlayerUi(true);
  feRenderInferDashboard();
  feRenderTrainDashboard(null);
  setTimeout(() => {
    feDrawInferCharts();
    feDrawTrainCharts();
    feDrawPpoCharts();
  }, 120);
  setInterval(tickMain, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { boot(); });
} else {
  boot();
}

