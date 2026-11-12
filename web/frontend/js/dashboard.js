/**
 * dashboard.js — Tab 切换 / 提示条 / 动画循环 / 主轮询：页面级编排。
 */

import { loadFeCheckpoints } from './checkpoint.js';
import { checkHealth, enterDemo, exitDemo, renderConn } from './connection.js';
import { feRestoreInferJobs } from './feInferRun.js';
import { feRenderInferDashboard } from './feInferUi.js';
import { feRenderTrainDashboard, renderFeTrainArtifacts } from './feTrain.js';
import { advanceSandbox, currentFrame, ensureSandboxContent, updatePlayerUi } from './playback.js';
import { feApplyTrainMode, feRenderPpoDashboard } from './ppo.js';
import { feRestorePpoJob } from './ppoRun.js';
import { stopRecording } from './recorder.js';
import { drawSandbox } from './render.js';
import { loadReplays } from './replays.js';
import { refreshRwppStatus } from './rwpp.js';
import { els, state } from './state.js';
import { loadMaps } from './terrain.js';

export function switchTab(name) {
  const resolved = name === 'train' ? 'train' : 'infer';
  if ((state.tab === 'sandbox' || state.tab === 'infer') && resolved !== 'infer') stopRecording();
  state.tab = resolved;
  const buttons = document.querySelectorAll('.tab-btn');
  for (const button of buttons) {
    const active = button.getAttribute('data-tab') === resolved;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  const panels = document.querySelectorAll('.tab-panel');
  for (const panel of panels) {
    const target = resolved === 'train' ? 'panel-train' : 'panel-infer';
    panel.classList.toggle('active', panel.id === target);
  }
  if (window.location.hash !== '#' + resolved) {
    try {
      window.history.replaceState(null, '', '#' + resolved);
    } catch (error) {
      window.location.hash = resolved;
    }
  }
  if (resolved === 'infer') {
    if (state.mode === 'demo') ensureSandboxContent();
    drawSandbox();
    refreshRwppStatus();
    loadReplays();
    void loadFeCheckpoints(true);
    feRenderInferDashboard();
    void feRestoreInferJobs();
  } else {
    refreshRwppStatus(true);
    feApplyTrainMode(state.fe.trainMode);
    feRenderPpoDashboard();
    void feRestorePpoJob();
    feRenderTrainDashboard(state.fe.trainJob);
    renderFeTrainArtifacts(state.fe.trainArtifacts);
    if (state.fe.trainStatus === 'idle' && state.rwpp.hostRunning > 0) {
      const hint = document.getElementById('fe-train-hint');
      if (hint) hint.textContent = '检测到宿主机有任务正在运行；若为训练任务，完成后回放会自动入库。';
    }
  }
}

export function tabFromHash() {
  const hash = String(window.location.hash || '').replace('#', '');
  if (hash === 'train') return 'train';
  return 'infer';
}

export function toast(message, kind) {
  const host = els.toastHost;
  if (!host) return;
  const node = document.createElement('div');
  node.className = 'toast' + (kind ? ' ' + kind : '');
  node.textContent = String(message);
  host.appendChild(node);
  setTimeout(() => {
    node.style.transition = 'opacity 0.25s';
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 280);
  }, 3800);
  while (host.childElementCount > 6) host.removeChild(host.firstElementChild);
}

export function handleSandboxClick(event) {
  const sandbox = state.sandbox;
  // 拖动平移后松手会先触发 pointerup 再冒出一个 click，这里做时间门控，避免拖完误选中单位
  if (Date.now() - (sandbox.dragMovedAt || 0) < 300) return;
  const canvas = els.sandboxCanvas;
  const view = sandbox.view;
  if (!canvas || !view || !sandbox.frames.length) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  const frame = currentFrame();
  if (!frame) return;
  let best = null;
  let bestDistance = 22;
  for (const unit of frame.units) {
    const projected = view.project ? view.project(unit.x, unit.y) : { x: view.ox + unit.x * view.scale, y: view.oy + unit.y * view.scale };
    const distance = Math.hypot(projected.x - mouseX, projected.y - mouseY);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = unit;
    }
  }
  sandbox.selectedId = best ? best.id : null;
  if (best) {
    sandbox.eventLog.push({
      kind: 'info',
      time: frame.time,
      text: '选中 #' + best.id + '（' + (best.team === 'own' ? '我方' : '敌方') + '）'
        + ' HP ' + Math.round(best.hp) + '/' + Math.round(best.maxHp)
        + ' · order=' + (best.order || '—')
        + (best.target === null || best.target === undefined ? '' : ' · target #' + best.target),
    });
    if (sandbox.eventLog.length > 300) sandbox.eventLog.shift();
    sandbox.feedCount = -1;
    updatePlayerUi(true);
  }
}

export let lastAnimationTs = 0;

export function animationLoop(timestamp) {
  requestAnimationFrame(animationLoop);
  const dt = lastAnimationTs ? Math.min(0.12, (timestamp - lastAnimationTs) / 1000) : 0;
  lastAnimationTs = timestamp;
  if (state.tab === 'sandbox' || state.tab === 'infer') {
    advanceSandbox(dt);
    drawSandbox();
  }
}

export async function tickMain() {
  if (state.mainBusy) return;
  state.mainBusy = true;
  try {
    await checkHealth();
    if (!state.connected) {
      if (state.mode !== 'demo') enterDemo();
      renderConn();
      return;
    }
    if (state.mode === 'demo') exitDemo();
    if (state.mode === 'boot') state.mode = 'live';
    if (!state.maps.ready) await loadMaps();
    await refreshRwppStatus();
    renderConn();
  } finally {
    state.mainBusy = false;
  }
}
