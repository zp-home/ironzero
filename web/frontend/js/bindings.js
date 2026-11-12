/**
 * bindings.js — DOM 事件绑定：按钮、下拉框、画布交互的统一注册（bindEvents）。
 */

import { handleChartHover } from './charts.js';
import { commitCamera, ensureCamera, panBy, zoomAt, zoomFactorFromWheel } from './camera.js';
import { loadFeCheckpoints, renderFeCheckpointOptions } from './checkpoint.js';
import { retryConnection } from './connection.js';
import { handleSandboxClick, switchTab, toast } from './dashboard.js';
import { feStartInfer, feStopInfer } from './feInferRun.js';
import { feDrawInferCharts, feDrawTrainCharts } from './feInferUi.js';
import { feRefreshTrainArtifacts, feStartTrain, feStopTrain } from './feTrain.js';
import { feSyncInferPolicyUi, feSyncPpoConfigUi, feSyncTrainPolicyUi } from './model.js';
import {
  initModelManagerState,
  loadModels,
  removeTempModelJobs,
  renderModelManager,
  toggleModelManagerCollapse,
  updateModelManagerCollapseUi,
} from './modelManager.js';
import { loadDemoBattle, updatePlayerUi } from './playback.js';
import { feApplyTrainMode, feRenderPpoDashboard } from './ppo.js';
import { feStartPpo, feStopPpo } from './ppoRun.js';
import { toggleRecording } from './recorder.js';
import { drawSandbox } from './render.js';
import { loadReplays, renderReplays, toggleReplayCollapse } from './replays.js';
import { refreshRwppStatus, startRwppRecord } from './rwpp.js';
import { submitInfer } from './sandbox.js';
import { els, state } from './state.js';
import { loadMaps, syncMapStatus } from './terrain.js';
import { stopCurrentRun, submitStartRun } from './trainRuns.js';
import { clamp, clampInt, num } from './util.js';
import { applyViewMode, resetSandboxView, selectMap, syncCameraUi, syncMapToggles } from './view.js';

export function bindEvents() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  for (const button of tabButtons) {
    button.addEventListener('click', () => switchTab(button.getAttribute('data-tab') || 'train'));
  }
  if (els.startForm) els.startForm.addEventListener('submit', submitStartRun);
  if (els.stopBtn) els.stopBtn.addEventListener('click', stopCurrentRun);
  if (els.inferForm) els.inferForm.addEventListener('submit', submitInfer);
  if (els.replayRefresh) {
    els.replayRefresh.addEventListener('click', () => {
      loadReplays();
      refreshRwppStatus(true);
      void loadFeCheckpoints(true);
      // 模型管理面板若已展开过，顺手跟一次（缓存的条目可能已经在别处被删）。
      if (state.modelManager && state.modelManager.data) void loadModels(true);
    });
  }
  if (els.replayToggleBtn) {
    els.replayToggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleReplayCollapse();
    });
  }
  if (els.replayCardHead) {
    els.replayCardHead.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      toggleReplayCollapse();
    });
  }
  if (els.replayPrevBtn) {
    els.replayPrevBtn.addEventListener('click', () => {
      if (state.replayPagination.page > 1) {
        state.replayPagination.page -= 1;
        renderReplays();
      }
    });
  }
  if (els.replayNextBtn) {
    els.replayNextBtn.addEventListener('click', () => {
      const totalPages = Math.ceil((state.replays || []).length / (state.replayPagination.pageSize || 8));
      if (state.replayPagination.page < totalPages) {
        state.replayPagination.page += 1;
        renderReplays();
      }
    });
  }
  if (els.replayPageSize) {
    els.replayPageSize.addEventListener('change', () => {
      state.replayPagination.pageSize = Number(els.replayPageSize.value) || 8;
      state.replayPagination.page = 1;
      renderReplays();
    });
  }
  if (els.rwppRecordBtn) els.rwppRecordBtn.addEventListener('click', startRwppRecord);

  // ---- 模型管理面板（删除过期 checkpoint / 作业目录）----
  // 折叠状态与渲染都走 modelManager.js；这里只负责接线，不在模块顶层做任何事
  // （该目录存在循环依赖，见 js/README.md 第 5 节）。
  initModelManagerState();
  if (els.modelManagerToggleBtn) {
    els.modelManagerToggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleModelManagerCollapse();
    });
  }
  if (els.modelManagerHead) {
    els.modelManagerHead.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      toggleModelManagerCollapse();
    });
  }
  if (els.modelManagerRefresh) {
    els.modelManagerRefresh.addEventListener('click', async () => {
      toast('正在重新扫描 checkpoint 目录…', 'info');
      await loadModels(true);
      const data = state.modelManager && state.modelManager.data;
      if (data) {
        toast('已刷新：' + data.count + ' 个权重，共 ' + data.total_mb + ' MB', 'ok');
      }
    });
  }
  if (els.modelManagerCleanTmp) {
    els.modelManagerCleanTmp.addEventListener('click', () => {
      void removeTempModelJobs();
    });
  }
  if (els.modelManagerMoreWeights) {
    els.modelManagerMoreWeights.addEventListener('click', () => {
      if (!state.modelManager) return;
      state.modelManager.showAllWeights = true;
      renderModelManager();
    });
  }
  if (els.modelManagerMoreJobs) {
    els.modelManagerMoreJobs.addEventListener('click', () => {
      if (!state.modelManager) return;
      state.modelManager.showAllJobs = true;
      renderModelManager();
    });
  }
  if (els.modelManagerCard) updateModelManagerCollapseUi();
  document.querySelectorAll('.rwpp-status').forEach((badge) => {
    badge.style.cursor = 'pointer';
    badge.addEventListener('click', async () => {
      toast('正在重新检测宿主机真机服务 (8300)…', 'info');
      await refreshRwppStatus(true);
      if (state.rwpp.available) {
        toast('真机服务就绪 (8300) · 空闲槽位: ' + (state.rwpp.freeSlots ?? '—'), 'ok');
      } else {
        toast('真机服务未就绪：请在宿主机终端执行 python scripts/rwpp_real_service.py', 'error');
      }
    });
  });
  if (els.feStartBtn) els.feStartBtn.addEventListener('click', feStartInfer);
  if (els.feInferStop) els.feInferStop.addEventListener('click', feStopInfer);
  if (els.feTrainStart) els.feTrainStart.addEventListener('click', feStartTrain);
  if (els.feTrainStop) els.feTrainStop.addEventListener('click', feStopTrain);
  if (els.feTrainRefresh) {
    els.feTrainRefresh.addEventListener('click', () => {
      feRefreshTrainArtifacts();
      void loadFeCheckpoints(true);
    });
  }
  for (const button of document.querySelectorAll('[data-train-mode-btn]')) {
    button.addEventListener('click', () => feApplyTrainMode(button.getAttribute('data-train-mode-btn')));
  }
  if (els.fePpoStart) els.fePpoStart.addEventListener('click', feStartPpo);
  if (els.fePpoStop) els.fePpoStop.addEventListener('click', feStopPpo);
  if (els.fePolicy) els.fePolicy.addEventListener('change', feSyncInferPolicyUi);
  if (els.feCheckpointSort) {
    els.feCheckpointSort.addEventListener('change', () => {
      renderFeCheckpointOptions();
    });
  }
  if (els.fePpoCheckpointSort) {
    els.fePpoCheckpointSort.addEventListener('change', () => {
      renderFeCheckpointOptions();
    });
  }
  if (els.feCheckpointRefresh) {
    els.feCheckpointRefresh.addEventListener('click', async () => {
      toast('正在重新扫描 Checkpoint 权重目录…', 'info');
      await loadFeCheckpoints(true);
      const count = (state.fe.checkpointList || []).length;
      toast('已刷新 Checkpoint 列表，共发现 ' + count + ' 个模型', 'ok');
    });
  }
  if (els.fePpoCheckpointRefresh) {
    els.fePpoCheckpointRefresh.addEventListener('click', async () => {
      toast('正在重新扫描 Checkpoint 权重目录…', 'info');
      await loadFeCheckpoints(true);
      const count = (state.fe.checkpointList || []).length;
      toast('已刷新 Checkpoint 列表，共发现 ' + count + ' 个模型', 'ok');
    });
  }
  if (els.feTrainPolicy) els.feTrainPolicy.addEventListener('change', feSyncTrainPolicyUi);
  if (els.fePpoMap) els.fePpoMap.addEventListener('change', feSyncPpoConfigUi);
  if (els.fePpoInitMode) els.fePpoInitMode.addEventListener('change', feSyncPpoConfigUi);
  if (els.fePpoUseLlm) {
    els.fePpoUseLlm.addEventListener('change', () => {
      if (els.fePpoLlmNote) els.fePpoLlmNote.hidden = !els.fePpoUseLlm.checked;
      feRenderPpoDashboard();
    });
  }
  if (els.demoRetry) els.demoRetry.addEventListener('click', retryConnection);
  if (els.mapStatus) {
    els.mapStatus.style.cursor = 'pointer';
    els.mapStatus.addEventListener('click', () => {
      toast('正在重新加载地图资源…', 'info');
      void loadMaps(true);
    });
  }
  if (els.recordBtn) els.recordBtn.addEventListener('click', toggleRecording);
  if (els.mapSelect) {
    els.mapSelect.addEventListener('change', () => {
      if (state.sandbox.simMode !== 'infer') {
        state.sandbox.simMode = 'terrain';
        state.sandbox.fogAvailable = true;
      }
      selectMap(els.mapSelect.value || '');
    });
  }
  if (els.fogToggle) {
    els.fogToggle.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.fogAvailable) return;
      sandbox.fogEnabled = !sandbox.fogEnabled;
      syncMapToggles();
      syncMapStatus();
      drawSandbox();
    });
  }
  if (els.itemsToggle) {
    els.itemsToggle.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.mapDetail) return;
      sandbox.itemsEnabled = !sandbox.itemsEnabled;
      syncMapToggles();
      syncMapStatus();
      drawSandbox();
    });
  }
  if (els.viewSelect) {
    els.viewSelect.addEventListener('change', () => {
      applyViewMode(els.viewSelect.value);
    });
  }

  if (els.playBtn) {
    els.playBtn.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.frames.length) {
        if (state.mode === 'demo') {
          loadDemoBattle();
        } else {
          toast('暂无可播放的对局，请先开始推理或从回放库加载', 'error');
          return;
        }
      }
      sandbox.playing = !sandbox.playing;
      if (sandbox.playing && sandbox.playback >= sandbox.frames.length - 1 && !sandbox.streaming) sandbox.playback = 0;
      updatePlayerUi(true);
    });
  }

  if (els.latestBtn) {
    els.latestBtn.addEventListener('click', () => {
      const sandbox = state.sandbox;
      if (!sandbox.frames.length) return;
      sandbox.playback = sandbox.frames.length - 1;
      sandbox.playing = !!sandbox.streaming;
      sandbox.follow = true;
      updatePlayerUi(true);
    });
  }

  if (els.speedSelect) {
    els.speedSelect.addEventListener('change', () => {
      const raw = els.speedSelect.value;
      state.sandbox.speed = raw === 'max' ? 30 : clamp(num(raw, 1), 0.5, 4);
      updatePlayerUi(true);
    });
  }

  if (els.seek) {
    els.seek.addEventListener('input', () => {
      const sandbox = state.sandbox;
      if (!sandbox.frames.length) return;
      sandbox.playing = false;
      sandbox.follow = false;
      sandbox.playback = clampInt(els.seek.value, 0, sandbox.frames.length - 1, 0);
      sandbox.effects = [];
      updatePlayerUi(true);
      drawSandbox();
    });
  }

  if (els.sandboxCanvas) {
    const canvas = els.sandboxCanvas;
    const wrap = els.stageWrap;
    // 拖动状态：仅在一次按下期间有效，pointerup / pointercancel 后清空
    let drag = null;

    els.sandboxCanvas.addEventListener('click', handleSandboxClick);

    // 滚轮缩放：以光标为不动点（camera.js 的 zoomAt 保证光标下的地图内容缩放前后不动）
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const sandbox = state.sandbox;
      const geom = sandbox.view && sandbox.view.geom;
      if (!geom) {
        // 无帧/无地图时基准几何不存在，重绘一次即可，不做缩放
        drawSandbox();
        return;
      }
      // 画布用 CSS 像素坐标系，geom 也是 CSS 像素 —— 不要乘 devicePixelRatio
      const rect = canvas.getBoundingClientRect();
      const screenX = event.clientX - rect.left;
      const screenY = event.clientY - rect.top;
      const camera = ensureCamera(sandbox);
      const targetZoom = camera.zoom * zoomFactorFromWheel(event.deltaY, event.deltaMode);
      commitCamera(sandbox, zoomAt(geom, camera, screenX, screenY, targetZoom));
      drawSandbox();
      syncCameraUi();
    }, { passive: false });

    // 拖动平移：左键或中键按住拖动；累计位移超过 3px 才算拖动，避免手抖把点击吃掉
    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 && event.button !== 1) return;
      const sandbox = state.sandbox;
      if (!sandbox.view || !sandbox.view.geom) return;
      if (event.button === 1) event.preventDefault(); // 阻止中键自动滚动
      drag = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
        moved: false,
      };
      if (canvas.setPointerCapture) {
        try {
          canvas.setPointerCapture(event.pointerId);
        } catch (error) {
          // 浏览器可能对已释放的 pointerId 抛错，忽略，不影响后续无 capture 拖动
        }
      }
    });

    canvas.addEventListener('pointermove', (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (!drag.moved) {
        const totalX = event.clientX - drag.startX;
        const totalY = event.clientY - drag.startY;
        if (Math.hypot(totalX, totalY) <= 3) return;
        drag.moved = true;
        if (wrap) wrap.classList.add('dragging');
      }
      const dx = event.clientX - drag.lastX;
      const dy = event.clientY - drag.lastY;
      drag.lastX = event.clientX;
      drag.lastY = event.clientY;
      const sandbox = state.sandbox;
      const geom = sandbox.view && sandbox.view.geom;
      if (!geom) return;
      event.preventDefault();
      // dx/dy 直接传入鼠标位移：地图跟着光标走（grab 语义），panBy 内部会做边界钳制
      commitCamera(sandbox, panBy(geom, ensureCamera(sandbox), dx, dy));
      drawSandbox();
    });

    const endDrag = (event) => {
      if (!drag) return;
      if (event && event.pointerId !== undefined && event.pointerId !== drag.pointerId) return;
      const moved = drag.moved;
      drag = null;
      if (event && canvas.hasPointerCapture && canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      if (wrap) wrap.classList.remove('dragging');
      // 供 handleSandboxClick 门控：拖完松手那一下不算点击，避免误选中单位
      if (moved) state.sandbox.dragMovedAt = Date.now();
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    // 双击复位到全览
    canvas.addEventListener('dblclick', (event) => {
      event.preventDefault();
      resetSandboxView();
      drawSandbox();
      syncCameraUi();
    });
  }

  if (els.viewResetBtn) {
    els.viewResetBtn.addEventListener('click', () => {
      resetSandboxView();
      drawSandbox();
      syncCameraUi();
    });
  }
  // 首帧就按当前相机状态刷一次读数/按钮可用性（全览时「复位」应为不可点）
  if (els.viewZoom || els.viewResetBtn) syncCameraUi();

  if (els.logPanel) {
    els.logPanel.addEventListener('scroll', () => {
      const panel = els.logPanel;
      state.autoScroll = panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 24;
    });
  }

  const chartPairs = [
    [els.chartWin, els.tipWin],
    [els.chartReturn, els.tipReturn],
  ];
  for (const pair of chartPairs) {
    const canvas = pair[0];
    const tip = pair[1];
    if (!canvas || !tip) continue;
    canvas.addEventListener('mousemove', (event) => handleChartHover(canvas, tip, event.clientX, event.clientY));
    canvas.addEventListener('mouseleave', () => { tip.hidden = true; });
  }

  window.addEventListener('resize', () => {
    if (state.tab === 'infer' || state.tab === 'sandbox') {
      drawSandbox();
      feDrawInferCharts();
    } else if (state.tab === 'train') {
      feDrawTrainCharts();
    }
  });

  window.addEventListener('keydown', (event) => {
    if (state.tab !== 'sandbox' && state.tab !== 'infer') return;
    const target = event.target;
    if (target && ['INPUT', 'SELECT', 'TEXTAREA'].indexOf(target.tagName) >= 0) return;
    const sandbox = state.sandbox;
    if (event.code === 'Space' || event.key === ' ') {
      event.preventDefault();
      if (!sandbox.frames.length) return;
      sandbox.playing = !sandbox.playing;
      if (sandbox.playing && sandbox.playback >= sandbox.frames.length - 1 && !sandbox.streaming) sandbox.playback = 0;
      updatePlayerUi(true);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      sandbox.playing = false;
      sandbox.playback = clamp(sandbox.playback + 1, 0, Math.max(0, sandbox.frames.length - 1));
      updatePlayerUi(true);
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      sandbox.playing = false;
      sandbox.playback = clamp(sandbox.playback - 1, 0, Math.max(0, sandbox.frames.length - 1));
      updatePlayerUi(true);
    } else if (event.key === '0') {
      // 与双击沙盘、工具条「复位」等价：相机回全览
      event.preventDefault();
      resetSandboxView(sandbox);
      drawSandbox();
      syncCameraUi();
    }
  });

  if (typeof ResizeObserver === 'function') {
    const stack = document.querySelector('.chart-stack');
    if (stack) {
      const observer = new ResizeObserver(() => {
        if (state.tab === 'infer' || state.tab === 'sandbox') drawSandbox();
      });
      observer.observe(stack);
    }
  }
}
