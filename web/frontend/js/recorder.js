/**
 * recorder.js — 对局沙盘 · 录制：canvas.captureStream + MediaRecorder 录制与 .webm 下载。
 */

import { toast } from './dashboard.js';
import { els, state } from './state.js';

/* ---------------- 录制（canvas.captureStream + MediaRecorder） ---------------- */
export function toggleRecording() {
  const recorder = state.sandbox.recorder;
  if (recorder && recorder.state !== 'inactive') stopRecording();
  else startRecording();
}

export function startRecording() {
  const canvas = els.sandboxCanvas;
  if (!canvas) return;
  if (typeof window.MediaRecorder === 'undefined' || typeof canvas.captureStream !== 'function') {
    toast('当前浏览器不支持 canvas 录制（需要 MediaRecorder + captureStream）', 'error');
    return;
  }
  let stream = null;
  try {
    stream = canvas.captureStream(30);
  } catch (error) {
    toast('无法捕获画布视频流：' + error.message, 'error');
    return;
  }
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  let mimeType = '';
  for (const candidate of candidates) {
    if (typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported(candidate)) {
      mimeType = candidate;
      break;
    }
  }
  let recorder = null;
  try {
    recorder = mimeType
      ? new MediaRecorder(stream, { mimeType: mimeType, videoBitsPerSecond: 6000000 })
      : new MediaRecorder(stream, { videoBitsPerSecond: 6000000 });
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    toast('录制初始化失败：' + error.message, 'error');
    return;
  }
  const sandbox = state.sandbox;
  sandbox.recorder = recorder;
  sandbox.recStream = stream;
  sandbox.recChunks = [];
  recorder.addEventListener('dataavailable', (event) => {
    if (event.data && event.data.size) sandbox.recChunks.push(event.data);
  });
  recorder.addEventListener('stop', () => finalizeRecording(sandbox));
  recorder.start(250);
  if (els.recordBtn) {
    els.recordBtn.textContent = '■ 停止录制';
    els.recordBtn.style.color = '#ff8f9b';
  }
  if (els.recBadge) els.recBadge.hidden = false;
  toast('开始录制沙盘画面（canvas.captureStream 30fps → .webm）', 'ok');
}

export function stopRecording() {
  const recorder = state.sandbox.recorder;
  if (recorder && recorder.state !== 'inactive') {
    state.sandbox.recCancel = false;
    recorder.stop();
  }
}

/** 重置沙盘时静默丢弃正在进行的录制（不触发下载） */
export function cancelRecording() {
  const sandbox = state.sandbox;
  const recorder = sandbox.recorder;
  if (recorder && recorder.state !== 'inactive') {
    sandbox.recCancel = true;
    recorder.stop();
  }
}

export function finalizeRecording(sandbox) {
  const chunks = sandbox.recChunks || [];
  const recorder = sandbox.recorder;
  const mimeType = (recorder && recorder.mimeType) || 'video/webm';
  const cancelled = !!sandbox.recCancel;
  if (sandbox.recStream) sandbox.recStream.getTracks().forEach((track) => track.stop());
  sandbox.recorder = null;
  sandbox.recStream = null;
  sandbox.recChunks = [];
  sandbox.recCancel = false;
  if (els.recordBtn) {
    els.recordBtn.textContent = '● 录制视频';
    els.recordBtn.style.color = '';
  }
  if (els.recBadge) els.recBadge.hidden = true;
  if (cancelled) return;
  if (!chunks.length) {
    toast('录制结束，但未捕获到视频数据', 'error');
    return;
  }
  const blob = new Blob(chunks, { type: mimeType.split(';')[0] || 'video/webm' });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = 'ironzero-sandbox-' + timestamp + '.webm';
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  toast('录制完成 · ' + filename + '（' + (blob.size / 1048576).toFixed(2) + ' MB）', 'ok');
}
