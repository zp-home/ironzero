/**
 * feTrain.js — 指挥中心大屏 · 实机训练（多线程数据采集）：启停、轮询、KPI、日志、产物列表。
 */

import { Api } from './api.js';
import { feSetStatusPill, feSetText } from './control.js';
import { switchTab, toast } from './dashboard.js';
import { feDrawTrainCharts, handleFeError } from './feInferUi.js';
import { resultMeta } from './normalize.js';
import { loadReplays } from './replays.js';
import { isRealReplay, openReplay, refreshRwppStatus } from './rwpp.js';
import { els, state } from './state.js';
import { basename, clampInt, fmtShortTime, num, numOrNull } from './util.js';

/* ---------------- 实机训练：多线程采集大屏 ---------------- */
export function feSetTrainRunning(running) {
  if (els.feTrainStart) {
    els.feTrainStart.disabled = running;
    els.feTrainStart.hidden = running;
    els.feTrainStart.textContent = running ? '实机训练中…' : '开始实机训练';
  }
  if (els.feTrainStop) {
    els.feTrainStop.hidden = !running;
    els.feTrainStop.disabled = !running;
  }
  if (els.feTrainEpisodes) els.feTrainEpisodes.disabled = running;
  if (els.feTrainTicks) els.feTrainTicks.disabled = running;
  if (els.feTrainMap) els.feTrainMap.disabled = running;
  if (els.feTrainConcurrency) els.feTrainConcurrency.disabled = running;
}

export function feRenderTrainKpis(job) {
  const fe = state.fe;
  const metrics = (fe.trainMetrics || []).slice().sort((left, right) => num(left.episode, 0) - num(right.episode, 0));
  const total = job && numOrNull(job.episodes_total) !== null ? num(job.episodes_total, fe.trainEpisodes) : fe.trainEpisodes;
  const done = job && numOrNull(job.episodes_done) !== null ? num(job.episodes_done, metrics.length) : metrics.length;
  const elapsed = job && numOrNull(job.elapsed) !== null ? num(job.elapsed, 0) : (job && numOrNull(job.seconds) !== null ? num(job.seconds, 0) : (fe.trainStartedAt ? (Date.now() - fe.trainStartedAt) / 1000 : 0));
  const framesTotal = job && numOrNull(job.frames) !== null ? num(job.frames, 0) : metrics.reduce((sum, metric) => sum + num(metric.frames, 0), 0);
  const ticksTotal = job && numOrNull(job.ticks_total) !== null
    ? num(job.ticks_total, 0)
    : metrics.reduce((sum, metric) => sum + (numOrNull(metric.ticks) !== null ? num(metric.ticks, 0) : num(job && job.ticks, 0)), 0);
  const avg = metrics.length
    ? metrics.reduce((sum, metric) => sum + num(metric.seconds, 0), 0) / metrics.length
    : (done > 0 ? elapsed / done : 0);
  feSetText(els.feTrainKpiProgress, done);
  feSetText(els.feTrainKpiProgressTotal, '/ ' + total + ' 局');
  feSetText(els.feTrainKpiSeconds, Math.round(elapsed));
  feSetText(els.feTrainKpiThreads, job && numOrNull(job.concurrency) !== null ? num(job.concurrency, fe.trainConcurrency) : fe.trainConcurrency);
  feSetText(els.feTrainKpiAvg, avg.toFixed(1));
  feSetText(els.feTrainKpiFrames, framesTotal);
  feSetText(els.feTrainKpiTicks, ticksTotal);
  if (els.feTrainEpisodesLabel) els.feTrainEpisodesLabel.textContent = done + ' / ' + total + ' 局';
  if (els.feTrainSeconds) els.feTrainSeconds.textContent = Math.round(elapsed) + 's';
  if (els.feTrainCurrent) {
    const current = job && numOrNull(job.current_episode) !== null ? num(job.current_episode, 0) : 0;
    if (total > 0 && done >= total) els.feTrainCurrent.textContent = '已完成';
    else if (current > 0) els.feTrainCurrent.textContent = '当前第 ' + current + ' 局';
    else if (job && job.status === 'running') els.feTrainCurrent.textContent = '采集中…';
    else els.feTrainCurrent.textContent = '等待开始';
  }
  const percent = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  if (els.feTrainProgress) els.feTrainProgress.style.width = percent + '%';
  if (job) {
    const labels = { running: ['运行中', 'running'], done: ['已完成', 'done'], failed: ['失败', 'bad'] };
    const pair = labels[job.status] || ['空闲', 'idle'];
    feSetStatusPill(els.feTrainStatus, pair[0], pair[1]);
    feSetText(els.feTrainLogStatus, 'status=' + job.status + ' · concurrency=' + (job.concurrency || '—'));
  } else if (els.feTrainStatus) {
    feSetStatusPill(els.feTrainStatus, '空闲', 'idle');
  }
  if (els.feTrainSummary) {
    const resultCounts = {};
    for (const metric of metrics) {
      if (!metric || !metric.result) continue;
      const label = resultMeta(metric.result).label;
      resultCounts[label] = (resultCounts[label] || 0) + 1;
    }
    const resultText = Object.keys(resultCounts).length
      ? ' · ' + Object.keys(resultCounts).map((label) => label + ' ' + resultCounts[label]).join(' / ')
      : '';
    if (job && job.status === 'done') {
      els.feTrainSummary.textContent = '汇总：' + done + ' 局 · ' + framesTotal + ' 帧 · ' + ticksTotal + ' tick · ' + Math.round(elapsed) + 's · 平均 ' + avg.toFixed(1) + 's/局' + resultText;
    } else if (job && job.status === 'failed') {
      els.feTrainSummary.textContent = '失败：' + String(job.error || '未知错误').slice(0, 80);
    } else {
      els.feTrainSummary.textContent = '—';
    }
  }
}

export function feRenderTrainLog(job) {
  const log = els.feTrainLog;
  if (!log) return;
  if (!job) {
    log.textContent = '等待训练任务…';
    return;
  }
  const lines = [];
  lines.push('status: ' + job.status);
  lines.push('episodes: ' + num(job.episodes_done, 0) + '/' + num(job.episodes_total, state.fe.trainEpisodes) + ' · concurrency: ' + (job.concurrency || '—'));
  const episodeResults = (Array.isArray(job.episodes_metrics) ? job.episodes_metrics : [])
    .map((metric) => metric && metric.result)
    .filter(Boolean);
  if (episodeResults.length) {
    const counts = {};
    for (const result of episodeResults) {
      const label = resultMeta(result).label;
      counts[label] = (counts[label] || 0) + 1;
    }
    lines.push('results: ' + Object.keys(counts).map((label) => label + ' ' + counts[label]).join(' · '));
  }
  lines.push('ticks: ' + num(job.ticks, 0) + ' · instances: ' + (job.instances || job.workers || '—'));
  lines.push('frames: ' + num(job.frames, 0) + ' · ticks_total: ' + num(job.ticks_total, 0));
  if (job.error) lines.push('');
  if (job.error) lines.push('ERROR: ' + job.error);
  const tail = job.last_log ? String(job.last_log).trim() : '';
  if (tail) {
    lines.push('');
    lines.push(tail.split('\n').slice(-10).join('\n'));
  }
  log.textContent = lines.join('\n');
}

export function feRenderTrainDashboard(job) {
  const fe = state.fe;
  const jobData = job === undefined ? fe.trainJob : job;
  if (jobData) {
    fe.trainJob = jobData;
    if (Array.isArray(jobData.episodes_metrics)) {
      fe.trainMetrics = jobData.episodes_metrics.slice().sort((left, right) => num(left.episode, 0) - num(right.episode, 0));
    }
  }
  feRenderTrainKpis(jobData);
  feRenderTrainLog(jobData);
  feDrawTrainCharts();
}

export async function feStartTrain() {
  const fe = state.fe;
  if (fe.trainStatus === 'running') return;
  if (state.mode === 'demo') {
    toast('演示模式下无法连接真机服务', 'error');
    return;
  }
  const mapName = els.feTrainMap ? els.feTrainMap.value : '';
  if (!mapName) {
    toast('请选择地图', 'error');
    return;
  }
  const episodes = clampInt(els.feTrainEpisodes && els.feTrainEpisodes.value, 1, 50, 3);
  const ticks = clampInt(els.feTrainTicks && els.feTrainTicks.value, 120, 216000, 1200);
  const concurrency = clampInt(els.feTrainConcurrency && els.feTrainConcurrency.value, 1, 4, 2);
  const policy = (els.feTrainPolicy && els.feTrainPolicy.value) || 'scripted';
  let checkpoint = null;
  if (policy === 'model') {
    checkpoint = (els.feTrainCheckpointSelect && els.feTrainCheckpointSelect.value) || null;
    if (!checkpoint) {
      toast('请选择用于采集对战的模型 Checkpoint', 'error');
      return;
    }
  }
  fe.trainMetrics = [];
  fe.trainEpisodes = episodes;
  fe.trainConcurrency = concurrency;
  fe.trainStatus = 'running';
  fe.trainStartedAt = Date.now();
  fe.trainArtifacts = [];
  feSetTrainRunning(true);
  feRenderTrainDashboard(null);
  feSetStatusPill(els.feTrainStatus, '启动中', 'running');
  feRenderTrainLog({ status: 'starting', episodes_done: 0, episodes_total: episodes, concurrency: concurrency, ticks: ticks, frames: 0, ticks_total: 0, last_log: '正在创建真机训练任务…' });
  feDrawTrainCharts();
  feSetText(els.feTrainSummary, '—');
  try {
    const response = await Api.rwppTrain({
      episodes: episodes,
      ticks: ticks,
      policy: policy,
      checkpoint: checkpoint,
      map_name: mapName,
      concurrency: concurrency,
    });
    const jobId = response && (response.job_id || response.id);
    if (!jobId) throw new Error('后端未返回训练任务 id');
    fe.trainJobId = String(jobId);
    fe.trainStatus = 'running';
    const hint = document.getElementById('fe-train-hint');
    if (hint) hint.textContent = '并行跑多个真实游戏进程，每线程独立采集一局；当前阶段为真机数据采集。';
    startFeTrainPolling();
    toast('实机训练已开始 · ' + episodes + ' 局 × ' + ticks + ' ticks · ' + concurrency + ' 路并发', 'ok');
  } catch (error) {
    feSetTrainRunning(false);
    fe.trainStatus = 'failed';
    if (error && error.status === 409) {
      toast('已有训练或实机任务正在运行，请先在控制台停止旧任务或稍候重试', 'error');
    } else {
      handleFeError(error, '实机训练');
    }
    feRenderTrainDashboard(null);
    feSetStatusPill(els.feTrainStatus, '失败', 'bad');
    feRenderTrainLog({ status: 'failed', episodes_done: 0, episodes_total: episodes, concurrency: concurrency, ticks: ticks, frames: 0, ticks_total: 0, error: error && error.message ? error.message : String(error) });
  }
}

export async function feStopTrain() {
  const fe = state.fe;
  if (!fe.trainJobId || fe.trainStatus !== 'running') return;
  if (els.feTrainStop) els.feTrainStop.disabled = true;
  try {
    await Api.rwppJobStop(fe.trainJobId);
    toast('已请求停止实机数据采集任务', 'ok');
  } catch (error) {
    handleFeError(error, '停止训练');
  }
}

export function startFeTrainPolling() {
  stopFeTrainPolling();
  state.fe.trainTimer = setInterval(fePollTrain, 2000);
  void fePollTrain();
}

export function stopFeTrainPolling() {
  const fe = state.fe;
  if (fe.trainTimer) {
    clearInterval(fe.trainTimer);
    fe.trainTimer = null;
  }
}

export async function fePollTrain() {
  const fe = state.fe;
  if (!fe.trainJobId || fe.trainPolling) return;
  fe.trainPolling = true;
  try {
    const job = await Api.rwppJob(fe.trainJobId);
    if (!job) return;
    const metrics = Array.isArray(job.episodes_metrics) ? job.episodes_metrics : [];
    const previous = fe.trainMetrics.length;
    fe.trainJob = job;
    fe.trainMetrics = metrics.slice().sort((left, right) => num(left.episode, 0) - num(right.episode, 0));
    feRenderTrainDashboard(job);
    if (fe.trainMetrics.length !== previous || job.status !== 'running') {
      await loadReplays();
      await renderFeTrainArtifactsFromJob(job);
    }
    if (job.status !== 'running') {
      stopFeTrainPolling();
      feSetTrainRunning(false);
      if (job.status === 'done') {
        fe.trainStatus = 'done';
        toast('实机训练完成（' + Math.round(num(job.elapsed, 0)) + 's · ' + num(job.episodes_done, fe.trainMetrics.length) + ' 局）', 'ok');
      } else {
        fe.trainStatus = 'failed';
        toast('实机训练失败：' + String(job.error || job.last_log || '未知错误').slice(0, 160), 'error');
      }
      await refreshRwppStatus(true);
      feRenderTrainDashboard(job);
    }
  } catch (error) {
    if (error && (error.status === 503 || error.status === 409)) {
      stopFeTrainPolling();
      feSetTrainRunning(false);
      fe.trainStatus = 'failed';
      handleFeError(error, '实机训练');
      feSetStatusPill(els.feTrainStatus, '失败', 'bad');
    }
  } finally {
    fe.trainPolling = false;
  }
}

export async function renderFeTrainArtifactsFromJob(job) {
  const names = [];
  for (const path of job && Array.isArray(job.replays) ? job.replays : []) names.push(basename(String(path).replace(/\\/g, '/')));
  for (const metric of job && Array.isArray(job.episodes_metrics) ? job.episodes_metrics : []) {
    if (metric && metric.replay) names.push(basename(String(metric.replay).replace(/\\/g, '/')));
  }
  const items = state.replays || [];
  let picked = names
    .map((name) => items.find((item) => item && item.name === name))
    .filter(Boolean);
  const unique = [];
  const seen = new Set();
  for (const item of picked) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    unique.push(item);
  }
  if (!unique.length) {
    const total = job && numOrNull(job.episodes_done) !== null ? num(job.episodes_done, 1) : Math.max(1, names.length);
    picked = items.filter(isRealReplay).slice(0, Math.max(1, total));
  } else {
    picked = unique;
  }
  state.fe.trainArtifacts = picked;
  renderFeTrainArtifacts(picked);
}

export function renderFeTrainArtifacts(items) {
  const body = els.feTrainArtifacts;
  if (!body) return;
  body.innerHTML = '';
  const list = Array.isArray(items) ? items : [];
  if (els.feTrainArtifactCount) els.feTrainArtifactCount.textContent = String(list.length);
  if (!list.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.className = 'empty';
    cell.textContent = '训练完成后显示本任务录制的回放';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const item of list) {
    const row = document.createElement('tr');
    row.className = 'data-row';
    const cellName = document.createElement('td');
    cellName.textContent = item.name || basename(item.id);
    if (isRealReplay(item)) {
      const tag = document.createElement('span');
      tag.className = 'real-tag';
      tag.textContent = '真机';
      cellName.appendChild(tag);
    }
    const cellSim = document.createElement('td');
    cellSim.textContent = (item.simMode === 'terrain' ? '地形' : '竞技场')
      + (item.simMode === 'terrain' && item.mapName ? ' · ' + item.mapName : '');
    const cellResult = document.createElement('td');
    const badge = document.createElement('span');
    const meta = resultMeta(item.result);
    badge.className = 'result-badge ' + meta.cls;
    badge.textContent = meta.label;
    cellResult.appendChild(badge);
    const cellFrames = document.createElement('td');
    cellFrames.textContent = item.frames === null || item.frames === undefined ? '—' : String(item.frames);
    const cellTime = document.createElement('td');
    cellTime.textContent = fmtShortTime(item.created_at) || '—';
    const cellActions = document.createElement('td');
    const playButton = document.createElement('button');
    playButton.type = 'button';
    playButton.className = 'btn btn-sm';
    playButton.textContent = '播放';
    playButton.addEventListener('click', (event) => {
      event.stopPropagation();
      void openFeReplay(item.id);
    });
    cellActions.appendChild(playButton);
    row.appendChild(cellName);
    row.appendChild(cellSim);
    row.appendChild(cellResult);
    row.appendChild(cellFrames);
    row.appendChild(cellTime);
    row.appendChild(cellActions);
    body.appendChild(row);
  }
}

export async function openFeReplay(id) {
  switchTab('infer');
  await openReplay(id);
}

export async function feRefreshTrainArtifacts() {
  await loadReplays();
  if (state.fe.trainJobId) {
    try {
      const job = await Api.rwppJob(state.fe.trainJobId);
      await renderFeTrainArtifactsFromJob(job);
      return;
    } catch (error) {
      // fall back to cached artifacts
    }
  }
  renderFeTrainArtifacts(state.fe.trainArtifacts);
}
