/**
 * modelManager.js — 模型管理面板：列出 checkpoint / 作业目录，删除过期权重或整个作业目录。
 *
 * 设计要点（都是「删除不可撤销」逼出来的）：
 * - 所有删除动作一律 `window.confirm` 二次确认，文案里必带**相对路径 + 大小**；
 * - 受保护条目（运行中作业、BC 预热权重、最新权重、下拉框当前选中）后端返回 409，
 *   前端直接把按钮 disable 掉并显示原因，用户点不到才最省事；
 * - 面板默认折叠，展开时才首次加载，避免给推理页首屏再压一个 2s 的扫描请求；
 * - 删除后主动 `loadFeCheckpoints(true)`，否则推理/训练下拉框还留着已经删掉的模型。
 */

import { Api } from './api.js';
import { toast } from './dashboard.js';
import { loadFeCheckpoints } from './checkpoint.js';
import { els, state } from './state.js';
import { fmtNum } from './util.js';

/** 面板状态：折叠状态、最近一次扫描结果、加载/错误标记。 */
export function initModelManagerState() {
  if (state.modelManager) return;
  state.modelManager = {
    collapsed: true,
    loading: false,
    data: null,
    error: null,
    jobsError: null,
    showAllWeights: false,
    showAllJobs: false,
    busy: false,
  };
}

const WEIGHT_ROW_LIMIT = 40;
const JOB_ROW_LIMIT = 30;

/** 把大小格式化成 MB 文本（后端已按 MB 返回）。 */
function mbText(value) {
  const num = Number(value);
  if (!isFinite(num)) return '—';
  if (num >= 1024) return fmtNum(num / 1024, 2) + ' GB';
  return fmtNum(num, 2) + ' MB';
}

/** 当前三个下拉框里选中的 checkpoint（绝对路径），用于「选中的不许删」。 */
function selectedCheckpoint() {
  const candidates = [
    els.feCheckpointSelect && els.feCheckpointSelect.value,
    els.fePpoInitCheckpoint && els.fePpoInitCheckpoint.value,
    els.feTrainCheckpointSelect && els.feTrainCheckpointSelect.value,
  ];
  for (const value of candidates) {
    if (value) return String(value);
  }
  return '';
}

/** 该条目是否是下拉框当前选中的模型（本地先判一次，省掉必然 409 的请求）。 */
function isLocallySelected(item) {
  const selected = selectedCheckpoint();
  if (!selected) return false;
  const normalised = selected.replace(/\\/g, '/');
  const suffix = String(item.path || '');
  return normalised.endsWith('/' + suffix) || normalised === suffix;
}

/** 状态标签文案与配色类。 */
function statusLabel(item) {
  if (item.protected) return { text: '受保护', cls: 'locked' };
  if (item.tmp) return { text: '临时', cls: 'tmp' };
  const status = String(item.status || '');
  if (status === 'running') return { text: '运行中', cls: 'running' };
  if (status === 'failed') return { text: '失败', cls: 'failed' };
  return { text: status || '可删', cls: 'free' };
}

/** 折行安全地塞一段文本到元素里。 */
function setText(node, text) {
  if (node) node.textContent = String(text == null ? '' : text);
}

/** 渲染总占用 / 临时占用 / 条数三个读数。 */
export function renderModelManagerSummary() {
  const panel = state.modelManager || {};
  const data = panel.data;
  if (!data) {
    setText(els.modelManagerTotal, '—');
    setText(els.modelManagerTmp, '—');
    setText(els.modelManagerCount, '—');
    return;
  }
  setText(els.modelManagerTotal, mbText(data.total_mb) + ' / ' + (data.count || 0) + ' 个权重');
  setText(els.modelManagerTmp, mbText(data.tmp_mb) + ' / ' + (data.tmp_jobs_count || 0) + ' 个临时作业');
  setText(
    els.modelManagerCount,
    (data.jobs_count || 0) + ' 个作业目录 · 共 ' + mbText(data.jobs_total_mb),
  );
}

/** 折叠状态同步到 DOM。 */
export function updateModelManagerCollapseUi() {
  const collapsed = !!(state.modelManager && state.modelManager.collapsed);
  if (els.modelManagerCard) els.modelManagerCard.classList.toggle('collapsed', collapsed);
  if (els.modelManagerToggleIcon) els.modelManagerToggleIcon.textContent = collapsed ? '▶' : '▼';
  if (els.modelManagerToggleText) els.modelManagerToggleText.textContent = collapsed ? '展开' : '收起';
}

/** 展开 / 收起；首次展开时触发一次加载。 */
export function toggleModelManagerCollapse() {
  initModelManagerState();
  state.modelManager.collapsed = !state.modelManager.collapsed;
  updateModelManagerCollapseUi();
  if (!state.modelManager.collapsed && !state.modelManager.data && !state.modelManager.loading) {
    void loadModels(false);
  }
}

/** 建一个按钮，点击时调用 handler。 */
function makeButton(label, cls, handler, disabled, title) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn btn-xs ' + cls;
  button.textContent = label;
  if (disabled) {
    button.disabled = true;
  } else {
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      handler();
    });
  }
  if (title) button.title = title;
  return button;
}

/** 渲染权重表：受保护/可删一眼可分，按后端顺序（mtime 倒序）。 */
export function renderModelManagerWeights() {
  const body = els.modelManagerWeightsBody;
  if (!body) return;
  const panel = state.modelManager || {};
  const data = panel.data;
  body.innerHTML = '';
  if (!data) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.className = 'empty';
    cell.textContent = panel.loading ? '正在扫描权重目录…' : (panel.error ? '读取失败：' + panel.error : '点击「刷新」加载模型列表');
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  const items = data.items || [];
  if (!items.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.className = 'empty';
    cell.textContent = '未发现 checkpoint 权重';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }

  const limit = panel.showAllWeights ? items.length : WEIGHT_ROW_LIMIT;
  // 可删的排在前面：清理时用户不用在最上面翻那些锁住的条目。
  const ordered = items.slice().sort((left, right) => Number(left.protected) - Number(right.protected));
  for (const item of ordered.slice(0, limit)) {
    const row = document.createElement('tr');
    row.className = 'data-row' + (item.protected ? ' locked-row' : '');

    const cellName = document.createElement('td');
    cellName.className = 'cell-name';
    cellName.textContent = item.job_name + '/' + item.file;
    cellName.title = item.path;
    row.appendChild(cellName);

    const cellSize = document.createElement('td');
    cellSize.textContent = mbText(item.size_mb);
    cellSize.title = String(item.size_mb) + ' MB';
    row.appendChild(cellSize);

    const cellTime = document.createElement('td');
    cellTime.textContent = item.mtime || '—';
    row.appendChild(cellTime);

    const cellState = document.createElement('td');
    const label = statusLabel(item);
    const badge = document.createElement('span');
    badge.className = 'mm-badge ' + label.cls;
    badge.textContent = label.text;
    if (item.protected_reason) badge.title = item.protected_reason;
    cellState.appendChild(badge);
    if (item.map) {
      const mapChip = document.createElement('span');
      mapChip.className = 'mm-map';
      mapChip.textContent = String(item.map).replace(/\[p\d+\]/, '').trim();
      cellState.appendChild(mapChip);
    }
    row.appendChild(cellState);

    const cellActions = document.createElement('td');
    const actions = document.createElement('div');
    actions.className = 'row-actions';
    const selectedHere = isLocallySelected(item);
    const weightTitle = selectedHere
      ? '这是下拉框当前选中的模型，不能删除'
      : (item.protected_reason || '删除该权重文件（保留 metrics/回放/日志）');
    actions.appendChild(makeButton(
      '删除权重',
      'btn-danger',
      () => removeModelWeight(item),
      item.protected || selectedHere,
      weightTitle,
    ));
    actions.appendChild(makeButton(
      '删除作业',
      'btn-danger',
      () => removeModelJob({ job_dir: item.job_dir }),
      false,
      '删除整个作业目录（权重 + 回放 + 日志）',
    ));
    cellActions.appendChild(actions);
    row.appendChild(cellActions);

    body.appendChild(row);
  }
  if (!panel.showAllWeights && items.length > WEIGHT_ROW_LIMIT && els.modelManagerMoreWeights) {
    els.modelManagerMoreWeights.hidden = false;
    els.modelManagerMoreWeights.textContent = '展开剩余 ' + (items.length - WEIGHT_ROW_LIMIT) + ' 个权重';
  } else if (els.modelManagerMoreWeights) {
    els.modelManagerMoreWeights.hidden = true;
  }
}

/** 渲染作业目录表：删整个目录腾空间更彻底。 */
export function renderModelManagerJobs() {
  const body = els.modelManagerJobsBody;
  if (!body) return;
  const panel = state.modelManager || {};
  const data = panel.data;
  body.innerHTML = '';
  if (!data) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.className = 'empty';
    cell.textContent = '—';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  const jobDirs = (data.jobs || []).slice();
  if (!jobDirs.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.className = 'empty';
    cell.textContent = '未发现作业目录';
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }

  const limit = panel.showAllJobs ? jobDirs.length : JOB_ROW_LIMIT;
  const ordered = jobDirs.slice().sort((left, right) => Number(left.protected) - Number(right.protected));
  for (const job of ordered.slice(0, limit)) {
    const row = document.createElement('tr');
    row.className = 'data-row' + (job.protected ? ' locked-row' : '');

    const cellName = document.createElement('td');
    cellName.className = 'cell-name';
    cellName.textContent = job.job_dir;
    cellName.title = job.job_dir;
    row.appendChild(cellName);

    const cellSize = document.createElement('td');
    cellSize.textContent = mbText(job.size_mb);
    row.appendChild(cellSize);

    const cellTime = document.createElement('td');
    cellTime.textContent = job.mtime || '—';
    row.appendChild(cellTime);

    const cellState = document.createElement('td');
    const label = statusLabel(job);
    const badge = document.createElement('span');
    badge.className = 'mm-badge ' + label.cls;
    badge.textContent = label.text;
    if (job.protected_reason) badge.title = job.protected_reason;
    cellState.appendChild(badge);
    if ((job.checkpoints || []).length) {
      const weightChip = document.createElement('span');
      weightChip.className = 'mm-map';
      weightChip.textContent = job.checkpoints.length + ' 个权重';
      cellState.appendChild(weightChip);
    }
    row.appendChild(cellState);

    const cellActions = document.createElement('td');
    const actions = document.createElement('div');
    actions.className = 'row-actions';
    actions.appendChild(makeButton(
      '删除整个作业',
      'btn-danger',
      () => removeModelJob(job),
      job.protected,
      job.protected_reason || '删除该作业目录（含权重、回放、日志、评估报告）',
    ));
    cellActions.appendChild(actions);
    row.appendChild(cellActions);

    body.appendChild(row);
  }
  if (!panel.showAllJobs && jobDirs.length > JOB_ROW_LIMIT && els.modelManagerMoreJobs) {
    els.modelManagerMoreJobs.hidden = false;
    els.modelManagerMoreJobs.textContent = '展开剩余 ' + (jobDirs.length - JOB_ROW_LIMIT) + ' 个作业';
  } else if (els.modelManagerMoreJobs) {
    els.modelManagerMoreJobs.hidden = true;
  }
}

/** 全量重绘（读数 + 两张表 + 错误行）。 */
export function renderModelManager() {
  initModelManagerState();
  updateModelManagerCollapseUi();
  renderModelManagerSummary();
  renderModelManagerWeights();
  renderModelManagerJobs();
  const panel = state.modelManager;
  if (els.modelManagerStatus) {
    if (panel.loading) {
      els.modelManagerStatus.textContent = '正在扫描权重目录…';
    } else if (panel.error) {
      els.modelManagerStatus.textContent = '读取失败：' + panel.error;
    } else if (panel.jobsError) {
      els.modelManagerStatus.textContent = '作业列表读取失败：' + panel.jobsError;
    } else if (panel.data) {
      els.modelManagerStatus.textContent = '根目录 ' + (panel.data.root || '—');
    } else {
      els.modelManagerStatus.textContent = '未加载';
    }
  }
  if (els.modelManagerCleanTmp) {
    const data = panel.data;
    const count = data ? data.tmp_jobs_count || 0 : 0;
    els.modelManagerCleanTmp.disabled = panel.loading || panel.busy || count <= 0;
  }
}

/** 拉取模型列表（两段式，`force` 绕过后端 25s 扫描缓存）。
 *
 * 冷扫在 Docker Desktop 的 Windows bind mount 上要 10-20s（两棵目录树各走一遍），
 * 一次请求把权重和作业一起等，面板就一直空白。所以分两段：
 *   1. `/api/checkpoints` —— 权重表，通常几秒，先渲染出来；
 *   2. `/api/models` —— 作业目录表 + 保护标记，慢的那一段，回来后再补齐。
 * 第一段失败才把面板判为失败；第二段失败只让作业区显示错误，权重表照常可用。
 */
export async function loadModels(force) {
  initModelManagerState();
  const panel = state.modelManager;
  if (panel.loading) return;
  panel.loading = true;
  panel.error = null;
  panel.jobsError = null;
  renderModelManager();

  // 第一段：权重列表。`/api/checkpoints` 与 `/api/models` 共用后端扫描缓存，
  // 先跑它能把后续 `/api/models` 的冷扫时间一并省下。
  try {
    await Api.checkpoints(!!force);
  } catch (error) {
    panel.error = error && error.message ? String(error.message) : String(error);
    panel.loading = false;
    renderModelManager();
    if (force) toast('读取模型列表失败：' + panel.error.slice(0, 160), 'error');
    return;
  }

  try {
    const payload = await Api.models(!!force);
    panel.data = payload && typeof payload === 'object' ? payload : null;
    panel.error = null;
    panel.jobsError = null;
  } catch (error) {
    panel.data = null;
    panel.jobsError = error && error.message ? String(error.message) : String(error);
    if (force) toast('读取模型列表失败：' + panel.jobsError.slice(0, 160), 'error');
  } finally {
    panel.loading = false;
    renderModelManager();
  }
}

/** 删除单个权重文件（已二次确认）。 */
export async function removeModelWeight(item) {
  initModelManagerState();
  const panel = state.modelManager;
  if (!item || panel.busy) return;
  const path = String(item.path || '');
  const size = mbText(item.size_mb);
  const confirmed = window.confirm(
    '确认删除该 checkpoint 权重？\n\n'
    + '路径：' + path + '\n'
    + '大小：' + size + '\n'
    + '作业：' + (item.job_dir || '—') + '\n\n'
    + '只会删掉这个 .pt 文件，同目录的 metrics / 回放 / 日志都保留。\n'
    + '此操作不可撤销。',
  );
  if (!confirmed) return;
  panel.busy = true;
  renderModelManager();
  try {
    const result = await Api.deleteModel(path, selectedCheckpoint());
    toast('已删除 ' + path + '（释放 ' + mbText(result && result.freed_mb) + '）', 'ok');
  } catch (error) {
    const message = error && error.message ? String(error.message) : String(error);
    toast('删除失败：' + message.slice(0, 200), 'error');
  } finally {
    panel.busy = false;
    await loadModels(true);
    void loadFeCheckpoints(true);
  }
}

/** 删除整个作业目录（已二次确认）。 */
export async function removeModelJob(job) {
  initModelManagerState();
  const panel = state.modelManager;
  if (!job || panel.busy) return;
  const path = String(job.job_dir || job.path || '');
  const size = mbText(job.size_mb);
  const weights = (job.checkpoints || []).length;
  const confirmed = window.confirm(
    '确认删除整个作业目录？\n\n'
    + '路径：' + path + '\n'
    + '大小：' + size + '\n'
    + '内含：' + (weights ? weights + ' 个权重文件 + ' : '')
    + 'metrics / 回放 / 日志 / 评估报告\n\n'
    + '这是一次递归删除，目录里所有文件都会消失，此操作不可撤销。',
  );
  if (!confirmed) return;
  panel.busy = true;
  renderModelManager();
  try {
    const result = await Api.deleteModelJob(path, selectedCheckpoint());
    toast('已删除作业目录 ' + path + '（释放 ' + mbText(result && result.freed_mb) + '）', 'ok');
  } catch (error) {
    const message = error && error.message ? String(error.message) : String(error);
    toast('删除作业失败：' + message.slice(0, 200), 'error');
  } finally {
    panel.busy = false;
    await loadModels(true);
    void loadFeCheckpoints(true);
  }
}

/** 批量清理全部临时作业（一次确认列出全部路径与总大小，然后逐个删）。 */
export async function removeTempModelJobs() {
  initModelManagerState();
  const panel = state.modelManager;
  if (!panel.data || panel.busy) return;
  const targets = (panel.data.jobs || []).filter((job) => job.tmp && !job.protected);
  if (!targets.length) {
    toast('没有可清理的临时作业', 'info');
    return;
  }
  const total = targets.reduce((sum, job) => sum + (Number(job.size_mb) || 0), 0);
  const preview = targets.slice(0, 25).map((job) => '· ' + job.job_dir).join('\n');
  const more = targets.length > 25 ? '\n… 其余 ' + (targets.length - 25) + ' 个' : '';
  const confirmed = window.confirm(
    '确认清理全部临时作业？\n\n'
    + '共 ' + targets.length + ' 个作业目录，合计 ' + mbText(total) + '：\n'
    + preview + more + '\n\n'
    + '名字里带 smoke / bench / probe / test / tmp / scratch / verify / debug 的作业\n'
    + '会被递归删除（含权重、回放、日志）。此操作不可撤销。',
  );
  if (!confirmed) return;

  panel.busy = true;
  renderModelManager();
  const failed = [];
  let freed = 0;
  let done = 0;
  try {
    for (const job of targets) {
      try {
        const result = await Api.deleteModelJob(job.job_dir, selectedCheckpoint());
        freed += Number(result && result.freed_mb) || 0;
        done += 1;
      } catch (error) {
        failed.push(job.job_dir + '：' + (error && error.message ? error.message : String(error)));
      }
    }
  } finally {
    panel.busy = false;
  }
  if (failed.length) {
    toast('清理完成 ' + done + '/' + targets.length + '，释放 ' + mbText(freed)
      + '；失败 ' + failed.length + ' 个：' + failed[0].slice(0, 120), 'error');
  } else {
    toast('已清理 ' + done + ' 个临时作业，释放 ' + mbText(freed), 'ok');
  }
  await loadModels(true);
  void loadFeCheckpoints(true);
}
