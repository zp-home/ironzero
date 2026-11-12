/**
 * model.js — 实机 PPO · 策略 UI 同步 / 奖励分解 / 战略指挥面板。
 */

import { loadFeCheckpoints } from './checkpoint.js';
import { feSetText } from './control.js';
import { fePpoUpdateEvents } from './ppo.js';
import { els, state } from './state.js';
import { num } from './util.js';

export function feSyncTrainPolicyUi() {
  const isModel = !!(els.feTrainPolicy && els.feTrainPolicy.value === 'model');
  if (els.feTrainCheckpointField) els.feTrainCheckpointField.hidden = !isModel;
}

export function feSyncPpoConfigUi() {
  const mapChoice = (els.fePpoMap && els.fePpoMap.value) || 'IronZero25';
  const isIronZero = mapChoice.toLowerCase().startsWith('ironzero');
  const isArtillery = mapChoice.toLowerCase().includes('artillery');

  if (els.fePpoUnitsField) {
    els.fePpoUnitsField.hidden = !isIronZero;
  }
  if (els.fePpoMapHint) {
    if (isIronZero) {
      els.fePpoMapHint.textContent = '竞技场纯微操：开局双方自动生成指定数量的纯坦克部队，直接进行集火与拉扯对决。';
    } else if (isArtillery) {
      els.fePpoMapHint.textContent = '原版战役挑战：开局双方预置 22 辆火炮部队与阵地建筑，直接爆发大兵团炮火决战。';
    } else {
      els.fePpoMapHint.textContent = '原版遭遇战规则：初始部队严格遵循原版地图设定（双方各 1 个主基地 + 1 辆建造车），无预置坦克部队。当前微操模型不包含采矿造工厂等宏观运营指令，双方开局不会产生部队交火。若要训练坦克走位集火，强烈建议选用上方【对称竞技场】或原版【Artillery】！';
    }
  }

  if (els.fePpoUnits) {
    if (mapChoice === 'IronZero25') els.fePpoUnits.value = '25';
    else if (mapChoice === 'IronZero20') els.fePpoUnits.value = '20';
    else if (mapChoice === 'IronZero15') els.fePpoUnits.value = '15';
    else if (mapChoice === 'IronZero10') els.fePpoUnits.value = '10';
    else if (mapChoice === 'IronZero5') els.fePpoUnits.value = '5';
    else if (isArtillery) els.fePpoUnits.value = '22';
    else {
      els.fePpoUnits.value = '2';
    }
  }
  const initMode = (els.fePpoInitMode && els.fePpoInitMode.value) || 'warmstart';
  if (els.fePpoCheckpointField) {
    els.fePpoCheckpointField.hidden = (initMode !== 'checkpoint');
  }
  if (initMode === 'checkpoint' && (!state.fe.checkpointList || !state.fe.checkpointList.length)) {
    void loadFeCheckpoints(true);
  }
  if (els.fePpoInitHint) {
    if (initMode === 'warmstart') {
      els.fePpoInitHint.textContent = '预热策略基于真实对战规则蒸馏（BC），坦克开局会自动集火与编队推进，可避免从零探索在前几百轮毫无击杀的问题。';
    } else if (initMode === 'checkpoint') {
      els.fePpoInitHint.textContent = '在上方选定的 Checkpoint 权重基础上继续进行 PPO 在环对抗强化学习。';
    } else {
      els.fePpoInitHint.textContent = '从完全随机参数的神经网络从零探索。RTS 微操动作空间庞大，初始几十轮胜率提升较慢。';
    }
  }
}

export function feSyncInferPolicyUi() {
  const isModel = !!(els.fePolicy && els.fePolicy.value === 'model');
  if (els.feCheckpointField) els.feCheckpointField.hidden = !isModel;
  if (els.feModelNote) els.feModelNote.hidden = !isModel;
  if (isModel && !(state.fe.checkpointList || []).length && els.feInferStatus) {
    els.feInferStatus.textContent = '没有可用 checkpoint：先到「实机训练」页跑一次 PPO';
    els.feInferStatus.className = 'hint-line bad';
  }
}

export function sandboxStrategyLabel(sandbox) {
  const infer = sandbox.source === 'infer';
  const realReplay = sandbox.checkpoint === 'rwpp-live';
  if (!infer && !realReplay) return '';
  if (infer && sandbox.policy === 'model') {
    const name = sandbox.modelCheckpoint ? String(sandbox.modelCheckpoint).replace(/\\/g, '/') : '';
    return '模型策略' + (name ? ' · ' + name : '') + ' · ';
  }
  return '脚本策略 · ';
}

export const REWARD_COMPONENT_LABELS = {
  potential: '势能 potential',
  engagement: '交战 engagement',
  strategy: '战略 strategy',
  kills: '击杀 kills',
  losses: '损失 losses',
  economy: '采矿经济 economy',
  factories: '军工建造 factories',
  production: '战车出厂 production',
  building_loss: '建筑折损 building_loss',
  invalid: '非法 invalid',
  idle: '空闲 idle',
  terminal: '终局 terminal',
  timeout: '超时 timeout',
};

export function feRenderPpoReward() {
  const box = els.fePpoReward;
  if (!box) return;
  const updates = fePpoUpdateEvents();
  const latest = updates.length ? updates[updates.length - 1] : null;
  const components = latest && latest.reward_components && typeof latest.reward_components === 'object'
    ? latest.reward_components
    : null;
  const keys = components ? Object.keys(components) : [];
  if (!components || !keys.length) {
    box.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = '完成一局后出现';
    box.appendChild(empty);
    feSetText(els.fePpoRewardChip, '—');
    return;
  }
  const ordered = [];
  for (const key of Object.keys(REWARD_COMPONENT_LABELS)) {
    if (Object.prototype.hasOwnProperty.call(components, key)) ordered.push(key);
  }
  for (const key of keys) {
    if (ordered.indexOf(key) < 0) ordered.push(key);
  }
  let total = 0;
  let maxAbs = 0.0001;
  for (const key of ordered) {
    const value = num(components[key], 0);
    total += value;
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  box.innerHTML = '';
  for (const key of ordered) {
    const value = num(components[key], 0);
    const row = document.createElement('div');
    row.className = 'reward-row';
    const label = document.createElement('span');
    label.className = 'reward-label';
    label.textContent = REWARD_COMPONENT_LABELS[key] || key;
    const bar = document.createElement('span');
    bar.className = 'reward-bar';
    const fill = document.createElement('i');
    fill.className = value >= 0 ? 'pos' : 'neg';
    fill.style.width = Math.max(2, Math.round((Math.abs(value) / maxAbs) * 100)) + '%';
    bar.appendChild(fill);
    const val = document.createElement('span');
    val.className = 'reward-value ' + (value >= 0 ? 'pos' : 'neg');
    val.textContent = (value >= 0 ? '+' : '') + value.toFixed(3);
    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(val);
    box.appendChild(row);
  }
  feSetText(els.fePpoRewardChip, (total >= 0 ? '+' : '') + total.toFixed(3));
}

export function feFormatStrategyValue(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value)) return value.map((item) => feFormatStrategyValue(item)).join(', ');
  if (typeof value === 'object') {
    return Object.keys(value).map((key) => key + '=' + feFormatStrategyValue(value[key])).join(', ');
  }
  return String(value);
}

export function feRenderPpoStrategy() {
  const box = els.fePpoStrategy;
  if (!box) return;
  const ppo = state.fe.ppo;
  const events = (ppo.events || []).filter((event) => event && event.kind === 'strategy');
  const latest = events.length ? events[events.length - 1] : null;
  if (!latest) {
    box.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = ppo.useLlm
      ? '已启用 LLM 战略层，等待第一条 directive…'
      : '未启用战略层（在 PPO 表单勾选「启用 LLM 战略层」）';
    box.appendChild(empty);
    feSetText(els.fePpoStrategyChip, ppo.useLlm ? '等待中' : '未启用');
    feSetText(els.fePpoStrategySource, '—');
    return;
  }
  const payload = latest.payload || {};
  const source = String(payload.source || 'rule');
  feSetText(els.fePpoStrategyChip, source === 'llm' ? 'LLM' : '规则');
  feSetText(els.fePpoStrategySource, String(payload.created_at || '') + ' · directive #' + num(latest.seq, 0));
  const rows = [
    ['intent', payload.intent],
    ['attack_region', payload.attack_region],
    ['rally_point', payload.rally_point],
    ['retreat_hp_threshold', payload.retreat_hp_threshold],
    ['priority_target', payload.priority_target],
    ['notes', payload.notes],
  ];
  box.innerHTML = '';
  for (const pair of rows) {
    const row = document.createElement('div');
    row.className = 'strategy-kv';
    const label = document.createElement('span');
    label.textContent = pair[0];
    const value = document.createElement('b');
    value.textContent = feFormatStrategyValue(pair[1]);
    row.appendChild(label);
    row.appendChild(value);
    box.appendChild(row);
  }
  if (ppo.useLlm && source !== 'llm') {
    const warn = document.createElement('div');
    warn.className = 'strategy-warn';
    warn.textContent = 'LLM 未配置（缺 IRONZERO_LLM_API_KEY）或调用失败，已退回规则策略。';
    box.appendChild(warn);
  }
  const logLines = events.slice(-5).map((event) => {
    const data = event.payload || {};
    return '#' + num(event.seq, 0) + ' ' + String(data.created_at || '')
      + ' · ' + String(data.intent || '—')
      + ' · attack=' + feFormatStrategyValue(data.attack_region)
      + ' · source=' + String(data.source || 'rule');
  });
  if (logLines.length) {
    const log = document.createElement('div');
    log.className = 'strategy-log';
    log.textContent = logLines.join('\n');
    box.appendChild(log);
  }
}

/**
 * state.fe 的默认字段（原为顶层 Object.assign 语句）。
 * ES Module 存在循环依赖，顶层求值顺序不保证与原文件一致，故由 main.js 在
 * 所有模块求值完成后按原顺序调用本函数。语句内容逐字保留。
 */
export function initFeModelState() {
  Object.assign(state.fe, {
    checkpointList: [],
    checkpointsLoaded: false,
    checkpointsLoading: false,
  });

  Object.assign(state.fe.ppo, {
    evalEvery: 0,
    evalEpisodes: 1,
    useLlm: false,
  });
}
