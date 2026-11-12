/**
 * ppoCharts.js — 实机 PPO 训练 · 六联图绘制（返回/胜率/HP/奖励/熵/耗时）。
 */

import { DASH_COLORS, drawDashBars, drawDashLine } from './control.js';
import { fePpoDurations, fePpoEvalEvents, fePpoSeriesRange, fePpoUpdateEvents } from './ppo.js';
import { els, state } from './state.js';
import { num, numOrNull } from './util.js';

export function feDrawPpoCharts() {
  const ppo = state.fe.ppo;
  const updates = fePpoUpdateEvents();
  const evals = fePpoEvalEvents();

  // 1. 对战得分趋势（训练得分 vs 实战盲测得分）
  const trainReturns = updates.map((update) => {
    const val = numOrNull(update.rolling_return) !== null
      ? num(update.rolling_return, 0)
      : num(update.mean_return, 0);
    return { x: num(update.update, 0), y: val };
  });
  const evalReturns = evals
    .filter((ev) => numOrNull(ev.mean_return) !== null)
    .map((ev) => ({ x: num(ev.update, 0), y: num(ev.mean_return, 0) }));

  const returnSeries = [{ label: '训练得分', color: DASH_COLORS.cyan, points: trainReturns }];
  if (evalReturns.length) {
    returnSeries.push({ label: '实战盲测', color: DASH_COLORS.green, points: evalReturns });
  }
  const returnRange = fePpoSeriesRange([trainReturns, evalReturns]);
  drawDashLine(els.fePpoChartReturn, {
    series: returnSeries,
    yMin: returnRange.yMin,
    yMax: returnRange.yMax,
    yFormatter: (value) => (value >= 0 ? '+' : '') + value.toFixed(2),
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待 update / eval 事件…',
  });

  // 2. 实战盲测与训练胜率
  const trainWins = updates
    .filter((u) => num(u.episodes, 0) > 0 || num(u.total_episodes, 0) > 0 || !!u.has_completed)
    .map((update) => ({
      x: num(update.update, 0),
      y: num(numOrNull(update.rolling_win_rate) !== null ? update.rolling_win_rate : update.win_rate, 0) * 100,
    }));
  const evalWins = evals.map((ev) => ({ x: num(ev.update, 0), y: num(ev.win_rate, 0) * 100 }));

  const winSeries = [];
  if (evalWins.length) {
    winSeries.push({ label: '盲测胜率', color: DASH_COLORS.green, points: evalWins });
  }
  if (trainWins.length) {
    winSeries.push({ label: '训练胜率', color: DASH_COLORS.cyan, points: trainWins });
  }
  drawDashLine(els.fePpoChartWin, {
    series: winSeries.length ? winSeries : [{ color: DASH_COLORS.green, points: evalWins }],
    yMin: 0,
    yMax: 100,
    yFormatter: (value) => Math.round(value) + '%',
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待对战胜率数据…',
  });

  // 3. 双方剩余血量比（战力走势）
  const ownHp = evals
    .filter((ev) => numOrNull(ev.mean_own_hp_frac) !== null)
    .map((ev) => ({ x: num(ev.update, 0), y: num(ev.mean_own_hp_frac, 0) * 100 }));
  const enemyHp = evals
    .filter((ev) => numOrNull(ev.mean_enemy_hp_frac) !== null)
    .map((ev) => ({ x: num(ev.update, 0), y: num(ev.mean_enemy_hp_frac, 0) * 100 }));
  drawDashLine(els.fePpoChartHp, {
    series: [
      { label: '我方HP%', color: DASH_COLORS.own, points: ownHp },
      { label: '敌方HP%', color: DASH_COLORS.enemy, points: enemyHp },
    ],
    yMin: 0,
    yMax: 100,
    yFormatter: (value) => Math.round(value) + '%',
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待评估对局产出战力血量比…',
  });

  // 4. 得分构成分解（Reward Breakdown）
  const killsPoints = updates
    .filter((u) => u.reward_components && numOrNull(u.reward_components.kills) !== null)
    .map((u) => ({ x: num(u.update, 0), y: num(u.reward_components.kills, 0) }));
  const engPoints = updates
    .filter((u) => u.reward_components && numOrNull(u.reward_components.engagement) !== null)
    .map((u) => ({ x: num(u.update, 0), y: num(u.reward_components.engagement, 0) }));
  const lossPoints = updates
    .filter((u) => u.reward_components && numOrNull(u.reward_components.losses) !== null)
    .map((u) => ({ x: num(u.update, 0), y: num(u.reward_components.losses, 0) }));
  const rewardRange = fePpoSeriesRange([killsPoints, engPoints, lossPoints]);
  drawDashLine(els.fePpoChartReward, {
    series: [
      { label: '击杀得分', color: DASH_COLORS.green, points: killsPoints },
      { label: '推进交火', color: DASH_COLORS.cyan, points: engPoints },
      { label: '战损扣分', color: DASH_COLORS.enemy, points: lossPoints },
    ],
    yMin: rewardRange.yMin,
    yMax: rewardRange.yMax,
    yFormatter: (value) => (value >= 0 ? '+' : '') + value.toFixed(2),
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待奖励分项统计…',
  });

  // 5. 策略探索与损失
  const entropy = updates.map((update) => ({ x: num(update.update, 0), y: num(update.entropy, 0) }));
  const policyLoss = updates.map((update) => ({ x: num(update.update, 0), y: num(update.policy_loss, 0) }));
  const entropyRange = fePpoSeriesRange([entropy, policyLoss]);
  drawDashLine(els.fePpoChartEntropy, {
    series: [
      { color: DASH_COLORS.cyan, points: entropy },
      { color: DASH_COLORS.amber, points: policyLoss },
    ],
    yMin: entropyRange.yMin,
    yMax: entropyRange.yMax,
    yFormatter: (value) => value.toFixed(2),
    xFormatter: (value) => '#' + Math.round(value),
    emptyText: '等待 update 事件…',
  });

  // 6. 每轮耗时柱状图
  const durations = fePpoDurations(updates);
  drawDashBars(els.fePpoChartDuration, {
    bars: updates.map((update, index) => ({
      label: '#' + num(update.update, 0),
      value: Math.max(0.01, durations[index] || 0),
      color: DASH_COLORS.cyan,
    })),
    yFormatter: (value) => value.toFixed(1) + 's',
    valueFormatter: (value) => value.toFixed(1),
    xFormatter: (label) => String(label),
    emptyText: '等待 update 事件…',
  });

  const evaluations = (ppo.job && Array.isArray(ppo.job.evaluations)) ? ppo.job.evaluations : [];
  const lastEval = evaluations.length ? evaluations[evaluations.length - 1] : null;
  if (els.fePpoEvalHint) {
    els.fePpoEvalHint.textContent = lastEval
      ? 'eval #' + num(lastEval.update, 0) + ' · win ' + (num(lastEval.win_rate, 0) * 100).toFixed(0)
        + '% · timeout ' + (num(lastEval.timeout_rate, 0) * 100).toFixed(0)
        + '% · return ' + num(lastEval.mean_return, 0).toFixed(2)
      : '—';
  }
}
