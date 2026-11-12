/**
 * demo.js — 演示数据：后端不可用时构造假的事件流/日志/对局供界面预览（界面明确标注 DEMO）。
 */

import { normalizeFrame } from './normalize.js';
import { clamp, fmtDateTime, fmtShortTime, mulberry32, num, numOrNull } from './util.js';

export const DEMO = {
  ready: false,
  runs: [],
  events: {},
  details: {},
  checkpoints: [],
  replays: [],
  battles: {},
};

export function buildDemoEvents(options) {
  const opts = options || {};
  const seed = num(opts.seed, 11);
  const updates = Math.max(8, Math.round(num(opts.updates, 96)));
  const evalEvery = Math.max(1, Math.round(num(opts.evalEvery, 8)));
  const random = mulberry32(seed * 7919 + 13);
  const noise = (scale) => (random() - 0.5) * scale;
  const events = [];
  let seq = 0;
  const push = (kind, payload) => {
    seq += 1;
    events.push({ seq: seq, kind: kind, payload: payload });
  };

  push('start', {
    config: {
      total_updates: updates,
      num_envs: 8,
      rollout_steps: 64,
      epochs: 4,
      lr: 0.0003,
      seed: seed,
      eval_every: evalEvery,
      eval_episodes: 8,
      opponent: 'scripted',
      hidden: 256,
    },
    device: 'cpu',
    random_baseline: { episodes: 8, win_rate: 0.0, mean_return: -1.31 },
    created_at: fmtDateTime(Date.now() - updates * 3400),
  });

  for (let update = 1; update <= updates; update += 1) {
    const winTrend = clamp(0.24 + 0.52 * (1 - Math.exp(-update / 30)) + noise(0.04), 0.02, 0.97);
    const returnTrend = -1.05 + 2.3 * (1 - Math.exp(-update / 26)) + noise(0.09);
    const entropy = 1.52 * Math.exp(-update / 55) + 0.74 + noise(0.02);
    const policyLoss = 0.32 * Math.exp(-update / 38) + 0.046 + random() * 0.014;
    const valueLoss = 0.9 * Math.exp(-update / 42) + 0.09 + random() * 0.025;
    push('update', {
      update: update,
      policy_loss: Number(policyLoss.toFixed(5)),
      value_loss: Number(valueLoss.toFixed(5)),
      entropy: Number(entropy.toFixed(4)),
      clip_fraction: Number((random() * 0.12).toFixed(4)),
      episodes: 8 * 64,
      mean_return: Number(returnTrend.toFixed(4)),
      win_rate: Number(winTrend.toFixed(4)),
      elapsed_s: Number((update * 3.4 + random()).toFixed(2)),
    });
    if (update % evalEvery === 0) {
      const evalWin = clamp(winTrend + noise(0.035), 0, 1);
      push('eval', {
        update: update,
        episodes: 8,
        win_rate: Number(evalWin.toFixed(4)),
        loss_rate: Number((1 - evalWin).toFixed(4)),
        timeout_rate: 0,
        mean_return: Number((returnTrend + noise(0.06)).toFixed(4)),
        mean_own_hp_frac: Number(clamp(evalWin * 0.92, 0, 1).toFixed(4)),
        mean_enemy_hp_frac: Number(clamp(1 - evalWin * 0.92, 0, 1).toFixed(4)),
        mean_decisions: Math.round(28 + random() * 8),
      });
    }
    if (update % 4 === 0) {
      push('log', {
        line: '[train] update ' + update + '/' + updates
          + '  return=' + returnTrend.toFixed(3)
          + '  win=' + winTrend.toFixed(2)
          + '  entropy=' + entropy.toFixed(2)
          + '  sps=' + Math.round(300 + random() * 90),
      });
    }
  }

  const finalEval = events.filter((event) => event.kind === 'eval').pop();
  push('done', {
    final_eval: finalEval ? finalEval.payload : null,
    total_seconds: Number((updates * 3.3).toFixed(1)),
    checkpoint: 'runs/demo/checkpoint.pt',
  });
  return events;
}

export function demoDetailFromEvents(config, events) {
  let latest = {};
  let startPayload = null;
  let finalEval = null;
  const evals = [];
  for (const event of events) {
    const payload = event.payload || {};
    if (event.kind === 'update' && numOrNull(payload.update) !== null) latest = Object.assign({}, latest, payload);
    else if (event.kind === 'eval') evals.push(payload);
    else if (event.kind === 'start') startPayload = payload;
    else if (event.kind === 'done') finalEval = payload.final_eval || finalEval;
  }
  if (evals.length) {
    const last = evals[evals.length - 1];
    latest = Object.assign({}, latest, {
      win_rate: last.win_rate,
      mean_return: last.mean_return,
    });
  }
  const startConfig = (startPayload && startPayload.config) || {};
  const detail = {
    id: config.id,
    name: config.name,
    kind: 'train',
    status: config.status,
    created_at: fmtDateTime(Date.now() - config.createdAgo),
    config: Object.assign({}, startConfig, {
      updates: config.updates,
      total_updates: config.updates,
      envs: 8,
      rollout_steps: 64,
      opponent: 'scripted',
      seed: config.seed,
      eval_every: config.evalEvery,
      eval_episodes: 8,
    }),
    latest: latest,
    evals: evals,
    final_eval: finalEval,
    events: events.length,
    log_tail: buildDemoLogTail(evals, config),
  };
  return detail;
}

export function buildDemoLogTail(evals, config) {
  const lines = [
    '[' + fmtShortTime(fmtDateTime(Date.now() - config.createdAgo)) + '] ironzero.train | device=cpu envs=8 updates=' + config.updates,
    '[' + fmtShortTime(fmtDateTime(Date.now() - config.createdAgo)) + '] opponent=scripted seed=' + config.seed + ' eval_every=' + config.evalEvery,
  ];
  const recent = evals.slice(-3);
  for (const item of recent) {
    lines.push('eval update=' + item.update + ' win_rate=' + Number(item.win_rate).toFixed(3) + ' mean_return=' + Number(item.mean_return).toFixed(3));
  }
  return lines;
}

export function buildDemoBattle(options) {
  const opts = options || {};
  const seed = num(opts.seed, 11);
  const ownPower = num(opts.ownPower, 1);
  const random = mulberry32(seed * 104729 + 7);
  const MAP = { width: 1000, height: 1000 };
  const TEAM_SIZE = 8;
  const MAX_HP = 210;
  const TICK_RATE = 60;
  const DECISION_INTERVAL = 30;
  const MAX_DECISIONS = 150;
  const units = [];

  for (let index = 0; index < TEAM_SIZE; index += 1) {
    const group = Math.floor(index / 2);
    const baseY = 120 + group * 220;
    units.push({
      id: index + 1, team: 0, group: group,
      x: 170 + (index % 2) * 30 + random() * 18,
      y: baseY + (random() - 0.5) * 40,
      hp: MAX_HP, maxHp: MAX_HP, alive: true, target: null, cooldown: 0,
    });
    units.push({
      id: index + 9, team: 1, group: group,
      x: 830 - (index % 2) * 30 - random() * 18,
      y: baseY + (random() - 0.5) * 40,
      hp: MAX_HP, maxHp: MAX_HP, alive: true, target: null, cooldown: 0,
    });
  }

  const frames = [];
  const eventLog = [];
  let result = null;
  const stepMove = 16;
  const engageRange = 230;

  for (let decision = 1; decision <= MAX_DECISIONS; decision += 1) {
    const shotPlans = [];
    for (const unit of units) {
      if (!unit.alive) continue;
      const foes = units.filter((other) => other.alive && other.team !== unit.team);
      if (!foes.length) break;
      let target = null;
      let bestDistance = Infinity;
      for (const foe of foes) {
        const distance = Math.hypot(foe.x - unit.x, foe.y - unit.y);
        if (distance < bestDistance) {
          bestDistance = distance;
          target = foe;
        }
      }
      if (!target) break;
      unit.target = target.id;
      const angle = Math.atan2(target.y - unit.y, target.x - unit.x);
      if (bestDistance > engageRange) {
        unit.x = clamp(unit.x + Math.cos(angle) * stepMove, 0, MAP.width);
        unit.y = clamp(unit.y + Math.sin(angle) * stepMove, 0, MAP.height);
      }
      if (unit.cooldown > 0) unit.cooldown -= 1;
      if (bestDistance <= engageRange + 24 && unit.cooldown <= 0) {
        unit.cooldown = 4;
        shotPlans.push({ source: unit, target: target });
      }
    }

    const events = [];
    for (const plan of shotPlans) {
      const target = plan.target;
      if (!target.alive) continue;
      const power = target.team === 1 ? ownPower : 1;
      const amount = Math.round(25 * power * (0.9 + random() * 0.2));
      target.hp = Math.max(0, target.hp - amount);
      events.push({
        type: 'shot', source: plan.source.id, target: target.id,
        team: plan.source.team, amount: amount, tick: decision * DECISION_INTERVAL,
      });
      if (target.hp <= 0) {
        target.alive = false;
        events.push({
          type: 'kill', source: plan.source.id, target: target.id, team: plan.source.team,
          x: Number(target.x.toFixed(1)), y: Number(target.y.toFixed(1)), tick: decision * DECISION_INTERVAL,
        });
      }
    }

    const ownAlive = units.filter((unit) => unit.team === 0 && unit.alive);
    const enemyAlive = units.filter((unit) => unit.team === 1 && unit.alive);
    if (!enemyAlive.length) result = 'victory';
    else if (!ownAlive.length) result = 'defeat';

    const ownHp = units.filter((unit) => unit.team === 0).reduce((sum, unit) => sum + Math.max(0, unit.hp), 0) / (TEAM_SIZE * MAX_HP);
    const enemyHp = units.filter((unit) => unit.team === 1).reduce((sum, unit) => sum + Math.max(0, unit.hp), 0) / (TEAM_SIZE * MAX_HP);
    const frame = {
      episode: 0,
      decision: decision,
      tick: decision * DECISION_INTERVAL,
      time: Number(((decision * DECISION_INTERVAL) / TICK_RATE).toFixed(3)),
      reward: Number(((ownHp - enemyHp) / 8).toFixed(4)),
      units: units.filter((unit) => unit.alive).map((unit) => ({
        id: unit.id, team: unit.team, group: unit.group,
        x: Number(unit.x.toFixed(1)), y: Number(unit.y.toFixed(1)),
        hp: Number(unit.hp.toFixed(1)), max_hp: unit.maxHp,
        order: unit.target === null ? 'attack_move' : 'attack_target', target: unit.target,
      })),
      events: events,
      stats: {
        own_hp_frac: Number(ownHp.toFixed(4)),
        enemy_hp_frac: Number(enemyHp.toFixed(4)),
        own_alive: ownAlive.length,
        enemy_alive: enemyAlive.length,
      },
      result: result,
      reward_total: Number((((ownHp - enemyHp) / 4)).toFixed(4)),
    };
    frames.push(normalizeFrame(frame));
    for (const event of events) {
      eventLog.push({
        kind: event.type,
        time: frame.time,
        text: event.type === 'shot'
          ? '命中 #' + event.source + ' → #' + event.target + '  -' + Math.round(event.amount) + ' HP'
          : '击毁 #' + event.target + '（#' + event.source + '）',
      });
    }
    if (result) break;
  }

  const lastFrame = frames.length ? frames[frames.length - 1] : null;
  const totalReward = lastFrame ? numOrNull(lastFrame.rewardTotal) : null;
  const meta = {
    schema: 'ironzero.replay/1',
    checkpoint: 'runs/demo/checkpoint.pt',
    opponent: 'scripted',
    seed: seed,
    episodes: 1,
    mode: 'greedy',
    created_at: fmtDateTime(Date.now()),
    map: MAP,
    tick_rate: TICK_RATE,
    decision_interval: DECISION_INTERVAL,
    units_per_team: TEAM_SIZE,
    groups: 4,
    episode: 0,
    result: result || 'timeout',
    return: totalReward === null ? 0 : totalReward,
    decisions: frames.length,
  };
  return { meta: meta, frames: frames, eventLog: eventLog };
}

export function initDemoData() {
  if (DEMO.ready) return;
  DEMO.ready = true;
  const configs = [
    { id: 'demo-microduel-a', name: 'micro_duel · 演示运行 A', status: 'running', seed: 11, updates: 96, evalEvery: 8, createdAgo: 5 * 60 * 1000 },
    { id: 'demo-microduel-b', name: 'micro_duel · 演示运行 B', status: 'succeeded', seed: 29, updates: 64, evalEvery: 8, createdAgo: 42 * 60 * 1000 },
  ];
  for (const config of configs) {
    const events = buildDemoEvents(config);
    const detail = demoDetailFromEvents(config, events);
    DEMO.events[config.id] = events;
    DEMO.details[config.id] = detail;
    DEMO.runs.push({
      id: detail.id,
      name: detail.name,
      kind: 'train',
      status: detail.status,
      created_at: detail.created_at,
      config: detail.config,
      latest: detail.latest,
      final_eval: detail.final_eval,
      eval_count: detail.evals.length,
    });
  }

  DEMO.checkpoints = [
    { name: 'demo/micro_duel/checkpoint.pt', path: 'runs/demo/micro_duel/checkpoint.pt', update: 96, eval: { win_rate: 0.76 }, mtime: fmtDateTime(Date.now()), size_mb: 0.66 },
    { name: 'demo/micro_probe/checkpoint.pt', path: 'runs/demo/micro_probe/checkpoint.pt', update: 64, eval: { win_rate: 0.63 }, mtime: fmtDateTime(Date.now() - 3600000), size_mb: 0.65 },
  ];

  const battleA = buildDemoBattle({ seed: 11, ownPower: 1.12 });
  const battleB = buildDemoBattle({ seed: 23, ownPower: 0.82 });
  DEMO.battles['demo-replay-a'] = battleA;
  DEMO.battles['demo-replay-b'] = battleB;
  DEMO.replays = [
    {
      id: 'demo-replay-a', name: 'demo-victory-scripted-seed11.json',
      checkpoint: battleA.meta.checkpoint, opponent: battleA.meta.opponent, seed: 11,
      result: battleA.meta.result, ret: battleA.meta.return, episodes: 1,
      frames: battleA.frames.length, created_at: battleA.meta.created_at,
    },
    {
      id: 'demo-replay-b', name: 'demo-defeat-scripted-seed23.json',
      checkpoint: battleB.meta.checkpoint, opponent: battleB.meta.opponent, seed: 23,
      result: battleB.meta.result, ret: battleB.meta.return, episodes: 1,
      frames: battleB.frames.length, created_at: battleB.meta.created_at,
    },
  ];
}

export function demoReplayPayload(id) {
  initDemoData();
  const battle = DEMO.battles[id];
  if (!battle) return null;
  return { meta: battle.meta, frames: battle.frames.map((frame) => frame), eventLog: battle.eventLog };
}
