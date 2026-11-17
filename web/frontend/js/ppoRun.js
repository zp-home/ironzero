/**
 * ppoRun.js — 实机 PPO 训练 · 调度：启动/停止、轮询、任务恢复。
 */

import { Api } from './api.js';
import { toast } from './dashboard.js';
import { difficultyLabel, feCatalogMap, feMapCatalog, loadFeMapCatalog } from './feMapCatalog.js';
import { handleFeError } from './feInferUi.js';
import { feMapList } from './feMaps.js';
import { feApplyTrainMode, feRenderPpoDashboard, feSetPpoRunning, feSetPpoStatus } from './ppo.js';
import { refreshRwppStatus } from './rwpp.js';
import { els, state } from './state.js';
import { clampInt, num, numOrNull } from './util.js';

export function startFePpoPolling() {
  stopFePpoPolling();
  state.fe.ppo.timer = setInterval(fePollPpo, 1500);
  void fePollPpo();
}

export function stopFePpoPolling() {
  const ppo = state.fe.ppo;
  if (ppo.timer) {
    clearInterval(ppo.timer);
    ppo.timer = null;
  }
}

export async function fePollPpo() {
  const ppo = state.fe.ppo;
  if (!ppo.jobId || ppo.polling) return;
  ppo.polling = true;
  try {
    const payload = await Api.rwppPpoEvents(ppo.jobId, ppo.cursor || 0);
    const events = Array.isArray(payload && payload.events) ? payload.events : [];
    for (const event of events) {
      if (!event || numOrNull(event.seq) === null) continue;
      const seq = num(event.seq, 0);
      ppo.cursor = Math.max(ppo.cursor || 0, seq);
      if (!ppo.events.some((item) => item && item.seq === seq)) ppo.events.push(event);
    }
    ppo.events.sort((left, right) => num(left.seq, 0) - num(right.seq, 0));
    const job = payload && payload.job ? payload.job : null;
    if (job) {
      ppo.job = job;
      if (numOrNull(job.updates_total) !== null) ppo.updatesTotal = num(job.updates_total, ppo.updatesTotal);
      if (numOrNull(job.updates_done) !== null) ppo.updatesDone = num(job.updates_done, ppo.updatesDone);
      if (job.latest_update) ppo.latest = job.latest_update;
      if (Array.isArray(job.evaluations)) ppo.evaluations = job.evaluations;
      if (numOrNull(job.frame_ms) !== null) ppo.frameMs = num(job.frame_ms, ppo.frameMs);
      if (numOrNull(job.vision_radius) !== null) ppo.visionRadius = num(job.vision_radius, ppo.visionRadius);
      if (numOrNull(job.eval_every) !== null) ppo.evalEvery = num(job.eval_every, ppo.evalEvery);
      if (numOrNull(job.eval_episodes) !== null) ppo.evalEpisodes = num(job.eval_episodes, ppo.evalEpisodes);
      if (job.use_llm !== undefined && job.use_llm !== null) ppo.useLlm = !!job.use_llm;
      // The *resolved* starting weights reported by the host service, so the
      // console can state what training actually began from instead of echoing
      // the form. A warmstart resolves to a different BC file per tank count.
      if (job.init_checkpoint !== undefined) ppo.initCheckpoint = job.init_checkpoint || null;
      if (job.init_mode !== undefined) ppo.initMode = job.init_mode || null;
      // Resolved scenario: the host reports the map it actually loaded and the
      // difficulty it really pinned. Echoing the form would hide a mismatch, and
      // a wrong pixel size does not fail — it just skews every observation.
      if (job.map_name) ppo.mapName = String(job.map_name);
      if (job.map_path) ppo.mapPath = String(job.map_path);
      if (numOrNull(job.map_width) !== null) ppo.mapWidth = num(job.map_width, ppo.mapWidth);
      if (numOrNull(job.map_height) !== null) ppo.mapHeight = num(job.map_height, ppo.mapHeight);
      if (numOrNull(job.enemy_ai_difficulty) !== null) {
        ppo.enemyAiDifficulty = num(job.enemy_ai_difficulty, ppo.enemyAiDifficulty);
      }
    }
    const doneEvent = ppo.events.filter((item) => item.kind === 'done').pop();
    if (doneEvent) ppo.donePayload = doneEvent.payload || {};
    const status = (job && job.status) || (payload && payload.status) || 'running';
    ppo.status = status;
    feRenderPpoDashboard();
    if (status !== 'running') {
      stopFePpoPolling();
      feSetPpoRunning(false);
      if (status === 'done') {
        const seconds = num(ppo.donePayload && ppo.donePayload.total_seconds, num(ppo.job && ppo.job.elapsed, 0));
        toast('PPO 训练完成（' + Math.round(seconds) + 's）', 'ok');
      } else {
        const stopped = ppo.stopRequested;
        const message = String((ppo.job && ppo.job.error) || '未知错误').slice(0, 140);
        toast(stopped ? 'PPO 训练已停止' : 'PPO 训练失败：' + message, stopped ? 'ok' : 'error');
      }
      await refreshRwppStatus(true);
      feRenderPpoDashboard();
    }
  } catch (error) {
    if (error && error.status === 404) {
      stopFePpoPolling();
      ppo.status = 'failed';
      feSetPpoRunning(false);
      toast('PPO 任务不存在或已结束', 'error');
      feRenderPpoDashboard();
    } else if (error && error.status === 503) {
      stopFePpoPolling();
      ppo.status = 'failed';
      feSetPpoRunning(false);
      handleFeError(error, 'PPO 训练');
      feRenderPpoDashboard();
    }
  } finally {
    ppo.polling = false;
  }
}

export async function feStartPpo() {
  const ppo = state.fe.ppo;
  if (ppo.status === 'running') {
    toast('PPO 训练正在进行中，请等待或先停止', 'error');
    return;
  }
  if (state.mode === 'demo') {
    toast('演示模式下无法连接真机服务', 'error');
    return;
  }
  // Every default below must equal the corresponding `value=` in index.html.
  // They used to disagree (updates 20 vs 40, rollout 20 vs 64, ticks 3600 vs
  // 4800, frame_ms 2 vs 1, vision 260 vs 0, eval_every 0 vs 10, eval_episodes 1
  // vs 4): the fallback only fires when the DOM value is missing or unparseable,
  // and then the run silently used a completely different config than the form
  // showed. A guard test pins these together (tests/test_web_form_defaults.py).
  const updates = clampInt(els.fePpoUpdates && els.fePpoUpdates.value, 1, 500, 40);
  const envs = clampInt(els.fePpoEnvs && els.fePpoEnvs.value, 1, 10, 4);
  const rollout = clampInt(els.fePpoRollout && els.fePpoRollout.value, 1, 512, 64);
  // 0 = "derive from the map". A fixed 4800 cut every original-map episode off
  // long before the engine could decide anything: measured on ``[p2]Lake (2p)``
  // the earliest native verdict ever seen was tick 46972, and the map-derived
  // value is 20056. Leaving this at a small constant was the single biggest
  // reason ``terminal`` stayed at 0.000 for hundreds of updates.
  const episodeTicks = clampInt(els.fePpoTicks && els.fePpoTicks.value, 0, 216000, 0);
  const device = (els.fePpoDevice && els.fePpoDevice.value) || 'auto';
  // 0 = unpaced, the shipping default: with `--physics-ms 16` pinning the engine
  // delta, `frame_ms` is *only* pacing, so 0 changes nothing about the battle and
  // measured 26.5x real time vs 10.7x at 1 ms (68% of the 1 ms budget was
  // Thread.sleep wake-up overhead). The old fallback of 1 would quietly cost 2.5x
  // whenever the DOM value was unreadable.
  const frameMs = clampInt(els.fePpoFrameMs && els.fePpoFrameMs.value, 0, 16, 0);
  // Engine-native game speed. 1 = off; N makes each tick advance N times the
  // game time, so a match needs 1/N the ticks -- at the cost of fidelity, which
  // is why it defaults to off rather than being folded into frame_ms.
  const fastForward = clampInt(els.fePpoFastForward && els.fePpoFastForward.value, 1, 32, 1);
  // Outcome adjudication. ``objective`` is the only mode that can produce a
  // verdict on a stock skirmish map (see IronZeroBridge.detectGameResult).
  // Until this existed the form had no control at all, so the host service fell
  // through to ``native`` -- which produced zero victories over 7129 real games.
  const winMode = (els.fePpoWinMode && els.fePpoWinMode.value) || 'objective';
  // Enemy AI difficulty. The form's `selected` option is 1 (困难, x1.40 income)
  // and the fallback must equal it, or a missing DOM value would quietly send a
  // different opponent than the one displayed. ``0`` is the *engine's* default
  // (x1.00, no bonus), which is exactly the level a difficulty control must not
  // silently select. The host service also defaults to 1 now, so both ends agree
  // even if this field is never sent.
  const enemyAiDifficulty = clampInt(
    els.fePpoEnemyAiDifficulty && els.fePpoEnemyAiDifficulty.value, -2, 3, 1,
  );
  const unlockAi = !!(els.fePpoUnlockAi && els.fePpoUnlockAi.checked);
  // The form default is 0 = no observation fog, which is what the BC warm starts
  // were trained with (bc_ironzero25 / bc_ironzero10 both record vision_radius=0).
  // Defaulting to 260 here meant a warm start could be asked to play a game it
  // had never seen, and the UI text advertised "默认 260px" to match.
  const visionRadius = clampInt(els.fePpoVision && els.fePpoVision.value, 0, 2000, 0);
  const evalEvery = clampInt(els.fePpoEvalEvery && els.fePpoEvalEvery.value, 0, 500, 10);
  const evalEpisodes = clampInt(els.fePpoEvalEpisodes && els.fePpoEvalEpisodes.value, 1, 16, 4);
  const useLlm = !!(els.fePpoUseLlm && els.fePpoUseLlm.checked);
  const selectedMap = (els.fePpoMap && els.fePpoMap.value) || 'Lake';
  let ppoMapName = selectedMap;
  let ppoMapPath = `maps/ironzero/${selectedMap}.tmx`;
  let unitsPerTeam = clampInt(els.fePpoUnits && els.fePpoUnits.value, 1, 25, 25);
  let mapWidth = 2700.0;
  let mapHeight = 2500.0;

  const isIronZero = selectedMap.toLowerCase().startsWith('ironzero');
  // The nine stock 2p maps come from `GET /maps` (via `/api/maps/catalog`): it is
  // the only place that knows their real pixel sizes. The old code read
  // `state.maps.list` (the *asset* listing) and fell back to 2700x2500 for
  // anything it did not recognise — which was every stock map when the asset
  // cache was cold, i.e. the observation normalised positions by the wrong
  // denominator without saying anything.
  if (!feMapCatalog()) await loadFeMapCatalog();
  const catalogMap = feCatalogMap(selectedMap);
  const originalMap = catalogMap
    ? null
    : feMapList().find((m) => m.name === selectedMap || m.id === selectedMap);

  if (isIronZero) {
    ppoMapName = `IronZero${unitsPerTeam}`;
    ppoMapPath = `maps/ironzero/${ppoMapName}.tmx`;
    mapWidth = 2700.0;
    mapHeight = 2500.0;
  } else if (catalogMap) {
    ppoMapName = catalogMap.name;
    ppoMapPath = catalogMap.map_path || '';
    mapWidth = num(catalogMap.map_width, 2700.0);
    mapHeight = num(catalogMap.map_height, 2500.0);
  } else if (originalMap) {
    ppoMapName = originalMap.name;
    ppoMapPath = originalMap.relative || '';
    mapWidth = num(originalMap.width_px, 2700.0);
    mapHeight = num(originalMap.height_px, 2500.0);
    if (selectedMap === 'Artillery' && els.fePpoUnits && els.fePpoUnits.value === '25') {
      unitsPerTeam = 22;
    }
  } else if (selectedMap === 'Artillery') {
    ppoMapName = 'Artillery';
    ppoMapPath = 'maps/challenge/Artillery.tmx';
    mapWidth = 2700.0;
    mapHeight = 2500.0;
    unitsPerTeam = 22;
  }

  const initMode = (els.fePpoInitMode && els.fePpoInitMode.value) || 'warmstart';
  // ``warmstart`` deliberately sends no path: the host service resolves the BC
  // weight for the *current* tank count (``resolve_bc_checkpoint`` knows about
  // runs/experiments/bc_ironzero{5,10,15,20,25}). This used to hard-code the
  // legacy 25v25 location under runs/, which only ever held an old copy — so
  // warmstarting a 10v10 run silently loaded 25v25 weights, or nothing at all
  // when that file was absent (training then quietly started from scratch).
  let initCheckpoint = null;
  if (initMode === 'checkpoint') {
    initCheckpoint = (els.fePpoInitCheckpoint && els.fePpoInitCheckpoint.value) || null;
    if (!initCheckpoint) {
      toast('请选择一个初始 Checkpoint，或把初始化方式改成「预热策略」/「从零随机」', 'error');
      return;
    }
  }

  ppo.jobId = null;
  ppo.job = null;
  ppo.events = [];
  ppo.cursor = 0;
  ppo.latest = null;
  ppo.evaluations = [];
  ppo.donePayload = null;
  ppo.status = 'running';
  ppo.stopRequested = false;
  ppo.restoreTried = true;
  ppo.startedAt = Date.now();
  ppo.updatesTotal = updates;
  ppo.updatesDone = 0;
  ppo.frameMs = frameMs;
  ppo.fastForward = fastForward;
  ppo.visionRadius = visionRadius;
  ppo.evalEvery = evalEvery;
  ppo.evalEpisodes = evalEpisodes;
  ppo.useLlm = useLlm;
  // Scenario actually requested, so the log line can show it before the host
  // starts reporting the resolved values back.
  ppo.mapName = ppoMapName;
  ppo.mapPath = ppoMapPath;
  ppo.mapWidth = mapWidth;
  ppo.mapHeight = mapHeight;
  ppo.enemyAiDifficulty = enemyAiDifficulty;
  // Reset the resolved-start fields: the previous run's checkpoint must not be
  // shown while this one is still starting up.
  ppo.initCheckpoint = initCheckpoint;
  ppo.initMode = initMode;
  feSetPpoRunning(true);
  feSetPpoStatus('启动中', 'running');
  feRenderPpoDashboard();
  try {
    const response = await Api.rwppPpo({
      updates: updates,
      envs: envs,
      rollout_steps: rollout,
      episode_ticks: episodeTicks,
      device: device,
      map_name: ppoMapName,
      map_path: ppoMapPath,
      map_width: mapWidth,
      map_height: mapHeight,
      win_mode: winMode,
      enemy_ai_difficulty: enemyAiDifficulty,
      unlock_ai: unlockAi,
      frame_ms: frameMs,
      vision_radius: visionRadius,
      eval_every: evalEvery,
      eval_episodes: evalEpisodes,
      use_llm: useLlm,
      units_per_team: unitsPerTeam,
      init_checkpoint: initCheckpoint,
      init_mode: initMode,
      // Physics delta is pinned explicitly: the trainer's default has moved
      // before, and with it omitted a web run silently inherited whatever the
      // current default happened to be, so ``episode_ticks`` stopped meaning
      // game ticks. frame_ms then only paces the wall clock.
      physics_ms: 16,
      // Engine-native acceleration (changes fidelity, unlike frame_ms): every
      // tick advances N times the game time, so the same match needs 1/N the
      // ticks and finishes N times sooner on the wall clock.
      fast_forward: fastForward,
      // Lockstep is /ppo's default and is stated explicitly here so the run does
      // not silently inherit a changed client-side default: the policy must
      // answer every observation, otherwise an unpaced engine (frame_ms=0)
      // advances past decisions the policy never made.
      lockstep: true,
    });
    const jobId = response && (response.job_id || response.id);
    if (!jobId) throw new Error('后端未返回 PPO 任务 id');
    ppo.jobId = String(jobId);
    ppo.status = 'running';
    startFePpoPolling();
    const initDesc = initMode === 'warmstart' ? '预热策略(BC)' : (initMode === 'checkpoint' ? '选定Checkpoint' : '从零随机');
    toast('PPO 真机训练已开始 · ' + ppoMapName + ' · ' + Math.round(mapWidth) + '×' + Math.round(mapHeight) + 'px'
      + (isIronZero ? ' · ' + unitsPerTeam + 'v' + unitsPerTeam + ' 坦克' : '')
      + ' · 起点: ' + initDesc
      + ' · ' + updates + ' 轮 × ' + envs + ' 实例 · frame_ms=' + frameMs
      + (fastForward > 1 ? ' · fast_forward=' + fastForward + '×' : '')
      + ' · 每局 ' + (episodeTicks > 0 ? episodeTicks + ' tick' : '按地图推导')
      + ' · 判定=' + winMode
      + ' · 对手难度=' + enemyAiDifficulty + ' ' + difficultyLabel(enemyAiDifficulty)
      + ' · fog=' + (visionRadius || 'off')
      + (useLlm ? ' · LLM 战略层' : ''), 'ok');
  } catch (error) {
    ppo.status = 'failed';
    feSetPpoRunning(false);
    if (error && error.status === 409) {
      toast('已有训练或实机任务正在运行中，请先在控制台点击【停止训练】或等待完成', 'error');
      void feRestorePpoJob(true);
    } else {
      handleFeError(error, 'PPO 训练');
    }
    feRenderPpoDashboard();
  }
}

export async function feStopPpo() {
  const ppo = state.fe.ppo;
  if (!ppo.jobId || ppo.status !== 'running') return;
  ppo.stopRequested = true;
  if (els.fePpoStop) els.fePpoStop.disabled = true;
  try {
    await Api.rwppPpoStop(ppo.jobId);
    toast('已请求停止 PPO 训练', 'ok');
  } catch (error) {
    handleFeError(error, '停止 PPO 训练');
  }
}

export async function feRestorePpoJob(force) {
  const ppo = state.fe.ppo;
  if (ppo.jobId && ppo.status === 'running') return;
  if (ppo.restoreTried && !force) return;
  ppo.restoreTried = true;
  try {
    const payload = await Api.rwppJobs();
    const jobs = ((payload && payload.jobs) || []).filter((job) => job && job.kind === 'ppo');
    // Prefer a running job, but fall back to the most recent finished one.
    //
    // Restoring *only* ``status === 'running'`` meant a training run that had
    // already finished was invisible after a page refresh: the job list showed it
    // (``done``, ``updates_done=80/80``, 163 live events readable from
    // ``/ppo/{id}/events``) while the page showed an empty form and no result. From
    // the operator's side that is indistinguishable from "the run never happened",
    // which is exactly the report "跑完了但是我在 web 端没看到进度".
    //
    // Attaching to a finished job is safe: ``fePollPpo`` reads its stored events to
    // rebuild the dashboard, then stops polling because the status is not
    // ``running``. The newest job wins, so a fresh run replaces an old one.
    const running = jobs.find((job) => job.status === 'running');
    const finished = jobs
      .filter((job) => job.status && job.status !== 'running')
      .sort((left, right) => num(right.started_ts, 0) - num(left.started_ts, 0))[0];
    const attach = running || finished;
    if (!attach) {
      if (state.rwpp.ppoRunning > 0 && els.fePpoHint) {
        els.fePpoHint.textContent = '检测到宿主机有 PPO 任务正在运行，请稍后刷新或重新进入本页查看。';
      }
      return;
    }
    const runningJob = attach;
    ppo.jobId = String(runningJob.id);
    ppo.status = runningJob.status || 'running';
    ppo.job = runningJob;
    ppo.stopRequested = false;
    ppo.updatesTotal = num(runningJob.updates_total, 0);
    ppo.updatesDone = num(runningJob.updates_done, 0);
    ppo.startedAt = Date.now() - num(runningJob.elapsed, 0) * 1000;
    if (els.fePpoUpdates && runningJob.updates_total) els.fePpoUpdates.value = String(runningJob.updates_total);
    if (els.fePpoEnvs && runningJob.envs) els.fePpoEnvs.value = String(runningJob.envs);
    if (els.fePpoRollout && runningJob.rollout_steps) els.fePpoRollout.value = String(runningJob.rollout_steps);
    if (els.fePpoTicks && runningJob.episode_ticks) els.fePpoTicks.value = String(runningJob.episode_ticks);
    if (els.fePpoDevice && runningJob.device) els.fePpoDevice.value = String(runningJob.device);
    if (numOrNull(runningJob.frame_ms) !== null) {
      ppo.frameMs = num(runningJob.frame_ms, 0);
      if (els.fePpoFrameMs) els.fePpoFrameMs.value = String(ppo.frameMs);
    }
    if (numOrNull(runningJob.fast_forward) !== null) {
      ppo.fastForward = num(runningJob.fast_forward, 1);
      if (els.fePpoFastForward) els.fePpoFastForward.value = String(ppo.fastForward);
    }
    if (numOrNull(runningJob.vision_radius) !== null) {
      ppo.visionRadius = num(runningJob.vision_radius, 260);
      if (els.fePpoVision) els.fePpoVision.value = String(ppo.visionRadius);
    }
    if (numOrNull(runningJob.eval_every) !== null) {
      ppo.evalEvery = num(runningJob.eval_every, 0);
      if (els.fePpoEvalEvery) els.fePpoEvalEvery.value = String(ppo.evalEvery);
    }
    if (numOrNull(runningJob.eval_episodes) !== null) {
      ppo.evalEpisodes = num(runningJob.eval_episodes, 1);
      if (els.fePpoEvalEpisodes) els.fePpoEvalEpisodes.value = String(ppo.evalEpisodes);
    }
    // Restore the scenario that job actually ran with (map + difficulty). A form
    // that keeps showing its own defaults next to another run's results is how
    // "对手难度=2" got attached to runs that executed at the engine default 0.
    if (runningJob.map_name && els.fePpoMap) {
      const options = [...els.fePpoMap.options].map((option) => option.value);
      if (options.includes(String(runningJob.map_name))) {
        els.fePpoMap.value = String(runningJob.map_name);
      }
    }
    if (numOrNull(runningJob.enemy_ai_difficulty) !== null) {
      ppo.enemyAiDifficulty = num(runningJob.enemy_ai_difficulty, ppo.enemyAiDifficulty);
      if (els.fePpoEnemyAiDifficulty) {
        els.fePpoEnemyAiDifficulty.value = String(ppo.enemyAiDifficulty);
      }
    }
    if (runningJob.use_llm !== undefined && runningJob.use_llm !== null) {
      ppo.useLlm = !!runningJob.use_llm;
      if (els.fePpoUseLlm) els.fePpoUseLlm.checked = ppo.useLlm;
      if (els.fePpoLlmNote) els.fePpoLlmNote.hidden = !ppo.useLlm;
    }
    // A finished job is attached read-only: show its dashboard and metrics, but
    // do not present the console as a live run (there is nothing to stop).
    const isRunning = ppo.status === 'running';
    feSetPpoRunning(isRunning);
    startFePpoPolling();
    if (state.tab === 'train') {
      feApplyTrainMode('ppo');
      toast(
        isRunning ? '已恢复运行中的 PPO 训练任务' : '已载入最近一次 PPO 训练结果',
        'ok'
      );
    }
  } catch (error) {
    // restore is best effort
  }
}
