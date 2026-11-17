// SPDX-License-Identifier: Apache-2.0
package ironzero;

import com.corrodinggames.rts.game.RWGameEngine;
import com.corrodinggames.rts.game.Team;
import com.corrodinggames.rts.game.ai.AI;
import com.corrodinggames.rts.game.units.AttackMode;
import com.corrodinggames.rts.game.units.MovementType;
import com.corrodinggames.rts.game.units.OrderableUnit;
import com.corrodinggames.rts.game.units.Unit;
import com.corrodinggames.rts.game.units.UnitType;
import com.corrodinggames.rts.game.units.UnitTypeInterface;
import com.corrodinggames.rts.game.units.Waypoint;
import com.corrodinggames.rts.game.units.WaypointType;
import com.corrodinggames.rts.game.units.actions.ActionId;
import com.corrodinggames.rts.game.units.actions.SpecialAction;
import com.corrodinggames.rts.game.units.buildings.Extractor;
import com.corrodinggames.rts.game.units.buildings.FactoryInterface;
import com.corrodinggames.rts.game.units.buildings.turrets.Turret;
import com.corrodinggames.rts.gameFramework.CommandControllerCommand;
import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;
import io.github.rwml.api.ExtensionContext;
import io.github.rwpp.api.game.impl.NativeMapLoader;
import io.github.rwpp.server.ServerExtension;
import io.github.rwpp.server.ServerRuntimeContext;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * IronZero RWPP bridge: runs a real headless Rusted Warfare 1.15 game, streams
 * ironzero/1-style observations over loopback NDJSON, injects policy actions and
 * reports the native game result (victory/defeat) when the battle is decided.
 * Extension args (after --):
 *   --bridge-port 5130  --ticks 3600  --map maps/challenge/Artillery.tmx  --frame-ms 2
 *   --win-mode native|annihilation   (default "native": rely on the map objective
 *       and Team defeat bookkeeping; "annihilation": ignore the native early
 *       signals and decide the episode from unit counts on a building-less map)
 *   --unlock-ai                      (default off: clear Unit.aiLimitedUse on the
 *       enemy AI teams so map-placed units join the AI's army. Deliberately does
 *       *not* set AI.aggressiveMode: that flag disables the AI's construction and
 *       base defence, and made the AI build nothing but builders and scouts)
 *   --ticks bounds simulated frames (1 frame = 1 tick; 60 ticks = 1 s).
 *   --frame-ms is the wall-clock pacing budget; by default it is *also* the
 *       engine's frame delta, so 16 = real time and 2 = ~8x faster.
 *   --physics-ms fixes the engine delta independently of --frame-ms, so an
 *       episode can keep a real-time simulation delta while the wall clock runs
 *       unpaced or fast. Omitted = old behaviour (delta == frame-ms).
 *   --fast-forward N                 (1..32, default 1 = off) sets the engine's
 *       own GameEngine.gameSpeed after the map loads, so every frame advances N
 *       times as much game time. physics-ms can only reach 3x (the engine clamps
 *       the frame delta to 3.0); gameSpeed is applied after that clamp and is not
 *       bounded by it. Changes fidelity, so it stays opt-in.
 *   --lockstep                       block after each obs until the client's
 *       step/shutdown arrives, so the control period is exactly 10 ticks. Free
 *       running (the default) is unchanged.
 *   --lockstep-timeout-ms N          (1000..600000, default 30000) lockstep wait
 *       budget; on expiry the bridge falls back to free running for the rest of
 *       the episode and counts the event in the ``bridge finished`` log.
 *   --ai-difficulty N                (-2..3: 非常简单/简单/正常/困难/非常困难/炼狱)
 *       pins every AI team's difficulty. Omitted = the game's own resolution,
 *       which in a headless single-player run is SettingsEngine.aiDifficulty
 *       (default 0 = 正常 = x1.00 income, no AI cheat bonus).
 */
public final class IronZeroBridge implements ServerExtension {
    private static final int DECISION_INTERVAL = 10;
    private static final int FRAME_PERIOD_MILLIS = 16;
    /**
     * How long {@link #closeBridge()} waits for the client's trailing messages
     * before the final {@code close()}, so the close is graceful instead of a RST.
     *
     * <p>Short on purpose: the client sends at most one step message per
     * observation, and we already called {@code shutdownOutput()}, so anything we
     * are still reading is a write that was in flight. Waiting longer only delays
     * the episode's end.
     */
    private static final int CLOSE_DRAIN_TIMEOUT_MILLIS = 250;
    /**
     * Engine physics delta in milliseconds; {@code 0} means "use {@code frameMs}".
     *
     * <p>By default the physics delta stays exactly what it has always been
     * ({@code frameMs}), so behaviour is bit-identical to previous builds. Passing
     * {@code --physics-ms N} decouples the two: the engine then always advances by
     * N and {@code frameMs} only paces the wall clock. That is a <em>semantic</em>
     * change (an episode of {@code --ticks} ticks covers a different amount of
     * simulated game time), so it is opt-in rather than a silent default.
     */
    private int physicsMs = 0;
    /**
     * Engine-native fast-forward, i.e. {@code GameEngine.gameSpeed}
     * ({@code --fast-forward N}, 1..32, default 1 = off).
     *
     * <p>This is <em>not</em> the same knob as {@link #physicsMs}. The delta we
     * pass to {@code updateFrameInternal} is clamped to {@code 3.0F} by the engine
     * ({@code RWGameEngine.java:1203}: {@code if (var1 > 3.0F) var1 = 3.0F;}),
     * so {@code physics_ms} alone can never exceed 3x (16 ms -> 50 ms). The
     * clamp happens <em>before</em> {@code updateAllGame} is reached, and
     * {@code updateAllGame} then multiplies the already-clamped delta by
     * {@code gameSpeed} ({@code RWGameEngine.java:1513-1515}:
     * {@code if (this.gameSpeed != 1.0F && !networked && !isRecordingOrPlaying())
     * var1 *= this.gameSpeed;}). The two multipliers therefore compose and the
     * {@code gameSpeed} factor is <em>not</em> subject to the 3x clamp: the
     * official UI itself goes to 64 for replays ({@code InterfaceEngine.java:955})
     * and caps live games at 5 ({@code :958}); the contract keeps us at 1..32.
     *
     * <p>The write must happen <em>after</em> the map is loaded: {@code loadMap}
     * calls {@code interfaceEngine.resetInterfaceState(false)}, which stamps
     * {@code gameSpeed = 1.0F} ({@code InterfaceEngine.java:335}), so a write
     * before the load is simply overwritten (same trap as {@link #seed}).
     */
    private int fastForward = 1;
    /**
     * {@code --lockstep}: block after every observation until the client sends a
     * {@code step} / {@code shutdown}, so the control period is exactly
     * {@link #DECISION_INTERVAL} ticks no matter how fast the engine runs.
     *
     * <p>{@link #lockstepActive} is the runtime latch: it starts as this value
     * and is cleared for good once the first wait times out (the contract's
     * "fall back to free-run" rule). Keeping the two separate means the start
     * log still reports what was <em>requested</em>.
     */
    private boolean lockstep = false;
    /** {@code --lockstep-timeout-ms N}: upper bound for one lockstep wait. */
    private int lockstepTimeoutMs = 30000;
    /** How many lockstep waits hit {@link #lockstepTimeoutMs}. */
    private int lockstepTimeouts = 0;
    /** Runtime latch for {@link #lockstep}; see there. */
    private boolean lockstepActive = false;
    /**
     * Optional fixed engine seed ({@code --seed N}); {@code 0} keeps the engine's
     * own random seed.
     *
     * <p>Why this has to be pinned *before* the map loads. Every combat/gameplay
     * roll goes through {@code CommonUtils.deterministicRandom*}, which reads
     * {@code GameEngine.randomSeed} as its base. {@code RWGameEngine.loadMap}
     * overwrites that field unconditionally for a local game
     * ({@code RWGameEngine.java:666}: {@code gameSetup.randomSeed =
     * CommonUtils.randomIntBetween(1, 1e9)}), and a headless bridge run is
     * neither networked nor a replay, so the condition there is always true.
     *
     * <p>Consequence measured during the 2026-11-14 audit: two identical
     * evaluations of the same checkpoint on the same map produced different
     * per-episode outcomes, and every game log printed a different
     * {@code global Seed:}. That is why "same weights, 20 games" moved from 0.700
     * to 0.900 and why no win rate could be audited or reproduced. Seeding before
     * the load is the only place it can still be honoured.
     */
    private int seed = 0;
    // Annihilation win-mode grace window: the map load/spawn settles during the
    // first simulated ticks, so unit counts are not authoritative yet.
    private static final int ANNIHILATION_GRACE_TICKS = 30;
    // Objective win-mode grace window: same reasoning, but this mode is used on
    // stock economy maps where an opening build queue must also be allowed to
    // start before "no production capability" means anything.
    private static final int OBJECTIVE_GRACE_TICKS = 120;
    // Objective win mode: a "military dominance" verdict.
    //
    // The engine objectives alone are impractical as *training* signals on a real
    // economy map. Measured, AI-vs-AI on Lake with our side on level 3 and the
    // opponent on level -2: ``mainBuildings``/``noConstructionOrTech`` victory
    // only arrived at tick 34701-39864, after our side had produced 196-218 units.
    // No tick budget a training run can afford reaches that, so a policy would
    // still never see a win.
    //
    // A battle is therefore also called once our army is overwhelmingly superior
    // *and* the enemy has stopped being able to hurt us: this is the point at
    // which a human player would resign. Both conditions are required, so the
    // verdict can never fire during a normal exchange of blows.
    private static final int DOMINANCE_OWN_MIN_UNITS = 8;
    private static final double DOMINANCE_ARMY_RATIO = 3.0;
    private static final double DOMINANCE_HP_RATIO = 3.0;
    // Action-space contract shared with MicroDuelConfig: 0..3 are unit groups,
    // 4 (== GROUP_BROADCAST) orders every living friendly unit.
    private static final int GROUP_COUNT = 4;
    private static final int GROUP_BROADCAST = GROUP_COUNT;
    /** Team id reported for map features that belong to no player (>= 0 per protocol). */
    private static final int NEUTRAL_TEAM_ID = 255;

    /** Economy unit types reported in ``info.builders[].can_build``. */
    private static final UnitType[] BUILDABLE_PROBE = {
        UnitType.extractor,
        UnitType.landFactory,
        UnitType.turret,
        UnitType.tank,
    };

    /**
     * Canonical economy names probed through {@code resolveUnitType}, i.e. the
     * same lookup {@code buildWithBuilders} dispatches with.
     *
     * <p>These cover the four names the Python action mask gates on -- the mask
     * asks for the literal string {@code "extractor"} and {@code "landFactory"}
     * -- plus the two remaining economy kinds. The registry scan in
     * {@link #buildableNamesFor} cannot produce {@code "extractor"} on a stock
     * map (the engine's own internal name there is {@code extractorT1}'), so
     * without this list the mask kept ``build_extractor`` closed even though the
     * engine accepted the order.
     *
     * <p><b>The anti-air turret was missing from this list, and that is why the
     * agent never had any anti-air at all.</b> Measured on {@code [p2]Lake (2p)}
     * ({@code .runtime/q1_build_menu.json}, {@code .runtime/q3_factory_menu.json}):
     * the reported menu was
     * {@code [landFactory, airFactory, seaFactory, laserDefence, repairbay,
     * experimentalLandFactory, extractor, turret]} -- no anti-air entry of any
     * spelling. The only defensive structure reachable from the action space was
     * {@code turret}, and that builds {@code c_turret_t1}, which the engine reports
     * with {@code can_attack_air: false} ({@code .runtime/q2_turret.json}). Meanwhile
     * the enemy AI builds {@code c_antiAirTurret} on the same map (1008 unit-frames
     * in a 1200-frame recording of
     * {@code runs/platform/real-rwpp-20261116-133006-d97c5a}). So the defect was
     * expressiveness, not strategy: the action space could not name the structure,
     * and no amount of training could have discovered it.
     *
     * <p>{@code antiAirTurret} exists in {@code UnitType} (see {@code UnitType.java})
     * but the {@code UnitType.values()} scan skips it, exactly as it skips
     * {@code extractor}: the typed units are registered behind a
     * {@code getSpecialAction} that the scan's probe does not match. Probing the
     * canonical name through the same resolution path {@code buildWithBuilders}
     * uses is what makes the reported menu a superset of the dispatchable set --
     * and if the engine does not accept it on a given map, the probe adds nothing
     * and no {@code BUILD_UNAVAILABLE} is created.
     *
     * <p>{@code antiAirTurretT2} is listed too: on maps where the T2 upgrade is
     * unlocked the same builder can queue it, and it is a different structure with
     * a different footprint than the T1.
     */
    private static final String[] CANONICAL_BUILD_NAMES = {
        "extractor",
        "landFactory",
        "turret",
        "antiAirTurret",
        "antiAirTurretT2",
    };

    /**
     * Unit types reported in ``info.factories[].can_produce``.
     *
     * <p>Kept only as a fast path. The authoritative answer is the engine's own
     * registry scan in {@link #producibleNamesFor(OrderableUnit)}: a factory's
     * menu is not derivable from a fixed type list (on Lake a
     * {@code landFactory} does *not* offer {@code tank}, while the
     * {@code commandCenter} offers {@code builder}), so probing a hard-coded
     * handful of names reported an empty menu and left {@code produce_unit}
     * permanently masked.
     *
     * <p>Separate from {@link #BUILDABLE_PROBE} because the two menus belong to
     * different units and are not interchangeable: on a stock skirmish map the
     * builder can queue only {@code landFactory} while the factory is precisely
     * what produces combat units.
     */
    private static final UnitType[] PRODUCIBLE_PROBE = {
        UnitType.tank,
        UnitType.artillery,
        UnitType.builder,
    };

    private volatile boolean stop = false;    private ServerSocket serverSocket;
    private Socket client;
    private BufferedWriter writer;
    private BufferedReader reader;
    private RWGameEngine engine;
    private int port = 5130;
    private int maxFrames = 900;
    private int frameMs = 16;
    private String mapPath = "maps/skirmish/[p2]Lake (2p).tmx";
    // "native" (default, backward compatible) or "annihilation" (25v25 arenas with
    // no buildings: the engine's Team.updateDefeatCheck marks any team without a
    // factory/builder as defeated, so result detection must bypass it).
    private String winMode = "native";
    private boolean unlockAi = false;
    /** ``--debug-build``: trace every gate that rejects a build order. */
    private boolean debugBuild = false;
    /**
     * ``--ai-pilot``: let the built-in AI drive the *controlled* team.
     *
     * <p>This is the behaviour-cloning teacher. The engine creates the player team
     * as a plain {@code Player(0)} and only teams that are {@code instanceof AI}
     * get an AI tick (see {@code Team.updateTeams} -> {@code AI.updateResourceTimers}).
     * Replacing {@code engine.playerTeam} with an {@code AI} instance therefore
     * makes the game play itself with the official AI, and
     * {@code --record-ai-orders} can then read back exactly which order it gave to
     * which unit -- a real expert demonstration instead of a hand-written script.
     */
    private boolean aiPilotTeam = false;
    /**
     * ``--record-ai-orders``: emit the AI's current orders in every observation.
     *
     * <p>Implies {@code --ai-pilot}. Orders are read from each own unit's active
     * {@link Waypoint} (move / attackMove / attack / build / repair), which is
     * what the AI actually queued, not what we guessed it would do.
     */
    private boolean recordAiOrders = false;
    /**
     * Last emitted order signature per own unit id, for ``--record-ai-orders``.
     *
     * <p>Labels are emitted on *change* only: a per-frame snapshot of the same
     * standing order is 87% ``hold`` and would teach a BC model to stand still.
     * Pruned to the units that are still alive on every emission so a recycled id
     * cannot inherit a stale signature.
     */
    private final java.util.HashMap<Long, String> aiOrderSignatures = new java.util.HashMap<>();
    /**
     * ``--ai-difficulty N``: force the AI difficulty level (-2..3), or ``null`` to
     * keep the game's own resolution order.
     *
     * <p>Rusted Warfare resolves an AI's difficulty through
     * {@code Team.getEffectiveAIDifficulty()}: a locked team uses its own
     * {@code aiDifficulty}; a networked/replay match uses its own value; everything
     * else (which includes this headless single-player bridge) falls back to
     * {@code aiDifficultyOverride}, and finally to the global
     * {@code SettingsEngine.aiDifficulty}. That global preference defaults to
     * {@code 0} ("正常" / Medium) when no settings file exists, so without this flag
     * every bridge run -- including the AI pilot used as a BC teacher -- plays at
     * level 0, i.e. the x1.00 income multiplier with no AI resource bonus.
     *
     * <p>Setting it locks every AI team it can find (via {@code aiDifficultyLocked})
     * so the value actually reaches {@code getAIDifficultyIncomeMultiplier()}.
     */
    private Integer aiDifficulty = null;
    /**
     * ``--enemy-ai-difficulty N``: override {@link #aiDifficulty} for the enemy
     * teams only, so teacher and opponent can be set independently.
     *
     * <p>Motivated by a measurement, not preference: with both sides on the same
     * level the two AIs mirror each other and never fight, so the recorded teacher
     * behaviour is "economy only". Raising only the enemy makes the teacher
     * respond to actual pressure.
     */
    private Integer enemyAiDifficulty = null;
    // Game-over plumbing: the native engine knows when the battle is decided
    // (TriggerEngine win/lose objectives and Team defeat checks). The bridge
    // polls that state every frame and reports it to the RL environment; it is
    // null while the battle is still running.
    private volatile String gameResult = null;
    private boolean enemySeen = false;
    // ``objective`` win mode: latched once the enemy has ever fielded a factory
    // or a builder, so a map whose AI never spawned cannot be scored a free win.
    private boolean objectiveEnemySeen = false;
    // ``objective`` win mode: latched once *we* have ever fielded production
    // capability, so losing it means something (an economy map must not report
    // "defeat" during the opening build queue).
    private boolean ownSawCapability = false;

    @Override
    public void onLoad(ExtensionContext context) {
        System.out.println("[ironzero] bridge extension loaded");
    }

    @Override
    public void runServer(ServerRuntimeContext context) {
        parseArguments(context.getArguments());
        RWGameEngine engine = context.createEngine(false);
        this.engine = engine;
        System.out.println("[ironzero] engine created, bridge port=" + port + " ticks=" + maxFrames);

        startBridge();
        waitForClient(30_000L);

        engine.currentMapPath = mapPath;
        // Pin the engine seed *before* the map loads. ``RWGameEngine.loadMap``
        // re-randomises ``gameSetup.randomSeed`` for every local game
        // (``RWGameEngine.java:666``), and every gameplay roll reads
        // ``GameEngine.randomSeed`` through ``CommonUtils.deterministicRandom*``.
        // Setting it here is the only window where the value survives.
        if (seed != 0) {
            engine.networkEngine.gameSetup.randomSeed = seed;
            System.out.println("[ironzero] seed pinned: " + seed);
        }
        NativeMapLoader.INSTANCE.loadMapWithAI(mapPath, true, 1, 0, true, false);
        long loadDeadline = System.currentTimeMillis() + 30_000L;
        while (System.currentTimeMillis() < loadDeadline) {
            context.pump();
            if (engine.map != null && engine.map.mapLoaded) {
                break;
            }
            Thread.yield();
        }
        engine.gamePaused = false;
        engine.touchInputDisabled = false;
        engine.gameJustLoaded = true;
        // Re-apply the pinned seed. Setting it before ``loadMapWithAI`` is not
        // enough: ``RWGameEngine.loadMap`` assigns
        // ``gameSetup.randomSeed = CommonUtils.randomIntBetween(1, 1e9)`` and then
        // copies it into ``this.randomSeed`` (:666-669) *during* the call, so a
        // pre-load write is simply overwritten. ``randomSeed`` is public and read
        // lazily by ``CommonUtils.deterministicRandom*`` on every roll, so writing
        // it right after the load is what actually pins the game.
        if (seed != 0) {
            engine.networkEngine.gameSetup.randomSeed = seed;
            engine.randomSeed = seed;
            pinGlobalRandom(seed);
            System.out.println("[ironzero] seed applied after load: " + seed);
        }
        if (unlockAi) {
            unlockAiTeams();
        }
        if (aiPilotTeam) {
            // Hand the controlled team to the built-in AI so its own commands can
            // be recorded as a behaviour-cloning teacher (``--record-ai-orders``).
            installAiPilot();
        }
        if (aiDifficulty != null || enemyAiDifficulty != null) {
            // Must run after the map finished loading (teams exist) and after the
            // pilot replaced the player team, otherwise the pilot would inherit the
            // old team's difficulty.
            if (aiDifficulty == null) {
                // Only an enemy override was given: leave the player side on the
                // engine's own resolution and pin the enemies.
                aiDifficulty = Integer.valueOf(defaultPlayerAiDifficulty());
            }
            applyAiDifficulty();
        }
        // Engine-native fast-forward. Deliberately *here*, i.e. after the map
        // finished loading: ``loadMap`` ends in
        // ``interfaceEngine.resetInterfaceState(false)``, which stamps
        // ``GameEngine.gameSpeed = 1.0F`` (InterfaceEngine.java:335), so a write
        // before the load is silently overwritten (exactly the trap ``--seed``
        // hits with ``gameSetup.randomSeed``).
        //
        // How this differs from ``--physics-ms``: our frame delta is clamped by
        // ``updateFrameInternal`` (RWGameEngine.java:1203: ``if (var1 > 3.0F)
        // var1 = 3.0F;``) *before* ``updateAllGame`` is called, which caps
        // physics_ms at 3x (16 ms -> 50 ms). ``updateAllGame`` then multiplies the
        // already-clamped delta by ``gameSpeed`` (RWGameEngine.java:1513-1515,
        // single-player and not recording/playing a replay -- both true for this
        // bridge). The gameSpeed factor is therefore *not* subject to the 3x
        // clamp, which is why the official UI can reach 64 for replays
        // (InterfaceEngine.java:955) and 5 for live play (:958). We use 1..32.
        //
        // Consequence for callers: each frame advances ``gameSpeed`` times as
        // much game time, so an episode of N ticks covers N*gameSpeed*delta of
        // simulated time and a decided battle needs proportionally fewer ticks.
        // That is a fidelity change, hence opt-in (default 1 = off, and the flag
        // is only on the command line when != 1).
        if (fastForward != 1) {
            engine.gameSpeed = (float) fastForward;
            System.out.println("[ironzero] fast-forward: gameSpeed=" + fastForward
                    + " (engine-native, applied after the map load)");
        }
        System.out.println("[ironzero] world ready: mapLoaded="
                + (engine.map != null && engine.map.mapLoaded)
                + " units=" + countUnits() + " teams=" + Team.getTeams().size()
                + " winMode=" + winMode + " unlockAi=" + unlockAi
                + " aiPilot=" + aiPilotTeam
                + " aiDifficulty=" + effectiveAiDifficultyDescription());

        // The one-line "what is actually pacing this run" record, emitted once so
        // a log reader never has to infer it from the argument list. Lockstep is
        // reported as *requested*, not as the runtime latch (which can fall back).
        System.out.println("[ironzero] pacing: frameMs=" + frameMs
                + " physicsMs=" + physicsMs
                + " physicsDelta=" + fmt(physicsMs > 0 ? (physicsMs / 16.666666f) : 1.0f)
                + " lockstep=" + lockstep
                + " fastForward=" + fastForward);
        lockstepActive = lockstep;

        sendHello();
        int frames = 0;
        int lastSentTick = -1;
        // Physics vs. pacing. `frameMs` historically played three roles at once:
        // the engine's time-scale argument, the engine's per-frame elapsed-ms
        // argument, and the wall-clock sleep budget. Measured on IronZero25 with
        // zero client actions, changing only this number changes the battle
        // (own_hp 1960 at 2 ms vs 690 at 16 ms over the same 3600 ticks), so it
        // is physics, not just speed: `--ticks` is a count of frames, and each
        // frame advances `frameMs` milliseconds of game time.
        //
        // `--physics-ms` is meant to pin the simulation delta so `frameMs` can
        // pace the wall clock independently. That only works if the delta reaches
        // the engine's *physics* argument.
        //
        // The engine signature is `updateFrameInternal(float var1, int var2)` and
        // (RWGameEngine.java:1217, 1371, 1506-1524) only `var1` advances the
        // simulation: `frameTimeMillis += var1 * 16.667`, `updateAllGame(var1)`,
        // `simulationTick++`. `var2` feeds nothing but `fpsTimeAccumulatorMs`.
        //
        // This call used to be `updateFrameInternal(1.0f, physicsElapsedMs)`, i.e.
        // the delta was hard-wired to 1.0 and `--physics-ms` was pushed into the
        // FPS counter. Consequence, measured directly: running `--physics-ms 2`
        // and `--physics-ms 16` for the same wall-clock budget simulated the *same*
        // number of ticks (191 vs 191; the 8x ratio that would prove the flag works
        // never appeared), so the documented "3600 tick x 16 ms = 57.6 s of game
        // time" claim was never true. The delta now goes where the engine reads it.
        //
        // `physicsMs == 0` keeps the historical 1.0 delta, so existing runs that
        // rely on it are unchanged.
        final float physicsDelta = physicsMs > 0 ? (physicsMs / 16.666666f) : 1.0f;
        final int pacingElapsedMs = physicsMs > 0 ? physicsMs : frameMs;
        long nextWake = System.nanoTime();
        while (!stop && context.isRunning() && frames < maxFrames) {
            context.pump();
            try {
                engine.updateFrameInternal(physicsDelta, pacingElapsedMs);
            } catch (Throwable failure) {
                System.out.println("[ironzero] frame error: " + failure);
                break;
            }
            frames++;
            // Annihilation arenas have no factory/builder, so Team.updateDefeatCheck
            // marks every team defeated on the first check. The engine keeps running
            // either way (defeated does not stop combat), but the flags must not leak
            // into result detection or map-script logic; clear them every frame.
            if ("annihilation".equals(winMode)) {
                clearNativeDefeatFlags();
            }
            // Poll the native game-over state after every simulated frame so a
            // decided battle ends the episode instead of being truncated at the
            // tick cap and reported as a timeout.
            detectGameResult(engine);
            boolean sentObservation = false;
            if (engine.simulationTick - lastSentTick >= DECISION_INTERVAL || frames == 1 || gameResult != null) {
                lastSentTick = engine.simulationTick;
                sendObservation(engine, frames);
                sentObservation = true;
            }
            // Lockstep: the decision boundary is a *rendezvous*, so block until the
            // client answers it instead of racing ahead and dropping frames.
            //
            // Only at a boundary (that is where a ``step`` belongs), only while the
            // battle is undecided (a terminal frame is followed by ``break`` below,
            // and waiting there would only burn the timeout), and only while the
            // latch is armed. ``readClientCommands()`` keeps running in between and
            // is untouched: free running (the default) is byte-for-byte what it was.
            if (lockstepActive && sentObservation && gameResult == null && reader != null) {
                if (awaitClientDecision(context)) {
                    // A step/shutdown arrived (or the socket ended): resume at once.
                } else {
                    // The contract's fallback: give up on lockstep for the rest of
                    // the episode so a silent client cannot hang the run. Counted
                    // and announced once, never per boundary.
                    lockstepActive = false;
                    lockstepTimeouts++;
                    System.out.println("[ironzero] lockstep timeout after " + lockstepTimeoutMs
                            + " ms (no step from the client); falling back to free running"
                            + " for the rest of this episode");
                }
            }
            readClientCommands();
            if (frames % 30 == 0 && writer == null) {
                break;
            }
            if (gameResult != null) {
                break;
            }
            // No pacing while lockstep is armed: the wait on the client already
            // bounds the frame rate, and sleeping *in addition* would just add the
            // sleep overshoot to every decision (the very cost --frame-ms 0 exists
            // to remove). Once lockstep has fallen back, pacing resumes.
            //
            // ``nextWake`` must be re-anchored while pacing is skipped, otherwise it
            // is left far in the past (it is only advanced inside the branch below).
            // Measured consequence of a stale value: after a lockstep fallback,
            // ``waitNanos`` stays negative for the next ``waitNs / frameMs`` frames,
            // so the bridge would silently run *unpaced* while still claiming to
            // honour ``--frame-ms``. Re-anchoring makes "fall back to free running"
            // mean "free running with the requested pacing". Free running never
            // reaches this branch, so its behaviour is unchanged.
            if (lockstepActive) {
                nextWake = System.nanoTime();
            }
            if (frameMs > 0 && !lockstepActive) {
                // Absolute deadlines instead of a fixed sleep per frame: the sleep
                // itself overshoots ~0.5 ms, and per-frame overshoot accumulated
                // into the wall clock (measured +1.2 ms per 1 ms budget). Sleeping
                // until "start + n*frameMs" keeps the average pacing honest.
                nextWake += frameMs * 1_000_000L;
                long waitNanos = nextWake - System.nanoTime();
                if (waitNanos > 0L) {
                    try {
                        long millis = waitNanos / 1_000_000L;
                        int nanos = (int) (waitNanos % 1_000_000L);
                        Thread.sleep(millis, nanos);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }
        sendDone(engine);
        closeBridge();
        System.out.println("[ironzero] bridge finished frames=" + frames + " tick=" + engine.simulationTick
                + " result=" + (gameResult != null ? gameResult : "timeout")
                + " fastForward=" + fastForward
                + " lockstepTimeouts=" + lockstepTimeouts);
    }

    /**
     * Block until the client answers the observation we just sent, or the lockstep
     * budget expires.
     *
     * <p>This is the blocking sibling of {@link #readClientCommands()}, which is
     * intentionally left alone: it polls {@code reader.ready()} and returns
     * immediately, and free running depends on exactly that. Here we instead set
     * a socket read timeout for the duration of the wait, consume lines until a
     * {@code step} / {@code shutdown} shows up, and then restore the previous
     * timeout so {@code closeBridge()}'s drain keeps its own budget.
     *
     * <p>Ordinary traffic is dispatched through the same {@link #applyStep}, so a
     * lockstep run and a free run execute identical commands. Non-step messages
     * (and unparsable lines) are consumed and ignored, matching
     * {@code readClientCommands()}.
     *
     * @return {@code true} when the client answered or the connection ended (the
     *     caller should keep going / stop naturally), {@code false} on timeout.
     */
    private boolean awaitClientDecision(ServerRuntimeContext context) {
        long deadline = System.currentTimeMillis() + lockstepTimeoutMs;
        int previousTimeout = -1;
        try {
            previousTimeout = client.getSoTimeout();
        } catch (IOException ignored) {
            // Socket already gone; the reads below will report it.
        }
        try {
            while (!stop && context.isRunning() && System.currentTimeMillis() < deadline) {
                long remaining = deadline - System.currentTimeMillis();
                if (remaining <= 0L) {
                    break;
                }
                try {
                    client.setSoTimeout((int) Math.min(remaining, Integer.MAX_VALUE));
                } catch (IOException failure) {
                    stop = true;
                    return true;
                }
                String line;
                try {
                    line = reader.readLine();
                } catch (java.net.SocketTimeoutException timedOut) {
                    break;
                } catch (IOException failure) {
                    // Peer closed: that is an answer, not a timeout.
                    stop = true;
                    return true;
                }
                if (line == null) {
                    stop = true;
                    return true;
                }
                if (line.isBlank()) {
                    continue;
                }
                JsonValue value;
                try {
                    value = Json.parse(line);
                } catch (RuntimeException failure) {
                    continue;
                }
                if (!value.isObject()) {
                    continue;
                }
                JsonObject message = value.asObject();
                String type = message.getString("type", "");
                if ("shutdown".equals(type)) {
                    stop = true;
                    return true;
                }
                if ("step".equals(type)) {
                    applyStep(message);
                    return true;
                }
                // Anything else is not a decision; keep waiting.
            }
            return stop || !context.isRunning();
        } finally {
            if (previousTimeout >= 0) {
                try {
                    client.setSoTimeout(previousTimeout);
                } catch (IOException ignored) {
                    // best effort
                }
            }
        }
    }

    @Override
    public void requestStop() {
        stop = true;
    }

    /**
     * Re-seed the engine's global {@link java.util.Random}.
     *
     * <p>{@code CommonUtils.random} is declared {@code static final Random
     * random = new Random()} (CommonUtils.java:31) with no seed, so every
     * {@code randomIntBetween / randomFloatBetween} call is non-deterministic
     * regardless of {@code GameEngine.randomSeed}. AI decisions, unit spawn jitter
     * and the map's own randomness all flow through it.
     *
     * <p>Reflection is the only way in (there is no setter and the field is
     * package-private), so it is wrapped defensively: a failure to pin the global
     * RNG must never break a training run, it only means the run is not exactly
     * reproducible.
     *
     * <p><b>What this does NOT fix.</b> The game calls {@code Math.random()} in
     * twelve places that never consult {@code GameEngine.randomSeed}:
     * {@code AI.java:1999/2029/2059} (target choice), {@code AIUnitGroup.java:261}
     * and {@code Zone.java:92/93/175/176} (scatter), and
     * {@code OrderableUnit.java:5412/5413/5438/5439} (waypoint jitter). Those use
     * the JVM-wide generator behind {@code Math.random()} and cannot be seeded
     * from outside the game, so a seeded episode is reproducible only up to the
     * first such call. Measured with an AI-driven episode: two runs with the same
     * seed diverged by frame 7. Treat {@code --seed} as "reduces variance and
     * makes the engine's own rolls repeatable", not as "bit-identical replays".
     */
    private void pinGlobalRandom(int value) {
        try {
            java.lang.reflect.Field field =
                    Class.forName("com.corrodinggames.rts.gameFramework.CommonUtils")
                            .getDeclaredField("random");
            field.setAccessible(true);
            Object instance = field.get(null);
            if (instance instanceof java.util.Random) {
                ((java.util.Random) instance).setSeed(value);
                System.out.println("[ironzero] global Random re-seeded: " + value);
            }
        } catch (Throwable failure) {
            System.out.println("[ironzero] could not re-seed global Random: " + failure);
        }
    }

    // ------------------------------------------------------------------
    private void parseArguments(List<String> args) {
        System.out.println("[ironzero] arguments=" + args);
        for (int i = 0; i < args.size(); i++) {
            String argument = args.get(i);
            if ("--bridge-port".equals(argument) && i + 1 < args.size()) {
                port = Integer.parseInt(args.get(i + 1));
            } else if ("--ticks".equals(argument) && i + 1 < args.size()) {
                maxFrames = Integer.parseInt(args.get(i + 1));
            } else if ("--map".equals(argument) && i + 1 < args.size()) {
                mapPath = args.get(i + 1);
            } else if ("--frame-ms".equals(argument) && i + 1 < args.size()) {
                // Wall-clock budget per simulated frame: 16 ms = real time (60 fps),
                // 0-4 ms lets the headless engine run several times faster for RL.
                //
                // 0 is legal and means "never sleep". It used to be rejected by the
                // *Python* side (``effective_frame_ms`` clamped it to 1 out of fear
                // that a zero delta would freeze the simulation), but that reasoning
                // predates ``--physics-ms``: with the delta pinned independently the
                // budget is pure pacing. Java always accepted 0 here and the main
                // loop only sleeps ``if (frameMs > 0)``, so nothing to change -- just
                // do not reintroduce a clamp.
                frameMs = Math.max(0, Integer.parseInt(args.get(i + 1)));
            } else if ("--physics-ms".equals(argument) && i + 1 < args.size()) {
                // Fixed simulation delta, independent of the pacing budget.
                // Omitted keeps the historical "delta == frameMs" behaviour.
                physicsMs = Math.max(1, Integer.parseInt(args.get(i + 1)));
            } else if ("--fast-forward".equals(argument) && i + 1 < args.size()) {
                // Engine-native speed multiplier (GameEngine.gameSpeed), applied
                // after the map loads. Clamped to the contract's 1..32 so a typo
                // cannot ask for a 1000x game; 1 keeps the argument off the command
                // line entirely (Python only emits it when != 1).
                try {
                    fastForward = clampInt(Integer.parseInt(args.get(i + 1).trim()), 1, 32);
                } catch (NumberFormatException ignored) {
                    System.out.println("[ironzero] ignoring non-numeric --fast-forward: "
                            + args.get(i + 1));
                }
            } else if ("--lockstep".equals(argument)) {
                lockstep = true;
            } else if ("--lockstep-timeout-ms".equals(argument) && i + 1 < args.size()) {
                try {
                    lockstepTimeoutMs = clampInt(
                            Integer.parseInt(args.get(i + 1).trim()), 1000, 600000);
                } catch (NumberFormatException ignored) {
                    System.out.println("[ironzero] ignoring non-numeric --lockstep-timeout-ms: "
                            + args.get(i + 1));
                }
            } else if ("--seed".equals(argument) && i + 1 < args.size()) {
                // Fixed engine seed. Applied before the map loads, because
                // ``loadMap`` re-randomises ``gameSetup.randomSeed`` for any
                // local (non-networked, non-replay) game.
                try {
                    seed = Integer.parseInt(args.get(i + 1).trim());
                } catch (NumberFormatException ignored) {
                    System.out.println("[ironzero] bad --seed: " + args.get(i + 1));
                }
            } else if ("--win-mode".equals(argument) && i + 1 < args.size()) {
                // How the episode is decided.
                //
                // This branch used to be *missing*: a refactor that added
                // ``--seed`` accidentally merged the win-mode parsing into the seed
                // branch, so this block read ``args.get(i+1)`` -- the seed's value --
                // as a mode name. The visible symptom was two log lines that looked
                // harmless:
                //
                //   [ironzero] unknown --win-mode: 11 (keeping native)
                //   [ironzero] world ready: ... winMode=native ...
                //
                // while the launcher had passed ``--win-mode objective``. Since
                // ``--seed`` is only emitted when non-zero, a seed-0 run logged
                // nothing at all and the flag simply never worked. That silently
                // reverted the fix for "the original maps cannot be won": on a stock
                // skirmish map the engine's own ``noConstructionOrTech`` objective
                // never fires and victory is unreachable, so every run went back to
                // ``timeout``.
                //
                // Parsing it here, in its own branch, is the fix. Deliberately does
                // *not* consume a following token, because every branch in this loop
                // reads ``args.get(i + 1)`` positionally and the loop does not skip
                // values -- so a branch that advanced ``i`` would desynchronise every
                // other one.
                String requested = args.get(i + 1).trim().toLowerCase(Locale.ROOT);
                if ("annihilation".equals(requested) || "native".equals(requested)
                        || "objective".equals(requested)) {
                    winMode = requested;
                } else {
                    System.out.println("[ironzero] unknown --win-mode: " + requested
                            + " (keeping " + winMode + ")");
                }
            } else if ("--unlock-ai".equals(argument)) {
                unlockAi = true;
            } else if ("--ai-pilot".equals(argument)) {
                aiPilotTeam = true;
            } else if ("--record-ai-orders".equals(argument)) {
                // Recording only makes sense while the AI is actually playing.
                recordAiOrders = true;
                aiPilotTeam = true;
            } else if ("--ai-difficulty".equals(argument) && i + 1 < args.size()) {
                try {
                    aiDifficulty = Integer.valueOf(Integer.parseInt(args.get(i + 1).trim()));
                } catch (NumberFormatException ignored) {
                    System.out.println("[ironzero] ignoring non-numeric --ai-difficulty: "
                            + args.get(i + 1));
                }
            } else if ("--enemy-ai-difficulty".equals(argument) && i + 1 < args.size()) {
                try {
                    enemyAiDifficulty = Integer.valueOf(
                            Integer.parseInt(args.get(i + 1).trim()));
                } catch (NumberFormatException ignored) {
                    System.out.println("[ironzero] ignoring non-numeric "
                            + "--enemy-ai-difficulty: " + args.get(i + 1));
                }
            } else if ("--debug-build".equals(argument)) {
                debugBuild = true;
            }
        }
    }

    private void startBridge() {
        try {
            serverSocket = new ServerSocket(port, 1, InetAddress.getByName("127.0.0.1"));
            System.out.println("[ironzero] bridge listening on 127.0.0.1:" + port);
        } catch (IOException failure) {
            System.out.println("[ironzero] bridge listen failed: " + failure);
            stop = true;
        }
    }

    private void waitForClient(long timeoutMillis) {
        if (serverSocket == null) {
            return;
        }
        try {
            serverSocket.setSoTimeout((int) timeoutMillis);
            client = serverSocket.accept();
            client.setTcpNoDelay(true);
            writer = new BufferedWriter(new OutputStreamWriter(client.getOutputStream(), StandardCharsets.UTF_8));
            reader = new BufferedReader(new InputStreamReader(client.getInputStream(), StandardCharsets.UTF_8));
            System.out.println("[ironzero] platform connected: " + client.getRemoteSocketAddress());
        } catch (IOException failure) {
            System.out.println("[ironzero] bridge accept failed: " + failure);
            stop = true;
        }
    }

    private void sendHello() {
        writeLine("{\"protocol\":\"ironzero/1\",\"type\":\"hello\",\"message_id\":\"bridge-hello\","
                + "\"adapter_version\":\"0.1.0-rwpp\",\"game_version\":\"1.15\","
                + "\"capabilities\":[\"obs\",\"tick\",\"real-game\",\"result\",\"building-flags\","
                + "\"annihilation\",\"unlock-ai\",\"unit-groups\"]}");
    }

    private void sendObservation(RWGameEngine engine, int frame) {
        if (writer == null) {
            return;
        }
        // The producible-units cache is a *within-one-observation* memo, not a
        // persistent one.
        //
        // ``isAvailableForUnit``/``canAffordForUnit`` depend on live instance state
        // (credits committed to the queue, tech tier, status effects), so a cached
        // menu goes stale as soon as the game advances. Measured on
        // ``[p2]Lake (2p)``: a landFactory advertised ``['builder', 'hoverTank']``
        // and every ``produce_unit`` for ``hoverTank`` was refused
        // ``PRODUCE_UNAVAILABLE`` for ~5800 ticks -- 104 of 131 sampled production
        // orders -- while funds sat at 353..426 credits, i.e. affordability was not
        // the problem. Keeping the memo only for the duration of one payload keeps
        // the cheap re-use (``factoriesJson`` asks about each factory once) while
        // making the reported menu track the engine.
        producibleCache.clear();
        Team player = engine.playerTeam;
        List<String> own = new ArrayList<>();
        List<String> enemies = new ArrayList<>();
        int alive = 0;
        int buildings = 0;
        int ownAlive = 0;
        int enemyAlive = 0;
        double ownHp = 0.0;
        double enemyHp = 0.0;
        // One pass over the unit list builds the entity payloads *and* the
        // counters. The previous version called resourceSlots() (which sorts the
        // map's ore pools and allocates a fresh list) plus slotExtractorLevel()
        // (another full scan) on every observation, so a 25v25 map paid several
        // extra O(units x pools) scans ten times per decision -- pure bridge CPU
        // that never reached the policy.
        List<double[]> slots = resourceSlots();
        int[] slotLevels = new int[slots.size()];
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            // Map features (trees, rocks) are "team -1": not a player. They are
            // neither ours nor an enemy. Counting them as enemies inflated
            // enemy_alive and polluted the target list with untargetable props.
            if (unit.team == null || unit.team.teamId < 0) {
                continue;
            }
            alive++;
            if (unit.isBuilding()) {
                buildings++;
            }
            String payload = entityJson(unit);
            if (isOwnUnit(player, unit)) {
                own.add(payload);
                ownAlive++;
                ownHp += unit.hp;
                // Fold this own building into the resource-slot occupancy the
                // observation reports, instead of re-scanning the unit list once
                // per slot afterwards.
                if (slotLevels.length > 0 && unit.isBuilding() && placeOnlyOnResPool(unit)) {
                    markSlotLevel(unit, slots, slotLevels);
                }
            } else {
                enemies.add(payload);
                if (isEnemyUnit(player, unit)) {
                    enemyAlive++;
                    enemyHp += unit.hp;
                }
            }
        }
        boolean terminated = gameResult != null;
        String resultJson = terminated ? "\"" + gameResult + "\"" : "null";
        String line = "{\"protocol\":\"ironzero/1\",\"type\":\"obs\",\"message_id\":\"obs-"
                + engine.simulationTick + "\",\"episode_id\":\"rwpp-bridge\",\"tick\":"
                + engine.simulationTick + ",\"decision_id\":\"d-" + engine.simulationTick
                + "\",\"game_time\":" + fmt(engine.simulationTick / 60.0)
                + ",\"controlled_team\":0,\"visibility_revision\":" + engine.simulationTick
                + ",\"observation_schema_version\":\"1\","
                + "\"global_state\":{\"unit_count\":" + alive + ",\"building_count\":" + buildings + ","
                + "\"funds\":" + fmt(player != null ? player.credits : 0.0) + ","
                + "\"income\":" + fmt(ownIncome()) + ","
                + "\"army_value\":0.0,\"visible_enemy_value\":0.0,\"kills\":0,\"losses\":0,"
                + "\"own_alive\":" + ownAlive + ",\"enemy_alive\":" + enemyAlive + ","
                + "\"own_hp\":" + fmt(ownHp) + ",\"enemy_hp\":" + fmt(enemyHp) + ","
                + "\"enemy_teams\":" + enemyTeamsJson(player) + "},"
                + "\"own_entities\":[" + String.join(",", own) + "],"
                + "\"visible_enemy_entities\":[" + String.join(",", enemies) + "],"
                + "\"remembered_enemies\":[],\"local_maps\":{},"
                + "\"resource_slots\":" + resourceSlotsJson(slots, slotLevels) + ","
                + "\"masks\":{\"action_mask\":[],\"unit_group_mask\":[],\"target_mask\":[],\"position_mask\":[]},"
                + "\"reward\":0.0,\"result\":" + resultJson + ",\"terminated\":" + terminated
                + ",\"truncated\":false,"
                + "\"info\":{\"source\":\"rwpp\",\"frame\":" + frame + ",\"units_alive\":" + alive
                + ",\"win_mode\":\"" + winMode + "\",\"unlock_ai\":" + unlockAi
                + ",\"ai_pilot\":" + aiPilotTeam
                + ",\"ai_difficulty\":" + ownAiDifficulty()
                + ",\"ai_teams\":" + aiTeamsJson()
                + ",\"builders\":" + buildersJson()
                + ",\"factories\":" + factoriesJson()                // Behaviour-cloning labels: the orders the expert (built-in AI)
                // is executing right now. Absent unless --record-ai-orders.
                + (recordAiOrders ? ",\"ai_orders\":" + aiOrdersJson() : "")
                + "}}";
        writeLine(line);
    }

    /**
     * Report, for each own builder, which economy unit types it can actually
     * queue. This is answered by the game itself rather than inferred from map
     * files, so operators can tell whether a map is economically playable
     * before training against it.
     */
    private String buildersJson() {
        StringBuilder out = new StringBuilder(256);
        out.append('[');
        boolean first = true;
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f || unit.team == null) {
                continue;
            }
            if (unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (!unit.isBuilder()) {
                continue;
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            String typeName = "?";
            try {
                typeName = unit.getUnitTypeInterface().getInternalName();
            } catch (RuntimeException ignored) {
                // Fall through with the placeholder name.
            }
            out.append("{\"entity_id\":").append(unit.id)
                    .append(",\"unit_type\":\"").append(typeName).append("\",\"can_build\":[");
            java.util.List<String> names = buildableNamesFor(unit);
            for (int index = 0; index < names.size(); index++) {
                if (index > 0) {
                    out.append(',');
                }
                out.append('"').append(names.get(index)).append('"');
            }
            out.append("]}");
        }
        out.append(']');
        return out.toString();
    }

    /**
     * Unit type names this builder can queue right now, in engine registry order.
     *
     * <p><b>This is the fix for "the agent never learns the economy on a stock
     * map".</b> The previous implementation probed a fixed four-name list
     * ({@link #BUILDABLE_PROBE}) with {@code unit.canQueueAction(...)}, and on
     * {@code [p2]Lake (2p)} that returned exactly {@code ["landFactory"]}. The
     * Python action mask gates {@code build_extractor} on the literal name
     * {@code "extractor"} appearing here, so the kind was masked away for the
     * whole episode -- the policy <i>could not express</i> "build an extractor".
     *
     * <p>Measured contradiction, same bridge build, same map, one order each
     * (see {@code .runtime/buildnames.json}):
     *
     * <pre>
     *   reported can_build = ["landFactory"]
     *   order structure_type="extractor"   -&gt; accepted
     *   order structure_type="extractorT1" -&gt; accepted
     *   order structure_type="turret"      -&gt; accepted
     *   order structure_type="tank"        -&gt; rejected BUILD_UNAVAILABLE
     * </pre>
     *
     * <p>So {@code canQueueAction} is <i>not</i> the question the mask needs
     * answered. The engine dispatches a build through {@code getSpecialAction}
     * plus {@code isValidNewWaypoint} -- exactly the pair {@link #buildWithBuilders}
     * uses before it reports an order as accepted -- so the honest menu is "the
     * types for which a build action resolves", scanned over the registry the same
     * way {@link #producibleNamesFor} does for factories. The reported menu and
     * the dispatchable set then agree by construction, and the mask stops
     * switching off actions the game would have executed.
     *
     * <p>Affordability is deliberately <i>not</i> baked in: the action mask already
     * gates the economy kinds on funds, and a menu that changed with the bank
     * balance would make the observation non-stationary for no benefit.
     */
    private java.util.List<String> buildableNamesFor(OrderableUnit builder) {
        String key;
        try {
            key = builder.getUnitTypeInterface().getInternalName();
        } catch (RuntimeException ignored) {
            key = "?";
        }
        java.util.List<String> cached = buildableCache.get(key);
        if (cached != null) {
            return cached;
        }
        java.util.List<String> names = new java.util.ArrayList<>();
        UnitType[] types;
        try {
            types = UnitType.values();
        } catch (RuntimeException ignored) {
            types = new UnitType[] {};
        }
        for (UnitType candidate : types) {
            if (candidate == null || candidate.isLocked()) {
                continue;
            }
            try {
                if (!builder.isBuilder() || !builder.canBeGivenBuildOrders()) {
                    continue;
                }
                // ``getSpecialAction`` is what ``buildWithBuilders`` dispatches
                // through; a null action means the engine has no build entry for
                // this type on this builder at all.
                SpecialAction action = builder.getSpecialAction(candidate, false);
                if (action == null) {
                    continue;
                }
                names.add(candidate.getInternalName());
            } catch (RuntimeException ignored) {
                // Unqueryable candidate: not part of this builder's menu.
            }
        }
        // The registry scan above misses names that dispatch *does* accept,
        // because a unit is dispatched through ``UnitType.findUnitType(name)``
        // (what ``resolveUnitType`` / ``buildWithBuilders`` use), which can
        // resolve a name that no ``UnitType.values()`` entry reports verbatim.
        // Measured on [p2]Lake (2p): the scan yields landFactory / airFactory /
        // seaFactory / laserDefence / repairbay / experimentalLandFactory, while
        // ``structure_type="extractor"`` is accepted by the very same builder.
        // The Python mask asks for the literal name ``extractor``, so the
        // canonical economy names are probed through the same resolution path
        // and added when they resolve, keeping the reported menu a superset of
        // the dispatchable one.
        for (String canonical : CANONICAL_BUILD_NAMES) {
            if (names.contains(canonical)) {
                continue;
            }
            try {
                UnitTypeInterface resolved = resolveUnitType(canonical);
                if (resolved == null || resolved.isLocked()) {
                    continue;
                }
                if (builder.getSpecialAction(resolved, false) == null) {
                    continue;
                }
                names.add(canonical);
            } catch (RuntimeException ignored) {
                // Not buildable under that name on this map.
            }
        }
        buildableCache.put(key, names);
        return names;
    }

    /** Cache of "which unit types can this *builder type* make" (see above). */
    private final java.util.HashMap<String, java.util.List<String>> buildableCache =
            new java.util.HashMap<>();

    /**
     * Cache of "which unit types can this factory type make", keyed by the
     * factory's internal unit-type name.
     *
     * <p>Rescanning {@code UnitType.values()} per factory per frame would be
     * wasteful: the observation is emitted every 10 ticks and there can be
     * several factories. The menu only depends on the factory's *type* and its
     * tech level, and a type never changes tier mid-game on the maps used here,
     * so caching by type name is safe. Cleared on reset.
     */
    private final java.util.HashMap<String, java.util.List<String>> producibleCache =
            new java.util.HashMap<>();

    /**
     * True when this factory has finished construction and the engine considers
     * it operational, i.e. when a {@code produce_unit} order can actually be
     * queued here right now.
     *
     * <p>Deliberately two independent checks rather than one:
     *
     * <ul>
     *   <li>{@code isNonOperational()} is the engine's own answer and also covers
     *       states a progress number cannot express (disabled, stunned, still
     *       waiting on a build order that has not been paid for);</li>
     *   <li>{@code buildProgress} catches the window where the engine has started
     *       constructing but has not yet flipped the operational flag.</li>
     * </ul>
     *
     * <p>Both are needed because a bridge that reports "you can queue a tank here"
     * for a half-built factory makes the action mask open {@code produce_unit}
     * hundreds of ticks early; every such order is rejected and the policy learns
     * to stop using the action entirely (measured: {@code bonus_produce} stayed at
     * exactly 0.000 across 200 updates on {@code [p2]Lake (2p)}).
     */
    private boolean factoryCanProduceNow(OrderableUnit factory) {
        try {
            if (!isOperational(factory)) {
                return false;
            }
        } catch (RuntimeException ignored) {
            // Unqueryable: fall through to the progress check rather than
            // reporting a possibly-unusable menu.
        }
        // ``buildProgress >= 1.0`` is the second half of "is it finished", but it
        // must not be the only test: an extractor (and any type whose progress
        // counter is never advanced by the engine) reports 0.0 forever while
        // working. For a *factory* the progress reading has historically been the
        // correct one, so it is kept -- but only as an additional veto when the
        // engine is actually mid-construction (a value strictly inside (0,1)).
        float progress = buildProgress(factory);
        return !(progress > 0.0f && progress < 1.0f);
    }

    /**
     * Unit type names this factory can queue, in engine registry order.
     *
     * <p>This is the authoritative answer and is deliberately a registry scan,
     * not a probe of a fixed list. A factory's menu is data, not something the
     * bridge can infer: measured on Lake, {@code landFactory} offers
     * ``builder`` (and no ``tank``), while {@code commandCenter} offers
     * ``builder`` too. Hard-coding candidate names therefore reported an empty
     * menu on the very map being trained and left production masked forever.
     *
     * <p>{@code findActionForUnitType} is exactly the lookup
     * {@link #produceUnit} uses to dispatch, so the reported menu and the
     * dispatchable set agree by construction.
     */
    private java.util.List<String> producibleNamesFor(OrderableUnit factory) {
        String key;
        try {
            // Per-instance key: ``isAvailableForUnit``/``canAffordForUnit`` are
            // properties of the individual building, and two factories of the same
            // type can legitimately differ (a T1 vs a T2 upgrade).
            key = factory.getUnitTypeInterface().getInternalName()
                    + "#" + factory.getTechLevel()
                    + "@" + factory.id;
        } catch (RuntimeException ignored) {
            key = "?";
        }
        java.util.List<String> cached = producibleCache.get(key);
        if (cached != null) {
            return cached;
        }
        java.util.List<String> names = new java.util.ArrayList<>();
        UnitType[] types;
        try {
            types = UnitType.values();
        } catch (RuntimeException ignored) {
            types = new UnitType[] {};
        }
        for (UnitType candidate : types) {
            if (candidate == null || candidate.isLocked()) {
                continue;
            }
            try {
                SpecialAction action = factory.findActionForUnitType(candidate);
                if (action == null) {
                    continue;
                }
                // The reported menu must be exactly what {@link #produceUnit} can
                // dispatch, otherwise the mask opens a kind the engine will refuse.
                // ``produceUnit`` requires BOTH checks; reporting on
                // ``isAvailableForUnit`` alone made the menu a superset of what was
                // dispatchable.
                if (action.isAvailableForUnit(factory)
                        && action.canAffordForUnit(factory, false)) {
                    names.add(candidate.getInternalName());
                }
            } catch (RuntimeException ignored) {
                // Unqueryable candidate: not part of this factory's menu.
            }
        }
        // The registry scan above misses units the *dispatch path* accepts, for
        // exactly the reason {@link #buildableNamesFor} documents: a unit is
        // dispatched through {@code UnitType.findUnitType(name)} / a factory's own
        // {@code findActionForUnitType(...)}, and a custom or aliased unit can be
        // resolvable by name without appearing in {@code UnitType.values()} under
        // that name.
        //
        // <p>Measured on {@code [p2]Lake (2p)} (``.runtime/q3_factory_menu.json``),
        // same finished T1 {@code landFactory}, same bridge build. The scan reported
        // two entries, yet the engine accepted and BUILT four different combat units:
        //
        // <pre>
        //   reported can_produce = ["builder", "hoverTank"]
        //   produce "hoverTank" -> built            (reported)
        //   produce "tank"      -> built c_tank     (NOT reported)
        //   produce "artillery" -> built c_artillery(NOT reported)
        //   produce "scout"     -> built            (NOT reported)
        //   produce "helicopter"-> PRODUCE_UNAVAILABLE
        //   produce "lightTank" -> PRODUCE_UNAVAILABLE
        // </pre>
        //
        // <p>Because the Python side indexes this list to decide *which* unit to
        // build, the omission was not cosmetic: it made three of the four buildable
        // units unreachable from the action space, which is precisely the reported
        // "only ever builds one kind of tank". Probing these canonical names through
        // the same {@code findActionForUnitType} that {@link #produceUnit}
        // dispatches with makes the reported menu a superset of the dispatchable set,
        // which is the direction the contract needs.
        for (String canonical : CANONICAL_PRODUCE_NAMES) {
            if (names.contains(canonical)) {
                continue;
            }
            try {
                UnitTypeInterface resolved = resolveUnitType(canonical);
                if (resolved == null || resolved.isLocked()) {
                    continue;
                }
                SpecialAction action = factory.findActionForUnitType(resolved);
                if (action == null) {
                    continue;
                }
                if (action.isAvailableForUnit(factory)
                        && action.canAffordForUnit(factory, false)) {
                    names.add(canonical);
                }
            } catch (RuntimeException ignored) {
                // Not producible under that name at this factory.
            }
        }
        producibleCache.put(key, names);
        return names;
    }

    /**
     * Canonical unit names probed through {@code findActionForUnitType}, i.e. the
     * same lookup {@link #produceUnit} dispatches with.
     *
     * <p>These are the stock combat unit families a land factory can field, plus the
     * economy entry. They are probed in addition to the registry scan because the
     * scan under-reports on this map (see {@link #producibleNamesFor}): measured,
     * the scan listed 2 entries while 4 were buildable.
     *
     * <p>Affordability is deliberately not baked in by the caller: the same
     * {@code canAffordForUnit} check the dispatch path uses is applied here, so a
     * menu entry is only reported when the order would actually be accepted. That
     * keeps the reported menu and the dispatchable set in agreement, which is what
     * stops the policy paying {@code invalid_penalty} for obeying its own mask.
     */
    private static final String[] CANONICAL_PRODUCE_NAMES = {
        "hoverTank",
        "tank",
        "artillery",
        "scout",
        "lightTank",
        "helicopter",
        "builder",
    };

    /**
     * Report, for each own factory, which unit types it can queue right now.
     *
     * <p>This is what makes {@code produce_unit} selectable: the Python mask
     * gates that kind on a factory, and asks the game -- not a hard-coded guess
     * -- which unit types are actually available. Reported separately from
     * {@link #buildersJson()} because a builder's menu and a factory's menu are
     * different capabilities.
     */
    private String factoriesJson() {
        StringBuilder out = new StringBuilder(256);
        out.append('[');
        boolean first = true;
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f || unit.team == null) {
                continue;
            }
            if (unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (!unit.isFactoryBuilding()) {
                continue;
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            String typeName = "?";
            try {
                typeName = unit.getUnitTypeInterface().getInternalName();
            } catch (RuntimeException ignored) {
                // Fall through with the placeholder name.
            }
            out.append("{\"entity_id\":").append(unit.id)
                    .append(",\"unit_type\":\"").append(typeName)
                    .append("\",\"can_produce\":[");
            // A factory that is still being built cannot queue anything. This is
            // NOT cosmetic: the Python action mask opens ``produce_unit`` when any
            // reported factory advertises a non-``builder`` entry, so an
            // under-construction factory used to advertise a full menu and the
            // kind became selectable hundreds of ticks too early.
            //
            // Measured on [p2]Lake (2p) (``.runtime/produce_gate.json``, 99 samples
            // while the landFactory was building):
            //
            //   progress 0.01..0.99, is_operational=false  ->  can_produce
            //   ["builder","hoverTank"] advertised, mask open in 99/99 samples
            //
            // Every such order came back ``PRODUCE_UNAVAILABLE``. The policy paid
            // ``invalid_penalty`` for obeying the mask and learned to avoid the
            // action, which is why ``bonus_produce`` stayed at 0.000 over 200
            // updates even though the same factory accepts the order in the same
            // episode once it finishes. Reported honestly as an empty menu instead:
            // an empty list is a real answer ("queue nothing yet"), and the mask
            // already treats it as "not producible".
            java.util.List<String> names = factoryCanProduceNow(unit)
                    ? producibleNamesFor(unit)
                    : java.util.Collections.<String>emptyList();
            for (int index = 0; index < names.size(); index++) {
                if (index > 0) {
                    out.append(',');
                }
                out.append('"').append(names.get(index)).append('"');
            }
            out.append(']');
            // What is *actually* queued, in order.
            //
            // ``can_produce`` answers "what could this factory start right now", which
            // is the question the action mask needs. It says nothing about work already
            // in progress, and that omission is why behaviour cloning had no production
            // labels: the teacher's army grows from 2 to 13 units while the recorded
            // orders contain zero ``produce_unit`` entries, because the AI's production
            // decisions live in the build queue rather than in waypoints.
            //
            // ``BuildQueueItem`` already carries everything a label needs -- the unit
            // type (``unitType``), how many of it (``buildCount``), and the engine's
            // progress -- so the queue is reported verbatim instead of being inferred.
            out.append(",\"queued\":[");
            java.util.List<com.corrodinggames.rts.game.units.buildings.BuildQueueItem> queueItems =
                    queuedItems(unit);
            for (int index = 0; index < queueItems.size(); index++) {
                if (index > 0) {
                    out.append(',');
                }
                com.corrodinggames.rts.game.units.buildings.BuildQueueItem queued =
                        queueItems.get(index);
                String queuedType = "?";
                try {
                    if (queued.unitType != null) {
                        queuedType = queued.unitType.getInternalName();
                    }
                } catch (RuntimeException ignored) {
                    // Keep the placeholder: an unreadable type must not drop the item,
                    // because the *length* of the queue is itself information.
                }
                float itemProgress = Float.isFinite(queued.progress) ? queued.progress : -1.0f;
                out.append("{\"unit_type\":\"").append(queuedType).append('"')
                        .append(",\"count\":").append(Math.max(0, queued.buildCount))
                        .append(",\"progress\":").append(fmt(itemProgress))
                        .append(",\"unit_production\":").append(queued.unitProduction)
                        .append('}');
            }
            out.append(']');
            out.append(",\"queue_length\":").append(queueLength(unit))
                    .append(",\"queue_progress\":").append(fmt(queueProgress(unit)));
            // Only the object brace closes here: the ``queued`` array was already
            // closed above. Shipping ``"]}"`` emitted a stray ``]`` and made the
            // whole observation line invalid JSON -- measured: every one of 900
            // messages failed to parse with ``Expecting ',' delimiter`` at the
            // byte before the end, i.e. the entire bridge output was discarded by
            // any strict JSON reader. The Python env's own reader is lenient
            // enough to hide it, which is exactly why this needed a protocol-level
            // check rather than a unit test over the encoder.
            out.append('}');
        }
        out.append(']');
        return out.toString();
    }

    /** Snapshot of one factory's build queue, empty for a non-factory. */
    private java.util.List<com.corrodinggames.rts.game.units.buildings.BuildQueueItem>
            queuedItems(Unit unit) {
        try {
            if (!(unit instanceof com.corrodinggames.rts.game.units.buildings.Factory)) {
                // Only ``Factory`` exposes ``getQueuedItems()``; the interface has the
                // counts but not the list. A custom producer still reports
                // ``queue_length``/``queue_progress`` through the interface.
                return java.util.Collections.emptyList();
            }
            com.corrodinggames.rts.game.units.buildings.Factory factory =
                    (com.corrodinggames.rts.game.units.buildings.Factory) unit;
            if (factory.isQueueEmpty()) {
                return java.util.Collections.emptyList();
            }
            com.corrodinggames.rts.gameFramework.utility.FastArrayList queue =
                    factory.getQueuedItems();
            if (queue == null) {
                return java.util.Collections.emptyList();
            }
            java.util.List<com.corrodinggames.rts.game.units.buildings.BuildQueueItem> queueItems =
                    new java.util.ArrayList<>(queue.size());
            for (Object entry : queue) {
                if (entry instanceof com.corrodinggames.rts.game.units.buildings.BuildQueueItem) {
                    queueItems.add(
                            (com.corrodinggames.rts.game.units.buildings.BuildQueueItem) entry);
                }
            }
            return queueItems;
        } catch (RuntimeException ignored) {
            return java.util.Collections.emptyList();
        }
    }

    private String entityJson(Unit unit) {
        // ``protocol.Entity.team_id`` requires >= 0. Unowned map features (trees,
        // rocks) either have a null team or carry the "-1 neutral" marker which
        // the engine uses internally; both must be mapped to a valid sentinel,
        // otherwise every observation fails protocol validation.
        int teamId = unit.team != null && unit.team.teamId >= 0 ? unit.team.teamId : NEUTRAL_TEAM_ID;
        boolean building = unit.isBuilding();
        return "{\"entity_id\":" + unit.id
                + ",\"entity_type_id\":" + stableTypeId(unit) + ",\"team_id\":" + teamId
                + ",\"x\":" + fmt(unit.x) + ",\"y\":" + fmt(unit.y)
                + ",\"hp\":" + fmt(unit.hp) + ",\"max_hp\":" + fmt(unit.maxHp)
                + ",\"unit_type\":\"" + internalName(unit) + "\""
                + ",\"movement_type\":\"" + movementType(unit) + "\""
                + ",\"is_air_unit\":" + isAirUnit(unit)
                + ",\"is_building\":" + building
                + ",\"is_factory\":" + unit.isFactoryBuilding()
                + ",\"is_builder\":" + unit.isBuilder()
                + ",\"is_operational\":" + isOperational(unit)
                // Completion fraction of a building under construction (1.0 for a
                // finished unit). Without it neither the observation nor the
                // reward can tell "a factory is going up" from "no factory yet",
                // and a policy that starts a build and then gets distracted has
                // no way to see that its structure is still half-built.
                + ",\"build_progress\":" + fmt(buildProgress(unit))
                // Whether the engine would accept an ``upgrade_building`` order
                // for this unit *right now* (a real answer, not a type guess).
                + ",\"can_upgrade\":" + canUpgrade(unit)
                // Action-group alignment: addOwnOrderableUnits() selects group members
                // with id % GROUP_COUNT, so expose exactly that value to the policy.
                + ",\"group_id\":" + (unit.id % GROUP_COUNT)
                // "Can this thing actually shoot?" was the single missing bit that
                // made a builder suicide into the enemy base: the observation could
                // describe where an entity was and how much hp it had, but not
                // whether walking into range of it was fatal. Reported as two
                // fields so the policy can both gate the decision (``can_attack``)
                // and reason about distance (``attack_range``).
                + ",\"can_attack\":" + canAttack(unit)
                + ",\"attack_range\":" + fmt(attackRange(unit))
                // ------------------------------------------------------------------
                // The information a real-time-strategy decision actually needs.
                //
                // Before this block the payload could say *where* an entity was and
                // *how much hp* it had, and whether it could shoot in principle. That
                // is not enough to answer any of the questions the game poses on
                // every decision:
                //
                //   * "can my tank catch that thing?"            -> needs ``move_speed``
                //   * "can it out-range my turret?"               -> needs both ranges
                //                                                  (enemy entities get the
                //                                                  same fields, so the
                //                                                  comparison is per-slot)
                //   * "is that a laser/AA turret or a tank?"      -> needs the defence
                //                                                  flags; ``can_attack``
                //                                                  alone calls them all
                //                                                  "armed"
                //   * "will it shoot my aircraft?"                -> needs ``can_attack_air``
                //   * "is its base upgraded, i.e. is a T2 army
                //      about to arrive?"                          -> needs ``tech_level``
                //   * "how fast does it shoot?"                   -> needs ``shoot_delay``
                //   * "does it have a shield I have to chew
                //      through first?"                            -> needs ``shield``
                //   * "what is it doing right now?"               -> needs ``order_type``
                //
                // The last one is not a nicety: ``observation.features._entity_features``
                // already reserves a column for ``order_type`` and reads it on every
                // entity, but the bridge never sent the key, so that column has been a
                // hard-coded constant 0 for every unit of every episode. Measured over
                // 40 real Lake decisions (``.runtime/contract_gap.json``):
                // ``order_type_values_seen = ["None"]`` -- the policy was structurally
                // blind to the difference between "idle" and "attacking".
                //
                // Every one of these is the engine's own answer, not a lookup table,
                // so it stays correct on custom units and mods.
                + ",\"order_type\":\"" + orderName(unit) + "\""
                + ",\"move_speed\":" + fmt(moveSpeed(unit))
                + ",\"tech_level\":" + techLevel(unit)
                + ",\"shoot_delay\":" + fmt(shootDelay(unit))
                + ",\"can_attack_air\":" + canAttackAir(unit)
                + ",\"can_attack_land\":" + canAttackLand(unit)
                + ",\"shield\":" + fmt(shield(unit))
                + ",\"max_shield\":" + fmt(maxShield(unit))
                + ",\"is_turret\":" + isTurretType(unit)
                + ",\"is_laser\":" + isLaserType(unit)
                // "A building whose purpose is to defend" -- deliberately NOT
                // ``isBuilding() && canAttack()``.
                //
                // ``LaserDefence.canAttack()`` returns **false**
                // (``LaserDefence.java:268``): it is a projectile *interceptor*, not a
                // gun, so the naive definition would classify the game's most
                // prominent defensive structure as a harmless farm. The honest test is
                // "is it a building AND (it can shoot OR it is a laser tower)"; a
                // turret, an AA turret and the laser tower all satisfy it, and an
                // extractor / factory / command center does not (unless the command
                // center, which genuinely does shoot).
                + ",\"is_defence\":"
                + (unit.isBuilding() && (canAttack(unit) || isLaserType(unit)))
                + ",\"queue_length\":" + queueLength(unit)
                + ",\"queue_progress\":" + fmt(queueProgress(unit))
                + ",\"is_visible\":true,\"is_alive\":true}";
    }

    /**
     * The unit's current order, spelled with the names
     * {@code engine.constants.ORDER_NAMES} uses so the encoder's lookup hits.
     *
     * <p>Read from the engine's live waypoint, the same source {@link #aiOrdersJson()}
     * uses for behaviour-cloning labels: a unit with no waypoint but an acquired
     * attack target is genuinely attacking, which is the state a turret sits in for
     * its whole life.
     */
    private String orderName(Unit unit) {
        if (!(unit instanceof OrderableUnit)) {
            return "idle";
        }
        try {
            Waypoint waypoint = ((OrderableUnit) unit).getCurrentWaypoint();
            if (waypoint != null) {
                WaypointType type = waypoint.getType();
                if (type != null) {
                    switch (type.name()) {
                        case "move":
                            return "move";
                        case "attackMove":
                        case "patrol":
                            return "attack_move";
                        case "attack":
                            return "attack_target";
                        case "guard":
                        case "guardAt":
                        case "follow":
                            return "hold";
                        default:
                            break;
                    }
                }
            }
            if (((OrderableUnit) unit).getEffectiveAttackTarget() != null) {
                return "attack_target";
            }
        } catch (RuntimeException ignored) {
            // An unreadable order is reported as idle rather than failing the frame.
        }
        return "idle";
    }

    /**
     * Engine movement speed, in world pixels per tick at the 60 Hz simulation loop.
     *
     * <p>{@code getMoveSpeed()} is declared on {@link OrderableUnit} and
     * {@code Building} overrides it to {@code 0.0F}, so a stationary structure and a
     * unit that merely *has not moved yet* are distinguishable -- which is exactly
     * the distinction "is that a turret or a tank" needs. The engine applies it as
     * {@code position += getMoveSpeed() * throttle * frameDelta}, so a stock tank's
     * {@code 1.0} is ~60 px/s.
     */
    private float moveSpeed(Unit unit) {
        try {
            if (!(unit instanceof OrderableUnit)) {
                return 0.0f;
            }
            float speed = ((OrderableUnit) unit).getMoveSpeed();
            return Float.isFinite(speed) && speed > 0.0f ? speed : 0.0f;
        } catch (RuntimeException ignored) {
            return 0.0f;
        }
    }

    /**
     * Building / unit tier: 1 for a base unit, 2 or 3 once upgraded.
     *
     * <p>{@code Unit.getTechLevel()} defaults to 1; {@code LandFactory} answers
     * {@code upgraded ? 2 : 1} and {@code Extractor} reports its own tier, so this is
     * the engine's answer to "is this already the upgraded version". It matters
     * because a T2 factory produces a different army than a T1 one.
     *
     * <p><b>Turrets are special-cased, and it is not cosmetic.</b>
     * {@code Turret} does <i>not</i> override {@code Unit.getTechLevel()}, so it
     * inherits the constant {@code 1} -- a gun T3 tower would report tier 1, i.e.
     * "nothing to worry about". Its real tier lives in
     * {@code Turret.getTurretLevel()} (which delegates to the current
     * {@code TurretType.getTechLevel()}). Since a turret's tier is exactly what
     * decides whether a push will work, reporting it wrong is worse than not
     * reporting it.
     */
    private int techLevel(Unit unit) {
        try {
            int level;
            if (unit instanceof com.corrodinggames.rts.game.units.buildings.turrets.Turret) {
                level = ((com.corrodinggames.rts.game.units.buildings.turrets.Turret) unit)
                        .getTurretLevel();
            } else {
                level = unit.getTechLevel();
            }
            return level > 0 ? Math.min(level, 3) : 1;
        } catch (RuntimeException ignored) {
            return 1;
        }
    }

    /**
     * Ticks between shots for the unit's first turret, or {@code 0} if it has none.
     *
     * <p>This is the engine's own fire cadence ({@code Tank} returns 75, i.e. 1.25 s
     * per shot). Exposed so "how dangerous is that thing" is answerable without the
     * policy having to memorise a per-type table from reward alone.
     *
     * <p>The {@code getTurretCount() < 1} guard is load-bearing, not defensive
     * padding: {@code CustomUnit.getTurretShootDelay(int)} indexes its
     * {@code turretTemplates} array unchecked and throws
     * {@code ArrayIndexOutOfBoundsException} for a custom unit with no turrets.
     * Stock units always have {@code turrets[0]} allocated, so without the guard a
     * single moddeed unit type would abort the whole observation.
     */
    private float shootDelay(Unit unit) {
        try {
            if (!(unit instanceof OrderableUnit)) {
                return 0.0f;
            }
            OrderableUnit orderable = (OrderableUnit) unit;
            if (orderable.getTurretCount() < 1 || !orderable.canAttack()) {
                return 0.0f;
            }
            float delay = orderable.getTurretShootDelay(0);
            return Float.isFinite(delay) && delay > 0.0f ? delay : 0.0f;
        } catch (RuntimeException ignored) {
            return 0.0f;
        }
    }

    /** Whether this unit's weapon can hit air units (engine default is true). */
    private boolean canAttackAir(Unit unit) {
        try {
            if (!(unit instanceof OrderableUnit) || !unit.canAttack()) {
                return false;
            }
            return ((OrderableUnit) unit).canAttackFlyingUnits();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /** Whether this unit's weapon can hit ground units (engine default is true). */
    private boolean canAttackLand(Unit unit) {
        try {
            if (!(unit instanceof OrderableUnit) || !unit.canAttack()) {
                return false;
            }
            return ((OrderableUnit) unit).canAttackLandUnits();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /**
     * Current shield points. Shields absorb damage before hull in
     * {@code Unit.damage(...)}, so they are the "defence" number that changes a
     * fight's arithmetic. Reported raw together with {@link #maxShield(Unit)} so the
     * encoder can form a fraction; a unit without a shield reports 0/0.
     */
    private float shield(Unit unit) {
        try {
            float value = unit.shield;
            return Float.isFinite(value) && value > 0.0f ? value : 0.0f;
        } catch (RuntimeException ignored) {
            return 0.0f;
        }
    }

    private float maxShield(Unit unit) {
        try {
            float value = unit.maxShield;
            return Float.isFinite(value) && value > 0.0f ? value : 0.0f;
        } catch (RuntimeException ignored) {
            return 0.0f;
        }
    }

    /** True for the engine's turret family (gun towers, AA turrets, artillery). */
    private boolean isTurretType(Unit unit) {
        try {
            return unit instanceof com.corrodinggames.rts.game.units.buildings.turrets.Turret;
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /** True for the laser defence tower specifically. */
    private boolean isLaserType(Unit unit) {
        try {
            return unit
                    instanceof com.corrodinggames.rts.game.units.buildings.LaserDefence;
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /**
     * How many items a factory has queued, or 0 for anything that is not a factory.
     *
     * <p>This is the single most important field the bridge was missing. The built-in
     * AI's own behaviour -- what a behaviour-cloning teacher would demonstrate -- is
     * almost entirely *production decisions*, and the payload reported none of them:
     * measured on a real AI-pilot Lake run over 6000 ticks
     * ({@code .runtime/ai_orders_probe.json}) the teacher issued 6
     * {@code build_structure} and 23 {@code hold} orders but **zero**
     * {@code produce_unit}, while its army grew from 2 to 13 units. It was
     * producing the whole time and the observation could not see it, so there was
     * nothing to imitate.
     *
     * <p>Cast to {@link com.corrodinggames.rts.game.units.buildings.FactoryInterface}
     * and not to {@code Factory}: {@code CustomUnit} implements the interface but does
     * not extend {@code Factory}, so an {@code instanceof Factory} test would silently
     * report "no queue" for every mod-produced unit type.
     *
     * <p>{@code getTotalQueuedCount(false)} excludes the instant-purchase slot: that
     * slot is a one-off purchase the player already paid for, not work the factory is
     * doing, and counting it would make "queue_length" jump for no visual reason.
     */
    private int queueLength(Unit unit) {
        try {
            if (!(unit instanceof com.corrodinggames.rts.game.units.buildings.FactoryInterface)) {
                return 0;
            }
            com.corrodinggames.rts.game.units.buildings.FactoryInterface factory =
                    (com.corrodinggames.rts.game.units.buildings.FactoryInterface) unit;
            if (factory.isQueueEmpty()) {
                return 0;
            }
            return Math.max(0, factory.getTotalQueuedCount(false));
        } catch (RuntimeException ignored) {
            return 0;
        }
    }

    /**
     * Completion fraction of the factory's current build, in {@code [0, 1]}.
     *
     * <p>{@code Factory.getConstructionProgress()} returns
     * {@code buildQueue.buildProgress} while the queue is non-empty and the factory is
     * built, and otherwise falls through to the building's own construction progress --
     * so on a half-built factory this reports the *building's* progress, which is
     * exactly right too. A non-factory reports 0.
     */
    private float queueProgress(Unit unit) {
        try {
            if (!(unit instanceof com.corrodinggames.rts.game.units.buildings.FactoryInterface)) {
                return 0.0f;
            }
            com.corrodinggames.rts.game.units.buildings.FactoryInterface factory =
                    (com.corrodinggames.rts.game.units.buildings.FactoryInterface) unit;
            if (factory.isQueueEmpty()) {
                return 0.0f;
            }
            float progress = unit.getConstructionProgress();
            if (!Float.isFinite(progress)) {
                return 0.0f;
            }
            return Math.max(0.0f, Math.min(1.0f, progress));
        } catch (RuntimeException ignored) {
            return 0.0f;
        }
    }

    /**
     * Stable numeric type id for one unit.
     *
     * <p>The bridge used to emit {@code building ? 2 : 1}, i.e. every tank, every
     * builder and every combat unit shared one id, and every building shared
     * another. The Python side could therefore not tell an extractor from a
     * factory (so {@code upgrade_building} always selected "no building" and
     * degraded into {@code hold}) and could not tell a unit type from another at
     * all. Ids for the economy units line up with the Python mock engine
     * ({@code engine/constants.py}: tank=1, builder=2, extractor=10,
     * landFactory=11, turret=12); every other type gets a deterministic id
     * derived from its internal name so {@code obs} values stay identical across
     * runs of the same map and mod set.
     */
    private int stableTypeId(Unit unit) {
        String name = internalName(unit).toLowerCase(Locale.ROOT);
        // Mods / stock content disagree on the prefix (``c_tank`` vs ``tank``),
        // so compare on the suffix that names the unit type.
        if (name.startsWith("c_")) {
            name = name.substring(2);
        }
        switch (name) {
            case "tank":
                return 1;
            case "builder":
                return 2;
            case "extractor":
                return 10;
            case "landfactory":
                return 11;
            case "turret":
                return 12;
            default:
                // Deterministic across JVMs (String.hashCode is specified), so a
                // checkpoint trained on a map keeps seeing the same ids.
                return 1000 + (internalName(unit).hashCode() & 0x7fffffff) % 1000000;
        }
    }

    /**
     * Completion fraction of a unit, clamped to {@code [0, 1]}.
     *
     * <p>A finished unit reports {@code 1.0}; a structure that is still being
     * placed reports the engine's own {@code buildProgress}. Anything the engine
     * cannot answer falls back to 1.0 so "unknown" never reads as "just started".
     */
    private float buildProgress(Unit unit) {
        try {
            if (!unit.isBuilding()) {
                return 1.0f;
            }
            float progress = unit.buildProgress;
            if (!Float.isFinite(progress)) {
                // An uninitialised float is not evidence of anything. Treating NaN
                // as "finished" keeps the field usable on engine builds that leave
                // it unset, which is what the old code was trying to do.
                return 1.0f;
            }
            // ``<= 0`` used to be folded into the NaN case above and reported as
            // 1.0. That is a *lie with consequences*: the engine uses
            // ``buildProgress < 1.0`` in ``isNonOperational()``, so a stalled or
            // not-yet-started structure has ``buildProgress == 0.0`` and is
            // genuinely unusable, while this helper claimed it was complete.
            //
            // Measured on ``[p2]Lake (2p)``: a landFactory sat at
            // ``build_progress: 1.0, is_operational: false`` from tick 1571 to the
            // 20056-tick cap, with ``can_produce: []`` the whole time -- the two
            // fields disagreed for 18000 ticks because one read the raw engine
            // value and the other read this helper. A policy (and the opening
            // curriculum) gating on ``build_progress`` therefore believed a
            // zero-progress factory was finished and stopped re-issuing the build
            // order, so the factory never completed and ``bonus_produce`` was
            // structurally unreachable.
            return Math.max(0.0f, Math.min(1.0f, progress));
        } catch (RuntimeException ignored) {
            return 1.0f;
        }
    }

    /**
     * Whether this structure is finished and usable -- the engine's answer,
     * corrected for the one case it gets wrong.
     *
     * <p><b>Why this is not just {@code !unit.isNonOperational()}.</b> The engine
     * derives {@code isNonOperational()} from {@code buildProgress < 1.0}, and an
     * extractor reports {@code buildProgress == 0.0} for its entire life even while
     * it is producing income. Measured on {@code [p2]Lake (2p)} with a working
     * extractor on slot 0:
     *
     * <pre>
     *   tick   161 .. 3961 : progress=0.0  is_operational=false
     *                        hp=800/800 (full)  order=idle  queue_length=0
     *                        income=26.0 the whole time
     * </pre>
     *
     * <p>Every field is identical from tick 161 onward -- the structure is done and
     * idle, but the engine's progress counter never moves for it, so
     * {@code isNonOperational()} answers "not finished" forever.
     *
     * <p>That answer is not cosmetic: the opening curriculum gates expansion on
     * "every standing structure is operational"
     * ({@code env.py} {@code _issue_late_expansion}), so a single finished extractor
     * blocked <b>584 of 600</b> decisions and the economy was pinned at one mine
     * (~26 credits/s) for the whole episode while 8 pools sat free.
     *
     * <p>The correction is deliberately narrow. A structure counts as operational
     * when the engine says so <b>or</b> when it satisfies all three of:
     * <ul>
     *   <li>it is a building,</li>
     *   <li>it is at full health ({@code hp >= maxHp}), and</li>
     *   <li>it has no queued construction order ({@code buildQueue} empty).</li>
     * </ul>
     * A site that is genuinely still being built fails at least one of those: the
     * engine reports {@code buildProgress} in {@code (0,1)} and the builder holds a
     * build order, so it keeps its {@code is_operational=false}. The rule therefore
     * cannot turn a half-built factory into a "finished" one -- it only rescues the
     * types whose progress counter is never updated.
     */
    private boolean isOperational(Unit unit) {
        try {
            if (!unit.isNonOperational()) {
                return true;
            }
            if (!unit.isBuilding()) {
                return false;
            }
            // Full health plus an idle build queue is what a completed structure
            // looks like; an unfinished one is either damaged (still accruing hp as
            // it builds) or still holds the construction order.
            if (unit.hp + 1e-3f < unit.maxHp) {
                return false;
            }
            if (buildProgress(unit) > 0.0f && buildProgress(unit) < 1.0f) {
                // A real, in-flight construction: trust the engine's "not yet".
                return false;
            }
            return true;
        } catch (RuntimeException ignored) {
            return true;
        }
    }

    /** Engine internal name of the unit's type, with a safe placeholder. */
    private String internalName(Unit unit) {        try {
            String name = unit.getUnitTypeInterface().getInternalName();
            if (name != null && !name.isEmpty()) {
                return name;
            }
        } catch (RuntimeException ignored) {
            // Fall through with the placeholder.
        }
        return "?";
    }

    /** ``NONE``/``LAND``/``BUILDING``/``AIR``/``WATER``/``HOVER``/``OVER_CLIFF_WATER``... */
    private String movementType(Unit unit) {
        try {
            com.corrodinggames.rts.game.units.MovementType movement = unit.getMovementType();
            return movement == null ? "NONE" : movement.name();
        } catch (RuntimeException ignored) {
            return "NONE";
        }
    }

    private boolean isAirUnit(Unit unit) {
        try {
            return unit.isAirUnit();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /**
     * Whether this entity can shoot at all — the engine's own answer.
     *
     * <p>This is the exact query asked for, not a heuristic: {@code Unit} declares
     * {@code public abstract boolean canAttack()}, so every unit type answers it
     * (a builder / an extractor / a command center return {@code false}, tanks and
     * turrets {@code true}). It is deliberately the type-level capability and not
     * "does it have a target right now": the policy needs to know that walking into
     * range of the thing in front of it is fatal, and that is a property of its
     * type, not of the current frame's targeting state.
     *
     * <p>Falls back to {@code false} on any engine exception: a single unreadable
     * unit type must never fail the whole frame's observation.
     */
    private boolean canAttack(Unit unit) {
        try {
            return unit.canAttack();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /**
     * The entity's maximum attack range in world pixels, or {@code 0} if it has none.
     *
     * <p>{@code getMaxAttackRange()} is declared on {@link OrderableUnit}, not on
     * {@link Unit}: only orderable units own turrets and firing logic, so a map item
     * or other non-orderable unit genuinely has no attack range and reports 0.
     *
     * <p>Gated on {@link #canAttack}: the engine's max-range helper can return a
     * non-zero number for units that cannot attack (a builder's nano/repair reach is
     * a different quantity with the same shape of API). Reporting 290 px for a
     * builder would tell the policy exactly the wrong lesson, so a non-combatant is
     * reported as range 0. The two fields therefore always agree.
     */
    private float attackRange(Unit unit) {
        try {
            if (!(unit instanceof OrderableUnit)) {
                return 0.0f;
            }
            if (!unit.canAttack()) {
                return 0.0f;
            }
            float range = ((OrderableUnit) unit).getMaxAttackRange();
            if (!Float.isFinite(range) || range <= 0.0f) {
                return 0.0f;
            }
            return range;
        } catch (RuntimeException ignored) {
            return 0.0f;
        }
    }

    /**
     * Whether {@code upgrade_building} would be accepted for this unit.
     *
     * <p>Rusted Warfare models a tier upgrade as the building's current build
     * action ({@code getBuildActionId}), so a real action id means "there is a
     * next tier". Reporting the capability lets the Python side select an
     * upgrade target by what the engine says is possible instead of matching a
     * hard-coded type id.
     *
     * <p><b>The unit must also be finished.</b> {@code getBuildActionId()}
     * answers "does this type have a next tier", *not* "may I upgrade it right
     * now": a structure that is still being placed reports a real upgrade action
     * the whole time it is under construction. Measured on ``[p2]Lake (2p)``
     * ({@code .runtime/raw_entity.json}) -- a half-built extractor:
     *
     * <pre>
     *   tick 161: build_progress 0.000  is_operational false  can_upgrade TRUE
     *   tick 301: build_progress 0.029  is_operational false  can_upgrade TRUE
     *   ...       (true for every one of the 18000 ticks of the build)
     * </pre>
     *
     * <p>The Python mask opens {@code upgrade_building} whenever *any* own entity
     * reports {@code can_upgrade}, and {@code decode_action} picks one of those
     * entities. So for the entire construction of the opening extractor the mask
     * advertised an upgrade the engine then refused -- the single largest
     * rejection bucket in every run. Measured over 24 PPO updates:
     * {@code UPGRADE_UNAVAILABLE = 798} against {@code BUILD_UNAVAILABLE = 157},
     * contributing most of {@code invalid = -17.49}, the dominant negative reward.
     *
     * <p>Gating on completion makes the reported capability agree with the
     * dispatch path, which is the whole point of reporting it.
     */
    private boolean canUpgrade(Unit unit) {
        if (!unit.isBuilding()) {
            return false;
        }
        try {
            // An unterminated structure cannot be upgraded. This used
            // ``unit.isNonOperational()`` directly, which is derived from
            // ``buildProgress < 1.0`` -- and an extractor reports
            // ``buildProgress == 0.0`` forever even when it is finished and
            // producing. The result was that a completed mine reported
            // ``can_upgrade = false`` for its entire life, so the policy and the
            // curriculum both saw "nothing here can be upgraded" and the economy
            // never advanced past T1 (measured: ``extractor_level 1`` on all 1991
            // decisions of a full Lake episode).
            //
            // ``isOperational`` applies the same rule the rest of the bridge now
            // uses: finished means the engine says so, or the structure is a
            // full-health building with no in-flight construction.
            if (!isOperational(unit)) {
                return false;
            }
            if (unit instanceof OrderableUnit) {
                ActionId upgradeId = unit.getBuildActionId();
                return upgradeId != null && SpecialAction.isRealActionId(upgradeId);
            }
        } catch (RuntimeException ignored) {
            return false;
        }
        return false;
    }

    private void sendDone(RWGameEngine engine) {
        String result = gameResult != null ? gameResult : "timeout";
        boolean terminated = !"timeout".equals(result);
        // Carry the terminal entity lists.
        //
        // This used to send only result/tick/terminated, so the last ``obs`` was
        // the last time the client saw any units. ``merge_done_message`` then had
        // nothing to merge and kept the *previous* state, which means the closing
        // decision was scored against pre-terminal army counts:
        //
        //   * ``losses`` (``previous_enemy_alive - enemy_alive``) scored 0 for the
        //     kills that actually ended the game;
        //   * ``potential`` used the pre-terminal hp, so the final shaping step
        //     was blind to the wipe-out it was being paid +1.0 for;
        //   * ``eval_report.json`` recorded ``own_alive``/``enemy_alive`` from the
        //     stale frame -- measured: a winning episode reported ``enemy_alive=3``
        //     while the engine had just declared victory.
        //
        // Sending the lists makes the terminal state self-describing, so the
        // Python side merges real numbers instead of inheriting an old frame. The
        // fields are exactly the ones ``observation`` already emits, so every
        // existing consumer keeps working.
        String done = "{\"protocol\":\"ironzero/1\",\"type\":\"done\",\"message_id\":\"done-"
                + engine.simulationTick + "\",\"episode_id\":\"rwpp-bridge\",\"tick\":"
                + engine.simulationTick + ",\"result\":\"" + result + "\",\"reward\":0.0,"
                + "\"terminated\":" + terminated + ",\"truncated\":" + (!terminated);
        try {
            java.util.List<String> own = new java.util.ArrayList<>();
            java.util.List<String> enemies = new java.util.ArrayList<>();
            collectEntities(own, enemies);
            done += ",\"own_entities\":[" + String.join(",", own) + "]"
                    + ",\"visible_enemy_entities\":[" + String.join(",", enemies) + "]";
        } catch (RuntimeException ignored) {
            // A failed snapshot must not stop the verdict from being delivered:
            // the result is the load-bearing part, the lists are an improvement.
        }
        writeLine(done + "}");
    }

    /**
     * Fill the own/enemy entity JSON lists for the current frame.
     *
     * <p>Split out of the observation writer so {@link #sendDone} can attach the
     * exact same lists to the terminal message.
     */
    private void collectEntities(
            java.util.List<String> own, java.util.List<String> enemies) {
        Team player = engine != null ? engine.playerTeam : null;
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (isOwnUnit(player, unit)) {
                own.add(entityJson(unit));
            } else if (isEnemyUnit(player, unit)) {
                enemies.add(entityJson(unit));
            }
        }
    }

    /**
     * Map the game state onto ``victory`` / ``defeat``.
     *
     * Native mode (default, backward compatible): the engine exposes two
     * independent end signals: the map objective check in ``TriggerEngine``
     * (``victoryScreenShown`` / ``defeatScreenShown``) and the per-team defeat
     * bookkeeping (``Team.defeated`` / ``wipedOut``). The challenge map used for
     * training declares ``winCondition=mainBuildings`` and the AI team is
     * eliminated when its main buildings are destroyed, so either signal is
     * authoritative. Returns null while the battle is live.
     *
     * Annihilation mode: on building-less arena maps the native bookkeeping is
     * unusable --``Team.updateDefeatCheck`` marks any team without a factory or
     * builder as ``defeated`` on its first check, and the objective checker finds
     * no objective at all. Result detection therefore ignores the native flags
     * entirely and counts surviving units each frame: every enemy team at zero
     * means victory, the controlled team at zero means defeat. The first
     * {@link #ANNIHILATION_GRACE_TICKS} ticks are a load/spawn grace window, and
     * victory additionally requires that an enemy unit was actually seen alive at
     * least once (so a map that failed to spawn its AI army reports a timeout
     * instead of a bogus instant win).
     */
    private String detectGameResult(RWGameEngine engine) {
        if (gameResult != null) {
            return gameResult;
        }
        if (engine == null || engine.playerTeam == null) {
            return null;
        }
        Team player = engine.playerTeam;
        if ("objective".equals(winMode)) {
            // Stock skirmish maps declare ``type=skirmish`` and no ``winCondition``,
            // so TriggerEngine falls through to
            // ``ObjectiveType.noConstructionOrTech``: "the enemy has no factory and
            // no builder left". That objective *is* reachable on an economy map --
            // you win by destroying the enemy's production capability, not by
            // erasing its last scout.
            //
            // Why this mode has to exist. Measured on ``maps/skirmish/[p2]Lake (2p)``
            // with our side handed to the built-in AI (``--ai-pilot``) for a full
            // map-derived 20056-tick episode: both sides reached a complete economy
            // (ours 22 units / 11 types, theirs 43 units / 23 types) and the native
            // path still reported ``timeout``. The native victory channel waits for
            // ``victoryScreenShown`` or "every enemy team defeated/wipedOut", and on
            // a real economy map neither happens inside any sane tick budget, so
            // training could never observe a win.
            //
            // Two objectives are evaluated, in the engine's own priority order:
            //
            //   1. ``mainBuildings`` -- the objective ``challenge/Artillery.tmx``
            //      uses (``winCondition=mainBuilings``): the enemy loses once it has
            //      no *factory building* that is not a turret or an extractor. This
            //      is the objective a real player is judged by and it is much
            //      cheaper to reach than a total wipe.
            //   2. ``noConstructionOrTech`` -- the stock-skirmish fallback: no
            //      factory *and* no builder at all.
            //
            // Measured cost of (2) alone, AI vs AI with our side on level 3 and the
            // opponent on level -2: ``victory`` only arrived at tick 34701, after
            // our side had grown to 196 units. (1) is what makes a decisive result
            // practical, and (2) still catches the "chewed through every builder
            // too" case.
            if (engine.simulationTick <= OBJECTIVE_GRACE_TICKS) {
                return null;
            }
            Capability own = capabilityOf(player);
            Capability enemies = capabilityOfEnemies(player);
            if (enemies.hasCapability) {
                objectiveEnemySeen = true;
            }
            if (own.hasCapability) {
                ownSawCapability = true;
            }
            boolean enemyDecided = enemies.mainBuildings == 0 || !enemies.hasCapability;
            if (objectiveEnemySeen && enemyDecided) {
                gameResult = "victory";
                System.out.println("[ironzero] game over: victory (objective) tick="
                        + engine.simulationTick + " enemyMainBuildings=" + enemies.mainBuildings
                        + " enemyFactories=" + enemies.factories
                        + " enemyBuilders=" + enemies.builders);
                return gameResult;
            }
            if (!own.hasCapability && ownSawCapability) {
                gameResult = "defeat";
                System.out.println("[ironzero] game over: defeat (objective) tick="
                        + engine.simulationTick + " ownFactories=" + own.factories
                        + " ownBuilders=" + own.builders);
                return gameResult;
            }
            // Military dominance: a force the enemy cannot come back from. See
            // DOMINANCE_* for why the objectives above are not enough on their own.
            Tally engagement = tally(player);
            if (
                objectiveEnemySeen
                && engagement.ownCombat >= DOMINANCE_OWN_MIN_UNITS
                && engagement.enemyCombat > 0
                && engagement.ownCombat >= engagement.enemyCombat * DOMINANCE_ARMY_RATIO
                && engagement.ownCombatHp >= engagement.enemyCombatHp * DOMINANCE_HP_RATIO
            ) {
                gameResult = "victory";
                System.out.println("[ironzero] game over: victory (dominance) tick="
                        + engine.simulationTick + " ownCombat=" + engagement.ownCombat
                        + " enemyCombat=" + engagement.enemyCombat
                        + " ownCombatHp=" + fmt(engagement.ownCombatHp)
                        + " enemyCombatHp=" + fmt(engagement.enemyCombatHp));
                return gameResult;
            }
            return null;
        }
        if ("annihilation".equals(winMode)) {
            if (engine.simulationTick <= ANNIHILATION_GRACE_TICKS) {
                return null;
            }
            Tally tally = tally(player);
            if (tally.enemyAlive > 0) {
                enemySeen = true;
            }
            if (enemySeen && tally.enemyAlive == 0) {
                gameResult = "victory";
                System.out.println("[ironzero] game over: victory (annihilation) tick="
                        + engine.simulationTick + " ownAlive=" + tally.ownAlive
                        + " enemyAlive=" + tally.enemyAlive + " ownHp=" + fmt(tally.ownHp)
                        + " enemyHp=" + fmt(tally.enemyHp));
                return gameResult;
            }
            if (tally.ownAlive == 0) {
                gameResult = "defeat";
                System.out.println("[ironzero] game over: defeat (annihilation) tick="
                        + engine.simulationTick + " ownAlive=" + tally.ownAlive
                        + " enemyAlive=" + tally.enemyAlive + " ownHp=" + fmt(tally.ownHp)
                        + " enemyHp=" + fmt(tally.enemyHp));
                return gameResult;
            }
            return null;
        }
        if (engine.defeatScreenShown || player.defeated || player.wipedOut) {
            gameResult = "defeat";
            System.out.println("[ironzero] game over: defeat tick=" + engine.simulationTick
                    + " defeated=" + player.defeated + " wipedOut=" + player.wipedOut
                    + " screen=" + engine.defeatScreenShown);
            return gameResult;
        }
        if (engine.victoryScreenShown) {
            gameResult = "victory";
            System.out.println("[ironzero] game over: victory tick=" + engine.simulationTick);
            return gameResult;
        }
        boolean anyEnemy = false;
        boolean enemyAlive = false;
        for (Object item : Team.getTeams()) {
            if (!(item instanceof Team)) {
                continue;
            }
            Team team = (Team) item;
            if (team == null || team == player || player.isSameTeam(team)) {
                continue;
            }
            anyEnemy = true;
            if (!team.defeated && !team.wipedOut) {
                enemyAlive = true;
                break;
            }
        }
        if (anyEnemy) {
            enemySeen = true;
        }
        if (enemySeen && !enemyAlive) {
            gameResult = "victory";
            System.out.println("[ironzero] game over: victory (all enemy teams defeated) tick="
                    + engine.simulationTick);
            return gameResult;
        }
        return null;
    }

    /** Surviving-unit snapshot used by annihilation win detection and the obs. */
    private static final class Tally {
        int ownAlive;
        int enemyAlive;
        double ownHp;
        double enemyHp;
        // Attack-capable subset, used by the dominance verdict: a builder or an
        // extractor must not count as "army" or a farm would read as a victory.
        int ownCombat;
        int enemyCombat;
        double ownCombatHp;
        double enemyCombatHp;
    }

    /**
     * Production capability of one side, for the ``objective`` win mode.
     *
     * Modelled directly on the engine's own skirmish objective
     * (``TriggerEngine.matchesObjective(ObjectiveType.noConstructionOrTech)``):
     * a factory building or a builder counts. ``sawCapability`` records that the
     * side *had* one at some point, so a map that simply spawned nothing can
     * never be scored as an instant win or loss.
     */
    private static final class Capability {
        int factories;
        int builders;
        /** Factory buildings that are neither turrets nor extractors (engine objective). */
        int mainBuildings;
        boolean hasCapability;
        boolean sawCapability;
    }

    /** Production capability of the controlled team. */
    private Capability capabilityOf(Team player) {
        Capability capability = new Capability();
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f || unit.dead || !isOperational(unit)) {
                continue;
            }
            if (!isOwnUnit(player, unit)) {
                continue;
            }
            if (unit.isFactoryBuilding()) {
                capability.factories++;
                // Mirrors TriggerEngine.matchesObjective(ObjectiveType.mainBuildings):
                // a factory building that is not a turret and not an extractor.
                if (!(unit instanceof Turret) && !(unit instanceof Extractor)) {
                    capability.mainBuildings++;
                }
            }
            if (unit.isBuilder()) {
                capability.builders++;
            }
        }
        capability.hasCapability = capability.factories > 0 || capability.builders > 0;
        capability.sawCapability = capability.hasCapability;
        return capability;
    }

    /** Production capability summed over every hostile team. */
    private Capability capabilityOfEnemies(Team player) {
        Capability capability = new Capability();
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f || unit.dead || !isOperational(unit)) {
                continue;
            }
            if (!isEnemyUnit(player, unit)) {
                continue;
            }
            if (unit.isFactoryBuilding()) {
                capability.factories++;
                if (!(unit instanceof Turret) && !(unit instanceof Extractor)) {
                    capability.mainBuildings++;
                }
            }
            if (unit.isBuilder()) {
                capability.builders++;
            }
        }
        capability.hasCapability = capability.factories > 0 || capability.builders > 0;
        capability.sawCapability = capability.hasCapability;
        return capability;
    }

    private Tally tally(Team player) {
        Tally tally = new Tally();
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            boolean combat = canFight(unit);
            if (isOwnUnit(player, unit)) {
                tally.ownAlive++;
                tally.ownHp += unit.hp;
                if (combat) {
                    tally.ownCombat++;
                    tally.ownCombatHp += unit.hp;
                }
            } else if (isEnemyUnit(player, unit)) {
                tally.enemyAlive++;
                tally.enemyHp += unit.hp;
                if (combat) {
                    tally.enemyCombat++;
                    tally.enemyCombatHp += unit.hp;
                }
            }
        }
        return tally;
    }

    /**
     * Whether this unit is part of the *army*, for the dominance verdict.
     *
     * Deliberately the same notion the reward shaping uses (``can_attack``):
     * buildings and builders are not army, so a player sitting on a big economy
     * with no units can never be scored as dominant. Turrets do count -- they
     * fire -- but the ratio thresholds in ``DOMINANCE_*`` keep a fortified
     * turtle from being called a decisive victory against a live enemy army.
     */
    private boolean canFight(Unit unit) {
        if (!(unit instanceof OrderableUnit)) {
            return false;
        }
        if (unit.isBuilder()) {
            return false;
        }
        // A building with a weapon (turret) can fight; a factory/extractor cannot.
        if (unit.isBuilding() && !unit.isFactoryBuilding()) {
            return true;
        }
        return !unit.isBuilding();
    }

    /**
     * True for units the controlled team owns or shares control with. Neutral
     * (gray) units are never "own".
     */
    private boolean isOwnUnit(Team player, Unit unit) {
        if (unit.team == null) {
            return false;
        }
        if (player == null) {
            return unit.team.teamId == 0;
        }
        return unit.team == player || player.isSameTeam(unit.team);
    }

    /**
     * True for hostile units. ``Team.isEnemy`` already excludes the neutral gray
     * team, so neutral map objects never count towards the enemy army.
     */
    private boolean isEnemyUnit(Team player, Unit unit) {
        if (unit.team == null || isOwnUnit(player, unit)) {
            return false;
        }
        return player == null ? unit.team.teamId != 0 : player.isEnemy(unit.team);
    }

    /** JSON array of the ids of every enemy team present on the map. */
    private String enemyTeamsJson(Team player) {
        StringBuilder json = new StringBuilder("[");
        boolean first = true;
        for (Object item : Team.getTeams()) {
            if (!(item instanceof Team)) {
                continue;
            }
            Team team = (Team) item;
            if (team == null || team == player || (player != null && player.isSameTeam(team))) {
                continue;
            }
            if (player != null && !player.isEnemy(team)) {
                continue;
            }
            if (!first) {
                json.append(',');
            }
            json.append(team.teamId);
            first = false;
        }
        return json.append(']').toString();
    }

    /**
     * Annihilation mode only: undo the engine's ``defeated`` / ``wipedOut``
     * bookkeeping after every simulated frame.
     *
     * ``Team.updateDefeatCheck`` treats "no factory and no builder" as defeat for
     * a team that still has units, which is exactly the 25v25 pure-tank arena
     * layout. The flags do not stop combat, but they leak into
     * ``countRemainingPlayers`` / winners announcements / map scripts and would
     * make the native result path report an immediate defeat. Combat state itself
     * (dead units, ``wipedOut`` unit removal) is untouched: a team that really
     * loses all its units is detected from the unit counts instead.
     */
    private void clearNativeDefeatFlags() {
        for (Object item : Team.getTeams()) {
            if (!(item instanceof Team)) {
                continue;
            }
            Team team = (Team) item;
            if (team == null) {
                continue;
            }
            team.defeated = false;
            team.wipedOut = false;
        }
    }

    /**
     * ``--unlock-ai``: hand the map-placed enemy army to the built-in AI.
     *
     * Units placed in a map editor are loaded with ``Unit.aiLimitedUse = true``,
     * which keeps them out of ``AIUnitGroup.recruitUnits`` and the AI's attacker
     * counts, so a 25v25 arena otherwise has a completely passive opponent.
     * Clearing that flag is the whole fix. Must run after the map finished loading.
     *
     * <p><b>Do not also set ``AI.aggressiveMode``.</b> This method used to, on the
     * theory that "aggressive" means "attacks". It does the opposite, because the
     * engine reads the flag in two places that both *disable* work
     * ({@code Base.chooseBuildingToBuild} returns {@code null} at
     * {@code Base.java:229} and {@code Base.defendBase()} is skipped at
     * {@code Base.java:1246} when it is set). Measured on Lake over 20056 ticks,
     * zero client actions, only the flag differing: with it the AI built *nothing
     * but* builders and scouts for the whole episode (3 unit types total, nearest
     * approach 1291 px); without it the AI built extractors, a land and an air
     * factory, turrets, tanks, artillery and hover tanks (12 types, nearest
     * approach 781 px). {@code aggressiveMode} is a scenario-authoring flag that
     * {@code TriggerEngine} sets for maps declaring {@code basicAI}; it is not the
     * AI's aggression switch for normal skirmish play.
     */
    private void unlockAiTeams() {
        Team player = engine != null ? engine.playerTeam : null;
        int teams = 0;
        int units = 0;
        for (Object item : Team.getTeams()) {
            if (!(item instanceof Team)) {
                continue;
            }
            Team team = (Team) item;
            if (team == null || team == player || !team.isAI) {
                continue;
            }
            teams++;
            for (Object unitItem : Unit.unitList) {
                if (!(unitItem instanceof Unit)) {
                    continue;
                }
                Unit unit = (Unit) unitItem;
                if (unit.team == team && unit.aiLimitedUse) {
                    unit.aiLimitedUse = false;
                    units++;
                }
            }
        }
        System.out.println("[ironzero] unlock-ai: aiTeams=" + teams + " unlockedUnits=" + units);
    }

    /**
     * Every AI team's resolved difficulty and income multiplier, as JSON.
     *
     * <p>Reported per observation so a run records -- rather than assumes -- which
     * teacher strength produced its labels. {@code getEffectiveAIDifficulty()} has
     * two surprises that only an explicit readback catches: a team-level
     * {@code aiDifficulty} is ignored unless {@code aiDifficultyLocked} is set, and
     * the fallback is a *global* persisted preference, so the same map can play at
     * different strengths on different machines.
     */
    private String aiTeamsJson() {
        StringBuilder out = new StringBuilder("[");
        boolean first = true;
        for (Object item : Team.getTeams()) {
            if (!(item instanceof Team)) {
                continue;
            }
            Team team = (Team) item;
            if (team == null || !team.isAI) {
                continue;
            }
            int level;
            double multiplier;
            try {
                level = team.getEffectiveAIDifficulty();
                multiplier = team.getAIDifficultyIncomeMultiplier();
            } catch (RuntimeException ignored) {
                continue;
            }
            if (!first) {
                out.append(',');
            }
            out.append("{\"team_id\":").append(team.teamId)
                    .append(",\"difficulty\":").append(level)
                    .append(",\"income_multiplier\":").append(fmt(multiplier))
                    .append(",\"locked\":").append(team.aiDifficultyLocked)
                    .append('}');
            first = false;
        }
        return out.append(']').toString();
    }

    /**
     * The engine's own difficulty for the controlled team, used when only
     * ``--enemy-ai-difficulty`` is requested.
     *
     * <p>Reads the value the engine already resolved instead of assuming a level,
     * so "pin the enemies, leave my teacher alone" really means leave it alone.
     */
    private int defaultPlayerAiDifficulty() {
        Team player = engine != null ? engine.playerTeam : null;
        if (player == null) {
            return 0;
        }
        try {
            return player.getEffectiveAIDifficulty();
        } catch (RuntimeException ignored) {
            return 0;
        }
    }

    /**
     * ``--ai-difficulty N``: pin the AI difficulty of every AI team on the map.
     *
     * <p>Rusted Warfare has six levels ({@code -2..3}, i.e. 非常简单 / 简单 / 正常 /
     * 困难 / 非常困难 / 炼狱模式). They are not cosmetic: the level feeds
     * {@code Team.getAIDifficultyIncomeMultiplier()}, which is the AI's resource
     * cheat, and the resolved level also drives {@code getAIDifficultyArmyValue()}
     * (how big an army the AI builds before attacking) and the AI's build order
     * delays. Measured multipliers: {@code -2 -> x0.40}, {@code -1 -> x0.70},
     * {@code 0 -> x1.00}, {@code 1 -> x1.40}, {@code 2 -> x1.80}, {@code 3 -> x3.70}
     * (level 3 also gets a flat +1.5 bonus, read from the bytecode and confirmed
     * on the Lake map).
     *
     * <p>Without the lock the team-level {@code aiDifficulty} is ignored and the
     * global {@code SettingsEngine.aiDifficulty} wins (default {@code 0}), which is
     * why the pilot measured x1.00 income.
     */
    private void applyAiDifficulty() {
        int requested = aiDifficulty.intValue();
        // ``--enemy-ai-difficulty`` overrides the enemy side only, so the teacher
        // and the opponent can differ. This is not tuning for taste: the built-in
        // AI plays *entirely differently* under pressure, and a BC teacher that
        // never gets attacked only ever demonstrates "build up and wait".
        Integer enemyRequested = enemyAiDifficulty;
        Team player = engine != null ? engine.playerTeam : null;
        int applied = 0;
        int skipped = 0;
        for (Object item : Team.getTeams()) {
            if (!(item instanceof Team)) {
                continue;
            }
            Team team = (Team) item;
            if (team == null) {
                continue;
            }
            if (!team.isAI) {
                // A non-AI team's difficulty is never consulted, so pinning it would
                // only produce a confusing log line.
                skipped++;
                continue;
            }
            int level = requested;
            if (enemyRequested != null && team != player) {
                level = enemyRequested.intValue();
            }
            try {
                team.aiDifficulty = level;
                team.aiDifficultyLocked = true;
                team.aiDifficultyOverride = Integer.valueOf(level);
                applied++;
            } catch (RuntimeException failure) {
                System.out.println("[ironzero] ai-difficulty: failed on team "
                        + team.teamId + ": " + failure);
            }
        }
        System.out.println("[ironzero] ai-difficulty: requested=" + requested
                + (enemyRequested != null ? " enemy=" + enemyRequested : "")
                + " appliedTeams=" + applied + " nonAiTeams=" + skipped
                + " incomeMultiplier=x" + fmt(incomeMultiplierFor(requested)));
    }

    /**
     * Every AI team's resolved difficulty and income multiplier, as a log string.
     *
     * <p>Reported at world-ready so an operator can tell from the log alone which
     * teacher strength a BC run actually used instead of inferring it from income.
     */
    private String effectiveAiDifficultyDescription() {
        StringBuilder out = new StringBuilder();
        out.append("none");
        boolean first = true;
        for (Object item : Team.getTeams()) {
            if (!(item instanceof Team)) {
                continue;
            }
            Team team = (Team) item;
            if (team == null || !team.isAI) {
                continue;
            }
            int level;
            double multiplier;
            try {
                level = team.getEffectiveAIDifficulty();
                multiplier = team.getAIDifficultyIncomeMultiplier();
            } catch (RuntimeException failure) {
                continue;
            }
            if (!first) {
                out.append(',');
            } else {
                out.setLength(0);
            }
            out.append("team").append(team.teamId)
                    .append("=level").append(level)
                    .append("/x").append(fmt(multiplier));
            first = false;
        }
        return out.toString();
    }

    /** The documented income cheat for one difficulty level (for log lines). */
    private static double incomeMultiplierFor(int level) {
        double multiplier = 1.0;
        if (level > 0) {
            multiplier += level * 0.4;
        } else {
            multiplier += level * 0.3;
        }
        if (level == 3) {
            // Verified against the bytecode: level 3 adds 1.5f, not 1.0f, so
            // 3 -> 1 + 3*0.4 + 1.5 = 3.7 (measured x3.7 on the Lake map).
            multiplier += 1.5;
        }
        return Math.max(0.1, multiplier);
    }

    /**
     * Replace the controlled team with a built-in AI instance (``--ai-pilot``).
     *
     * <p>The engine only ticks teams that are {@code instanceof AI}; the controlled
     * team is normally a plain {@code Player(0)}, so nothing drives it and the
     * policy has to learn macro play from scratch. Swapping in an {@code AI} for
     * the same team id gives us a competent player whose orders we can record.
     *
     * <p>The map is loaded with {@code loadMapWithAI(..., aiCount=1, ...)}, which
     * already created an AI for the *enemy* slot. The controlled team keeps its
     * own units either way: {@code Team} membership is by {@code teamId}, and unit
     * ownership is resolved from {@code unit.team}, so replacing the team object
     * only adds the AI brain -- it does not move or drop units.
     */
    private void installAiPilot() {
        if (engine == null) {
            return;
        }
        Team current = engine.playerTeam;
        if (current instanceof AI) {
            System.out.println("[ironzero] ai-pilot: team " + current.teamId + " already an AI");
            return;
        }
        int teamId = current != null ? current.teamId : 0;
        try {
            // The second constructor argument is ``replaceInTeamArray``: it must
            // be true, otherwise the AI is created but never registered in
            // ``Team.teamArray`` and therefore never ticked by
            // ``Team.updateTeams`` -- it looks installed (playerTeam points at it)
            // but plays nothing, which is exactly the silent failure measured on
            // the first attempt (own funds frozen at 4000 for 192 s, orders=0).
            AI pilot = new AI(teamId, true);
            pilot.isAI = true;
            // NOT ``aggressiveMode``: the engine treats that flag as "this is a
            // scripted scenario actor", and it stops the AI from building anything
            // (``Base.chooseBuildingToBuild`` returns null) and from defending
            // (``Base.updateBuildLogic`` skips ``defendBase``). A pilot taught with
            // it on only ever produced builders and scouts -- see unlockAiTeams.
            pilot.aiDifficulty = current != null ? current.aiDifficulty : pilot.aiDifficulty;
            // Carry the economy over: AI.updateResourceTimers reads credits from
            // its own Team instance, so a fresh team would start at 0 funds and
            // never build anything.
            if (current != null) {
                pilot.credits = current.credits;
                pilot.antiLagCredits = current.antiLagCredits;
                pilot.name = current.name != null ? current.name : "IronZeroPilot";
                pilot.startingUnitsOverride = current.startingUnitsOverride;
            }
            engine.playerTeam = pilot;
            // Re-home every own unit onto the new team object. Unit ownership is
            // compared with reference equality (``unit.team == this`` in
            // ``Team.updateDefeatCheck``, ``unit.team == team`` in every bridge
            // helper), so a fresh AI that nobody points at looks like an empty
            // team: the engine declared "Player had no starting units" and called
            // the game at tick=1. Re-pointing the units is what actually hands the
            // army over.
            int unlocked = 0;
            int moved = 0;
            for (Object item : Unit.unitList) {
                if (!(item instanceof Unit)) {
                    continue;
                }
                Unit unit = (Unit) item;
                if (unit.team == current) {
                    unit.team = pilot;
                    moved++;
                }
                if (unit.team == pilot && unit.aiLimitedUse) {
                    // Map-placed units arrive with aiLimitedUse=true, which keeps
                    // them out of AIUnitGroup.recruitUnits; without clearing it the
                    // pilot would sit on its units forever.
                    unit.aiLimitedUse = false;
                    unlocked++;
                }
            }
            // Rebuild the team caches from the re-homed units and mark the team as
            // having had units, otherwise ``updateDefeatCheck``/``getUnitsMax``
            // still reason about an empty roster.
            try {
                Team.markAllUnitCachesDirty();
                pilot.hasHadUnits = moved > 0;
                pilot.defeated = false;
                pilot.wipedOut = false;
                pilot.eliminated = false;
                pilot.unitCachesDirty = true;
            } catch (RuntimeException ignored) {
                // Cache bookkeeping is best effort; the reference swap above is
                // what makes the army usable.
            }
            System.out.println("[ironzero] ai-pilot: team=" + teamId
                    + " credits=" + fmt(pilot.credits)
                    + " movedUnits=" + moved + " unlockedUnits=" + unlocked);
        } catch (RuntimeException failure) {
            // Fall back to the normal human-controlled team: an unusable pilot is
            // worse than no pilot, and the caller can still drive the policy.
            System.out.println("[ironzero] ai-pilot FAILED: " + failure);
            engine.playerTeam = current;
            aiPilotTeam = false;
            recordAiOrders = false;
        }
    }

    /**
     * The controlled team's resolved AI difficulty, or {@code null} for a human team.
     *
     * <p>Exported in ``info.ai_difficulty`` so a run's logs and dataset metadata
     * record which teacher strength produced the labels.
     */
    private String ownAiDifficulty() {
        Team player = engine != null ? engine.playerTeam : null;
        if (player == null || !player.isAI) {
            return "null";
        }
        try {
            return Integer.toString(player.getEffectiveAIDifficulty());
        } catch (RuntimeException ignored) {
            return "null";
        }
    }

    /**
     * The controlled team's *changed* orders since the last observation.
     *
     * <p>Read from each own unit's active {@link Waypoint}. This is the label a
     * behaviour-cloning dataset needs: the action the expert actually took, in the
     * same vocabulary the policy emits (move / attack_move / attack_target /
     * build / repair / hold), plus the target coordinates or entity id.
     *
     * <p><b>Only changes are emitted.</b> A per-frame snapshot is the wrong label
     * distribution: measured over 15065 frames of AI play, 87% of entries were
     * {@code hold} ({@code hold: 13057, attack_move: 1519, move_group: 489}),
     * because a unit keeps the same order for many seconds. Training on that
     * teaches "stand still". Emitting only changes turns the same run into ~1600
     * genuine decisions instead.
     *
     * <p>The signature deliberately excludes position: the {@code hold} "target" is
     * the unit's own {@code x/y}, which changes every frame as it moves, so
     * including it would defeat the de-duplication entirely.
     */
    private String aiOrdersJson() {
        Team player = engine != null ? engine.playerTeam : null;
        StringBuilder out = new StringBuilder("[");
        boolean first = true;
        java.util.HashSet<Long> alive = new java.util.HashSet<>();
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (!isOwnUnit(player, unit)) {
                continue;
            }
            Waypoint waypoint = null;
            try {
                waypoint = unit.getCurrentWaypoint();
            } catch (RuntimeException ignored) {
                continue;
            }
            String kind = "hold";
            String typeName = "";
            double tx = unit.x;
            double ty = unit.y;
            long targetId = -1L;
            String buildType = "";
            int buildLevel = 0;
            if (waypoint != null) {
                try {
                    WaypointType type = waypoint.getType();
                    typeName = type == null ? "" : type.name();
                    tx = waypoint.getTargetX();
                    ty = waypoint.getTargetY();
                    Unit target = waypoint.getTargetUnit();
                    if (target != null) {
                        targetId = target.id;
                    }
                    UnitTypeInterface buildUnit = waypoint.getUnitType();
                    if (buildUnit != null) {
                        buildType = buildUnit.getInternalName();
                        buildLevel = waypoint.getBuildTechLevel();
                    }
                    kind = waypointKind(type);
                } catch (RuntimeException ignored) {
                    // Keep the "hold" default: an unreadable waypoint is not an
                    // order we can label.
                }
            }
            // Attack target is reported separately from the waypoint: a unit can
            // hold position and still have an auto-acquired attack target, and
            // that is exactly what a micro-management teacher demonstrates.
            Unit attackTarget = null;
            try {
                attackTarget = unit.getEffectiveAttackTarget();
            } catch (RuntimeException ignored) {
                // No target information available.
            }
            long attackTargetId = attackTarget != null ? attackTarget.id : -1L;
            alive.add(Long.valueOf(unit.id));

            // Signature = everything that identifies the order. Coordinates are
            // only meaningful for orders that actually carry a destination: for a
            // standing order the "target" is the unit's own x/y, which changes
            // every frame while it walks, so including it re-emits a "new" hold
            // order on every step -- measured: 478 change events of which 91% were
            // still `hold`, i.e. the de-duplication did not work at all. Whole
            // pixels are used for the rest so sub-pixel drift is not a decision.
            StringBuilder signature = new StringBuilder(64);
            signature.append(kind).append('|').append(typeName).append('|')
                    .append(targetId).append('|').append(attackTargetId).append('|')
                    .append(buildType).append('|').append(buildLevel);
            if (waypoint != null && !"hold".equals(kind)) {
                signature.append('|').append(Math.round(tx)).append(',').append(Math.round(ty));
            }
            String signatureText = signature.toString();
            String previous = aiOrderSignatures.put(Long.valueOf(unit.id), signatureText);
            if (signatureText.equals(previous)) {
                continue;
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            out.append("{\"entity_id\":").append(unit.id)
                    .append(",\"kind\":\"").append(kind).append('"')
                    .append(",\"waypoint_type\":\"").append(typeName).append('"')
                    .append(",\"x\":").append(fmt(tx))
                    .append(",\"y\":").append(fmt(ty))
                    .append(",\"target_entity_id\":").append(targetId);
            if (!buildType.isEmpty()) {
                out.append(",\"build_type\":\"").append(buildType).append('"')
                        .append(",\"build_tech_level\":").append(buildLevel);
            }
            out.append(",\"attack_target_id\":").append(attackTargetId).append('}');
        }
        // Forget dead units, otherwise an id reused by a later unit would inherit
        // the old signature and its first real order would be swallowed.
        aiOrderSignatures.keySet().retainAll(alive);
        out.append(']');
        return out.toString();
    }

    /** Map an engine {@code WaypointType} onto the policy's action vocabulary. */
    private String waypointKind(WaypointType type) {
        if (type == null) {
            return "hold";
        }
        switch (type.name()) {
            case "move":
                return "move_group";
            case "attackMove":
            case "patrol":
                return "attack_move";
            case "attack":
                return "attack_target";
            case "build":
                return "build_structure";
            case "repair":
                return "repair_target";
            case "guard":
            case "guardAt":
            case "follow":
                return "hold";
            default:
                return "hold";
        }
    }

    private void readClientCommands() {
        if (reader == null) {
            return;
        }
        try {
            while (reader.ready()) {
                String line = reader.readLine();
                if (line == null) {
                    stop = true;
                    return;
                }
                if (line.isBlank()) {
                    continue;
                }
                JsonValue value;
                try {
                    value = Json.parse(line);
                } catch (RuntimeException failure) {
                    continue;
                }
                if (!value.isObject()) {
                    continue;
                }
                JsonObject message = value.asObject();
                String type = message.getString("type", "");
                if ("shutdown".equals(type)) {
                    stop = true;
                    return;
                }
                if ("step".equals(type)) {
                    applyStep(message);
                }
            }
        } catch (IOException failure) {
            stop = true;
        }
    }

    /** Translate an ironzero/1 step into native unit commands (stage 2). */
    private void applyStep(JsonObject message) {
        String decisionId = message.getString("decision_id", "d-0");
        JsonArray actions = message.get("actions") instanceof JsonArray
                ? message.get("actions").asArray()
                : new JsonArray();
        // One entry per action: null = accepted, otherwise the ErrorCode name.
        List<String> actionOutcomes = new ArrayList<>();
        for (JsonValue item : actions) {
            if (!item.isObject()) {
                actionOutcomes.add("INVALID_MESSAGE");
                continue;
            }
            JsonObject action = item.asObject();
            String kind = action.getString("kind", "");
            int unitGroup = action.getInt("unit_group", GROUP_BROADCAST);
            CommandControllerCommand command = null;
            if ("move_group".equals(kind) || "attack_move".equals(kind)) {
                float x = (float) action.getDouble("x", 0.0);
                float y = (float) action.getDouble("y", 0.0);
                command = engine.commandController.createCommandForTeam(engine.playerTeam);
                addOwnOrderableUnits(command, unitGroup);
                if ("move_group".equals(kind)) {
                    command.setMoveWaypoint(x, y);
                } else {
                    command.setAttackMoveWaypoint(x, y);
                }
            } else if ("attack_target".equals(kind)) {
                long targetId = action.getLong("target_entity_id", -1L);
                Unit target = findUnitById(targetId);
                if (target != null) {
                    command = engine.commandController.createCommandForTeam(engine.playerTeam);
                    addOwnOrderableUnits(command, unitGroup);
                    command.setAttackUnitWaypoint(target);
                } else {
                    // The named target is gone. This is not a mask bug: the policy
                    // picked an enemy slot that was visible when the observation
                    // was produced, but the unit died (or left vision) before the
                    // order arrived. Reported separately from ACTION_MASKED so the
                    // histogram can tell the two apart -- they need different
                    // fixes (resend-time revalidation vs. mask repair).
                    actionOutcomes.add("TARGET_NOT_VISIBLE");
                    continue;
                }
            } else if ("stop".equals(kind) || "hold".equals(kind)) {
                // Hold ground: one command per unit so each keeps its own position.
                // "hold" additionally switches the unit into guard-area fire mode.
                int held = holdGround(unitGroup, "hold".equals(kind));
                // The selected group has no living unit. The mask opens a group
                // slot only while that group has at least one alive unit, so this
                // means the group died between the observation and the order.
                actionOutcomes.add(held > 0 ? null : "UNIT_NOT_OWNED");
                continue;
            } else if ("build_extractor".equals(kind) || "build_structure".equals(kind)) {
                String structure = "build_extractor".equals(kind)
                        ? "extractor"
                        : action.getString("structure_type", "landFactory");
                // Accept both the protocol's snake_case names and the engine's
                // camelCase enum constants (landFactory / airFactory / turret).
                structure = resolveUnitTypeName(structure);
                if (resolveUnitType(structure) == null) {
                    actionOutcomes.add("TARGET_INVALID");
                    continue;
                }
                float x = (float) action.getDouble("x", 0.0);
                float y = (float) action.getDouble("y", 0.0);
                if ("build_extractor".equals(kind)) {
                    // The policy names a resource slot; resolve it to map coordinates.
                    double[] slot = resourceSlotPosition(action.getInt("slot_id", 0));
                    if (slot != null) {
                        x = (float) slot[0];
                        y = (float) slot[1];
                    }
                }
                int built = buildWithBuilders(unitGroup, structure, x, y);
                // Distinguish "no builder can build this at all" from
                // "builders exist but are out of range / busy". Both are
                // rejections, but the first is a permanent capability gap.
                actionOutcomes.add(built > 0 ? null : "BUILD_UNAVAILABLE");
                continue;
            } else if ("produce_unit".equals(kind)) {
                // Factory production. The factory is named by entity id; when the
                // policy leaves it out (or names a dead unit) the nearest own
                // factory in the selected group is used instead.
                long factoryId = action.getLong("factory_entity_id", -1L);
                String unitTypeName = action.getString("unit_type", "tank");
                actionOutcomes.add(produceUnit(factoryId, unitTypeName) ? null : "PRODUCE_UNAVAILABLE");
                continue;
            } else if ("upgrade_building".equals(kind)) {
                // Extractor / factory tier upgrade: the engine exposes it as the
                // building's current build action id.
                long buildingId = action.getLong("building_entity_id", -1L);
                actionOutcomes.add(upgradeBuilding(buildingId) ? null : "UPGRADE_UNAVAILABLE");
                continue;
            } else if ("noop".equals(kind)) {
                // Explicit "do nothing". Construction and production need many
                // ticks of uninterrupted progress; every order cancels the
                // in-flight one, so the policy must be able to skip a decision.
                // A no-op dispatches no command at all (an empty action list is
                // treated the same way).
                actionOutcomes.add(null);
                continue;
            } else {
                // Unknown kind: name it explicitly instead of a generic mask error.
                actionOutcomes.add("INVALID_MESSAGE");
                continue;
            }
            if (command != null && command.getUnitCount() > 0) {
                actionOutcomes.add(null);
            } else {
                // A move/attack-move whose unit group has no living member. The
                // group head is only open while the group has an alive unit, so
                // this is the same observation-vs-order race as above. Naming it
                // separately keeps ACTION_MASKED meaningful for the economy kinds.
                actionOutcomes.add("UNIT_NOT_OWNED");
            }
        }
        engine.commandController.dispatchQueuedCommands();
        // Wire format: emit one receipt per action so the policy can attribute
        // success/failure precisely. Only fields declared by
        // ``protocol.ReceiptMessage`` are allowed (extra="forbid"): the earlier
        // ``accepted_count``/``rejected_count`` extras made every official
        // client fail to parse the line.
        int receiptIndex = 0;
        for (String reason : actionOutcomes) {
            writeLine(receiptLine(decisionId, reason, receiptIndex++));
        }
        if (actionOutcomes.isEmpty()) {
            writeLine(receiptLine(decisionId, null, receiptIndex));
        }
    }

    /** One protocol-valid receipt line; {@code reason == null} means accepted. */
    private String receiptLine(String decisionId, String reason, int index) {
        boolean ok = reason == null;
        StringBuilder out = new StringBuilder(160);
        out.append("{\"protocol\":\"ironzero/1\",\"type\":\"receipt\",\"message_id\":\"receipt-")
                .append(engine.simulationTick).append('-').append(index)
                .append("\",\"episode_id\":\"rwpp-bridge\",\"tick\":").append(engine.simulationTick)
                .append(",\"decision_id\":\"").append(decisionId)
                .append("\",\"accepted\":").append(ok)
                .append(",\"executed_at_tick\":").append(ok ? Integer.toString(engine.simulationTick + 1) : "null");
        if (!ok) {
            out.append(",\"rejected_reason\":\"").append(reason).append('"');
        }
        out.append('}');
        return out.toString();
    }

    /**
     * Resolve a policy-side unit-type name to the game's type descriptor.
     *
     * <p>Uses {@code UnitType.findUnitType}, which resolves both the built-in
     * enum constants (case-insensitively) and units contributed by mods /
     * custom unit definitions. The previous implementation called
     * {@code UnitType.valueOf}, so anything that was not an exact enum constant
     * -- including every custom unit -- resolved to {@code null} and the order
     * was rejected before it reached the engine.
     */
    private UnitTypeInterface resolveUnitType(String name) {
        if (name == null || name.isEmpty()) {
            return null;
        }
        try {
            return UnitType.findUnitType(name);
        } catch (RuntimeException unknown) {
            return null;
        }
    }

    /**
     * Queue one unit at an own factory ({@code produce_unit}).
     *
     * <p>The engine's production path is: pick the factory's
     * {@link SpecialAction} for the requested type
     * ({@code Factory.findActionForUnitType}), then issue it as a command whose
     * {@code actionId} names that action. {@code Factory.executeSpecialAction}
     * routes the command into {@code FactoryBuildQueue.queueOrCancel}, which
     * charges the price and enqueues the build.
     *
     * <p>Returns false when the type is unknown, the factory is missing, or the
     * factory cannot queue it (tech-locked, unaffordable, unit cap) so the
     * receipt reports a rejection instead of a phantom success.
     */
    private boolean produceUnit(long factoryId, String unitTypeName) {
        UnitTypeInterface unitType = resolveUnitType(unitTypeName);
        if (unitType == null) {
            return false;
        }
        OrderableUnit factory = findOwnFactory(factoryId);
        if (factory == null) {
            return false;
        }
        SpecialAction action;
        try {
            action = factory.findActionForUnitType(unitType);
        } catch (RuntimeException failure) {
            return false;
        }
        if (action == null) {
            // Not in this factory's build menu (e.g. a T2 unit at a T1 factory).
            return false;
        }
        try {
            if (!action.isAvailableForUnit(factory) || !action.canAffordForUnit(factory, false)) {
                return false;
            }
        } catch (RuntimeException failure) {
            return false;
        }
        CommandControllerCommand command =
                engine.commandController.createCommandForTeam(engine.playerTeam);
        command.addUnit(factory);
        command.setActionId(action.getActionId());
        return true;
    }

    /**
     * Upgrade an own building to its next tier ({@code upgrade_building}).
     *
     * <p>Rusted Warfare models upgrades as a build-queue item whose action id is
     * the building's current {@code getBuildActionId()} (T1 extractor -> T2 ->
     * T3, T1 factory -> T2). The action lives in the building's own special
     * action list, so the same command path used for production applies.
     */
    private boolean upgradeBuilding(long buildingId) {
        Unit building = findUnitById(buildingId);
        if (building == null || !(building instanceof OrderableUnit)) {
            return false;
        }
        if (building.team == null || building.team.teamId != engine.playerTeam.teamId) {
            return false;
        }
        ActionId upgradeId;
        try {
            upgradeId = building.getBuildActionId();
        } catch (RuntimeException failure) {
            return false;
        }
        if (upgradeId == null || !SpecialAction.isRealActionId(upgradeId)) {
            // Already at the top tier, or this building type has no upgrade.
            return false;
        }
        OrderableUnit orderable = (OrderableUnit) building;
        SpecialAction action;
        try {
            action = orderable.getSpecialAction(upgradeId);
        } catch (RuntimeException failure) {
            return false;
        }
        if (action == null) {
            return false;
        }
        try {
            if (!action.isAvailableForUnit(orderable) || !action.canAffordForUnit(orderable, false)) {
                return false;
            }
        } catch (RuntimeException failure) {
            return false;
        }
        CommandControllerCommand command =
                engine.commandController.createCommandForTeam(engine.playerTeam);
        command.addUnit(orderable);
        command.setActionId(upgradeId);
        return true;
    }

    /**
     * Own factory for {@code factoryId}; when the id is absent <em>or stale</em>,
     * the first own factory is used so a policy that only says "produce a tank"
     * is not rejected for omitting an entity id -- or for naming one that died
     * between the observation and the order.
     *
     * <p>The old last line was {@code factoryId >= 0 ? null : fallback}, i.e. a
     * named-but-missing factory returned {@code null} instead of the fallback,
     * which directly contradicts this method's own docstring. That turned every
     * order naming a just-destroyed factory into a failure even when another own
     * factory could have served it.
     */
    private OrderableUnit findOwnFactory(long factoryId) {
        OrderableUnit fallback = null;
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (!(unit instanceof FactoryInterface)) {
                continue;
            }
            if (factoryId >= 0 && unit.id == factoryId) {
                return unit;
            }
            if (fallback == null) {
                fallback = unit;
            }
        }
        return fallback;
    }

    /**
     * Map protocol snake_case structure names onto engine enum constants.
     * ``land_factory`` -> ``landFactory``, ``air_factory`` -> ``airFactory``.
     * Unknown names are returned unchanged so {@link #resolveUnitType} rejects them.
     */
    private String resolveUnitTypeName(String name) {
        if (name == null || name.isEmpty() || name.indexOf('_') < 0) {
            return name;
        }
        StringBuilder out = new StringBuilder(name.length());
        boolean upper = false;
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (c == '_') {
                upper = true;
                continue;
            }
            out.append(upper ? Character.toUpperCase(c) : c);
            upper = false;
        }
        return out.toString();
    }

    /**
     * Issue build orders with every builder in the selected group.
     * Returns the number of builders that accepted the order.
     *
     * The engine rejects an order after dispatch when the builder cannot queue
     * the unit type (tech-locked, no build entry, invalid spot). We therefore
     * pre-check with {@code isLocked()} and {@code canQueueAction()} so that an
     * impossible build is reported as rejected instead of being counted as
     * accepted and silently dropped by the engine a frame later.
     */
    private int buildWithBuilders(int unitGroup, String unitTypeName, float x, float y) {
        UnitTypeInterface unitType = resolveUnitType(unitTypeName);
        if (unitType == null) {
            debugBuild(-1, "resolveUnitType=null", unitTypeName);
            return 0;
        }
        // A locked unit type cannot be built by anyone right now.
        if (unitType.isLocked()) {
            debugBuild(-1, "unitType.isLocked", unitTypeName);
            return 0;
        }
        int ordered = 0;
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (unitGroup >= 0 && unitGroup < GROUP_COUNT && (unit.id % GROUP_COUNT) != unitGroup) {
                continue;
            }
            if (!unit.isBuilder() || !unit.canBeGivenBuildOrders()) {
                continue;
            }
            if (!unit.canQueueAction(unitType, false)) {
                debugBuild(unit.id, "canQueueAction=false", unitTypeName);
                continue;
            }
            // The tech level is not a constant: ``SpecialActionBuild`` for an
            // extractor is techLevel 1, and ``getSpecialAction(type, level,
            // ...)`` matches the level exactly. Passing 0 (the previous value)
            // made every build waypoint fail validation, so no build ever
            // reached the engine. Resolve the action first and reuse its level,
            // which is exactly what the built-in AI does in ``Base.buildBuilding``.
            SpecialAction buildAction;
            try {
                buildAction = unit.getSpecialAction(unitType, false);
            } catch (RuntimeException failure) {
                continue;
            }
            if (buildAction == null) {
                continue;
            }
            int techLevel = buildAction.getTechLevel();
            // ``canQueueAction`` only reflects the build menu. Whether the
            // order is legal (availability, affordability) is decided by
            // ``isValidNewWaypoint``; without this probe the engine drops the
            // order one frame later and we would report a false "accepted".
            Waypoint probe = new Waypoint();
            probe.setBuildTo(x, y, unitType, techLevel);
            if (!unit.isValidNewWaypoint(probe, false)) {
                debugBuild(unit.id, "isValidNewWaypoint=false", unitTypeName);
                continue;
            }
            // Terrain / placement legality, which ``isValidNewWaypoint`` does NOT
            // check.
            //
            // ``isValidNewWaypoint`` validates the *order* (build type present, a
            // builder in range, waypoint well formed); it does not ask whether the
            // target tile is a legal place for the thing being built. The engine
            // therefore accepts a ``seaFactory`` on dry sand, reports success, and
            // then silently drops it when the build finally executes -- producing
            // **no entity at all**.
            //
            // <p>Measured on {@code [p2]Lake (2p)} (``.runtime/q7b_seafactory_land.json``,
            // ``.runtime/terrain_gate.json``):
            // <pre>
            //   (1080,2340) landFactory  -> built, complete tick 1371
            //   (1080,2340) seaFactory   -> accepted=true, 9000 ticks, NO ENTITY
            // </pre>
            // Same pad, same builder, so the pad and the walk were never the obstacle.
            // A false "accepted" is worse than a rejection: the policy is told its
            // action succeeded, pays no ``invalid_penalty``, and can never learn that a
            // shipyard needs water.
            //
            // <p>{@code Building.getPlacementError} is the engine's **own**
            // authoritative placement test -- it is what the game itself runs before it
            // lets a building exist. It already special-cases exactly this situation:
            //
            // <pre>
            //   if (var1 != UnitType.seaFactory && var2 != MovementType.WATER) { ... }
            //   else { return !isBlocked(costsWater, tx, ty) ? null : "{3}"; }
            // </pre>
            //
            // <p>Re-deriving that rule in the bridge would be a second copy of engine
            // logic that drifts; calling it means the acceptance test and the game's own
            // rule are the same statement. An earlier attempt used
            // ``GameUtils.isBlockedForMovementType`` with the built type's own
            // ``MovementType`` and had no effect, because a *building* reports
            // ``MovementType.NONE`` -- which is why the guard needs the engine's
            // placement-specific predicate rather than the generic movement one.
            //
            // <p>Only a non-null error blocks the order. A null means the tile is
            // legal, so every previously-accepted build behaves exactly as before; this
            // gate can only turn a silent no-op into an honest rejection.
            MovementType placementMovement;
            try {
                // The *built type's* movement type, not the builder's: the question is
                // "may a structure of this kind stand here", and the engine's rule keys
                // off exactly this value (``MovementType.WATER`` forces the water
                // branch). A building reports ``NONE``, which the engine treats as the
                // ordinary land path -- that is correct and expected.
                placementMovement = unitType.getMovementType();
            } catch (RuntimeException ignored) {
                placementMovement = MovementType.NONE;
            }
            try {
                // Convert world pixels to a tile index with the engine's *own*
                // conversion rather than arithmetic of my own. ``pixelToTileCoords``
                // writes into ``Map.fogRevealTileX/Y`` and uses
                // ``(int)(pixel * inverseTileWidth)``, so it truncates exactly the way
                // the placement test expects; floor-dividing by ``tileWidth`` would
                // disagree by one tile on negative or non-integral edges and turn a
                // legal build into a spurious rejection.
                engine.map.pixelToTileCoords(x, y);
                int tileX = engine.map.fogRevealTileX;
                int tileY = engine.map.fogRevealTileY;
                java.lang.String placementError =
                        com.corrodinggames.rts.game.units.buildings.Building.getPlacementError(
                                unit, unitType, placementMovement,
                                tileX, tileY, true, engine.playerTeam);
                if (placementError != null) {
                    debugBuild(unit.id,
                            "placement illegal at (" + (int) x + "," + (int) y
                                    + "): " + placementError,
                            unitTypeName);
                    continue;
                }
            } catch (RuntimeException ignored) {
                // Unqueryable placement (older engine): fall through to the dispatch
                // path rather than rejecting every build on a missing API.
            }
            CommandControllerCommand command =
                    engine.commandController.createCommandForTeam(engine.playerTeam);
            command.addUnit(unit);
            command.setBuildWaypoint(x, y, unitType, techLevel);
            ordered++;
        }
        if (ordered == 0) {
            debugBuild(-1, "no builder passed every gate", unitTypeName);
        } else {
            debugBuild(-1, "ordered=" + ordered + " at (" + (int) x + "," + (int) y + ")", unitTypeName);
        }
        return ordered;
    }

    /** Build-path tracing, enabled with ``--debug-build``. */
    private void debugBuild(long unitId, String what, String unitTypeName) {
        if (debugBuild) {
            System.out.println("[ironzero] build " + unitTypeName
                    + " unit=" + unitId + " : " + what);
        }
    }

    /**
     * Position of the n-th extractor spot on the current map, or {@code null}
     * when the map exposes none. Derived from the loaded map's resource list so
     * the policy's {@code slot_id} maps onto a real location.
     */
    private double[] resourceSlotPosition(int slotId) {
        java.util.List<double[]> slots = resourceSlots();
        if (slots.isEmpty()) {
            return null;
        }
        return slots.get(Math.floorMod(slotId, slots.size()));
    }

    /**
     * Positions of the map's ore pools, in map pixels, ordered <b>nearest to our
     * own base first</b> so a policy's {@code slot_id} always names a sensible
     * spot.
     *
     * <p>Resource pools are <b>terrain</b> in Rusted Warfare 1.15, not units:
     * {@code MapLayer} appends every tile carrying the {@code res_pool} property
     * to {@code Map.resourcePoolPoints} (see {@code MapLayer.java}), and the
     * built-in AI finds ore exclusively through that list
     * ({@code AI.getNearestResourcePoint}). They never appear in
     * {@code Unit.unitList}, so the earlier implementation -- which scanned
     * units and asked {@code OrderableUnit.isResourceUnit} -- returned an empty
     * list on every stock map. That made {@code build_extractor} fall back to
     * map coordinate (0,0), which the engine rejects, so no build ever
     * succeeded and the economy reward stayed at zero.
     *
     * <p>Ordering used to be tile-row major (smallest y first). On a north/south
     * skirmish map that puts the <i>enemy's</i> half first -- measured on
     * {@code [p2]Lake (2p)}: our base sits at (990, 2010) while slot 0 was
     * (1130, 390), next to the enemy base at (1230, 210). A policy that emitted
     * {@code slot_id = 0} therefore marched its builder across the map into the
     * enemy base and lost it. Sorting by distance to our own base keeps the
     * order stable (both the map and our starting base are fixed) while making
     * {@code slot_id = 0} the pool right next to home.
     */
    private java.util.List<double[]> resourceSlots() {
        java.util.List<double[]> slots = new java.util.ArrayList<>();
        if (engine == null || engine.map == null) {
            return slots;
        }
        try {
            java.util.List<android.graphics.Point> points = engine.map.resourcePoolPoints;
            if (points == null || points.isEmpty()) {
                return slots;
            }
            for (android.graphics.Point tile : points) {
                // Centred pixel coordinate of the tile: the extractor must sit on
                // the pool, and tileToPixelCoordsCentered is the engine's own
                // convention for "the point inside this tile".
                engine.map.tileToPixelCoordsCentered(tile.x, tile.y);
                slots.add(new double[]{engine.map.fogRevealTileX, engine.map.fogRevealTileY});
            }
            double[] anchor = ownBaseAnchor();
            if (anchor != null) {
                // Stable sort on a static map with a static base: the same map
                // always yields the same slot ids, so masks and rewards stay
                // consistent across decisions and episodes.
                java.util.Collections.sort(slots, (left, right) -> Double.compare(
                        distanceSquared(left, anchor), distanceSquared(right, anchor)));
            } else {
                // No own base yet (pre-spawn observation): keep the previous
                // deterministic order instead of an arbitrary one.
                java.util.Collections.sort(slots, (left, right) -> {
                    int byY = Double.compare(left[1], right[1]);
                    return byY != 0 ? byY : Double.compare(left[0], right[0]);
                });
            }
        } catch (RuntimeException ignored) {
            return slots;
        }
        return slots;
    }

    private static double distanceSquared(double[] point, double[] anchor) {
        double dx = point[0] - anchor[0];
        double dy = point[1] - anchor[1];
        return dx * dx + dy * dy;
    }

    /**
     * Our own base position: the first own factory building (the command centre
     * reports {@code isFactoryBuilding()}), else the centroid of our units.
     * Returns {@code null} when we own nothing yet.
     */
    private double[] ownBaseAnchor() {
        double sumX = 0.0;
        double sumY = 0.0;
        int count = 0;
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (unit.isFactoryBuilding()) {
                // The command centre is unique: prefer it outright so the anchor
                // never drifts as outlying factories get built.
                return new double[]{unit.x, unit.y};
            }
            sumX += unit.x;
            sumY += unit.y;
            count++;
        }
        return count > 0 ? new double[]{sumX / count, sumY / count} : null;
    }

    /**
     * True when an own extractor already sits on this pool, so the policy is not
     * rewarded twice for the same spot and the reported occupancy is honest.
     */
    private int slotExtractorLevel(double[] slot) {
        if (slot == null) {
            return 0;
        }
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f || !unit.isBuilding()) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (!placeOnlyOnResPool(unit)) {
                continue;
            }
            double dx = unit.x - slot[0];
            double dy = unit.y - slot[1];
            if (dx * dx + dy * dy <= 40.0 * 40.0) {
                try {
                    return Math.max(1, Math.min(3, unit.getTechLevel()));
                } catch (RuntimeException ignored) {
                    return 1;
                }
            }
        }
        return 0;
    }

    /**
     * {@code resource_slots} array for the observation message.
     *
     * <p>Without this the policy cannot know where it is allowed to mine, and the
     * Python encoder's {@code extractor_slots_owned} feature stays 0 forever.
     * {@code owner_team} is only emitted for an occupied slot: claiming team 0 on
     * an empty pool would read as "somebody already built here".
     */
    private String resourceSlotsJson() {
        java.util.List<double[]> slots = resourceSlots();
        int[] levels = new int[slots.size()];
        for (int index = 0; index < slots.size(); index++) {
            levels[index] = slotExtractorLevel(slots.get(index));
        }
        return resourceSlotsJson(slots, levels);
    }

    /**
     * Serialise already-collected slots/occupancy levels. {@code sendObservation}
     * uses this overload so the per-observation work stays one unit scan.
     */
    private String resourceSlotsJson(java.util.List<double[]> slots, int[] levels) {
        StringBuilder out = new StringBuilder(256);
        out.append('[');
        for (int index = 0; index < slots.size(); index++) {
            double[] slot = slots.get(index);
            if (index > 0) {
                out.append(',');
            }
            int level = index < levels.length ? levels[index] : 0;
            out.append("{\"slot_id\":").append(index)
                    .append(",\"x\":").append(fmt(slot[0]))
                    .append(",\"y\":").append(fmt(slot[1]))
                    .append(",\"occupied\":").append(level > 0)
                    .append(",\"extractor_level\":").append(level);
            if (level > 0) {
                out.append(",\"owner_team\":").append(engine.playerTeam.teamId);
            }
            out.append('}');
        }
        out.append(']');
        return out.toString();
    }

    /**
     * Whether this unit's *type* can only be placed on a resource pool.
     *
     * <p><b>This is a type-level question, and asking the instance was the bug.</b>
     * {@code Unit.placeOnlyOnResPool()} returns a hard-coded {@code false}
     * ({@code Unit.java:1922}) and **no stock subclass overrides it** -- not
     * {@code Extractor}, not any building. The truth lives on the unit *type*:
     * {@code UnitType$1.placeOnlyOnResPool()} returns {@code true} for the extractor,
     * and {@code UnitType.placeOnlyOnResPool()} returns {@code false} for everything
     * else.
     *
     * <p>The consequences of asking the instance were visible for a whole training
     * campaign. Every caller used it to decide "is this building an extractor", so
     * <b>every resource pool in every observation reported {@code occupied: false}</b>
     * and {@code extractor_level: 0} even while an extractor stood on it. Measured on
     * {@code [p2]Lake (2p)}: an extractor at (1170, 2050) sat directly on slot 0 while
     * slot 0 reported {@code occupied=false, extractor_level=0}
     * ({@code .runtime/raw_entity.json}). The Python encoder counts occupied slots to
     * build its economy feature, and {@code build_extractor} deliberately picks a
     * *free* slot, so the policy could neither see the extractors it owned nor stop
     * re-ordering the same pool.
     */
    private boolean placeOnlyOnResPool(Unit unit) {
        try {
            return unit.getUnitTypeInterface().placeOnlyOnResPool();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    /**
     * Record that a own extractor sits on one of {@code slots}.
     *
     * <p>Same radius/threshold rule as {@link #slotExtractorLevel}: the occupancy
     * reported for a pool must mean exactly what the single-slot scan meant, so
     * the reward path (which pays once per occupied pool) cannot double-count.
     */
    private void markSlotLevel(Unit unit, java.util.List<double[]> slots, int[] levels) {
        int level;
        try {
            level = Math.max(1, Math.min(3, unit.getTechLevel()));
        } catch (RuntimeException ignored) {
            level = 1;
        }
        for (int index = 0; index < slots.size() && index < levels.length; index++) {
            double[] slot = slots.get(index);
            double dx = unit.x - slot[0];
            double dy = unit.y - slot[1];
            if (dx * dx + dy * dy <= 40.0 * 40.0) {
                if (level > levels[index]) {
                    levels[index] = level;
                }
                return;
            }
        }
    }

    /**
     * Income per second for the controlled team, summed from the units that
     * actually generate credits.
     *
     * <p>{@code global_state.income} used to be the literal {@code 0.0}, so the
     * encoder's {@code income} feature (normalised by
     * {@code MicroDuelConfig.income_scale = 40}) and the {@code income} reward
     * term were dead inputs. {@code Unit.getGenerationCredits()} is the engine's
     * own figure and is already a rate, so it is reported as-is; multiplying it
     * by 60 would push an active extractor past the scale and saturate the
     * feature at 1.0.
     *
     * <p>The AI difficulty multiplier is applied here even though
     * {@code getGenerationCredits()} does not include it. The engine multiplies in
     * {@code Team.updateIncome}: {@code addIncome(getAIDifficultyIncomeMultiplier()
     * * delta)}, so the raw sum is what a *non-AI* team would earn. Reporting it
     * unscaled made the {@code income} feature (and the {@code income} reward term)
     * blind to the AI's resource cheat -- measured: identical 18.0/s at difficulty
     * -2, 0 and 3, while the same teams' real multiplier ranged x0.40 to x3.70.
     */
    private double ownIncome() {
        double income = 0.0;
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            try {
                income += unit.getGenerationCredits();
            } catch (RuntimeException ignored) {
                // A unit type that does not report generation simply adds nothing.
            }
        }
        Team player = engine != null ? engine.playerTeam : null;
        if (player != null && player.isAI) {
            try {
                income *= player.getAIDifficultyIncomeMultiplier();
            } catch (RuntimeException ignored) {
                // Keep the raw sum rather than dropping the whole feature.
            }
        }
        return income;
    }

    private OrderableUnit firstOwnBuilder() {
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (unit.isBuilder()) {
                return unit;
            }
        }
        return null;
    }

    private void addOwnOrderableUnits(CommandControllerCommand command, int unitGroup) {
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (unitGroup >= 0 && unitGroup < GROUP_COUNT && (unit.id % GROUP_COUNT) != unitGroup) {
                continue;
            }
            command.addUnit(unit);
        }
    }

    /**
     * Stop/hold: one command per selected unit so each keeps its current position.
     * Returns the number of units that accepted the order.
     */
    private int holdGround(int unitGroup, boolean guardArea) {
        int ordered = 0;
        for (Object item : Unit.unitList) {
            if (!(item instanceof OrderableUnit)) {
                continue;
            }
            OrderableUnit unit = (OrderableUnit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            if (unit.team == null || unit.team.teamId != engine.playerTeam.teamId) {
                continue;
            }
            if (unitGroup >= 0 && unitGroup < GROUP_COUNT && (unit.id % GROUP_COUNT) != unitGroup) {
                continue;
            }
            CommandControllerCommand command = engine.commandController.createCommandForTeam(engine.playerTeam);
            command.addUnit(unit);
            command.setMoveWaypoint(unit.x, unit.y);
            if (guardArea) {
                command.setAttackMode(AttackMode.guardArea);
            }
            ordered++;
        }
        return ordered;
    }

    private Unit findUnitById(long entityId) {
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.id == entityId && !unit.removed && unit.hp > 0.0f) {
                return unit;
            }
        }
        return null;
    }

    private void writeLine(String line) {
        if (writer == null) {
            return;
        }
        try {
            writer.write(line);
            writer.write("\n");
            writer.flush();
        } catch (IOException failure) {
            writer = null;
        }
    }

    private int countUnits() {
        int count = 0;
        for (Object item : Unit.unitList) {
            if (item instanceof Unit) {
                Unit unit = (Unit) item;
                if (!unit.removed && unit.hp > 0.0f) {
                    count++;
                }
            }
        }
        return count;
    }

    private void closeBridge() {
        // Graceful shutdown, not ``client.close()`` in a hurry.
        //
        // Closing a TCP socket that still has *unread inbound* data makes the OS
        // send RST instead of FIN. RST instructs the peer to discard its receive
        // queue -- and at this point that queue holds the last observations and
        // the ``done`` message we just flushed. The consumer is by design a
        // free-running client that lags behind the simulation, so the queue is
        // never empty when the match ends.
        //
        // Measured symptom (2026-11-13, 40-update run): Python recorded
        // ``timeout`` with terminal = 0.0 for all 40 updates / win_rate = 0.0,
        // while these very logs said 38 victories and 5 defeats. The engine's
        // verdict was written, flushed, and then destroyed by the RST.
        //
        // Order matters: flush what we wrote, half-close (FIN) so the peer can
        // read it and then see EOF, and only then drain the peer's pending steps
        // so the final close() does not reset either.
        try {
            if (writer != null) {
                writer.flush();
            }
        } catch (IOException ignored) {
            // best effort
        }
        try {
            if (client != null) {
                client.shutdownOutput();
            }
        } catch (IOException ignored) {
            // best effort
        }
        try {
            if (reader != null && client != null) {
                client.setSoTimeout(CLOSE_DRAIN_TIMEOUT_MILLIS);
                String pending;
                while ((pending = reader.readLine()) != null) {
                    if (!pending.isBlank()) {
                        continue;
                    }
                }
            }
        } catch (IOException ignored) {
            // Timeout or reset while draining: the bytes we care about are already
            // handed to the OS, so proceed to close regardless.
        }
        try {
            if (client != null) {
                client.close();
            }
        } catch (IOException ignored) {
            // best effort
        }
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
        } catch (IOException ignored) {
            // best effort
        }
    }

    private static String fmt(double value) {
        return String.format(Locale.ROOT, "%.3f", value);
    }

    /**
     * Clamp a parsed numeric flag into its contract range, reporting the coercion.
     *
     * <p>Ranges live in {@code audit/加速方案_契约.md} §1 and must not drift: a
     * silently accepted out-of-range value would make the log line disagree with
     * the semantics the Python side documents.
     */
    private static int clampInt(int value, int min, int max) {
        if (value < min) {
            System.out.println("[ironzero] clamped " + value + " to " + min);
            return min;
        }
        if (value > max) {
            System.out.println("[ironzero] clamped " + value + " to " + max);
            return max;
        }
        return value;
    }
}
