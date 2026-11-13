// SPDX-License-Identifier: Apache-2.0
package ironzero;

import com.corrodinggames.rts.game.RWGameEngine;
import com.corrodinggames.rts.game.Team;
import com.corrodinggames.rts.game.ai.AI;
import com.corrodinggames.rts.game.units.AttackMode;
import com.corrodinggames.rts.game.units.OrderableUnit;
import com.corrodinggames.rts.game.units.Unit;
import com.corrodinggames.rts.game.units.UnitType;
import com.corrodinggames.rts.game.units.UnitTypeInterface;
import com.corrodinggames.rts.game.units.Waypoint;
import com.corrodinggames.rts.game.units.WaypointType;
import com.corrodinggames.rts.game.units.actions.ActionId;
import com.corrodinggames.rts.game.units.actions.SpecialAction;
import com.corrodinggames.rts.game.units.buildings.FactoryInterface;
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
 *       enemy AI teams and enable AI.aggressiveMode so map-placed units attack)
 *   --ticks bounds simulated frames (1 frame = 1 tick; 60 ticks = 1 s).
 *   --frame-ms is the wall-clock pacing budget; by default it is *also* the
 *       engine's frame delta, so 16 = real time and 2 = ~8x faster.
 *   --physics-ms fixes the engine delta independently of --frame-ms, so an
 *       episode can keep a real-time simulation delta while the wall clock runs
 *       unpaced or fast. Omitted = old behaviour (delta == frame-ms).
 *   --ai-difficulty N                (-2..3: 非常简单/简单/正常/困难/非常困难/炼狱)
 *       pins every AI team's difficulty. Omitted = the game's own resolution,
 *       which in a headless single-player run is SettingsEngine.aiDifficulty
 *       (default 0 = 正常 = x1.00 income, no AI cheat bonus).
 */
public final class IronZeroBridge implements ServerExtension {
    private static final int DECISION_INTERVAL = 10;
    private static final int FRAME_PERIOD_MILLIS = 16;
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
    // Annihilation win-mode grace window: the map load/spawn settles during the
    // first simulated ticks, so unit counts are not authoritative yet.
    private static final int ANNIHILATION_GRACE_TICKS = 30;
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

    private volatile boolean stop = false;
    private ServerSocket serverSocket;
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
    // Game-over plumbing: the native engine knows when the battle is decided
    // (TriggerEngine win/lose objectives and Team defeat checks). The bridge
    // polls that state every frame and reports it to the RL environment; it is
    // null while the battle is still running.
    private volatile String gameResult = null;
    private boolean enemySeen = false;

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
        if (unlockAi) {
            unlockAiTeams();
        }
        if (aiPilotTeam) {
            // Hand the controlled team to the built-in AI so its own commands can
            // be recorded as a behaviour-cloning teacher (``--record-ai-orders``).
            installAiPilot();
        }
        if (aiDifficulty != null) {
            // Must run after the map finished loading (teams exist) and after the
            // pilot replaced the player team, otherwise the pilot would inherit the
            // old team's difficulty.
            applyAiDifficulty();
        }
        System.out.println("[ironzero] world ready: mapLoaded="
                + (engine.map != null && engine.map.mapLoaded)
                + " units=" + countUnits() + " teams=" + Team.getTeams().size()
                + " winMode=" + winMode + " unlockAi=" + unlockAi
                + " aiPilot=" + aiPilotTeam
                + " aiDifficulty=" + effectiveAiDifficultyDescription());

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
        // `--physics-ms` pins both engine arguments so an episode always simulates
        // the same number of game-milliseconds, while `frameMs` only paces the
        // wall clock.
        final int physics = physicsMs > 0 ? physicsMs : frameMs;
        final int physicsElapsedMs = physicsMs > 0 ? physicsMs : frameMs;
        long nextWake = System.nanoTime();
        while (!stop && context.isRunning() && frames < maxFrames) {
            context.pump();
            try {
                engine.updateFrameInternal(1.0f, physicsElapsedMs);
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
            if (engine.simulationTick - lastSentTick >= DECISION_INTERVAL || frames == 1 || gameResult != null) {
                lastSentTick = engine.simulationTick;
                sendObservation(engine, frames);
            }
            readClientCommands();
            if (frames % 30 == 0 && writer == null) {
                break;
            }
            if (gameResult != null) {
                break;
            }
            if (frameMs > 0) {
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
                + " result=" + (gameResult != null ? gameResult : "timeout"));
    }

    @Override
    public void requestStop() {
        stop = true;
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
                frameMs = Math.max(0, Integer.parseInt(args.get(i + 1)));
            } else if ("--physics-ms".equals(argument) && i + 1 < args.size()) {
                // Fixed simulation delta, independent of the pacing budget.
                // Omitted keeps the historical "delta == frameMs" behaviour.
                physicsMs = Math.max(1, Integer.parseInt(args.get(i + 1)));
            } else if ("--win-mode".equals(argument) && i + 1 < args.size()) {
                String requested = args.get(i + 1).trim().toLowerCase(Locale.ROOT);
                if ("annihilation".equals(requested) || "native".equals(requested)) {
                    winMode = requested;
                } else {
                    System.out.println("[ironzero] unknown --win-mode: " + requested
                            + " (keeping native)");
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
                if (slotLevels.length > 0 && unit.isBuilding() && unit.placeOnlyOnResPool()) {
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
                // Behaviour-cloning labels: the orders the expert (built-in AI)
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
            boolean firstType = true;
            for (UnitType candidate : BUILDABLE_PROBE) {
                boolean ok;
                try {
                    ok = candidate != null && !candidate.isLocked()
                            && unit.canQueueAction(candidate, false);
                } catch (RuntimeException ignored) {
                    ok = false;
                }
                if (!ok) {
                    continue;
                }
                if (!firstType) {
                    out.append(',');
                }
                firstType = false;
                out.append('"').append(candidate.getInternalName()).append('"');
            }
            out.append("]}");
        }
        out.append(']');
        return out.toString();
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
                + ",\"is_operational\":" + !unit.isNonOperational()
                // Whether the engine would accept an ``upgrade_building`` order
                // for this unit *right now* (a real answer, not a type guess).
                + ",\"can_upgrade\":" + canUpgrade(unit)
                // Action-group alignment: addOwnOrderableUnits() selects group members
                // with id % GROUP_COUNT, so expose exactly that value to the policy.
                + ",\"group_id\":" + (unit.id % GROUP_COUNT)
                + ",\"is_visible\":true,\"is_alive\":true}";
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

    /** Engine internal name of the unit's type, with a safe placeholder. */
    private String internalName(Unit unit) {
        try {
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
     * Whether {@code upgrade_building} would be accepted for this unit.
     *
     * <p>Rusted Warfare models a tier upgrade as the building's current build
     * action ({@code getBuildActionId}), so a real action id means "there is a
     * next tier". Reporting the capability lets the Python side select an
     * upgrade target by what the engine says is possible instead of matching a
     * hard-coded type id.
     */
    private boolean canUpgrade(Unit unit) {
        if (!unit.isBuilding()) {
            return false;
        }
        try {
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
        writeLine("{\"protocol\":\"ironzero/1\",\"type\":\"done\",\"message_id\":\"done-"
                + engine.simulationTick + "\",\"episode_id\":\"rwpp-bridge\",\"tick\":"
                + engine.simulationTick + ",\"result\":\"" + result + "\",\"reward\":0.0,"
                + "\"terminated\":" + terminated + ",\"truncated\":" + (!terminated) + "}");
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
            if (isOwnUnit(player, unit)) {
                tally.ownAlive++;
                tally.ownHp += unit.hp;
            } else if (isEnemyUnit(player, unit)) {
                tally.enemyAlive++;
                tally.enemyHp += unit.hp;
            }
        }
        return tally;
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
     * Clearing the flag and enabling ``AI.aggressiveMode`` (attack mode
     * ``aggressive`` instead of ``outOfRange``) makes those units advance and
     * engage. Must run after the map finished loading.
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
            if (team instanceof AI) {
                ((AI) team).aggressiveMode = true;
            }
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
            try {
                team.aiDifficulty = requested;
                team.aiDifficultyLocked = true;
                team.aiDifficultyOverride = Integer.valueOf(requested);
                applied++;
            } catch (RuntimeException failure) {
                System.out.println("[ironzero] ai-difficulty: failed on team "
                        + team.teamId + ": " + failure);
            }
        }
        System.out.println("[ironzero] ai-difficulty: requested=" + requested
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
            ((AI) current).aggressiveMode = true;
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
            pilot.aggressiveMode = true;
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
                actionOutcomes.add(built > 0 ? null : "ACTION_MASKED");
                continue;
            } else if ("produce_unit".equals(kind)) {
                // Factory production. The factory is named by entity id; when the
                // policy leaves it out (or names a dead unit) the nearest own
                // factory in the selected group is used instead.
                long factoryId = action.getLong("factory_entity_id", -1L);
                String unitTypeName = action.getString("unit_type", "tank");
                actionOutcomes.add(produceUnit(factoryId, unitTypeName) ? null : "ACTION_MASKED");
                continue;
            } else if ("upgrade_building".equals(kind)) {
                // Extractor / factory tier upgrade: the engine exposes it as the
                // building's current build action id.
                long buildingId = action.getLong("building_entity_id", -1L);
                actionOutcomes.add(upgradeBuilding(buildingId) ? null : "ACTION_MASKED");
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
     * Own factory for {@code factoryId}; when the id is absent or stale, the
     * first own factory is used so a policy that only says "produce a tank" is
     * not rejected for omitting an entity id.
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
        return factoryId >= 0 ? null : fallback;
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
            if (!unit.placeOnlyOnResPool()) {
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
}
