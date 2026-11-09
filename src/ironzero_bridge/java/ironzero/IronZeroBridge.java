// SPDX-License-Identifier: Apache-2.0
package ironzero;

import com.corrodinggames.rts.game.RWGameEngine;
import com.corrodinggames.rts.game.Team;
import com.corrodinggames.rts.game.ai.AI;
import com.corrodinggames.rts.game.units.AttackMode;
import com.corrodinggames.rts.game.units.OrderableUnit;
import com.corrodinggames.rts.game.units.Unit;
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
 *   --frame-ms is a wall-clock pacing budget only; it does not change the horizon.
 */
public final class IronZeroBridge implements ServerExtension {
    private static final int DECISION_INTERVAL = 10;
    private static final int FRAME_PERIOD_MILLIS = 16;
    // Annihilation win-mode grace window: the map load/spawn settles during the
    // first simulated ticks, so unit counts are not authoritative yet.
    private static final int ANNIHILATION_GRACE_TICKS = 30;
    // Action-space contract shared with MicroDuelConfig: 0..3 are unit groups,
    // 4 (== GROUP_BROADCAST) orders every living friendly unit.
    private static final int GROUP_COUNT = 4;
    private static final int GROUP_BROADCAST = GROUP_COUNT;

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
        System.out.println("[ironzero] world ready: mapLoaded="
                + (engine.map != null && engine.map.mapLoaded)
                + " units=" + countUnits() + " teams=" + Team.getTeams().size()
                + " winMode=" + winMode + " unlockAi=" + unlockAi);

        sendHello();
        int frames = 0;
        int lastSentTick = -1;
        while (!stop && context.isRunning() && frames < maxFrames) {
            context.pump();
            try {
                engine.updateFrameInternal(1.0f, frameMs);
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
                try {
                    Thread.sleep(frameMs);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    break;
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
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f) {
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
                + "\"army_value\":0.0,\"visible_enemy_value\":0.0,\"kills\":0,\"losses\":0,"
                + "\"own_alive\":" + ownAlive + ",\"enemy_alive\":" + enemyAlive + ","
                + "\"own_hp\":" + fmt(ownHp) + ",\"enemy_hp\":" + fmt(enemyHp) + ","
                + "\"enemy_teams\":" + enemyTeamsJson(player) + "},"
                + "\"own_entities\":[" + String.join(",", own) + "],"
                + "\"visible_enemy_entities\":[" + String.join(",", enemies) + "],"
                + "\"remembered_enemies\":[],\"local_maps\":{},"
                + "\"masks\":{\"action_mask\":[],\"unit_group_mask\":[],\"target_mask\":[],\"position_mask\":[]},"
                + "\"reward\":0.0,\"result\":" + resultJson + ",\"terminated\":" + terminated
                + ",\"truncated\":false,"
                + "\"info\":{\"source\":\"rwpp\",\"frame\":" + frame + ",\"units_alive\":" + alive
                + ",\"win_mode\":\"" + winMode + "\",\"unlock_ai\":" + unlockAi + "}}";
        writeLine(line);
    }

    private String entityJson(Unit unit) {
        int teamId = unit.team != null ? unit.team.teamId : -1;
        boolean building = unit.isBuilding();
        return "{\"entity_id\":" + unit.id
                + ",\"entity_type_id\":" + (building ? 2 : 1) + ",\"team_id\":" + teamId
                + ",\"x\":" + fmt(unit.x) + ",\"y\":" + fmt(unit.y)
                + ",\"hp\":" + fmt(unit.hp) + ",\"max_hp\":" + fmt(unit.maxHp)
                + ",\"is_building\":" + building
                + ",\"is_factory\":" + unit.isFactoryBuilding()
                + ",\"is_builder\":" + unit.isBuilder()
                + ",\"is_operational\":" + !unit.isNonOperational()
                // Action-group alignment: addOwnOrderableUnits() selects group members
                // with id % GROUP_COUNT, so expose exactly that value to the policy.
                + ",\"group_id\":" + (unit.id % GROUP_COUNT)
                + ",\"is_visible\":true,\"is_alive\":true}";
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
     * unusable — ``Team.updateDefeatCheck`` marks any team without a factory or
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
        int accepted = 0;
        int rejected = 0;
        for (JsonValue item : actions) {
            if (!item.isObject()) {
                rejected++;
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
                }
            } else if ("stop".equals(kind) || "hold".equals(kind)) {
                // Hold ground: one command per unit so each keeps its own position.
                // "hold" additionally switches the unit into guard-area fire mode.
                accepted += holdGround(unitGroup, "hold".equals(kind));
                continue;
            }
            if (command != null && command.getUnitCount() > 0) {
                accepted++;
            } else {
                rejected++;
            }
        }
        engine.commandController.dispatchQueuedCommands();
        writeLine("{\"protocol\":\"ironzero/1\",\"type\":\"receipt\",\"message_id\":\"receipt-"
                + engine.simulationTick + "\",\"episode_id\":\"rwpp-bridge\",\"tick\":"
                + engine.simulationTick + ",\"decision_id\":\"" + decisionId + "\","
                + "\"accepted\":" + (accepted > 0) + ",\"executed_at_tick\":"
                + (accepted > 0 ? engine.simulationTick + 1 : null) + ","
                + "\"accepted_count\":" + accepted + ",\"rejected_count\":" + rejected
                + (accepted > 0 ? "" : ",\"rejected_reason\":\"ACTION_MASKED\"") + "}");
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
