// SPDX-License-Identifier: Apache-2.0
package ironzero;

import com.corrodinggames.rts.game.RWGameEngine;
import com.corrodinggames.rts.game.Team;
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
 * IronZero RWPP bridge (stage 1): runs a real headless Rusted Warfare 1.15
 * game and streams ironzero/1-style observations over loopback NDJSON.
 *
 * Actions are not injected yet; the platform uses this for real-game state
 * capture and replay recording. Extension args (after --):
 *   --bridge-port 5130  --ticks 600  --map maps/skirmish/[p2]Lake (2p).tmx
 */
public final class IronZeroBridge implements ServerExtension {
    private static final int DECISION_INTERVAL = 10;
    private static final int FRAME_PERIOD_MILLIS = 16;
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
        System.out.println("[ironzero] world ready: mapLoaded="
                + (engine.map != null && engine.map.mapLoaded)
                + " units=" + countUnits() + " teams=" + Team.getTeams().size());

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
            if (engine.simulationTick - lastSentTick >= DECISION_INTERVAL || frames == 1) {
                lastSentTick = engine.simulationTick;
                sendObservation(engine, frames);
            }
            readClientCommands();
            if (frames % 30 == 0 && writer == null) {
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
        System.out.println("[ironzero] bridge finished frames=" + frames + " tick=" + engine.simulationTick);
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
                + "\"capabilities\":[\"obs\",\"tick\",\"real-game\"]}");
    }

    private void sendObservation(RWGameEngine engine, int frame) {
        if (writer == null) {
            return;
        }
        List<String> own = new ArrayList<>();
        List<String> enemies = new ArrayList<>();
        int alive = 0;
        for (Object item : Unit.unitList) {
            if (!(item instanceof Unit)) {
                continue;
            }
            Unit unit = (Unit) item;
            if (unit.removed || unit.hp <= 0.0f) {
                continue;
            }
            alive++;
            String payload = entityJson(unit);
            if (unit.team != null && unit.team.teamId == 0) {
                own.add(payload);
            } else {
                enemies.add(payload);
            }
        }
        String line = "{\"protocol\":\"ironzero/1\",\"type\":\"obs\",\"message_id\":\"obs-"
                + engine.simulationTick + "\",\"episode_id\":\"rwpp-bridge\",\"tick\":"
                + engine.simulationTick + ",\"decision_id\":\"d-" + engine.simulationTick
                + "\",\"game_time\":" + fmt(engine.simulationTick / 60.0)
                + ",\"controlled_team\":0,\"visibility_revision\":" + engine.simulationTick
                + ",\"observation_schema_version\":\"1\","
                + "\"global_state\":{\"unit_count\":" + alive + ",\"building_count\":0,"
                + "\"army_value\":0.0,\"visible_enemy_value\":0.0,\"kills\":0,\"losses\":0},"
                + "\"own_entities\":[" + String.join(",", own) + "],"
                + "\"visible_enemy_entities\":[" + String.join(",", enemies) + "],"
                + "\"remembered_enemies\":[],\"local_maps\":{},"
                + "\"masks\":{\"action_mask\":[],\"unit_group_mask\":[],\"target_mask\":[],\"position_mask\":[]},"
                + "\"reward\":0.0,\"terminated\":false,\"truncated\":false,"
                + "\"info\":{\"source\":\"rwpp\",\"frame\":" + frame + ",\"units_alive\":" + alive + "}}";
        writeLine(line);
    }

    private String entityJson(Unit unit) {
        int teamId = unit.team != null ? unit.team.teamId : -1;
        return "{\"entity_id\":" + unit.id
                + ",\"entity_type_id\":1,\"team_id\":" + teamId
                + ",\"x\":" + fmt(unit.x) + ",\"y\":" + fmt(unit.y)
                + ",\"hp\":" + fmt(unit.hp) + ",\"max_hp\":" + fmt(unit.maxHp)
                + ",\"is_building\":false,\"is_visible\":true,\"is_alive\":true}";
    }

    private void sendDone(RWGameEngine engine) {
        writeLine("{\"protocol\":\"ironzero/1\",\"type\":\"done\",\"message_id\":\"done-"
                + engine.simulationTick + "\",\"episode_id\":\"rwpp-bridge\",\"tick\":"
                + engine.simulationTick + ",\"result\":\"timeout\",\"reward\":0.0,"
                + "\"terminated\":false,\"truncated\":true}");
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
