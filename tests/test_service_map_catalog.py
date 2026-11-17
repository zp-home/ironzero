# SPDX-License-Identifier: Apache-2.0
"""Web 端地图选择器 + 对手难度固定为 1。

这一组用例锁住四个曾经静默失效的行为，每一个都不会抛异常、只是让训练在错误的
世界上进行：

1. ``GET /maps`` 不存在 —— 前端只能从静态 HTML 里读地图，拿不到真实像素尺寸；
2. ``map_width``/``map_height`` 写死 ``2700x2500`` —— 观测层用像素尺寸归一化
   每个单位坐标，在 ``4000x3200`` 的地图上等于把整场战斗压进左三分之一；
   ``Lake`` 一直被写成 ``2600x2500``（真实 ``2600x2600``），y 方向差 100px；
3. ``enemy_ai_difficulty`` 缺省是 ``None`` —— 引擎回落到全局
   ``SettingsEngine.aiDifficulty``，那个默认是 **0**（正常，x1.00，毫无加成）。
   界面上"对手难度"这一栏看着有值，实际跑的是最容易的一档；
4. 前端只发 ``map_name`` 不发尺寸 —— 宿主服务只能猜几何。

最后一个用例把链路闭起来：payload → ``_start_ppo`` → ``build_ppo_command`` →
``train_rwpp.build_game`` → ``RwppGameConfig.map_width_px``，断言的是训练器
**真正拿到**的那个数，而不是中间某一层的字符串。
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "scripts", REPO / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rwpp_service import launch as launch_module  # noqa: E402
from rwpp_service import maps as service_maps  # noqa: E402
from rwpp_service import ppo as ppo_module  # noqa: E402
from rwpp_service import state  # noqa: E402
from rwpp_service.handler import Handler  # noqa: E402

#: 九张原版 2p 地图的真实像素尺寸（``.tmx`` 头部 ``width/height`` × 20px）。
#: 这张表是任务书给的实测值，测试的第二来源是它而不是被测代码自己。
EXPECTED_SIZES = {
    "Beach landing": (4000.0, 3200.0),
    "Big Island": (3600.0, 3600.0),
    "Dire Straight": (2200.0, 2200.0),
    "Fire Bridge": (2400.0, 2400.0),
    "Hills": (2600.0, 2600.0),
    "Ice Island": (2900.0, 2900.0),
    "Lake": (2600.0, 2600.0),
    "Small Island": (2200.0, 2200.0),
    "Two cold sides": (2700.0, 2500.0),
}

MAP_FIELDS = ("name", "map_path", "map_width", "map_height", "water_ratio")


@pytest.fixture(autouse=True)
def isolated_job_state(tmp_path, monkeypatch):
    """Keep these tests out of the *real* ``runs/_rwpp_service/jobs/`` directory.

    ``_start_ppo`` / ``_start`` call ``paths._persist_job``, which writes to
    ``REPO_ROOT/runs/_rwpp_service/jobs/<id>.json``. Without this fixture every
    case here would leave a job record on disk, and the next host-service start
    would revalidate them — a ``running`` PPO record with no artifacts blocks
    ``/ppo`` with a 409 ("a training job is already running") even though nothing
    is running. That happened for real while this file was being written, so the
    isolation is not hypothetical.

    ``paths``/``recover`` bind ``RUNS``/``JOB_STATE_DIR`` at call time, so patching
    the module attributes is enough; ``JOBS`` itself is mutated in place (its
    production contract), hence the explicit clear.
    """

    from rwpp_service import paths, recover

    state.JOBS.clear()
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    monkeypatch.setattr(paths, "JOB_STATE_DIR", tmp_path / "_state")
    monkeypatch.setattr(recover, "RUNS", tmp_path)
    monkeypatch.setattr(recover, "JOB_STATE_DIR", tmp_path / "_state")
    yield
    state.JOBS.clear()


# ---------------------------------------------------------------------------
# GET /maps
# ---------------------------------------------------------------------------
@pytest.fixture()
def host_server():
    """A real ``Handler`` on a loopback socket, so the route is exercised via HTTP.

    Calling ``maps_payload()`` directly would pass even if ``do_GET`` never routed
    ``/maps`` — which is exactly how a "backend has the data but the endpoint
    404s" regression hides.
    """

    from http.server import ThreadingHTTPServer

    state.MAX_PARALLEL = 4
    state.PORT_SLOTS.queue.clear()
    for slot in range(4):
        state.PORT_SLOTS.put(5130 + slot)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_maps_endpoint_returns_the_nine_stock_maps(host_server) -> None:
    import json
    import urllib.request

    with urllib.request.urlopen(host_server + "/maps", timeout=15) as response:
        assert response.status == 200
        payload = json.loads(response.read().decode("utf-8"))

    names = [entry["name"] for entry in payload["maps"]]
    assert len(names) == 9, names
    assert "Lake" in names
    assert sorted(names) == sorted(EXPECTED_SIZES), names

    for entry in payload["maps"]:
        missing = [field for field in MAP_FIELDS if field not in entry]
        assert not missing, f"{entry.get('name')} is missing {missing}"
        assert isinstance(entry["map_path"], str) and entry["map_path"].endswith(".tmx")
        assert entry["map_width"] > 0 and entry["map_height"] > 0
        assert 0.0 <= entry["water_ratio"] <= 1.0


def test_maps_endpoint_publishes_the_enemy_difficulty_contract(host_server) -> None:
    import json
    import urllib.request

    with urllib.request.urlopen(host_server + "/maps", timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    assert payload["default"] == "Lake"
    assert payload["difficulty_choices"] == [-2, -1, 0, 1, 2, 3]
    # 1 (困难, x1.40 income), *not* None: None resolves to the engine global,
    # whose default is 0 (x1.00) — i.e. "no choice" would mean the easiest level.
    assert payload["default_enemy_ai_difficulty"] == 1


def test_unknown_paths_still_404(host_server) -> None:
    import urllib.error
    import urllib.request

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(host_server + "/definitely-not-a-route", timeout=10)
    assert caught.value.code == 404


# ---------------------------------------------------------------------------
# Pixel sizes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", sorted(EXPECTED_SIZES.items()))
def test_each_map_resolves_to_its_real_pixel_size(name: str, expected) -> None:
    assert service_maps.resolve_map_dimensions(name) == expected


def test_lake_is_square() -> None:
    """Lake is 2600x2600, not the 2600x2500 every old config wrote.

    A 100 px shortfall on y is not cosmetic: positions are divided by these
    numbers, so every y coordinate was normalised against a world 4% shorter than
    the real one.
    """

    assert service_maps.resolve_map_dimensions("Lake") == (2600.0, 2600.0)
    assert service_maps.resolve_map_dimensions("Lake") != (2600.0, 2500.0)


@pytest.mark.parametrize(
    "written",
    [
        "Lake",
        "lake",
        "[p2]Lake (2p).tmx",
        "maps/skirmish/[p2]Lake (2p).tmx",
        "maps\\skirmish\\[p2]Lake (2p).tmx",
    ],
)
def test_a_map_is_found_by_any_of_its_spellings(written: str) -> None:
    """Form values, CLI args and asset manifests use different spellings."""

    assert service_maps.resolve_map_dimensions(written) == (2600.0, 2600.0)


@pytest.mark.parametrize("written", ["Dire Straight", "Dire_Straight", "[p2]Dire_Straight (2p) [by uber].tmx"])
def test_underscored_file_names_resolve_to_the_documented_name(written: str) -> None:
    entry = service_maps.lookup_map(written)
    assert entry is not None
    assert entry["name"] == "Dire Straight"
    assert (entry["map_width"], entry["map_height"]) == (2200.0, 2200.0)


def test_arena_maps_keep_the_callers_fallback() -> None:
    """``IronZero25`` is not one of the nine; the caller's own default must win."""

    assert service_maps.lookup_map("IronZero25") is None
    assert service_maps.resolve_map_dimensions("IronZero25", fallback=(2700.0, 2500.0)) == (2700.0, 2500.0)


def test_fallback_catalog_matches_the_registry() -> None:
    """The built-in constants must not drift from the repository registry.

    ``map_catalog()`` overlays the registry's *measurements* onto the built-in
    names, so this compares sizes: a typo in either source shows up here instead
    of as a silently skewed normalisation.
    """

    for entry in service_maps.map_catalog():
        assert (entry["map_width"], entry["map_height"]) == EXPECTED_SIZES[entry["name"]]


def test_catalog_survives_a_broken_registry(monkeypatch) -> None:
    """The endpoint must not 500 just because someone broke ``ironzero.rwpp.maps``."""

    service_maps._reset_cache()

    def explode():
        raise RuntimeError("registry is on fire")

    module = service_maps._load_registry()
    if module is None:  # pragma: no cover - registry always ships with the repo
        pytest.skip("registry module unavailable")
    monkeypatch.setattr(module, "map_specs", explode, raising=True)
    entries = service_maps.map_catalog()
    assert len(entries) == 9
    assert service_maps.resolve_map_dimensions("Lake") == (2600.0, 2600.0)


# ---------------------------------------------------------------------------
# Payload -> command line
# ---------------------------------------------------------------------------
def _command_from_payload(payload: dict) -> list[str]:
    """Run ``_start_ppo`` and return the argv it hands to ``run_ppo``.

    Parses the *real* command line rather than inspecting kwargs: the bug class
    here is "the value was read correctly and then never written into argv".
    """

    captured: list[dict] = []
    real = launch_module.run_ppo
    launch_module.run_ppo = lambda *args, **kwargs: captured.append(kwargs)  # type: ignore[assignment]
    state.JOBS.clear()
    try:
        handler = Handler.__new__(Handler)
        handler.sent = []  # type: ignore[attr-defined]

        def _send(payload_sent, status=200):  # noqa: ANN001
            handler.sent.append((payload_sent, status))  # type: ignore[attr-defined]

        handler._send = _send  # type: ignore[assignment]
        handler._start_ppo(payload)
        for thread in threading.enumerate():
            if thread.daemon and thread is not threading.current_thread():
                thread.join(timeout=5.0)
        assert handler.sent and handler.sent[0][1] == 200, handler.sent  # type: ignore[attr-defined]
        assert captured, "run_ppo was never called"
        kwargs = captured[-1]
    finally:
        launch_module.run_ppo = real  # type: ignore[assignment]
        state.JOBS.clear()

    return ppo_module.build_ppo_command(
        job_dir=Path("runs/x"),
        live_path=Path("runs/x/live.jsonl"),
        **{
            key: value
            for key, value in kwargs.items()
            if key not in ("init_checkpoint",)
        },
        init_checkpoint=kwargs.get("init_checkpoint"),
    )


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command, f"{flag} missing from {command}"
    return command[command.index(flag) + 1]


def test_omitted_enemy_difficulty_is_sent_as_one() -> None:
    """No ``enemy_ai_difficulty`` key must still produce ``--enemy-ai-difficulty 1``.

    ``None`` (the old default) made the engine resolve the global
    ``SettingsEngine.aiDifficulty``, which is 0. The dashboard therefore ran its
    easiest real opponent while showing a difficulty control.
    """

    command = _command_from_payload(
        {"updates": 1, "envs": 1, "map_name": "Lake",
         "map_path": "maps/skirmish/[p2]Lake (2p).tmx"}
    )
    index = command.index("--enemy-ai-difficulty")
    assert command[index:index + 2] == ["--enemy-ai-difficulty", "1"]


@pytest.mark.parametrize("level", [-2, -1, 0, 1, 2, 3])
def test_an_explicit_enemy_difficulty_is_forwarded_verbatim(level: int) -> None:
    command = _command_from_payload(
        {"updates": 1, "envs": 1, "map_name": "Lake",
         "map_path": "maps/skirmish/[p2]Lake (2p).tmx",
         "enemy_ai_difficulty": level}
    )
    assert _flag_value(command, "--enemy-ai-difficulty") == str(level)


def test_null_enemy_difficulty_falls_back_to_one() -> None:
    """A JSON ``null`` from the dashboard means "unset", not "engine default"."""

    assert launch_module.resolve_enemy_ai_difficulty({"enemy_ai_difficulty": None}) == 1
    assert launch_module.resolve_enemy_ai_difficulty({"enemy_ai_difficulty": ""}) == 1
    assert launch_module.resolve_enemy_ai_difficulty({}) == 1
    assert launch_module.resolve_enemy_ai_difficulty({"enemy_ai_difficulty": 0}) == 0


# ---------------------------------------------------------------------------
# The full chain: payload -> argv -> RwppGameConfig
# ---------------------------------------------------------------------------
def _config_from_command(command: list[str]):
    """Parse the trainer argv with the trainer's own parser and build its config."""

    from ironzero.training import train_rwpp

    assert command[:3] == [sys.executable, "-m", "ironzero.training.train_rwpp"]
    args = train_rwpp.build_parser().parse_args(command[3:])
    return args, train_rwpp.build_game(args)


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_SIZES.items()))
def test_the_trainer_receives_the_maps_real_width(name: str, expected) -> None:
    """payload``map_path`` only -> ``RwppGameConfig.map_width_px`` is the real width."""

    entries = {entry["name"]: entry for entry in service_maps.map_catalog()}
    entry = entries[name]
    command = _command_from_payload(
        # Only the path: the size is deliberately NOT supplied by the caller.
        {"updates": 1, "envs": 1, "map_name": name, "map_path": entry["map_path"], "init_mode": "scratch"}
    )
    _args, game = _config_from_command(command)
    width, height = expected
    assert game.map_width_px == pytest.approx(width)
    assert game.map_height_px == pytest.approx(height)
    assert game.map_path == entry["map_path"] if hasattr(game, "map_path") else True  # noqa: SIM210


def test_lake_reaches_the_trainer_as_a_square_map() -> None:
    command = _command_from_payload(
        {"updates": 1, "envs": 1, "map_name": "Lake",
         "map_path": "maps/skirmish/[p2]Lake (2p).tmx", "init_mode": "scratch"}
    )
    _args, game = _config_from_command(command)
    assert game.map_width_px == pytest.approx(2600.0)
    assert game.map_height_px == pytest.approx(2600.0), (
        "2600x2500 was the old constant; the observation would normalise y "
        "against a world 100 px shorter than the real one"
    )


def test_an_explicit_payload_size_still_wins() -> None:
    """A caller that pins a crop must not be overridden by the catalog."""

    command = _command_from_payload(
        {"updates": 1, "envs": 1, "map_name": "Lake",
         "map_path": "maps/skirmish/[p2]Lake (2p).tmx",
         "map_width": 1234.0, "map_height": 567.0, "init_mode": "scratch"}
    )
    _args, game = _config_from_command(command)
    assert game.map_width_px == pytest.approx(1234.0)
    assert game.map_height_px == pytest.approx(567.0)


def test_a_zero_size_is_treated_as_unspecified() -> None:
    """``0`` would divide by zero in the observation; it must mean "resolve it"."""

    assert launch_module.resolve_map_geometry(
        {"map_width": 0, "map_height": 0}, "Lake", "maps/skirmish/[p2]Lake (2p).tmx",
        fallback=(1.0, 1.0),
    ) == (2600.0, 2600.0)


def test_the_job_record_carries_the_resolved_geometry() -> None:
    """The dashboard needs the resolved size to notice a wrong-geometry run."""

    state.JOBS.clear()
    real = launch_module.run_ppo
    launch_module.run_ppo = lambda *a, **k: None  # type: ignore[assignment]
    try:
        handler = Handler.__new__(Handler)
        handler.sent = []  # type: ignore[attr-defined]
        handler._send = lambda payload, status=200: handler.sent.append((payload, status))  # type: ignore[assignment]
        handler._start_ppo(
            {"updates": 1, "envs": 1, "map_name": "Beach landing",
             "map_path": "maps/skirmish/[p2]Beach landing (2p) [by hxyy].tmx"}
        )
        for thread in threading.enumerate():
            if thread.daemon and thread is not threading.current_thread():
                thread.join(timeout=5.0)
        job = state.JOBS[handler.sent[0][0]["job_id"]]  # type: ignore[attr-defined]
        assert job["map_width"] == pytest.approx(4000.0)
        assert job["map_height"] == pytest.approx(3200.0)
        assert job["enemy_ai_difficulty"] == 1
    finally:
        launch_module.run_ppo = real  # type: ignore[assignment]
        state.JOBS.clear()


# ---------------------------------------------------------------------------
# The same rule for /infer and /train
# ---------------------------------------------------------------------------
def test_infer_resolves_the_geometry_too() -> None:
    """``_start`` used to default to 2600x2600 for every map."""

    captured: list[dict] = []
    real = launch_module.run_infer
    launch_module.run_infer = lambda *a, **k: captured.append(k)  # type: ignore[assignment]
    state.JOBS.clear()
    try:
        handler = Handler.__new__(Handler)
        handler.sent = []  # type: ignore[attr-defined]
        handler._send = lambda payload, status=200: handler.sent.append((payload, status))  # type: ignore[assignment]
        handler._start(
            kind="infer",
            payload={"ticks": 600, "map_name": "Big Island",
                     "map_path": "maps/skirmish/[p2]Big Island (2p).tmx"},
        )
        for thread in threading.enumerate():
            if thread.daemon and thread is not threading.current_thread():
                thread.join(timeout=5.0)
        assert captured, "run_infer was never called"
        assert captured[-1]["map_width"] == pytest.approx(3600.0)
        assert captured[-1]["map_height"] == pytest.approx(3600.0)
    finally:
        launch_module.run_infer = real  # type: ignore[assignment]
        state.JOBS.clear()


# ---------------------------------------------------------------------------
# The backend proxy and the frontend
# ---------------------------------------------------------------------------
def test_backend_proxies_the_catalog_from_the_host_service() -> None:
    """The dashboard cannot reach the host service, only ``/api/...``."""

    from fastapi.testclient import TestClient

    import web.backend.app.routers.maps as maps_router
    from web.backend.app import app

    real = maps_router._rwpp_call
    captured: dict[str, object] = {}

    def fake_call(path, payload=None, timeout=10.0):  # noqa: ANN001
        captured["path"] = path
        return service_maps.maps_payload()

    maps_router._rwpp_call = fake_call  # type: ignore[assignment]
    try:
        response = TestClient(app).get("/api/maps/catalog")
    finally:
        maps_router._rwpp_call = real  # type: ignore[assignment]
    assert response.status_code == 200, response.text
    assert captured["path"] == "/maps"
    body = response.json()
    assert len(body["maps"]) == 9
    assert body["default"] == "Lake"
    assert body["default_enemy_ai_difficulty"] == 1


def test_catalog_route_is_declared_before_the_map_id_route() -> None:
    """FastAPI matches in declaration order; ``catalog`` is not a base64 map id."""

    from web.backend.app import app

    paths = [route.path for route in app.routes if getattr(route, "path", "").startswith("/api/maps")]
    assert paths.index("/api/maps/catalog") < paths.index("/api/maps/{map_id}")


def _frontend(name: str) -> str:
    return (REPO / "web" / "frontend" / "js" / name).read_text(encoding="utf-8")


def test_frontend_map_dropdown_is_fed_by_the_catalog() -> None:
    """A dropdown built from a constant cannot know the real sizes."""

    catalog = _frontend("feMapCatalog.js")
    assert "/api/maps/catalog" in (REPO / "web" / "frontend" / "js" / "api.js").read_text(encoding="utf-8")
    assert "loadFeMapCatalog" in catalog
    # Both forms must consult it, not just the PPO one.
    assert "feCatalogMap" in _frontend("ppoRun.js")
    assert "feCatalogMap" in _frontend("feTrain.js")


def test_frontend_submits_the_geometry_with_the_ppo_request() -> None:
    """``map_name`` alone leaves the host guessing the world's size."""

    run = _frontend("ppoRun.js")
    for key in ("map_name:", "map_path:", "map_width:", "map_height:", "enemy_ai_difficulty:"):
        assert key in run, f"the PPO request must send {key}"


def test_frontend_does_not_hard_code_a_map_size() -> None:
    """``2700.0``/``2500.0`` as *fallbacks* are fine; as the answer they are not.

    The old code set ``mapWidth = 2700.0`` and only overwrote it when the asset
    listing happened to contain the map — which it did not when the asset cache
    was cold, so a stock map silently ran at Artillery's size.
    """

    import re

    run = _frontend("ppoRun.js")
    block = run.split("const catalogMap = feCatalogMap(selectedMap);")[0]
    # No assignment may pin the size before the catalog is consulted.
    assert not re.search(r"let mapWidth = 2700\.0;\s*let mapHeight = 2500\.0;\s*const originalMap", block), (
        "the size must come from the catalog before any constant is used"
    )
    # And the catalog branch must actually assign it.
    tail = run.split("} else if (catalogMap) {")[1].split("} else if")[0]
    assert "mapWidth = num(catalogMap.map_width" in tail
    assert "mapHeight = num(catalogMap.map_height" in tail


def test_frontend_default_map_is_lake() -> None:
    """The default in the markup and the catalog's ``default`` must agree."""

    import re

    html = (REPO / "web" / "frontend" / "index.html").read_text(encoding="utf-8")
    select = html.split('id="fe-ppo-map"')[1].split("</select>")[0]
    selected = re.search(r'<option value="([^"]*)"[^>]*\bselected\b', select)
    assert selected, "fe-ppo-map has no option marked selected"
    assert selected.group(1) == service_maps.DEFAULT_MAP_NAME


# ---------------------------------------------------------------------------
# A job record with no run directory must not block /ppo forever
# ---------------------------------------------------------------------------
def test_ppo_record_without_an_out_dir_is_revalidated_as_stale(monkeypatch, tmp_path) -> None:
    """``/ppo`` answers 409 "already running" off a record that never ran.

    ``_start_ppo`` persists the job record *before* the runner thread creates the
    run directory, so a crash in between leaves a ``running`` PPO record with an
    empty ``out_dir``. Nothing will ever write its ``rwpp_summary.json``, so the
    old artifact-based revalidation had no evidence to act on and left it
    ``running`` — the "one PPO job at a time" guard then refused every later
    training start with a 409 that waiting could never clear.

    This was hit for real: a test run wrote such records into the live
    ``runs/_rwpp_service/jobs/`` and the next service start resurrected them.
    """

    from rwpp_service import recover

    state.JOBS.clear()
    real_state_dir = recover.JOB_STATE_DIR
    monkeypatch.setattr(recover, "JOB_STATE_DIR", tmp_path)
    monkeypatch.setattr(recover, "_trainer_pids", lambda: {})
    try:
        (tmp_path / "ppo-orphan.json").write_text(
            '{"id": "ppo-orphan", "kind": "ppo", "status": "running", "out_dir": null,'
            ' "started_ts": 0, "replays": []}',
            encoding="utf-8",
        )
        assert recover.load_persisted_jobs() == 1
        assert state.JOBS["ppo-orphan"]["status"] == "stale"
        assert state.JOBS["ppo-orphan"]["error"]
    finally:
        recover.JOB_STATE_DIR = real_state_dir
        state.JOBS.clear()


def test_a_live_ppo_run_is_still_adopted_not_marked_stale(monkeypatch, tmp_path) -> None:
    """The fix must not turn a genuinely running trainer into ``stale``."""

    from rwpp_service import recover

    state.JOBS.clear()
    real_state_dir = recover.JOB_STATE_DIR
    monkeypatch.setattr(recover, "JOB_STATE_DIR", tmp_path)
    job_dir = tmp_path / "runs" / "rwpp-ppo-ppo-live"
    job_dir.mkdir(parents=True)
    (job_dir / "live.jsonl").write_text('{"seq":1,"kind":"start","payload":{}}\n', encoding="utf-8")
    monkeypatch.setattr(
        recover, "_trainer_pids", lambda: {str(job_dir.resolve()): 4321}
    )
    monkeypatch.setattr(recover, "REPO_ROOT", tmp_path)
    try:
        (tmp_path / "ppo-live.json").write_text(
            '{"id": "ppo-live", "kind": "ppo", "status": "running",'
            ' "out_dir": "runs/rwpp-ppo-ppo-live", "started_ts": 0, "replays": []}',
            encoding="utf-8",
        )
        assert recover.load_persisted_jobs() == 1
        record = state.JOBS["ppo-live"]
        assert record["status"] == "running", "a live trainer must stay running"
        assert record["adopted"] is True
        assert record["pid"] == 4321
    finally:
        recover.JOB_STATE_DIR = real_state_dir
        state.JOBS.clear()


def test_frontend_enemy_difficulty_default_is_one_everywhere() -> None:
    """Markup, JS fallback and state must all say 1 (困难).

    Three places carry the "current value"; if they disagree the run executes a
    different opponent than the form displays. That is the same defect that made
    ``frame_ms``/``vision`` drift apart, and it is silent by construction.
    """

    import re

    html = (REPO / "web" / "frontend" / "index.html").read_text(encoding="utf-8")
    select = html.split('id="fe-ppo-enemy-ai-difficulty"')[1].split("</select>")[0]
    selected = re.search(r'<option value="(-?\d+)"[^>]*\bselected\b', select)
    assert selected, "fe-ppo-enemy-ai-difficulty has no selected option"
    assert selected.group(1) == "1"

    run = _frontend("ppoRun.js")
    match = re.search(
        r"clampInt\(\s*els\.fePpoEnemyAiDifficulty[^;]*?,\s*(-?\d+)\s*,?\s*\)", run, re.S
    )
    assert match, "no clampInt fallback found for fePpoEnemyAiDifficulty"
    assert int(match.group(1)) == 1

    state_source = _frontend("state.js")
    assert (
        "'fe-ppo-enemy-ai-difficulty'" in state_source
        or '"fe-ppo-enemy-ai-difficulty"' in state_source
    ), "the control must be cached in els"
    ppo_source = _frontend("ppo.js")
    assert "enemyAiDifficulty: 1" in ppo_source
