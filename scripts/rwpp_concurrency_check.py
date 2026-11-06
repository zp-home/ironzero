"""Ad-hoc concurrency validation for the host-side real-game service (temporary)."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8300"


def call(path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def wait(job_id, limit=240):
    job = {}
    for _ in range(limit):
        job = call("/jobs/" + job_id, timeout=60)
        if job.get("status") != "running":
            return job
        time.sleep(1)
    return job


def main() -> int:
    print("health:", call("/health"))
    infer_ids = []
    for i in range(2):
        r = call("/infer", {"ticks": 300, "policy": "scripted"})
        infer_ids.append(r["job_id"])
        print("infer start:", r["job_id"])
    train = call(
        "/train",
        {"episodes": 2, "ticks": 240, "policy": "scripted", "concurrency": 2},
    )
    print("train start:", train["job_id"])
    ok = True
    for jid in infer_ids:
        job = wait(jid)
        print(
            "infer",
            jid,
            job.get("status"),
            "frames=", job.get("frames"),
            "port=", job.get("port"),
            "err=", job.get("error"),
        )
        if job.get("status") != "done" or not (job.get("frames") or 0) > 0:
            ok = False
    train_job = wait(train["job_id"])
    eps = train_job.get("episodes_metrics") or []
    replays = train_job.get("replays") or []
    print(
        "train",
        train_job.get("status"),
        "done=", train_job.get("episodes_done"),
        "metrics=", len(eps),
        "replays=", len(replays),
        "unique=", len(set(replays)) == len(replays),
        "err=", train_job.get("error"),
    )
    for m in eps:
        print("  episode:", json.dumps(m, ensure_ascii=False)[:200])
    if train_job.get("status") != "done" or len(eps) != 2:
        ok = False
    print("health:", call("/health"))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
