"""Smoke test for a running offer API (standard library only, so it runs inside the image).

    python -m nectar.serving.smoke --url http://localhost:8000

Exits non-zero if any check fails. Used by CI after starting the Docker container.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from nectar.serving.demo_bundle import synthetic_features


def _call(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def run(base: str) -> list[str]:
    failures = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail if not ok else ''}".rstrip())
        if not ok:
            failures.append(name)

    status, health = _call(f"{base}/health")
    check("health is ok", status == 200 and health.get("status") == "ok", health)
    check("ready", _call(f"{base}/ready")[0] == 200)
    status, model = _call(f"{base}/model")
    check("model manifest", status == 200 and "bundle_version" in model, model)

    status, offers = _call(f"{base}/members/1/offers")
    check("member offers", status == 200 and isinstance(offers.get("offers"), list), offers)
    check("unknown member is 404", _call(f"{base}/members/99999999/offers")[0] == 404)

    row = synthetic_features(1).drop(columns="commodity").head(3)
    rows = [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        for r in row.to_dict("records")
    ]
    status, scored = _call(f"{base}/score", {"rows": rows})
    check("score 3 rows", status == 200 and len(scored.get("results", [])) == 3, scored)
    bad = dict(rows[0], c_trip_share=2.0)
    check("invalid row is 422", _call(f"{base}/score", {"rows": [bad]})[0] == 422)

    t = time.perf_counter()
    for _ in range(50):
        _call(f"{base}/members/1/offers")
    ms = (time.perf_counter() - t) / 50 * 1000
    print(f"INFO  plan lookup mean latency over 50 calls: {ms:.1f} ms")
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    failures = run(p.parse_args().url)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
