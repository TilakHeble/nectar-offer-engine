"""Latency benchmark for a running offer API.

    python -m nectar.serving.benchmark --url http://localhost:8000

Measures end-to-end HTTP latency (client side) for the three request types the app makes:
plan lookups, single on-demand scores and 100-row batch scores.
"""

from __future__ import annotations

import argparse
import time

import httpx
import numpy as np

from nectar.serving.demo_bundle import synthetic_features


def _rows(n: int, seed: int) -> list[dict]:
    df = synthetic_features(n_members=max(1, n // 18 + 1), seed=seed).drop(columns="commodity")
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        for r in df.head(n).to_dict("records")
    ]


def _time(fn, n: int) -> np.ndarray:
    out = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        out.append((time.perf_counter() - t) * 1000)
        r.raise_for_status()
    return np.array(out)


def run(url: str, n: int = 300, seed: int = 0) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    with httpx.Client(base_url=url, timeout=30) as c:
        manifest = c.get("/model").json()
        n_members = manifest["plan"]["eligible_members"]
        members = c.get("/members/1/offers")  # warm-up
        members = [int(m) for m in rng.integers(1, n_members + 1, n)]
        ok = [m for m in members if c.get(f"/members/{m}/offers").status_code == 200][:n]
        single, batch = _rows(1, seed), _rows(100, seed)
        it = iter(ok * 2)
        results = {
            "plan lookup": _time(lambda: c.get(f"/members/{next(it)}/offers"), len(ok)),
            "score 1 row": _time(lambda: c.post("/score", json={"rows": single}), n),
            "score 100 rows": _time(
                lambda: c.post("/score", json={"rows": batch}), max(20, n // 10)
            ),
        }
    return {
        k: {
            "n": len(v),
            "p50_ms": float(np.percentile(v, 50)),
            "p95_ms": float(np.percentile(v, 95)),
            "p99_ms": float(np.percentile(v, 99)),
        }
        for k, v in results.items()
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("-n", type=int, default=300)
    args = p.parse_args()
    for name, s in run(args.url, args.n).items():
        print(
            f"{name:15s} n={s['n']:4d}  p50 {s['p50_ms']:6.1f} ms  p95 {s['p95_ms']:6.1f} ms  "
            f"p99 {s['p99_ms']:6.1f} ms"
        )


if __name__ == "__main__":
    main()
