"""Nectar offer API.

Endpoints
---------
GET  /health                     liveness: always 200, reports whether a bundle is loaded
GET  /ready                      readiness: 503 until a bundle is loaded
GET  /model                      bundle manifest: version, champion, constraints, plan summary
GET  /members/{member_id}/offers this cycle's allocated offers for a member (plan lookup)
POST /score                      score up to 1,000 (member, offer) feature rows on demand

Run locally:
    uvicorn nectar.serving.app:app --reload
Configure with NECTAR_ARTIFACTS_DIR (default: ./artifacts).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nectar.features.schema import OFFER_IDS
from nectar.serving.bundle import DEFAULT_DIR, Bundle

log = logging.getLogger("nectar.api")

MAX_BATCH = 1_000
OfferId = Literal[tuple(OFFER_IDS)]  # type: ignore[valid-type]


# ---------- request / response contracts ----------


class FeatureRow(BaseModel):
    """One (member, offer) pair. Mirrors the Pandera feature contract used in training."""

    model_config = ConfigDict(extra="forbid")

    member_id: int = Field(ge=1)
    offer_id: OfferId
    discount_pct: Literal[0.1, 0.2, 0.3]
    m_trips_per_week: float = Field(gt=0)
    m_spend_per_week: float = Field(gt=0)
    m_weeks_since_last_trip: int = Field(ge=0)
    m_member_discount_rate: float = Field(ge=0, le=1)
    m_private_label_share: float = Field(ge=0, le=1)
    m_median_basket: float = Field(gt=0)
    m_trips_per_week_13w: float = Field(ge=0)
    c_trip_share: float = Field(ge=0, le=1)
    c_trip_share_13w: float = Field(ge=0, le=1)
    c_spend_per_week: float = Field(ge=0)
    c_spend_per_trip: float = Field(gt=0)
    c_weeks_since_last: int = Field(ge=0)
    c_discount_line_share: float = Field(ge=0, le=1)
    c_ever_bought: bool
    o_spend_per_trip: float = Field(gt=0)


class ScoreRequest(BaseModel):
    rows: list[FeatureRow] = Field(min_length=1, max_length=MAX_BATCH)


class ScoreResult(BaseModel):
    member_id: int
    offer_id: str
    predicted_incremental_profit: float
    predicted_discount_cost: float
    recommend: bool


class ScoreResponse(BaseModel):
    bundle_version: str
    results: list[ScoreResult]


class PlannedOffer(BaseModel):
    offer_id: str
    commodity: str
    discount_pct: float
    predicted_incremental_profit: float
    predicted_discount_cost: float


class MemberOffers(BaseModel):
    member_id: int
    bundle_version: str
    offer_window_weeks: list[int]
    offers: list[PlannedOffer]


# ---------- app ----------


RELOAD_CHECK_SECONDS = float(os.environ.get("NECTAR_RELOAD_CHECK_SECONDS", "5"))


def _manifest_mtime(path: Path) -> float | None:
    try:
        return (path / "manifest.json").stat().st_mtime
    except OSError:
        return None


def _load(path: Path) -> tuple[Bundle | None, str | None]:
    try:
        return Bundle.load(path), None
    except FileNotFoundError as e:
        return None, f"bundle not found: {e.filename}"
    except Exception as e:  # noqa: BLE001 - surface any load failure through /health
        return None, f"bundle failed to load: {e!r}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("NECTAR_ARTIFACTS_DIR", DEFAULT_DIR))
    app.state.artifacts_dir = path
    app.state.bundle_mtime = _manifest_mtime(path)
    app.state.last_check = time.monotonic()
    app.state.bundle, app.state.load_error = _load(path)
    if app.state.bundle:
        log.info("Loaded bundle %s from %s", app.state.bundle.manifest["bundle_version"], path)
    else:
        log.warning("Starting without a bundle: %s", app.state.load_error)
    yield


app = FastAPI(
    title="Nectar Offer Engine API",
    version="1.0.0",
    description="Personalised offer decisions from an uplift model and a budget optimiser.",
    lifespan=lifespan,
)


def maybe_reload(state) -> None:
    """Pick up a newly published bundle without a restart.

    Each worker process checks the manifest's modification time at most every few seconds
    (one stat call), so every worker switches over on its own: a reload endpoint would reach
    only the one worker that received the request. A bundle that fails to load is ignored and
    the previous one keeps serving.
    """
    now = time.monotonic()
    if now - state.last_check < RELOAD_CHECK_SECONDS:
        return
    state.last_check = now
    mtime = _manifest_mtime(state.artifacts_dir)
    if mtime is None or mtime == state.bundle_mtime:
        return
    new, error = _load(state.artifacts_dir)
    if new is None:
        log.error("New bundle failed to load, keeping the current one: %s", error)
        return
    old = state.bundle.manifest["bundle_version"] if state.bundle else None
    state.bundle, state.load_error, state.bundle_mtime = new, None, mtime
    log.info("Reloaded bundle %s (was %s)", new.manifest["bundle_version"], old)


@app.middleware("http")
async def timing(request: Request, call_next):
    maybe_reload(request.app.state)
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    log.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


def _bundle(request: Request) -> Bundle:
    b = request.app.state.bundle
    if b is None:
        raise HTTPException(503, detail=request.app.state.load_error or "bundle not loaded")
    return b


@app.get("/health")
def health(request: Request) -> dict:
    b = request.app.state.bundle
    return {
        "status": "ok" if b else "degraded",
        "bundle_loaded": b is not None,
        "bundle_version": b.manifest["bundle_version"] if b else None,
        "error": request.app.state.load_error,
    }


@app.get("/ready")
def ready(request: Request) -> dict:
    _bundle(request)
    return {"ready": True}


@app.get("/model")
def model_info(request: Request) -> dict:
    return _bundle(request).manifest


@app.get("/members/{member_id}/offers", response_model=MemberOffers)
def member_offers(member_id: int, request: Request) -> MemberOffers:
    b = _bundle(request)
    if not b.has_member(member_id):
        raise HTTPException(404, detail=f"member {member_id} is not eligible this cycle")
    rows = b.offers_for(member_id)
    return MemberOffers(
        member_id=member_id,
        bundle_version=b.manifest["bundle_version"],
        offer_window_weeks=b.manifest["offer_window_weeks"],
        offers=[
            PlannedOffer(
                offer_id=r.offer_id,
                commodity=r.commodity,
                discount_pct=r.discount_pct,
                predicted_incremental_profit=r.value,
                predicted_discount_cost=r.cost,
            )
            for r in rows.itertuples()
        ],
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    b = _bundle(request)
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    # Pydantic has validated every field; the one cross-row rule left is uniqueness.
    dupes = df.duplicated(["member_id", "offer_id"], keep=False)
    if dupes.any():
        pairs = df.loc[dupes, ["member_id", "offer_id"]].drop_duplicates().to_dict("records")
        raise HTTPException(
            422, detail={"error": "duplicate (member_id, offer_id) rows", "pairs": pairs}
        )
    out = b.score(df, validate=False)
    return ScoreResponse(
        bundle_version=b.manifest["bundle_version"],
        results=[ScoreResult(**r) for r in out.to_dict("records")],
    )
