"""Feature and prediction drift monitoring.

Compares this cycle's scoring population with the population the model was trained on.

Measure: the Population Stability Index (PSI), standard in credit-risk model monitoring:

    PSI = sum_b (cur_b - ref_b) * ln(cur_b / ref_b)

over bins b fixed on the reference data (deciles for continuous features, one bin per value
for flags). Conventional reading: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
The Kolmogorov-Smirnov statistic is reported alongside; with tens of thousands of rows its
p-value is almost always tiny, so decisions use PSI, not significance.

PSI also has sampling noise, roughly (bins - 1) / n for n independent members: with ~2,400
members it is about 0.01, but with a few hundred, noise alone can cross the 0.10 line. Small
segments therefore need wider thresholds or fewer bins.

The reference is stored as a small *profile* (bin edges and proportions) rather than raw data,
so it can ship with the model and be compared against without the training set.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nectar.features.schema import BOOL_FEATURES, NUMERIC_FEATURES

WARN_PSI = 0.10
ALERT_PSI = 0.25
EPS = 1e-4  # floor for empty bins so the log term stays finite
PREDICTION = "__prediction__"
# Offer-level attributes (discount depth, the category's average spend per trip) are set by the
# offer catalogue and take one value per offer, so their distribution is not member drift.
OFFER_LEVEL = {"discount_pct", "o_spend_per_trip"}
MONITORED = [c for c in NUMERIC_FEATURES if c not in OFFER_LEVEL] + BOOL_FEATURES
STATUS_ORDER = {"ok": 0, "warn": 1, "alert": 2}


def status_for(psi: float) -> str:
    if psi >= ALERT_PSI:
        return "alert"
    if psi >= WARN_PSI:
        return "warn"
    return "ok"


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Interior quantile edges; duplicate edges (from point masses such as zero) are merged."""
    qs = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(qs)


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # side="left": a value equal to an edge falls in the lower bin, so a point mass at an edge
    # (e.g. many zeros) stays in one bin for reference and current alike.
    idx = np.searchsorted(edges, values, side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi_from_props(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(np.asarray(ref, float), EPS, None)
    cur = np.clip(np.asarray(cur, float), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def ks_statistic(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    """Max gap between the binned cumulative distributions (a binned KS statistic)."""
    return float(np.max(np.abs(np.cumsum(cur_props) - np.cumsum(ref_props))))


@dataclass
class FeatureProfile:
    name: str
    kind: str  # "continuous" or "binary"
    edges: list[float]
    proportions: list[float]
    mean: float

    @classmethod
    def fit(cls, name: str, values: np.ndarray, bins: int = 10) -> FeatureProfile:
        values = np.asarray(values, float)
        if set(np.unique(values)) <= {0.0, 1.0}:
            edges = np.array([0.5])
            kind = "binary"
        else:
            edges = _edges(values, bins)
            kind = "continuous"
        return cls(
            name, kind, edges.tolist(), _proportions(values, edges).tolist(), float(values.mean())
        )

    def compare(self, values: np.ndarray) -> dict:
        values = np.asarray(values, float)
        cur = _proportions(values, np.asarray(self.edges))
        psi = psi_from_props(self.proportions, cur)
        return {
            "feature": self.name,
            "kind": self.kind,
            "psi": psi,
            "ks": ks_statistic(np.asarray(self.proportions), cur),
            "ref_mean": self.mean,
            "cur_mean": float(values.mean()),
            "status": status_for(psi),
            "ref_proportions": self.proportions,
            "cur_proportions": cur.tolist(),
        }


@dataclass
class ReferenceProfile:
    """Distribution summary of the training population (and the model's scores on it)."""

    created_at: str
    source: str
    rows: int
    members: int
    features: dict[str, FeatureProfile] = field(default_factory=dict)

    @classmethod
    def fit(
        cls, df: pd.DataFrame, predictions: np.ndarray | None = None, source: str = ""
    ) -> ReferenceProfile:
        prof = cls(
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
            rows=len(df),
            members=int(df.member_id.nunique()),
        )
        for col in MONITORED:
            prof.features[col] = FeatureProfile.fit(col, df[col].astype(float).to_numpy())
        if predictions is not None:
            prof.features[PREDICTION] = FeatureProfile.fit(PREDICTION, predictions)
        return prof

    def save(self, path: Path) -> None:
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: Path) -> ReferenceProfile:
        d = json.loads(Path(path).read_text())
        feats = {k: FeatureProfile(**v) for k, v in d.pop("features").items()}
        return cls(**d, features=feats)


@dataclass
class DriftReport:
    created_at: str
    reference: dict
    current: dict
    results: list[dict]

    @property
    def status(self) -> str:
        worst = max((r["status"] for r in self.results), key=STATUS_ORDER.get, default="ok")
        return worst

    def summary(self) -> dict:
        drifted = [r["feature"] for r in self.results if r["status"] != "ok"]
        pred = next((r for r in self.results if r["feature"] == PREDICTION), None)
        return {
            "status": self.status,
            "features_checked": len([r for r in self.results if r["feature"] != PREDICTION]),
            "features_drifted": drifted,
            "max_feature_psi": max(
                (r["psi"] for r in self.results if r["feature"] != PREDICTION), default=0.0
            ),
            "prediction_psi": pred["psi"] if pred else None,
            "reference_rows": self.reference["rows"],
            "current_rows": self.current["rows"],
            "row_change": self.current["rows"] / max(self.reference["rows"], 1) - 1,
        }

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "summary": self.summary(),
            "reference": self.reference,
            "current": self.current,
            "results": self.results,
        }

    def save(self, directory: Path, name: str = "drift_report") -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        j = directory / f"{name}.json"
        h = directory / f"{name}.html"
        j.write_text(json.dumps(self.to_dict(), indent=2))
        h.write_text(render_html(self))
        return j, h


def compare(
    profile: ReferenceProfile,
    df: pd.DataFrame,
    predictions: np.ndarray | None = None,
    label: str = "",
) -> DriftReport:
    results = [profile.features[c].compare(df[c].astype(float).to_numpy()) for c in MONITORED]
    if predictions is not None and PREDICTION in profile.features:
        results.append(profile.features[PREDICTION].compare(predictions))
    results.sort(key=lambda r: r["psi"], reverse=True)
    return DriftReport(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reference={
            "source": profile.source,
            "rows": profile.rows,
            "members": profile.members,
            "created_at": profile.created_at,
        },
        current={"label": label, "rows": len(df), "members": int(df.member_id.nunique())},
        results=results,
    )


# ---------- HTML report ----------

_LABELS = {
    PREDICTION: "Model prediction (incremental profit)",
    "m_trips_per_week": "Trips per week",
    "m_spend_per_week": "Spend per week",
    "m_weeks_since_last_trip": "Weeks since last trip",
    "m_member_discount_rate": "Member-price reliance",
    "m_private_label_share": "Own-brand share",
    "m_median_basket": "Median basket",
    "m_trips_per_week_13w": "Trips per week (13w)",
    "c_trip_share": "Category share of trips",
    "c_trip_share_13w": "Category share of trips (13w)",
    "c_spend_per_week": "Category spend per week",
    "c_spend_per_trip": "Category spend per trip",
    "c_weeks_since_last": "Weeks since last category buy",
    "c_discount_line_share": "Category bought on deal",
    "c_ever_bought": "Ever bought category",
    "o_spend_per_trip": "Offer category spend per trip",
}


def _bars(ref: list[float], cur: list[float]) -> str:
    """Tiny inline SVG: reference (grey) vs current (blue) bin proportions."""
    n = len(ref)
    w, h, gap = 132, 30, 2
    bw = (w - gap * (n - 1)) / n
    top = max(max(ref), max(cur), 1e-9)
    parts = []
    for i, (r, c) in enumerate(zip(ref, cur, strict=True)):
        x = i * (bw + gap)
        rh, ch = r / top * h, c / top * h
        parts.append(
            f'<rect x="{x:.1f}" y="{h - rh:.1f}" width="{bw:.1f}" height="{rh:.1f}" class="ref"/>'
        )
        parts.append(
            f'<rect x="{x + bw * 0.3:.1f}" y="{h - ch:.1f}" width="{bw * 0.4:.1f}" '
            f'height="{ch:.1f}" class="cur"/>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        f'aria-label="bin proportions">{"".join(parts)}</svg>'
    )


def render_html(report: DriftReport) -> str:
    s = report.summary()
    rows = []
    for r in report.results:
        label = html.escape(_LABELS.get(r["feature"], r["feature"]))
        rows.append(
            f'<tr><td>{label}<div class="code">{html.escape(r["feature"])}</div></td>'
            f'<td><span class="chip {r["status"]}">{r["status"].upper()}</span></td>'
            f'<td class="num">{r["psi"]:.3f}</td><td class="num">{r["ks"]:.3f}</td>'
            f'<td class="num">{r["ref_mean"]:,.3f}</td><td class="num">{r["cur_mean"]:,.3f}</td>'
            f"<td>{_bars(r['ref_proportions'], r['cur_proportions'])}</td></tr>"
        )
    drifted = ", ".join(_LABELS.get(f, f) for f in s["features_drifted"]) or "none"
    pred = "n/a" if s["prediction_psi"] is None else f"{s['prediction_psi']:.3f}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drift report</title><style>
:root {{ --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --line:#e4e3df; --blue:#2a78d6;
  --grey:#c3c2b7; --ok:#008300; --warn:#a86b00; --alert:#c62828; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --fg:#f0efec; --muted:#c3c2b7;
  --line:#3a3936; --blue:#3987e5; --grey:#5c5b56; --ok:#4caf50; --warn:#e0a030; --alert:#ef5350; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,sans-serif;
  margin:0 auto; max-width:1040px; padding:24px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} .muted {{ color:var(--muted); }}
.cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:12px 16px; min-width:150px; }}
.card b {{ display:block; font-size:20px; }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ padding:8px; border-bottom:1px solid var(--line);
  text-align:left; vertical-align:middle; }} th {{ color:var(--muted); font-weight:600; font-size:13px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.code {{ color:var(--muted); font-size:12px; font-family:ui-monospace,monospace; }}
.chip {{ font-size:12px; font-weight:700; padding:2px 8px; border-radius:10px; border:1px solid; }}
.chip.ok {{ color:var(--ok); }} .chip.warn {{ color:var(--warn); }} .chip.alert {{ color:var(--alert); }}
svg .ref {{ fill:var(--grey); }} svg .cur {{ fill:var(--blue); }}
.wrap {{ overflow-x:auto; }}
</style></head><body>
<h1>Drift report: <span class="chip {s["status"]}">{s["status"].upper()}</span></h1>
<div class="muted">Current: {html.escape(report.current["label"])} · Reference:
{html.escape(report.reference["source"])} · Generated {report.created_at}</div>
<div class="cards">
<div class="card"><span class="muted">Features drifted</span><b>{len(s["features_drifted"])} of {s["features_checked"]}</b></div>
<div class="card"><span class="muted">Max feature PSI</span><b>{s["max_feature_psi"]:.3f}</b></div>
<div class="card"><span class="muted">Prediction PSI</span><b>{pred}</b></div>
<div class="card"><span class="muted">Rows (vs reference)</span><b>{s["current_rows"]:,} ({s["row_change"]:+.0%})</b></div>
</div>
<p>Drifted: {html.escape(drifted)}.</p>
<p class="muted">PSI under {WARN_PSI} is stable, {WARN_PSI}–{ALERT_PSI} is a moderate shift
(warn), {ALERT_PSI} and above is a major shift (alert). The bars show each bin's share:
grey is the reference, blue is now.</p>
<div class="wrap"><table><thead><tr><th>Feature</th><th>Status</th><th class="num">PSI</th>
<th class="num">KS</th><th class="num">Ref mean</th><th class="num">Current mean</th>
<th>Distribution</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
</body></html>"""
