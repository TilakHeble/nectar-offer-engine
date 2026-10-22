"""Nectar Offer Engine dashboard.

Reads only the aggregated snapshot in dashboards/snapshot/ (no raw data, no ML libraries), so it
runs anywhere, including Streamlit Community Cloud.

    streamlit run dashboards/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SNAP = Path(__file__).parent / "snapshot"
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#a8a7a2", "#1baf7a"
STATUS_COLOURS = {"ok": GREEN, "warn": "#eda100", "alert": ORANGE}
# Fixed colour per model, so a model keeps its colour whatever is selected.
MODEL_ORDER = [
    "X-learner",
    "Causal forest",
    "Ensemble",
    "S-learner",
    "DR-learner",
    "T-learner",
    "Response model",
    "Random (blanket)",
    "Perfect ranking (truth)",
]
MODEL_COLOURS = [BLUE, GREEN, "#6250d6", "#e87ba4", "#eda100", "#008300", ORANGE, GREY, "#52514e"]

st.set_page_config(page_title="Nectar Offer Engine", page_icon="🛒", layout="wide")


@st.cache_data
def load() -> dict:
    data = {p.stem: pd.read_csv(p) for p in SNAP.glob("*.csv")}
    data["meta"] = json.loads((SNAP / "meta.json").read_text())
    return data


def money(x: float, decimals: int = 0) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


d = load()
meta, plan = d["meta"], d["meta"]["plan"]
models = d["model_comparison"].set_index("model")
champion = meta["champion"]

st.title("Nectar Offer Engine")
st.caption(
    "Personalised offers chosen by an uplift model and a budget optimiser · "
    f"bundle {meta['bundle_version']} · amounts in {meta['currency']}"
)

tab_overview, tab_plan, tab_models, tab_fair, tab_monitor = st.tabs(
    ["Overview", "Offer plan", "Models", "Fairness", "Monitoring"]
)

# ---------------- Overview ----------------
with tab_overview:
    champ, blanket, resp = (models.loc[m] for m in (champion, "Random (blanket)", "Response model"))
    st.subheader("Uplift targeting turns a loss-making offer programme profitable")
    st.write(
        "On held-out test members (one 4-week cycle, true incremental profit from the "
        "randomised experiment):"
    )
    best = models[models.kind == "uplift"].true_incremental_profit.idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{champion} (champion)", money(champ.true_incremental_profit))
    c1.caption(f"{champ.share_of_oracle:.0%} of the best possible profit")
    c2.metric("Blanket offers (random 20%)", money(blanket.true_incremental_profit))
    c2.caption("discount mostly given to people who'd buy anyway")
    c3.metric("'Likely buyer' model", money(resp.true_incremental_profit))
    c3.caption("targets exactly the people who need offers least")
    c4.metric(f"Best uplift model ({best})", money(models.loc[best, "true_incremental_profit"]))
    c4.caption(f"{models.loc[best, 'share_of_oracle']:.0%} of the best possible profit")

    st.divider()
    st.subheader("This cycle's plan")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Offers sent", f"{plan['offers_sent']:,}")
    k2.metric("Members reached", f"{plan['members_reached']:,}")
    k2.caption(
        f"{plan['members_reached'] / plan['eligible_members']:.0%} of "
        f"{plan['eligible_members']:,} eligible"
    )
    k3.metric("Expected discount cost", money(plan["expected_cost"]))
    k3.caption(f"budget {money(meta['constraints']['budget'])}")
    k4.metric("Predicted incremental profit", money(plan["predicted_profit"]))
    k4.caption("model estimate; validate with a holdout")
    st.caption(
        f"Max {meta['constraints']['max_per_member']} offers per member · features to week "
        f"{meta['features_cutoff_week'] - 1} · decision rule: offer only when predicted "
        "incremental profit is positive."
    )

# ---------------- Offer plan ----------------
with tab_plan:
    st.subheader("Where the discount goes")
    seg = d["plan_by_segment"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Reach by RFM segment**: lapsed and occasional members get the most offers")
        chart = (
            alt.Chart(seg)
            .mark_bar(color=BLUE, cornerRadiusEnd=3)
            .encode(
                x=alt.X("reach:Q", axis=alt.Axis(format="%"), title="Members receiving an offer"),
                y=alt.Y("segment:N", sort="-x", title=None),
                tooltip=[
                    "segment",
                    "members",
                    alt.Tooltip("reach:Q", format=".0%"),
                    alt.Tooltip("offers_per_member:Q", format=".2f"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown(f"**Offers per member** (capped at {meta['constraints']['max_per_member']})")
        dist = d["offers_per_member"]
        chart = (
            alt.Chart(dist)
            .mark_bar(color=BLUE, cornerRadiusEnd=3)
            .encode(
                x=alt.X("offers:O", title="Offers received"),
                y=alt.Y("members:Q", title="Members"),
                tooltip=["offers", "members"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("**By offer**")
    cat = d["plan_by_category"].copy()
    cat["commodity"] = cat.commodity.str.title()
    st.dataframe(
        cat[
            [
                "offer_id",
                "commodity",
                "discount_pct",
                "offers_sent",
                "eligible_members",
                "expected_discount_cost",
                "predicted_incremental_profit",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "discount_pct": st.column_config.NumberColumn("Discount", format="percent"),
            "offers_sent": "Offers sent",
            "eligible_members": "Eligible",
            "expected_discount_cost": st.column_config.NumberColumn(
                "Expected cost", format="$%.0f"
            ),
            "predicted_incremental_profit": st.column_config.NumberColumn(
                "Predicted profit", format="$%.2f"
            ),
        },
    )

    st.divider()
    st.subheader("How much budget is worth spending?")
    bc = d["budget_curve"]
    st.write(
        "Predicted incremental profit flattens as the budget grows: each extra dollar buys "
        "less (the *shadow price* falls)."
    )
    base = alt.Chart(bc).encode(x=alt.X("budget:Q", title="Discount budget ($)"))
    profit = base.mark_line(color=BLUE, point=True).encode(
        y=alt.Y("predicted_profit:Q", title="Predicted incremental profit ($)"),
        tooltip=[
            "budget",
            alt.Tooltip("predicted_profit:Q", format="$.0f"),
            "offers_sent",
            "members_reached",
            alt.Tooltip("shadow_price:Q", format=".2f", title="profit per extra $"),
        ],
    )
    st.altair_chart(profit, use_container_width=True)

# ---------------- Models ----------------
with tab_models:
    st.subheader("Which model to trust with the discount budget")
    show = models.reset_index().sort_values("true_incremental_profit", ascending=False)
    table = pd.DataFrame(
        {
            "Model": [f"{m} (champion)" if m == champion else m for m in show.model],
            "Type": show.kind.str.title(),
            "Share offered": show.share_offered.map(lambda v: f"{v:.0%}"),
            "Observed profit (A/B estimate)": show.observed_incremental_profit.map(money),
            "True profit": show.true_incremental_profit.map(money),
            "Share of best possible": [
                f"{v:.0%}" if k == "uplift" else "—"
                for v, k in zip(show.share_of_oracle, show.kind, strict=True)
            ],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        f"Champion ({champion}) was chosen on a validation split using observed outcomes "
        "only; ground truth was never used for selection."
    )

    curves = d["uplift_curves"]
    pick = st.multiselect(
        "Models",
        [m for m in curves.model.unique()],
        default=[
            champion,
            "Causal forest",
            "Response model",
            "Random (blanket)",
            "Perfect ranking (truth)",
        ],
    )
    sub = curves[curves.model.isin(pick)]
    chart = (
        alt.Chart(sub)
        .mark_line()
        .encode(
            x=alt.X(
                "share_offered:Q",
                axis=alt.Axis(format="%"),
                title="Share of member-offer pairs offered (ranked by model)",
            ),
            y=alt.Y("true_profit:Q", title="True cumulative incremental profit ($)"),
            color=alt.Color(
                "model:N",
                legend=alt.Legend(orient="bottom", title=None, columns=3),
                scale=alt.Scale(domain=MODEL_ORDER, range=MODEL_COLOURS),
            ),
            strokeDash=alt.condition(
                alt.datum.model == "Perfect ranking (truth)", alt.value([4, 4]), alt.value([1, 0])
            ),
            tooltip=[
                "model",
                alt.Tooltip("share_offered:Q", format=".0%"),
                alt.Tooltip("true_profit:Q", format="$.0f"),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------- Fairness ----------------
with tab_fair:
    st.subheader("Do offers skew by customer group?")
    st.write(
        "The model never sees demographics, but behaviour-based targeting can still skew who "
        "receives offers. Groups are compared on reach (share receiving any offer). A group "
        "below 80% of the best-reached group is flagged (the four-fifths rule); groups under "
        "30 members are too small to judge. Demographics exist for one third of members."
    )
    fair = d["fairness"]
    attr = st.selectbox(
        "Attribute", fair.attribute.unique(), format_func=lambda a: a.replace("_", " ").title()
    )
    sub = fair[fair.attribute == attr].copy()
    sub["status"] = sub.apply(
        lambda r: "flagged" if r.flag else ("too small" if not r.judged else "ok"), axis=1
    )
    chart = (
        alt.Chart(sub)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "reach_ratio:Q", title="Reach vs best-reached group", axis=alt.Axis(format="%")
            ),
            y=alt.Y("group:N", sort="x", title=None),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["ok", "flagged", "too small"], range=[BLUE, ORANGE, GREY]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                "group",
                "members",
                alt.Tooltip("reach:Q", format=".0%"),
                alt.Tooltip("reach_ratio:Q", format=".0%"),
                alt.Tooltip("discount_per_member:Q", format="$.2f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [0.8]}))
        .mark_rule(strokeDash=[4, 4], color=GREY)
        .encode(x="x:Q")
    )
    st.altair_chart(chart + rule, use_container_width=True)
    flagged = sub[sub.flag.astype(bool)]
    if len(flagged):
        st.warning(
            "Flagged for review: "
            + ", ".join(
                f"{g} ({r:.0%} of best)"
                for g, r in zip(flagged.group, flagged.reach_ratio, strict=True)
            )
        )
    else:
        st.success("No judged group falls below the four-fifths threshold.")

# ---------------- Monitoring ----------------
with tab_monitor:
    st.subheader("Is the model still looking at the customers it was trained on?")
    latest = d["drift_latest"]
    hist = d["drift_history"]
    now = hist.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Drift status", now.status.upper())
    m2.metric("Largest feature PSI", f"{now.max_feature_psi:.3f}", help="< 0.10 stable")
    m3.metric("Prediction PSI", f"{now.prediction_psi:.3f}")
    band = pd.DataFrame({"y0": [0.10, 0.25], "y1": [0.25, 1.0], "level": ["warn", "alert"]})
    bands = (
        alt.Chart(band)
        .mark_rect(opacity=0.12)
        .encode(
            y="y0:Q",
            y2="y1:Q",
            color=alt.Color(
                "level:N",
                scale=alt.Scale(domain=["warn", "alert"], range=["#eda100", ORANGE]),
                legend=None,
            ),
        )
    )
    line = (
        alt.Chart(hist)
        .mark_line(color=BLUE, point=True)
        .encode(
            x=alt.X("cutoff_week:Q", title="Feature cutoff week", scale=alt.Scale(zero=False)),
            y=alt.Y(
                "max_feature_psi:Q",
                title="Largest feature PSI vs training",
                scale=alt.Scale(domain=[0, max(0.3, hist.max_feature_psi.max() * 1.1)], clamp=True),
            ),
            tooltip=["cutoff_week", "status", alt.Tooltip("max_feature_psi:Q", format=".3f")],
        )
    )
    st.altair_chart(bands + line, use_container_width=True)
    st.markdown("**Latest check, by feature**")
    st.dataframe(
        latest,
        hide_index=True,
        use_container_width=True,
        column_config={
            "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
            "ks": st.column_config.NumberColumn("KS", format="%.3f"),
        },
    )
    if "run_history" in d:
        st.markdown("**Pipeline runs**")
        st.dataframe(d["run_history"], hide_index=True, use_container_width=True)
