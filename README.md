# Nectar Offer Engine

**Which loyalty members should get which personalised offer, so that each £ of discount buys the most *incremental* profit?**

An end-to-end, production-style project modelled on Sainsbury's **Nectar Prices**: real grocery transaction data, a randomised offer experiment layered on top, uplift models, a budget-constrained offer optimiser, a scoring API and monitoring.

> Status: **Week 1 complete** — repo, ingestion pipeline, data-quality tests, exploratory analysis.

## Why uplift, not "who is likely to buy"

A propensity model sends offers to people who would have bought anyway, so the discount is lost margin. Uplift modelling targets the members whose behaviour *changes because of* the offer.

The EDA shows why this matters on real data: the retailer's historical campaigns went to members who **already spent 4.8x more per week before any campaign started**. Comparing targeted with non-targeted members would measure *who was chosen*, not *what the offer did*.

![Selection bias in historical campaigns](reports/figures/06_campaign_selection_bias.png)

## Data

[dunnhumby — *The Complete Journey*](https://www.dunnhumby.com/source-files/): 2,500 households, ~2.6m transaction lines over 711 days (102 weeks), plus products, demographics, 30 campaigns, coupons, redemptions and in-store promotions. Amounts are in US dollars, as recorded.

| Dataset concept | Nectar framing |
|---|---|
| Household | Nectar member |
| Retail (loyalty-card) discount | Nectar Prices member price |
| Campaign / coupon | Personalised offer campaign |
| Basket | Shopping trip |

Real Nectar data is not public. Because historical campaigns were not randomised, a **simulated randomised experiment** is generated on top of the real members (week 3), using their real behavioural features to drive a known ground-truth uplift.

## Week 1 findings

| # | Finding | Decision |
|---|---|---|
| 1 | Members join the panel over weeks 1–16 | Model windows start at week 17 |
| 2 | Fuel / misc lines have non-unit quantities (up to ~90k) | Excluded from the `grocery_lines` view |
| 3 | Top 20% of members = 52% of grocery spend | Report impact by value segment |
| 4 | Median basket $16; half of members shop < once a week | Frequency and recency are core features |
| 5 | 50% of lines carry a member discount (15.6% of shelf value); rate per member spans 10%–23% | Price sensitivity drives simulated offer response |
| 6 | Historical campaigns targeted heavier spenders (median $31 vs $7 a week, pre-campaign) | Build a randomised simulator for clean uplift |
| 7 | Demographics for 32% of members only, 95% of whom were targeted | Behavioural features for modelling; demographics for fairness checks |

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (HTML export in `reports/`).

## Quick start

```bash
pip install -e ".[dev]"
# put the 8 Complete Journey CSVs in data/raw/
make ingest      # CSV -> typed Parquet + DuckDB warehouse (~20s)
make test        # data-quality contract + unit tests
make eda         # re-run the EDA notebook
```

## Repository layout

```
├── src/nectar/
│   ├── ingest/load_raw.py   # typed, renamed CSV -> Parquet -> DuckDB
│   ├── config.py, viz.py    # paths, shared chart style
│   ├── simulate/ features/ models/ evaluate/ optimise/ serving/ monitoring/   # upcoming weeks
├── notebooks/01_eda.ipynb
├── tests/                   # data-quality contract + ingestion unit tests
├── reports/figures/         # charts used in README and write-up
├── dbt/ pipelines/ dashboards/ docs/
└── .github/workflows/ci.yml # ruff + pytest on every push
```

## Roadmap

| Week | Deliverable |
|---|---|
| 1 ✅ | Repo, ingestion, data-quality tests, EDA |
| 2 | dbt staging and marts, RFM segmentation, redemption funnel |
| 3 | Randomised offer simulator with known ground-truth uplift |
| 4 | Feature pipeline with validation, time-based split |
| 5 | Propensity baseline + T-learner, MLflow tracking |
| 6 | X / DR-learner, causal forest, Qini and profit curves (**v1**) |
| 7 | Budget-constrained offer optimiser |
| 8 | FastAPI scoring service in Docker |
| 9 | Prefect pipeline, CI/CD, drift monitoring |
| 10 | Dashboards, model card, business write-up |

## Tech stack

Python · SQL · DuckDB · Parquet · dbt · scikit-learn · LightGBM · EconML / CausalML · MLflow · PuLP · FastAPI · Docker · Prefect · Evidently · GitHub Actions
