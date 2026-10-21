# Nectar Offer Engine

**Which loyalty members should get which personalised offer, so that each £ of discount buys the most *incremental* profit?**

An end-to-end, production-style project modelled on Sainsbury's **Nectar Prices**: real grocery transaction data, a randomised offer experiment layered on top, uplift models, a budget-constrained offer optimiser, a scoring API and monitoring.

> Status: **Week 2 complete** — ingestion, dbt warehouse (14 models, 42 data tests), EDA, RFM segmentation and offer-funnel analysis.

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

## Week 2: warehouse, segments and the offer funnel

**dbt data model** (`dbt/`, DuckDB). Every model is tested: uniqueness, relationships, value ranges, and reconciliation checks such as "panel spend equals basket spend to the cent".

| Layer | Model | Grain |
|---|---|---|
| Staging | `stg_transactions`, `stg_products`, `stg_households`, `stg_campaigns`, `stg_campaign_members`, `stg_coupons`, `stg_coupon_redemptions` | Cleaned, renamed source tables |
| Facts | `fct_line` | Grocery basket line |
| | `fct_basket` | Shopping trip |
| | `fct_member_week` | Member × week (dense, weeks 17–101, zeros included) |
| | `fct_member_campaign` | Member × campaign offer funnel |
| Dimensions | `dim_member`, `dim_product` | Member / product |
| Analytics | `mart_member_rfm` | Member: 52-week RFM scores and segment |

![Segment value](reports/figures/07_segment_value.png)

| # | Finding | Implication |
|---|---|---|
| 1 | Champions are 26% of members and 50% of 52-week spend; with Loyal, 38% of members and 71% of spend | Blanket offers to them leak the most margin |
| 2 | 77% of targeted members bought an offer product, but only 12% redeemed a coupon | Buying promoted products is mostly not driven by the coupon |
| 3 | 84% of *untargeted* shoppers bought Type A offer products anyway | Redemption and purchase rates overstate offer impact |
| 4 | Redemption rate is 8% for the lightest pre-campaign spend quintile, 17% for the heaviest | A propensity model would chase heavy shoppers who need offers least |

Full analysis: [`notebooks/02_segments_and_funnel.ipynb`](notebooks/02_segments_and_funnel.ipynb).

## Quick start

```bash
pip install -e ".[dev]"
# put the 8 Complete Journey CSVs in data/raw/
make ingest      # CSV -> typed Parquet + DuckDB warehouse (~20s)
make dbt         # build and test the staging + mart models (~10s)
make test        # data-quality contract, mart checks, unit tests
make eda         # re-run the analysis notebooks
make dbt-docs    # browse model documentation and lineage
```

## Repository layout

```
├── src/nectar/
│   ├── ingest/load_raw.py   # typed, renamed CSV -> Parquet -> DuckDB
│   ├── config.py, viz.py    # paths, shared chart style
│   ├── simulate/ features/ models/ evaluate/ optimise/ serving/ monitoring/   # upcoming weeks
├── dbt/                     # staging, facts, dims, RFM mart + 42 data tests
├── notebooks/               # 01_eda, 02_segments_and_funnel
├── tests/                   # data-quality contract, mart checks, unit tests
├── reports/figures/         # charts used in README and write-up
├── pipelines/ dashboards/ docs/
└── .github/workflows/ci.yml # ruff + pytest on every push
```

## Roadmap

| Week | Deliverable |
|---|---|
| 1 ✅ | Repo, ingestion, data-quality tests, EDA |
| 2 ✅ | dbt staging and marts, RFM segmentation, offer funnel |
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
