# Nectar Offer Engine

[![ci](https://github.com/TilakHeble/nectar-offer-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/TilakHeble/nectar-offer-engine/actions/workflows/ci.yml)

**Which loyalty members should get which personalised offer, so that each £ of discount buys the most *incremental* profit?**

An end-to-end, production-style project modelled on Sainsbury's **Nectar Prices**: real grocery transaction data, a randomised offer experiment layered on top, uplift models, a budget-constrained offer optimiser, a scoring API and monitoring.

> Status: **weeks 1–7 complete**: data pipeline, dbt warehouse, analysis, randomised offer experiment, uplift models with honest selection, and a budget-constrained offer optimiser.

## Headline result

On a held-out test set, **uplift targeting turns a loss-making offer programme profitable**. The selected model (X-learner) earns +$107 of incremental profit where blanket offers lose $209 and a "likely buyer" model loses $777. The best model captures 77% of the best possible profit. The champion was chosen on validation data without access to the ground truth.

![Model comparison](reports/figures/14_model_comparison.png)

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

## Week 3: the randomised offer experiment

Because the real campaigns were not randomised, `src/nectar/simulate/simulator.py` runs a **randomised experiment on top of the real members**:

- 2,407 active members × 18 category offers (10/20/30% off, 4 weeks) = 43,326 member-offer pairs, each randomly assigned to offer or control (50/50).
- **Baseline buying is calibrated to reality:** a logistic model fitted on what members really bought in weeks 92–95 (cross-validated AUC 0.84). The simulated control group matches the real purchase rate offer by offer, within 3 percentage points.
- **The true effect is known:** each pair's uplift comes from a documented persuadability function (price sensitivity, category familiarity, lapsed buying, spend level, discount depth, plus an unobserved member effect). Ground truth is stored in `gt_` columns that models never see.
- **No leakage:** features use weeks 17–91 only. A test rebuilds them on data truncated at week 92 and checks they are identical.
- **Economics:** 30% gross margin, half of each discount funded by the brand (stated assumptions in `SimConfig`).

![Targeting economics](reports/figures/13_targeting_economics.png)

| Check | Result |
|---|---|
| Balance | All standardised mean differences < 0.05 |
| A/B recovery | Difference in means recovers the true uplift (4.0 pts) and true profit effect within the 95% CI |
| Offers on average | +4 pts conversion, but they **lose money** when sent to everyone |
| Who moves | Uplift peaks among occasional buyers; the 10% most likely buyers move by only 0.6 pts |
| The prize | Perfect uplift targeting turns −$4,097 (offer everyone) into +$658 (offer 19% of pairs); propensity targeting does worse than random |

Full validation: [`notebooks/03_offer_experiment.ipynb`](notebooks/03_offer_experiment.ipynb).

## Weeks 4–6: feature contract, uplift models and selection

- **Feature contract** (`features/schema.py`, Pandera): types, ranges, uniqueness and known offers are checked before training and scoring.
- **Models** (`models/learners.py`), all behind one `fit / score` interface and predicting *incremental gross profit*:
  random (blanket) · response model P(buy | offer) · T-, S-, X-, DR-learner · causal forest · ensemble. LightGBM base learners via EconML.
- **Protocol** (`models/train.py`): member-level 50/25/25 split. The champion is chosen on **validation** using an inverse-propensity-weighted estimate of incremental profit from observed outcomes only. Every run is logged to MLflow (`make mlflow-ui`).
- **Decision rule:** offer when predicted incremental profit > 0. Baselines offer to their top 20%.

| Model (test set) | Share offered | Observed incremental profit | True incremental profit | Share of best possible |
|---|---|---|---|---|
| Response model | 20% | −$994 | −$777 | — |
| Random (blanket) | 20% | −$225 | −$209 | — |
| T-learner | 36% | −$138 | $12 | 7% |
| DR-learner | 25% | $92 | $78 | 43% |
| S-learner | 20% | $64 | $104 | 58% |
| **X-learner (champion)** | 28% | $113 | $107 | 60% |
| Ensemble | 22% | $110 | $120 | 67% |
| Causal forest | 18% | $110 | $138 | 77% |

**What to take from it**
- Targeting "likely buyers" is the worst option: they buy anyway, so the discount is pure margin loss.
- Uplift models make money by holding back offers from habitual category buyers. Weekly category spend is the strongest signal (surrogate tree, R² 0.89).
- **Selection is uncertain:** validation 95% intervals for the top models overlap (chart below). The causal forest was truly best but not distinguishable on validation. The X-learner stays the champion because changing it after seeing test results would be selection on the test set. In production this argues for a larger test or a champion/challenger split.

![Selection uncertainty](reports/figures/16_selection_uncertainty.png)

Full analysis: [`notebooks/04_uplift_models.ipynb`](notebooks/04_uplift_models.ipynb).

## Week 7: budget-constrained offer allocation

Model scores say which offers pay; the optimiser (`optimise/`) decides which to **send** under business limits:

- **value** = champion's predicted incremental profit; **cost** = predicted retailer-funded discount cost from a separate cost model (totals within 2–5% of the truth).
- **Integer programme** (PuLP + CBC): maximise value subject to a discount budget, a per-member cap and optional per-offer (supplier) caps; never send a loss-making pair. A value-per-$ greedy rule is the fallback.
- The **shadow price** of the budget (from the LP relaxation) gives the predicted profit of one more $1 of budget.

![Budget curves](reports/figures/19_budget_curves.png)

| Finding (test set, 602 members, one 4-week cycle) | Implication |
|---|---|
| True profit peaks at a $200 budget (+$120) and then falls | A budget limits the damage from model error; more budget is not always better |
| Capping offers at 3 per member keeps only 42% of uncapped profit at $100 | Persuadable members respond in many categories; the cap is a business cost to quantify |
| With caps, the ILP earns +$37 where the greedy rule loses $6 (at $100) | Exact optimisation is worth it and solves in under a second |
| The plan predicts 34% more profit than it really delivers (at $100) | Optimiser's curse: measure plan impact on a randomised holdout |
| Recommended $150 plan: 2,066 offers to 420 members, +$113 true profit | Offers go mostly to lapsed and occasional buyers; Champions (26% of members) get 8% of offers |

Full analysis: [`notebooks/05_offer_allocation.ipynb`](notebooks/05_offer_allocation.ipynb).

## Quick start

```bash
pip install -e ".[dev,ml]"
# put the 8 Complete Journey CSVs in data/raw/
make ingest      # CSV -> typed Parquet + DuckDB warehouse (~20s)
make dbt         # build and test the staging + mart models (~10s)
make simulate    # generate the randomised offer experiment (~6s)
make train       # train, compare and select models, log to MLflow (~1 min)
make optimise    # allocate offers under a $150 budget
make mlflow-ui   # browse experiment runs
make test        # data-quality contract, mart checks, unit tests
make eda         # re-run the analysis notebooks
make dbt-docs    # browse model documentation and lineage
```

### Windows (no `make`)

```powershell
git clone https://github.com/TilakHeble/nectar-offer-engine.git
cd nectar-offer-engine
python -m venv .venv
.venv\Scripts\activate          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev,ml]"
# copy the 8 Complete Journey CSVs into data\raw\
python -m nectar.ingest.load_raw
cd dbt; dbt build --profiles-dir .; cd ..
python -m nectar.simulate.simulator
python -m nectar.models.train
python -m nectar.optimise.run
pytest -q
```

## Repository layout

```
├── src/nectar/
│   ├── ingest/load_raw.py   # typed, renamed CSV -> Parquet -> DuckDB
│   ├── config.py, viz.py    # paths, shared chart style
│   ├── features/offer_features.py   # point-in-time member x category features
│   ├── simulate/simulator.py        # randomised experiment with known ground truth
│   ├── features/schema.py           # Pandera feature contract + model matrix
│   ├── models/learners.py, train.py # uplift learners, training + selection + MLflow
│   ├── evaluate/metrics.py          # IPW policy value, uplift curves, oracle share
│   ├── optimise/                    # cost model, ILP + greedy allocation, CLI
│   ├── serving/ monitoring/         # upcoming weeks
├── dbt/                     # staging, facts, dims, RFM mart + 42 data tests
├── notebooks/               # 01 EDA · 02 segments & funnel · 03 experiment · 04 uplift models · 05 allocation
├── tests/                   # data quality, marts, simulator, schema, metrics, learners, optimiser (63)
├── models/                  # champion.json, model_comparison.csv (+ champion.joblib, generated)
├── reports/figures/         # charts used in README and write-up
├── pipelines/ dashboards/ docs/
└── .github/workflows/ci.yml # ruff + pytest on every push
```

## Roadmap

| Week | Deliverable |
|---|---|
| 1 ✅ | Repo, ingestion, data-quality tests, EDA |
| 2 ✅ | dbt staging and marts, RFM segmentation, offer funnel |
| 3 ✅ | Randomised offer simulator with known ground-truth uplift |
| 4 ✅ | Feature pipeline with validation, member-level split |
| 5 ✅ | Response-model baseline + T-learner, MLflow tracking |
| 6 ✅ | S / X / DR-learner, causal forest, ensemble, uplift and profit curves (**v1**) |
| 7 ✅ | Budget-constrained offer optimiser |
| 8 | FastAPI scoring service in Docker |
| 9 | Prefect pipeline, CI/CD, drift monitoring |
| 10 | Dashboards, model card, business write-up |

## Tech stack

Python · SQL · DuckDB · Parquet · dbt · scikit-learn · LightGBM · EconML · Pandera · MLflow · PuLP · FastAPI · Docker · Prefect · Evidently · GitHub Actions
