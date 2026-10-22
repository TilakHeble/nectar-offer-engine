# Model card: Nectar offer uplift model

| | |
|---|---|
| **Model** | X-learner (EconML) with LightGBM base learners |
| **Predicts** | Incremental gross profit of sending one category offer to one member over a 4-week window |
| **Decision rule** | Offer only when predicted incremental profit > 0. A budget optimiser then chooses which positive offers to send |
| **Inputs** | 34 columns: 15 behavioural features + 1 flag + 18 offer indicators (no demographics) |
| **Training data** | 21,654 member–offer pairs (50% of members) from a randomised offer experiment on Dunnhumby *The Complete Journey* |
| **Selected by** | Observed incremental profit on a 25% validation split. Ground truth was never used |
| **Code** | `src/nectar/models/learners.py`, `src/nectar/models/train.py` |
| **Status** | Portfolio project. Not deployed on real customers |

## Intended use

- **For:** choosing which members receive which personalised category discounts (a Nectar Prices-style programme) so that each pound of discount buys the most *additional* profit.
- **Users:** loyalty, pricing and CRM analysts, with the weekly pipeline publishing a plan the member app looks up.
- **Not for:**
  - setting shelf prices, or charging anyone more;
  - any credit, eligibility or employment decision;
  - scoring populations unlike the training one (the drift gate blocks this);
  - estimating offer effects from non-randomised history (that history is selection-biased, see below).

## How it was built and why

Historical campaigns in this data went to heavier shoppers: targeted households spent **4.8× more per week before any campaign started**. Their redemption rates therefore mix up *who was chosen* with *what the offer did*. To learn the causal effect, the project runs a **randomised experiment** on top of the real members: 2,407 members × 18 offers, each pair offered or held out at 50/50.

The outcomes are simulated from a documented model:
- **Baseline behaviour** is a logistic model fitted to members' *real* purchases in weeks 92–95 (cross-validated AUC 0.84). Simulated control purchase rates match the real ones within 3 points for every offer.
- **Response to an offer** comes from a stated persuadability function, which includes a hidden member-level effect so that no model can recover it exactly.

The true effect of every pair is known, so models are graded against it.

## Performance (held-out test members, one 4-week cycle)

| Policy | Share offered | Observed incremental profit (A/B estimate) | True incremental profit | Share of best possible |
|---|---|---|---|---|
| **X-learner (this model)** | 28% | $113 | **$107** | **60%** |
| Causal forest | 18% | $110 | $138 | 77% |
| Blanket offers (random 20%) | 20% | −$225 | −$209 | — |
| "Likely buyer" response model (top 20%) | 20% | −$994 | −$777 | — |

- Rank correlation with true profit uplift: **0.79**.
- **Calibration:** across score deciles, predicted and observed incremental profit track each other (notebook 04). Only the top decile is clearly profitable.
- **Selection uncertainty:** on validation, the 95% intervals of the top four uplift models overlap. The causal forest was truly better but indistinguishable on validation. The X-learner was kept because swapping after seeing test results would be selection on the test set.

## Fairness

The model uses no demographic data. Demographics exist for 801 of 2,500 households (32%) and are anonymised labels, so this check covers only that subset. The **published plan** (week 102 cutoff, $600 budget, at most 3 offers per member) was compared across groups on **reach** (the share of eligible members receiving at least one offer). A group is flagged when its reach is below 80% of the best-reached group (the four-fifths rule). Groups under 30 members are reported but not judged.

| Attribute | Groups judged | Reach range | Lowest ratio to best | Flagged |
|---|---|---|---|---|
| Age group | 6 of 6 | 65%–76% | 0.85 | none |
| Household size | 5 of 5 | 66%–75% | 0.88 | none |
| Children | 4 of 4 | 65%–72% | 0.90 | none |
| Income level | 9 of 12 | 57%–79% | **0.71** | **Level 8 (58%), Level 9 (57%)** |

**Reading the income flag.** Levels 8 and 9 have only 38 and 30 members, so their reach is uncertain by roughly ±17 points (95%). The flag says *review*, not *proven bias*. A likely mechanism is that the model gives fewer offers to members who already buy habitually or rely less on member prices, and those traits can correlate with income. The labels are anonymised, so it isn't known which levels are higher incomes. Before any real use:
1. re-run the check on a population with reliable income data and enough members per group;
2. decide, as a business, whether lower reach for a group is acceptable when it reflects lower price sensitivity;
3. if it is not, add a minimum-reach constraint per group to the optimiser.

Offers are a benefit, never a surcharge: no member pays more because of this model.

## Limitations

- **Simulated outcomes.** Results show the method works when the world behaves like the stated persuadability function. A live randomised test is required before trusting any profit figure.
- **Small data, one market.** 2,500 US households in USD; Nectar has many millions of members. Effects such as cross-category cannibalisation, pantry loading over several cycles, and offer fatigue are not modelled.
- **One 4-week window** for training. Seasonality and repeated offers to the same member are not covered.
- **Optimiser's curse.** Picking the offers with the highest *predicted* profit selects some of the most over-estimated ones too: at a $100 budget the plan predicts 34% more profit than it truly delivers. Measure real impact with a randomised holdout, never with the model's own estimate.
- **Economics are assumptions:** a 30% gross margin, and half of each discount funded by the brand.

## Monitoring and maintenance

- **Weekly drift gate:** PSI on every behavioural feature and on predictions against a stored training profile. Under 0.10 is stable, a WARN publishes and schedules retraining, and an ALERT (0.25 or more) blocks publication.
- **Current status:** OK (largest feature PSI 0.012 at week 102). In a simulated cost-of-living squeeze, a 10% squeeze is tolerated, 15% raises a WARN, and 20% or more blocks.
- **Plan gates:** budget, per-member cap, no loss-making offers, and plan size within ±50% of the live plan.
- **Retrain** when drift is WARN for consecutive weeks or after any ALERT, using a fresh randomised holdout, and re-run this card's fairness check on every new plan.
