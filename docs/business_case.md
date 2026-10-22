# Spending personalised offers where they change behaviour

*A two-page summary of the Nectar Offer Engine for a commercial audience. Technical detail is in the [README](../README.md) and the [model card](model_card.md).*

## The problem

Personalised member prices are a large, recurring investment. The usual way to target them is to send offers to the members most likely to buy, and that wastes much of the budget: those members would have bought anyway, so the discount is given away. The question that matters commercially is different. **Which members change what they buy *because* of the offer, and does the extra margin pay for the discount?**

Answering it from campaign history doesn't work. In the data used here (2,500 households, two years of real grocery transactions), past campaigns went to members who already spent **4.8× more per week** before any campaign began. Their high redemption mostly measures *who was chosen*, not *what the offer did*.

## What was built

1. **A randomised test on real members.** 2,407 members × 18 category offers (10%, 20% or 30% off for 4 weeks), each offered or held back at random. Members' normal buying is taken from what they really bought, and the response to offers is simulated from a documented model, so the true effect of every offer is known and every method can be graded against it.
2. **Uplift models** that predict the *extra* gross profit of each offer for each member, after the discount. An offer is only considered if it is expected to pay for itself.
3. **A risk-aware budget optimiser** that chooses which offers to send under a discount budget and a limit of 3 offers per member. It favours offers that are *reliably* profitable, not just those with the highest estimate, which stops the plan from over-promising.
4. **A weekly production cycle** that refreshes the data, checks it, checks for drift, builds the plan, checks the plan and publishes it to an API the member app could call. Each step has a safety gate, so a bad week never replaces a good plan.

## Results (held-out members, one 4-week cycle)

| Targeting approach | Incremental profit | |
|---|---|---|
| Send to members most likely to buy | **−$777** | targets exactly the people who need offers least |
| Blanket offers to a random 20% | **−$209** | most discount goes to people who'd buy anyway |
| **Uplift model (selected)** | **+$107** | 60% of the best achievable |
| Best uplift model in the comparison | +$138 | 77% of the best achievable |

- **The same offers swing from a loss to a profit** depending only on who receives them. Compared with blanket offers the swing is $316, about **$0.53 per member per 4-week cycle**. As a rough illustration only, with simulated economics and one small US dataset, that would be around $0.5m per cycle for every million members. The figure that matters is the one a live test produces.
- **Being careful earns more.** Ranking offers by a cautious estimate (the prediction minus half its uncertainty) instead of the raw prediction raises true profit by **37%** at a $100 budget with the 3-offer cap, and it wins at every budget, cap and random seed tested. It also makes the plan's forecast honest: over-promising falls from 133% to 45%.
- **Where the money goes changes.** Lapsed and occasional members receive the most offers (over 90% of *Lost* and *Hibernating* members). *Champions*, who make up 26% of members and 50% of spend, are reached least (42%), because they buy anyway. Habitual categories such as milk almost never get an offer (2 of 4,158 offers).
- **More budget isn't always better.** At a $100 budget, one more dollar of discount returns about $0.72 of forecast profit. At $300 that falls to $0.18, and at $400 to $0.05. The plan stops at $443 of its $600 budget, because no further offer is reliably profitable.

## How to prove it in a live programme

The model's own estimates are optimistic: a plan built from the highest predictions overstates its profit (by 34–133% in the simulation, cut to 6–45% by risk-aware allocation). **Real impact should only be measured against a randomised holdout.** A rollout would:

1. **Hold out 10% of eligible members** from personalised offers each cycle, as the permanent baseline.
2. **Run uplift-targeted offers against the current targeting** on the remaining members, randomised at member level.
3. Track **incremental gross margin per member** (primary), plus discount cost, visit frequency and a fairness check of reach by customer group.
4. **Size the test properly, and use CUPED.** Members' 4-week margin varies a lot ($52.85 standard deviation), so detecting a 50p effect at 80% power needs **175,394 members per arm**. Adjusting each member's result for their own spending before the test (CUPED) removes 63% of that noise and cuts the requirement to **64,284 per arm**, about a third. At Nectar's scale that fits in a single cycle.

## Risks and how they are handled

| Risk | Control in the system |
|---|---|
| Customers change (prices, competition, cost of living) | Weekly drift check against the training population: a 20% spending squeeze blocks publication automatically |
| A bad data load or model produces a bad plan | Data contract checks, plan checks (budget, caps, no loss-making offers, size vs last week) and atomic publishing |
| Offers skew against a customer group | The model uses no demographics, and every plan gets a group-reach check. The cautious plan reaches fewer frequent shoppers, and two small income groups and one household size are flagged for review. Options: a minimum-reach rule per group, or a less cautious setting (see model card) |
| Over-trusting simulated results | Every figure is labelled as simulated, and the rollout plan above makes a randomised holdout the source of truth |

## What I would do next

- **Supplier-funded offers:** let brands co-fund offers that create incremental sales for them, a natural fit for a retail media business such as Nectar360. The optimiser already supports per-offer caps for brand budgets.
- **Multi-cycle effects:** model pantry loading and offer fatigue across consecutive cycles, not just one 4-week window.
- **Real-time context:** feed basket signals from scan-as-you-go shopping (for example SmartShop) into in-store offers, using the same API.

---
*Data: dunnhumby "The Complete Journey" (US grocery, amounts in USD). Offer responses are simulated from a documented model; see the model card for limitations.*
