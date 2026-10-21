-- Panel spend for weeks 17-101 must equal basket spend for the same weeks (to the cent).
with panel as (
    select sum(spend) s from {{ ref('fct_member_week') }}
),
baskets as (
    select sum(basket_value) s from {{ ref('fct_basket') }}
    where week_no between {{ var('first_stable_week') }} and {{ var('last_full_week') }}
)
select * from panel, baskets where abs(panel.s - baskets.s) > 0.01
