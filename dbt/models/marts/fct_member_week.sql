-- Dense member x week panel from each member's first stable week to the last full week.
-- Weeks with no shopping appear with zero spend: needed for churn, recency and outcome windows.
with weekly as (
    select member_id, week_no,
           count(*)          as trips,
           sum(basket_value) as spend,
           sum(member_discount) as member_discount
    from {{ ref('fct_basket') }}
    group by 1, 2
),
member_span as (
    select member_id, greatest(min(week_no), {{ var('first_stable_week') }}) as first_week
    from weekly group by 1
),
weeks as (
    select range as week_no
    from range({{ var('first_stable_week') }}, {{ var('last_full_week') }} + 1)
)
select
    s.member_id,
    w.week_no,
    coalesce(wk.trips, 0)            as trips,
    coalesce(wk.spend, 0)            as spend,
    coalesce(wk.member_discount, 0)  as member_discount
from member_span s
join weeks w on w.week_no >= s.first_week
left join weekly wk on wk.member_id = s.member_id and wk.week_no = w.week_no
