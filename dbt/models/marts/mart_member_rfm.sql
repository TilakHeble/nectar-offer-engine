-- RFM over the last 52 weeks of data, scored 1-5 by percentile rank (ties share a score).
-- Segment rules follow the common retail RFM grid on Recency x average(Frequency, Monetary).
{% set window_start = var('last_day') - var('rfm_window_days') + 1 %}

with window_baskets as (
    select member_id,
           count(*)           as frequency,
           sum(basket_value)  as monetary
    from {{ ref('fct_basket') }}
    where day between {{ window_start }} and {{ var('last_day') }}
    group by 1
),
base as (
    select
        m.member_id,
        {{ var('last_day') }} - m.last_day      as recency_days,
        coalesce(w.frequency, 0)                as frequency,
        coalesce(w.monetary, 0)                 as monetary
    from {{ ref('dim_member') }} m
    left join window_baskets w using (member_id)
),
scored as (
    select *,
        least(5, floor(percent_rank() over (order by recency_days desc) * 5) + 1)::int as r_score,
        least(5, floor(percent_rank() over (order by frequency) * 5) + 1)::int         as f_score,
        least(5, floor(percent_rank() over (order by monetary) * 5) + 1)::int          as m_score
    from base
),
graded as (
    select *, round((f_score + m_score) / 2.0)::int as fm_score from scored
)
select
    member_id,
    recency_days,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    fm_score,
    r_score::varchar || f_score::varchar || m_score::varchar as rfm_cell,
    case
        when r_score >= 4 and fm_score >= 4 then 'Champions'
        when r_score >= 4 and fm_score >= 2 then 'Potential loyalists'
        when r_score >= 4                   then 'Recent, low value'
        when r_score = 3  and fm_score >= 4 then 'Loyal'
        when r_score = 3  and fm_score >= 2 then 'Needs attention'
        when r_score = 3                    then 'About to sleep'
        when fm_score >= 4                  then 'At risk'
        when r_score = 2                    then 'Hibernating'
        else 'Lost'
    end as segment
from graded
