-- One row per member: lifetime behaviour, offer history and (where available) demographics.
with baskets as (
    select
        member_id,
        min(day)                                    as first_day,
        max(day)                                    as last_day,
        count(*)                                    as trips,
        sum(basket_value)                           as total_spend,
        median(basket_value)                        as median_basket_value,
        sum(member_discount) / sum(shelf_value)     as member_discount_rate,
        sum(coupon_discount)                        as coupon_savings,
        sum(basket_value * private_label_share) / sum(basket_value) as private_label_share
    from {{ ref('fct_basket') }}
    group by 1
),
departments as (
    select member_id, count(distinct department) as n_departments_bought
    from {{ ref('fct_line') }} group by 1
),
campaigns as (
    select member_id, count(*) as n_campaigns_received
    from {{ ref('stg_campaign_members') }} group by 1
),
redemptions as (
    select member_id,
           count(*)                    as n_coupons_redeemed,
           count(distinct campaign_id) as n_campaigns_redeemed
    from {{ ref('stg_coupon_redemptions') }} group by 1
)
select
    b.member_id,
    b.first_day,
    b.last_day,
    b.last_day - b.first_day + 1                         as tenure_days,
    b.trips,
    b.trips / ((b.last_day - b.first_day + 1) / 7.0)     as trips_per_week,
    b.total_spend,
    b.median_basket_value,
    b.member_discount_rate,
    b.coupon_savings,
    b.private_label_share,
    d.n_departments_bought,
    coalesce(c.n_campaigns_received, 0)                  as n_campaigns_received,
    coalesce(c.n_campaigns_received, 0) > 0              as ever_targeted,
    coalesce(r.n_coupons_redeemed, 0)                    as n_coupons_redeemed,
    coalesce(r.n_campaigns_redeemed, 0)                  as n_campaigns_redeemed,
    h.member_id is not null                              as has_demographics,
    h.age_group, h.marital_group, h.income_level, h.household_size,
    h.hh_composition_group, h.homeowner, h.kids
from baskets b
left join departments d using (member_id)
left join campaigns c using (member_id)
left join redemptions r using (member_id)
left join {{ ref('stg_households') }} h using (member_id)
