-- One row per (member, campaign) for every member who was targeted by the campaign or shopped
-- during its window. Flags trace the offer funnel:
--   targeted -> shopped in window -> bought an offer product -> redeemed a coupon.
with shoppers as (
    select distinct b.member_id, c.campaign_id
    from {{ ref('fct_basket') }} b
    join {{ ref('stg_campaigns') }} c on b.day between c.start_day and c.end_day
),
active as (
    select member_id, campaign_id from shoppers
    union
    select member_id, campaign_id from {{ ref('stg_campaign_members') }}
),
offer_product_purchases as (
    select l.member_id, c.campaign_id,
           count(*)          as offer_product_lines,
           sum(l.sales_value) as offer_product_spend
    from {{ ref('fct_line') }} l
    join {{ ref('stg_campaigns') }} c on l.day between c.start_day and c.end_day
    join (select distinct campaign_id, product_id from {{ ref('stg_coupons') }}) cp
      on cp.campaign_id = c.campaign_id and cp.product_id = l.product_id
    group by 1, 2
),
redeemed as (
    select member_id, campaign_id, count(*) as coupons_redeemed
    from {{ ref('stg_coupon_redemptions') }} group by 1, 2
)
select
    a.member_id,
    a.campaign_id,
    c.campaign_type,
    tm.member_id is not null                    as is_targeted,
    sh.member_id is not null                    as shopped_in_window,
    coalesce(p.offer_product_lines, 0) > 0      as bought_offer_product,
    coalesce(p.offer_product_spend, 0)          as offer_product_spend,
    coalesce(r.coupons_redeemed, 0) > 0         as redeemed,
    coalesce(r.coupons_redeemed, 0)             as coupons_redeemed
from active a
join {{ ref('stg_campaigns') }} c using (campaign_id)
left join {{ ref('stg_campaign_members') }} tm using (member_id, campaign_id)
left join shoppers sh using (member_id, campaign_id)
left join offer_product_purchases p using (member_id, campaign_id)
left join redeemed r using (member_id, campaign_id)
