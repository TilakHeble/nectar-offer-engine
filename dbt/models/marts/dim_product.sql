with sales as (
    select product_id,
           count(distinct member_id) as n_buyers,
           sum(quantity)             as units_sold,
           sum(sales_value)          as total_sales,
           avg((member_discount > 0)::int) as share_lines_member_discounted
    from {{ ref('fct_line') }}
    group by 1
),
couponed as (
    select product_id, count(distinct campaign_id) as n_campaigns_couponed
    from {{ ref('stg_coupons') }} group by 1
)
select
    p.*,
    coalesce(s.n_buyers, 0)              as n_buyers,
    coalesce(s.units_sold, 0)            as units_sold,
    coalesce(s.total_sales, 0)           as total_sales,
    s.share_lines_member_discounted,
    coalesce(c.n_campaigns_couponed, 0)  as n_campaigns_couponed
from {{ ref('stg_products') }} p
left join sales s using (product_id)
left join couponed c using (product_id)
