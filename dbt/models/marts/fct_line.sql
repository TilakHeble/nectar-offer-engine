-- Grocery basket lines only (fuel, misc, returns and voids removed), enriched with product data.
select
    t.member_id,
    t.basket_id,
    t.product_id,
    t.store_id,
    t.day,
    t.week_no,
    t.hour_of_day,
    t.quantity,
    t.sales_value,
    t.member_discount,
    t.coupon_discount,
    t.shelf_value,
    p.department,
    p.commodity,
    p.sub_commodity,
    p.is_private_label
from {{ ref('stg_transactions') }} t
join {{ ref('stg_products') }} p using (product_id)
where t.is_grocery
