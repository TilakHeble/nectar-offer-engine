-- One row per basket line. Adds money columns with positive signs and a grocery flag.
select
    household_id                                        as member_id,
    basket_id,
    t.product_id,
    store_id,
    day,
    week_no,
    trans_time_hhmm // 100                              as hour_of_day,
    quantity,
    sales_value,
    -retail_disc                                        as member_discount,
    -(coupon_disc + coupon_match_disc)                  as coupon_discount,
    sales_value - retail_disc - coupon_match_disc       as shelf_value,
    -- Fuel and misc lines record gallons/odometer-style quantities, and non-positive lines are
    -- returns or voids: keep them in staging, flag them out of grocery analysis.
    (p.department not in ('KIOSK-GAS', 'MISC SALES TRAN')
        and quantity > 0 and sales_value > 0)           as is_grocery
from {{ source('raw', 'transactions') }} t
left join {{ source('raw', 'products') }} p using (product_id)
