-- One row per shopping trip (grocery lines only).
select
    basket_id,
    any_value(member_id)                                    as member_id,
    min(day)                                                as day,
    min(week_no)                                            as week_no,
    any_value(store_id)                                     as store_id,
    min(hour_of_day)                                        as hour_of_day,
    count(*)                                                as n_lines,
    sum(quantity)                                           as n_items,
    count(distinct department)                              as n_departments,
    sum(sales_value)                                        as basket_value,
    sum(member_discount)                                    as member_discount,
    sum(coupon_discount)                                    as coupon_discount,
    sum(shelf_value)                                        as shelf_value,
    sum(case when is_private_label then sales_value else 0 end)
        / sum(sales_value)                                  as private_label_share
from {{ ref('fct_line') }}
group by basket_id
