select
    household_id                                        as member_id,
    campaign_id,
    coupon_upc,
    day                                                 as redeemed_day
from {{ source('raw', 'coupon_redemptions') }}
