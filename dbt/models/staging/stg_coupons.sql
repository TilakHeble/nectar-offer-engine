-- A coupon (UPC) in a campaign can be valid on many products.
select distinct
    coupon_upc,
    campaign_id,
    product_id
from {{ source('raw', 'coupons') }}
