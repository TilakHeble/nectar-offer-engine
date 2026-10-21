select
    product_id,
    manufacturer_id,
    department,
    brand_type,
    (brand_type = 'Private')                            as is_private_label,
    commodity,
    sub_commodity,
    pack_size,
    department not in ('KIOSK-GAS', 'MISC SALES TRAN')  as is_grocery_department
from {{ source('raw', 'products') }}
