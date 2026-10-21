-- Which members received which campaign (the "offer sent" event).
select
    household_id                                        as member_id,
    campaign_id
from {{ source('raw', 'campaign_members') }}
