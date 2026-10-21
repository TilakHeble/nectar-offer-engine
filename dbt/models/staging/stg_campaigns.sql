select
    campaign_id,
    campaign_type,
    start_day,
    end_day,
    end_day - start_day + 1                             as duration_days
from {{ source('raw', 'campaigns') }}
