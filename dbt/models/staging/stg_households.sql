-- Demographics exist for 801 of 2,500 members and are anonymised in this copy of the data.
select
    household_id                                        as member_id,
    age_group,
    marital_group,
    income_level,
    household_size,
    hh_composition_group,
    homeowner,
    kids
from {{ source('raw', 'households') }}
