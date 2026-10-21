-- Funnel stages must not grow: offer-product buyers are a subset of window shoppers.
select * from {{ ref('fct_member_campaign') }} where bought_offer_product and not shopped_in_window
