-- Every coupon redemption must come from a member who received that campaign.
select * from {{ ref('fct_member_campaign') }} where redeemed and not is_targeted
