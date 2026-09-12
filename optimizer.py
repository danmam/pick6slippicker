import streamlit as st
import itertools
import math

# --- UTILITY FUNCTIONS ---
def american_to_prob(value):
    try:
        value = float(value)
        if 0 < value < 100:
            return value / 100.0
        if value > 0:
            return 100 / (value + 100)
        else:
            return abs(value) / (abs(value) + 100)
    except ValueError:
        return 0.0

# --- LEG MULTIPLIER FORMATS ---
# Most sites express a leg modifier on a 1.0x scale: 1.0x means "unmodified"
# and the slip payout is scaled by the product of the leg values.
#
# Underdog Fantasy instead displays the *price* of each leg, where an
# unmodified leg prices at 1.87x. Those prices are NOT proportional to the
# 1.0x scale -- dividing by 1.87 undershoots every observed slip:
#   1.87 + 2.02        -> 3.81x (3.5x unmodified; 3.5 * 2.02/1.87 = 3.78x)
#   1.87 + 2.04        -> 3.85x (3.5x unmodified; 3.5 * 2.04/1.87 = 3.82x)
#   1.87 + 2.55        -> 4.90x (3.5x unmodified; 3.5 * 2.55/1.87 = 4.77x)
#   1.87 + 1.87 + 2.04 -> 7.15x (6.5x unmodified; 6.5 * 2.04/1.87 = 7.09x)
#   1.87 + 1.87 + 2.55 -> 9.10x (6.5x unmodified; 6.5 * 2.55/1.87 = 8.86x)
# The prices fit an affine map instead -- every price carries a fixed offset
# that does not scale with the modifier:
#   price = SHIFT + (BASE - SHIFT) * mult      (BASE = 1.87, SHIFT = 0.17)
#   mult  = (price - SHIFT) / (BASE - SHIFT)
# giving 1.87x -> 1.000x, 2.02x -> 1.088x, 2.04x -> 1.100x, 2.55x -> 1.400x,
# which reproduces every observed payout exactly (3.5 * 1.088 = 3.81,
# 3.5 * 1.1 = 3.85, 3.5 * 1.4 = 4.90, 6.5 * 1.1 = 7.15, 6.5 * 1.4 = 9.10).
# The 2.55x observations pin the slope at 1/1.70 from far outside the cluster
# near 1.87x; the 2.02x/2.04x pair pins the offset. Offsets in [0.167, 0.172]
# fit every point, and 0.17 is the value that lands observed prices on round
# modifiers (2.04x = 1.10x, 2.55x = 1.40x).
#
# Note that Underdog rounds the displayed price to two decimals, so a price
# read off a slip converts to within ~0.3% of the true modifier -- e.g. a leg
# shown as 2.02x is anywhere in 2.015-2.025x, worth 1.085-1.091x.
UNDERDOG_BASE_PRICE = 1.87   # displayed price of an unmodified (1.0x) leg
UNDERDOG_PRICE_SHIFT = 0.17  # fixed portion of a leg price that does not scale


def underdog_price_to_leg_mult(price, base=UNDERDOG_BASE_PRICE, shift=UNDERDOG_PRICE_SHIFT):
    """Convert an Underdog-style leg price (e.g. 2.04x) to a 1.0x-scale multiplier."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return 1.0
    denom = float(base) - float(shift)
    if denom <= 0:
        return 1.0
    return max(0.0, (price - float(shift)) / denom)


def leg_mult_to_underdog_price(mult, base=UNDERDOG_BASE_PRICE, shift=UNDERDOG_PRICE_SHIFT):
    """Inverse of underdog_price_to_leg_mult: 1.0x-scale multiplier -> leg price."""
    try:
        mult = float(mult)
    except (TypeError, ValueError):
        return float(base)
    return float(shift) + (float(base) - float(shift)) * mult


def solve_general_kelly(outcomes):
    """
    Solves for optimal Kelly fraction 'f' given a list of (probability, net_odds) tuples.
    net_odds = (gross_payout - 1).
    """
    # Expected Value check:
    ev = sum(p * b for p, b in outcomes)
    if ev <= 0:
        return 0.0

    def kelly_derivative(f):
        s = 0.0
        for p, b in outcomes:
            # Avoid division by zero if f is too high causing 1+fb <= 0
            if 1 + f * b <= 0:
                return -float('inf')
            s += (p * b) / (1 + f * b)
        return s

    low, high = 0.0, 0.9999
    for _ in range(50):
        mid = (low + high) / 2
        val = kelly_derivative(mid)
        if val > 0:
            low = mid
        else:
            high = mid
    return low

def calculate_expected_growth(outcomes, stake_fraction):
    """
    Calculates expected growth rate G = sum(p_i * ln(1 + f * b_i)).
    Returns value in basis points (bps).
    """
    if stake_fraction <= 0:
        return 0.0
    growth_sum = 0.0
    for prob, net_odds in outcomes:
        term = 1 + stake_fraction * net_odds
        if term <= 0:
             # Bankruptcy risk, log undefined (-inf growth)
            return -float('inf')
        growth_sum += prob * math.log(term)
    return growth_sum * 10000

# --- DK PICK6 PARIMUTUEL MODEL ---
# DK Pick6 pays a guaranteed floor per tier plus a parimutuel "extra winnings"
# overage that depends on how the rest of the pool did. The preset top-tier
# values are 30-day AVERAGE payouts (floor + average overage baked together);
# intermediate tiers only have published floors. The model decomposes:
#
#   overage(top) = max(0, avg(top) - floor(top))
#   payout(tier) = floor*legmults + boost_cash(on the floor part ONLY) + overage
#
# Boosts multiply the guaranteed component only -- the overage is never
# boosted. The all-correct overage is scaled by a "chalk factor": winning with
# picks the pool also holds means sharing the pool with more winners, so
# chalkier-than-typical slips get less overage and contrarian slips more.
#
# Lower tiers have no published averages. Their overage is estimated by
# apportioning the top-tier overage by per-winner parimutuel intensity:
# share_k / w_k, where share_k is the assumed tier split of the prize pool and
# w_k = C(N,k) * p^k * (1-p)^(N-k) is the fraction of a pool of typical
# (p ~ 50% per leg) entries finishing in tier k. Fewer expected winners and a
# bigger pool share both mean more overage per winner. The tier splits below
# follow DK's published 80/20 (two paying tiers) and 70/20/10 (three paying
# tiers) structure and are an editable assumption.
_TIER_SHARE_SPLITS = {1: [1.0], 2: [0.8, 0.2], 3: [0.7, 0.2, 0.1]}
POOL_LEG_PROB = 0.5      # assumed pool-typical per-leg win rate (DK curates ~coinflips)
CHALK_FACTOR_MIN = 0.25  # clamp on the chalk adjustment
CHALK_FACTOR_MAX = 4.0


def chalk_overage_factor(probs, leg_multipliers=None, beta=1.0,
                         lo=CHALK_FACTOR_MIN, hi=CHALK_FACTOR_MAX):
    """
    Chalk adjustment for the all-correct overage: (q0/q)**beta, clamped.

    q  = product of the true leg win probabilities (your inputs).
    q0 = product of DK's implied reference prob per leg: 0.5 for a standard
         1.0x leg, ~0.5/mult for multiplier legs (capped at 1.0), so a properly
         priced multiplier pick is chalk-neutral and difficulty isn't counted
         twice (the leg multiplier already scales the payout).

    q > q0 (chalkier than the typical pool slip) shrinks the overage;
    q < q0 (contrarian) inflates it. beta=0 disables the adjustment.
    Returns 1.0 when any leg prob is unusable.
    """
    if beta == 0:
        return 1.0
    mults = leg_multipliers if leg_multipliers is not None else [1.0] * len(probs)
    q = 1.0
    q0 = 1.0
    for p, m in zip(probs, mults):
        if p <= 0.0 or p >= 1.0:
            return 1.0
        q *= p
        ref = 0.5 / m if m > 0 else 0.5
        q0 *= min(1.0, ref)
    if q <= 0:
        return 1.0
    factor = (q0 / q) ** beta
    return max(lo, min(hi, factor))


def build_tier_components(payout_structure, floor_structure, n_legs,
                          estimate_lower_tier_overage=True,
                          pool_leg_prob=POOL_LEG_PROB):
    """
    Decompose per-tier payouts into (floor_mult, overage_mult).

    Args:
        payout_structure: {wins: avg payout mult}. Top tier = 30-day average;
                          intermediate tiers = guaranteed floors (no avg data).
        floor_structure:  {wins: guaranteed floor mult}. For fixed-payout sites
                          floor == payout, giving zero overage everywhere.
        n_legs: slip size.
        estimate_lower_tier_overage: apportion the top-tier overage to lower
                          tiers by parimutuel intensity (see module docstring).
                          False = lower tiers pay floors only (conservative).

    Returns:
        {wins: (floor_mult, overage_mult)} for paying tiers.
    """
    paying = sorted(
        {k for k, v in payout_structure.items() if v > 0} |
        {k for k, v in floor_structure.items() if v > 0},
        reverse=True)
    comps = {}
    if not paying:
        return comps

    top = n_legs
    top_avg = payout_structure.get(top, 0.0)
    top_floor = floor_structure.get(top, 0.0)
    if top_floor <= 0:
        top_floor = top_avg
    top_over = max(0.0, top_avg - top_floor)

    splits = _TIER_SHARE_SPLITS.get(len(paying))
    if splits is None:
        splits = [1.0] + [0.0] * (len(paying) - 1)

    def _w(k):
        return math.comb(n_legs, k) * pool_leg_prob ** k * (1.0 - pool_leg_prob) ** (n_legs - k)

    w_top = _w(top)
    top_intensity = (splits[0] / w_top) if w_top > 0 else 0.0

    for idx, k in enumerate(paying):
        if k == top:
            comps[k] = (top_floor, top_over)
            continue
        floor_k = floor_structure.get(k, payout_structure.get(k, 0.0))
        over_k = 0.0
        if estimate_lower_tier_overage and top_over > 0 and top_intensity > 0:
            w_k = _w(k)
            if w_k > 0:
                over_k = top_over * (splits[idx] / w_k) / top_intensity
        comps[k] = (floor_k, over_k)
    return comps


def calculate_complex_outcomes(probs, leg_multipliers, tier_components, global_boost,
                               max_boost_amount=0.0, stake=1.0, boost_on_gross=True,
                               sweat_free_fraction=0.0, stake_back_on_win=False,
                               refund_partial_wins=True, chalk_factor=1.0):
    """
    Generates all 2^N scenarios to accurately calculate EV with specific leg multipliers.

    Args:
        probs: List of win probabilities for each leg.
        leg_multipliers: List of payout multipliers for each leg (if it wins).
        tier_components: Dict mapping number of wins (k) to
                         (floor_mult, overage_mult), from build_tier_components.
        global_boost: Boost multiplier. Applies ONLY to the guaranteed floor
                      component (after leg multipliers); the parimutuel overage
                      is added on top unboosted.
        max_boost_amount: Maximum dollar amount the boost can add to payout (0 = unlimited).
        stake: The stake amount used to calculate the dollar cap on boost.
        boost_on_gross: If True, boost multiplies the full guaranteed payout.
                        If False, boost multiplies only its net-profit part
                        (never negative on sub-1x tiers -- a profit boost
                        cannot reduce a payout).
        sweat_free_fraction: Fraction of stake returned on a complete loss (outcome not in
                             tier_components). 0.0 = standard loss, 1.0 = full refund.
        stake_back_on_win: If True, sweat_free_fraction is also stacked on top of winning
                           tiers whose payout already covers the stake (gross_payout >= 1.0).
        refund_partial_wins: For winning tiers that pay out less than the stake, True
                            (default) tops the payout up by sweat_free_fraction of the
                            shortfall. False leaves partial-win tiers untouched.
        chalk_factor: Scales the overage of the all-correct tier only
                      (see chalk_overage_factor).

    Returns:
        List of (probability, net_outcome) tuples.
    """
    num_legs = len(probs)
    outcomes = []
    max_boost_per_dollar = (max_boost_amount / stake) if (max_boost_amount > 0 and stake > 0) else None

    # Iterate through all 2^N combinations (0=Loss, 1=Win)
    for scenario in itertools.product([0, 1], repeat=num_legs):
        scenario_prob = 1.0
        scenario_leg_mult_product = 1.0
        wins = 0

        for i, is_win in enumerate(scenario):
            if is_win:
                scenario_prob *= probs[i]
                scenario_leg_mult_product *= leg_multipliers[i]
                wins += 1
            else:
                scenario_prob *= (1 - probs[i])

        if wins in tier_components:
            floor_mult, overage_mult = tier_components[wins]

            if floor_mult > 0 or overage_mult > 0:
                # Guaranteed component: floor scaled by winning-leg multipliers.
                base_gross = floor_mult * scenario_leg_mult_product
                # Overage scales proportionally with the leg multipliers
                # (harder picks earn more standings points -> larger pool share)
                # and, on the all-correct tier, with the chalk factor.
                overage = overage_mult * scenario_leg_mult_product
                if wins == num_legs:
                    overage *= chalk_factor

                # Boost cash on the guaranteed component ONLY.
                if boost_on_gross:
                    boost_cash = (global_boost - 1.0) * base_gross
                else:
                    boost_cash = (global_boost - 1.0) * max(0.0, base_gross - 1.0)
                if max_boost_per_dollar is not None and boost_cash > max_boost_per_dollar:
                    boost_cash = max_boost_per_dollar

                gross_payout = base_gross + boost_cash + overage
            else:
                # Explicit 0.0 payout in structure (rare but possible)
                gross_payout = 0.0

            if gross_payout < 1.0:
                # Partial win/loss: the payout alone is worth less than the stake.
                if refund_partial_wins:
                    gross_payout += (1.0 - gross_payout) * sweat_free_fraction
            elif stake_back_on_win:
                # Full win (payout already covers the stake).
                gross_payout += sweat_free_fraction

            net_outcome = gross_payout - 1.0

        else:
            # Outcome NOT defined in structure (typically a Loss)
            gross_payout = sweat_free_fraction
            net_outcome = sweat_free_fraction - 1.0

        outcomes.append((scenario_prob, net_outcome))

    return outcomes

def compute_payout_details(tier_components, n_legs, global_boost, boost_on_gross,
                           max_boost_amount, stake, leg_mult_product=1.0,
                           chalk_factor=1.0, sweat_free_fraction=0.0,
                           stake_back_on_win=False, refund_partial_wins=True):
    """
    Compute payout details per win tier for display purposes.
    Multiplier columns assume standard (1.0x) legs; prize/profit dollars for
    the full-win tier are adjusted by leg_mult_product. Dollars also apply the
    stake-back/refund settings, mirroring calculate_complex_outcomes.
    """
    details = []
    max_delta = (max_boost_amount / stake) if (max_boost_amount > 0 and stake > 0) else None

    def apply_refund(payout_mult):
        # Mirrors the stake-back/refund adjustment in calculate_complex_outcomes.
        if payout_mult < 1.0:
            if refund_partial_wins:
                return payout_mult + (1.0 - payout_mult) * sweat_free_fraction
            return payout_mult
        elif stake_back_on_win:
            return payout_mult + sweat_free_fraction
        return payout_mult

    def boost_cash_for(base_gross):
        # Boost cash on the guaranteed component only, with the dollar cap.
        if boost_on_gross:
            bc = (global_boost - 1.0) * base_gross
        else:
            bc = (global_boost - 1.0) * max(0.0, base_gross - 1.0)
        capped = max_delta is not None and bc > max_delta
        return (max_delta if capped else bc), bc, capped

    for wins in sorted(tier_components.keys(), reverse=True):
        floor_mult, overage_mult = tier_components[wins]
        if floor_mult <= 0 and overage_mult <= 0:
            continue

        ov = overage_mult * (chalk_factor if wins == n_legs else 1.0)

        # Display multipliers: standard (1.0x) legs
        bc_eff, bc_raw, capped = boost_cash_for(floor_mult)
        boosted = floor_mult + bc_raw + ov
        effective = floor_mult + bc_eff + ov

        # Dollar amounts: apply leg_mult_product to the full-win tier
        lm = leg_mult_product if wins == n_legs else 1.0
        base_gross_lm = floor_mult * lm
        bc_eff_lm, _bc_raw_lm, capped_lm = boost_cash_for(base_gross_lm)
        eff_lm = base_gross_lm + bc_eff_lm + ov * lm
        refunded_eff_lm = apply_refund(eff_lm)

        details.append({
            'tier': f"{wins}/{n_legs}",
            'base_mult': floor_mult,
            'overage_mult': ov,
            'avg_mult': floor_mult + ov,
            'boosted_mult': boosted,
            'effective_mult': effective,
            'capped': capped or capped_lm,
            'prize_dollars': refunded_eff_lm * stake if stake > 0 else 0,
            'profit_dollars': (refunded_eff_lm - 1) * stake if stake > 0 else 0,
            'boost_value_dollars': bc_eff_lm * stake if stake > 0 else 0,
        })
    return details

# --- PRESETS DATA ---
PRESETS = {
    "Custom": None,
    "Betr Nukes": {
        "p2": 6.0,
        "p3": 10.0,
        "p4": 20.0, "p4_i": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
    "Betr Picks": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 6.0, "p4_i": 1.5,
        "p5": 10.0, "p5_i": 2.0, "p5_i2": 0.4,
        "p6": 20.0, "p6_i": 1.5, "p6_i2": 1.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
    "Dabble": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 10.0, "p4_i": 0.0,
        "p5": 20.0, "p5_i": 0.0, "p5_i2": 0.0,
        "p6": 0.0, "p6_i": 0.0, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
    "DK Pick6 NBA": {
        "p2": 3.08,
        "p3": 6.175,
        "p4": 10.94, "p4_i": 0.0,
        "p5": 14.82, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 31.08, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 58.52, "p7_i": 2.0, "p7_i2": 0.0,
        "p8": 116.9, "p8_i": 3.0, "p8_i2": 1.0,
        "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 12.0, "f6": 25.0, "f7": 40.0, "f8": 80.0,
    },
    "DK Pick6 NBA Promo": {
        "p2": 3.0,
        "p3": 6.16,
        "p4": 10.29, "p4_i": 0.0,
        "p5": 13.13, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 30.88, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 49.73, "p7_i": 2.0, "p7_i2": 0.0,
        "p8": 116.9, "p8_i": 3.0, "p8_i2": 1.0,
        "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 12.0, "f6": 25.0, "f7": 40.0, "f8": 80.0,
    },
    "DK Pick6 CBB": {
        "p2": 3.22,
        "p3": 5.51,
        "p4": 11.83, "p4_i": 0.0,
        "p5": 21.01, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 42.89, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 2.7, "f3": 5.0, "f4": 8.0, "f5": 12.0, "f6": 25.0,
    },
    "DK Pick6 CBB Promo": {
        "p2": 2.7,
        "p3": 5.51,
        "p4": 8.81, "p4_i": 0.0,
        "p5": 17.74, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 42.89, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 2.7, "f3": 5.0, "f4": 8.0, "f5": 12.0, "f6": 25.0,
    },
    "DK Pick6 WNBA": {
        "p2": 3.1,
        "p3": 5.55,
        "p4": 10.05, "p4_i": 0.0,
        "p5": 13.55, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 26.75, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 3.0, "f3": 5.5, "f4": 10.0, "f5": 12.0, "f6": 25.0,
    },
    "DK Pick6 WNBA Promo": {
        "p2": 3,
        "p3": 5.55,
        "p4": 10.05, "p4_i": 0.0,
        "p5": 12.06, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 26.75, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 3.0, "f3": 5.5, "f4": 10.0, "f5": 12.0, "f6": 25.0,
    },
    "DK Pick6 UFC": {
        "p2": 3.98,
        "p3": 8.48,
        "p4": 15.9, "p4_i": 0.0,
        "p5": 21.62, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 63.68, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 2.7, "f3": 5.0, "f4": 8.0, "f5": 10.0, "f6": 18.0,
    },
    "DK Pick6 NHL": {
        "p2": 3.82,
        "p3": 7.06,
        "p4": 12.74, "p4_i": 0.0,
        "p5": 16.54, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 28.13, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 12.0, "f6": 25.0,
    },
    "DK Pick6 NHL Promo": {
        "p2": 3,
        "p3": 7.06,
        "p4": 11.77, "p4_i": 0.0,
        "p5": 15.29, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 28.13, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 12.0, "f6": 25.0,
    },
    "DK Pick6 PGA": {
        "p2": 3.46,
        "p3": 6.38,
        "p4": 12.42, "p4_i": 0.0,
        "p5": 15.3, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 24.26, "p6_i": 1.2, "p6_i2": 0.0,  # 5/6 floor is 1.2 for PGA
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 2.5, "f3": 4.0, "f4": 6.0, "f5": 8.0, "f6": 12.0,
    },
    "DK Pick6 MLB": {
        "p2": 3.4,
        "p3": 6.64,
        "p4": 10.64, "p4_i": 0.0,
        "p5": 15.56, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 26.78, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 40, "p7_i": 2.0, "p7_i2": 0.0,
        "p8": 80, "p8_i": 3.0, "p8_i2": 1.0,
        "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 12.0, "f6": 25.0, "f7": 40.0, "f8": 80.0,
    },
    "DK Pick6 MLB Promo": {
        "p2": 3.0,
        "p3": 6.64,
        "p4": 10.64, "p4_i": 0.0,
        "p5": 12.77, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 26.78, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 40, "p7_i": 2.0, "p7_i2": 0.0,
        "p8": 80, "p8_i": 3.0, "p8_i2": 1.0,
        "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 12.0, "f6": 25.0, "f7": 40.0, "f8": 80.0,
    },
    "DK Pick6 Soccer": {
        "p2": 3.6,
        "p3": 6.66,
        "p4": 12.56, "p4_i": 0.0,
        "p5": 19.4, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 31.2, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        # TODO: 5/6-pick floors blank on DK sheet -- set = avg (zero overage) until known
        "f2": 3.0, "f3": 5.5, "f4": 10.0, "f5": 19.4, "f6": 31.2,
    },
    "DK Pick6 Soccer Promo": {
        "p2": 3,
        "p3": 6.6,
        "p4": 12.11, "p4_i": 0.0,
        "p5": 15.07, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 31.2, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        # TODO: 5/6-pick floors blank on DK sheet -- set = avg (zero overage) until known
        "f2": 3.0, "f3": 5.5, "f4": 10.0, "f5": 15.07, "f6": 31.2,
    },
    "DK Pick6 CS2": {
        "p2": 3.1,
        "p3": 5.27,
        "p4": 8.16, "p4_i": 0.0,
        "p5": 14.94, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 20.29, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        # TODO: 5/6-pick floors blank on DK sheet -- set = avg (zero overage) until known
        "f2": 2.5, "f3": 4.0, "f4": 6.0, "f5": 14.94, "f6": 20.29,
    },
    "DK Pick6 CS2 Promo": {
        "p2": 2.5,
        "p3": 4.94,
        "p4": 7.90, "p4_i": 0.0,
        "p5": 0.00, "p5_i": 0.0, "p5_i2": 0.0,
        "p6": 0.00, "p6_i": 0.0, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 2.5, "f3": 4.0, "f4": 6.0,
    },
    "DK Pick6 Valorant": {
        "p2": 3.51,
        "p3": 8.98,
        "p4": 10.47, "p4_i": 0.0,
        "p5": 0, "p5_i": 0, "p5_i2": 0.0,
        "p6": 0, "p6_i": 0, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 2.5, "f3": 4.0, "f4": 6.0,
    },
    "DK Pick6 COD": {
        "p2": 3.31,
        "p3": 6.14,
        "p4": 9.09, "p4_i": 0.0,
        "p5": 0, "p5_i": 0.0, "p5_i2": 0.0,
        "p6": 0, "p6_i": 0, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 2.5, "f3": 4.0, "f4": 6.0,
    },
    "DK Pick6 LOL": {
        "p2": 3.77,
        "p3": 6.91,
        "p4": 10.94, "p4_i": 0.0,
        "p5": 15.9, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 20.32, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        # TODO: 5/6-pick floors blank on DK sheet -- set = avg (zero overage) until known
        "f2": 2.5, "f3": 4.0, "f4": 6.0, "f5": 15.9, "f6": 20.32,
    },
    "DK Pick6 NFL": {
        "p2": 3.38,
        "p3": 6.64,
        "p4": 11.74, "p4_i": 0.0,
        "p5": 18.26, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 35.72, "p6_i": 1.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
        "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 12.0, "f6": 25.0,
    },
    "Prizepicks": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 6.0, "p4_i": 1.5,
        "p5": 10.0, "p5_i": 2.0, "p5_i2": 0.4,
        "p6": 25.0, "p6_i": 2.0, "p6_i2": 0.4,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
    "RTSports (Mulligan)": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 10.0, "p4_i": 0.0,
        "p5": 12.0, "p5_i": 2.0, "p5_i2": 0.0,
        "p6": 25.0, "p6_i": 2.5, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
    "RTSports (Power)": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 10.0, "p4_i": 0.0,
        "p5": 12.0, "p5_i": 2.0, "p5_i2": 0.0,
        "p6": 40.0, "p6_i": 0.0, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
    "Drafters": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 4.0, "p4_i": 2.0,
        "p5": 20.0, "p5_i": 0.0, "p5_i2": 0.0,
        "p6": 10.0, "p6_i": 2.5, "p6_i2": 1.5,
        "p7": 65.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 50.0, "p8_i": 5.0, "p8_i2": 2.5,
    },
    "Drafters (Power Only)": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 10.0, "p4_i": 0.0,
        "p5": 20.0, "p5_i": 0.0, "p5_i2": 0.0,
        "p6": 40.0, "p6_i": 0.0, "p6_i2": 0.0,
        "p7": 65.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 100.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
    "Underdog Fantasy": {
        "p2": 3.5,
        "p3": 6.5,
        "p4": 7.2, "p4_i": 1.8,
        "p5": 0.0, "p5_i": 0.0, "p5_i2": 0.0,
        "p6": 0.0, "p6_i": 0.0, "p6_i2": 0.0,
        "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
        "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    },
}

# Leg-multiplier input format per preset. Presets not listed use the standard
# 1.0x scale; "underdog" presets take leg prices as Underdog displays them.
_PRESET_LEG_MULT_FORMATS = {
    "Underdog Fantasy": "underdog",
}

# Max stake defaults per preset (0 means no cap)
_PRESET_MAX_STAKES = {
    "Betr Nukes": 10.0,
    "Betr Picks": 10.0,
    "Prizepicks": 5.0,
    "Drafters": 10.0,
    "Drafters (Power Only)": 10.0,
    "Underdog Fantasy": 25.0,
}

# --- PROMO PRESETS DATA ---
# Each entry sets: max_stake_input, boost_mult, max_boost_dollars, boost_on_gross
# sweat_free_enabled is always False and use_std_leg_mults always True for all promo presets
PROMO_PRESETS = {
    "Custom": None,
    "DK Pick6 30% Boost": {
        "max_stake_input": 25.0,
        "boost_mult": 1.30,
        "max_boost_dollars": 150.0,
        "boost_on_gross": True,
    },
    "DK Pick6 Slashed Line": {
        "max_stake_input": 0.0,
        "use_tiered_stakes": True,
        "max_stake_small": 10.0,
        "max_stake_large": 20.0,
        "boost_mult": 1.00,
        "max_boost_dollars": 0.0,
        "boost_on_gross": True,
    },
    "Betr Picks Discount": {
        "max_stake_input": 10.0,
        "boost_mult": 1.00,
        "max_boost_dollars": 0.0,
        "boost_on_gross": True,
    },
    "Betr Picks 23% Boost": {
        "max_stake_input": 10.0,
        "boost_mult": 1.23,
        "max_boost_dollars": 20.0,
        "boost_on_gross": False,
    },
    "Betr Nukes": {
        "max_stake_input": 10.0,
        "boost_mult": 1.00,
        "max_boost_dollars": 0.0,
        "boost_on_gross": True,
    },
}

# --- SESSION STATE INITIALIZATION ---
_SS_DEFAULTS = {
    "boost_mult": 1.0,
    "max_boost_dollars": 0.0,
    "max_stake_input": 0.0,
    "sweat_free_enabled": False,
    "boost_on_gross": True,
    "use_std_leg_mults": True,
    "use_tiered_stakes": False,
    "max_stake_small": 0.0,
    "max_stake_large": 0.0,
    "show_78": False,
    "p7": 0.0, "p7_i": 0.0, "p7_i2": 0.0,
    "p8": 0.0, "p8_i": 0.0, "p8_i2": 0.0,
    "f2": 3.0, "f3": 6.0, "f4": 10.0, "f5": 20.0, "f6": 40.0, "f7": 0.0, "f8": 0.0,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# --- CANONICAL WIDGET STATE ---
# Streamlit discards the session_state entry of any widget it does not render
# on a given run -- a hidden section, a preset that swaps the inputs out, or
# (on some versions) an ordinary rerun. A widget key therefore cannot be the
# store for anything that lives behind a toggle: the input snaps back to its
# default the next time it appears. Canonical values live under "_cv_" keys,
# which are never widget keys and so are never discarded. Widgets are seeded
# from them before rendering and write back to them afterwards.

def _cv_get(name, default):
    """Read the canonical value for a widget, seeding it on first use."""
    ckey = f"_cv_{name}"
    if ckey not in st.session_state:
        st.session_state[ckey] = default
    return st.session_state[ckey]


def _cv_key(name):
    """Current widget key for a canonical name (see _cv_reset for the suffix)."""
    gen = st.session_state.get(f"_cvgen_{name}", 0)
    return name if gen == 0 else f"{name}__g{gen}"


def _cv_reset(name, default):
    """Force a value back to its default.

    A value already registered against a live widget cannot be overwritten in
    place -- Streamlit keeps (or clamps) the registered one. Bumping the
    generation gives the widget a fresh key, so it is built from scratch.
    """
    st.session_state[f"_cvgen_{name}"] = st.session_state.get(f"_cvgen_{name}", 0) + 1
    st.session_state[f"_cv_{name}"] = default
    st.session_state[_cv_key(name)] = default
    return default


def _cv_set(name, value):
    """Set the canonical value and the widget key together (before rendering)."""
    st.session_state[f"_cv_{name}"] = value
    st.session_state[_cv_key(name)] = value
    return value


def _cv_number_input(container, label, name, default, minimum=None, **kwargs):
    """number_input whose value survives runs where the widget isn't rendered."""
    wkey = _cv_key(name)
    if wkey not in st.session_state:
        st.session_state[wkey] = _cv_get(name, default)
    # Repair anything unusable (a stale 0.0 left by an earlier build, a value
    # under the floor) before it reaches the widget.
    cur = st.session_state[wkey]
    if not isinstance(cur, (int, float)) or isinstance(cur, bool) or \
            (minimum is not None and cur < minimum):
        _cv_reset(name, default)
        wkey = _cv_key(name)
    if minimum is not None:
        kwargs["min_value"] = minimum
    val = float(container.number_input(label, key=wkey, **kwargs))
    st.session_state[f"_cv_{name}"] = val
    return val


def _cv_selectbox(container, label, name, options, default, **kwargs):
    """selectbox whose value survives runs where the widget isn't rendered."""
    wkey = _cv_key(name)
    if wkey not in st.session_state:
        st.session_state[wkey] = _cv_get(name, default)
    if st.session_state[wkey] not in options:
        _cv_reset(name, default)
        wkey = _cv_key(name)
    val = container.selectbox(label, options=options, key=wkey, **kwargs)
    st.session_state[f"_cv_{name}"] = val
    return val


# --- STREAMLIT LAYOUT ---
st.set_page_config(page_title="Pick6/DFS Optimizer", layout="wide")
st.title("Pick6 & DFS Props EV Optimizer")

# --- SIDEBAR ---
st.sidebar.header("Configuration")

# --- PRESET SELECTOR (top of sidebar) ---
_prev_preset = st.session_state.get("_prev_preset", None)
selected_preset = st.sidebar.selectbox("Load Payout Preset", list(PRESETS.keys()))

if selected_preset != _prev_preset:
    st.session_state["_prev_preset"] = selected_preset
    if selected_preset != "Custom":
        # Reset promo settings on every new preset selection
        st.session_state["boost_mult"] = 1.00
        st.session_state["max_boost_dollars"] = 0.0
        st.session_state["sweat_free_enabled"] = False
        st.session_state["boost_on_gross"] = True
        st.session_state["use_std_leg_mults"] = True
        _cv_set("leg_mult_format", _PRESET_LEG_MULT_FORMATS.get(selected_preset, "standard"))
        st.session_state["use_tiered_stakes"] = False
        st.session_state["max_stake_small"] = 0.0
        st.session_state["max_stake_large"] = 0.0
        st.session_state["max_stake_input"] = _PRESET_MAX_STAKES.get(selected_preset, 0.0)
        # Load payout multipliers
        for _key, _val in PRESETS[selected_preset].items():
            st.session_state[_key] = _val
        # Load guaranteed floors; presets without explicit floors (fixed-payout
        # sites) default to floor == payout, i.e. zero overage.
        _pdata = PRESETS[selected_preset]
        for _n in range(2, 9):
            st.session_state[f"f{_n}"] = _pdata.get(f"f{_n}", _pdata.get(f"p{_n}", 0.0))

# --- PROMO PRESET SELECTOR ---
_prev_promo = st.session_state.get("_prev_promo_preset", None)
selected_promo = st.sidebar.selectbox("Load Promo Preset", list(PROMO_PRESETS.keys()))

if selected_promo != _prev_promo:
    st.session_state["_prev_promo_preset"] = selected_promo
    if selected_promo != "Custom":
        # Reset first so presets can selectively override
        st.session_state["sweat_free_enabled"] = False
        st.session_state["use_std_leg_mults"] = True
        st.session_state["use_tiered_stakes"] = False
        st.session_state["max_stake_small"] = 0.0
        st.session_state["max_stake_large"] = 0.0
        data = PROMO_PRESETS[selected_promo]
        for _key, _val in data.items():
            st.session_state[_key] = _val

st.sidebar.markdown("---")
bankroll = st.sidebar.number_input("Bankroll ($)", value=8000.0)
kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.0, 1.0, 0.25)
_tiered_help = (
    "When checked, set separate max stake caps for 2-3 pick and 4-8 pick slips."
)
use_tiered_stakes = st.sidebar.checkbox(
    "Different max stakes by slip size",
    key="use_tiered_stakes",
    help=_tiered_help,
)
if use_tiered_stakes:
    max_stake_input = 0.0
    _sc1, _sc2 = st.sidebar.columns(2)
    max_stake_small = _sc1.number_input(
        "Max: 2-3 picks",
        key="max_stake_small",
        min_value=0.0,
        step=5.0,
        help="Max stake cap for 2 and 3-pick slips."
    )
    max_stake_large = _sc2.number_input(
        "Max: 4-8 picks",
        key="max_stake_large",
        min_value=0.0,
        step=5.0,
        help="Max stake cap for 4, 5, 6, 7, and 8-pick slips."
    )
else:
    max_stake_input = st.sidebar.number_input(
        "Max Stake ($)",
        key="max_stake_input",
        help="Cap the recommended stake. If Kelly suggests a smaller stake, it will use Kelly. If Kelly suggests more, it caps at this value."
    )
    max_stake_small = 0.0
    max_stake_large = 0.0

st.sidebar.markdown("---")
st.sidebar.subheader("Promo Settings")
sweat_free_enabled = st.sidebar.checkbox(
    "Sweat Free Mode",
    key="sweat_free_enabled",
    help="If checked, a fraction of your stake is returned based on the selected mode."
)
if sweat_free_enabled:
    sweat_free_mode = st.sidebar.radio(
        "Mode",
        ["Refund on Loss", "Stake Back Win or Lose"],
        help="Refund on Loss: stake fraction returned only when the outcome isn't in the payout structure. "
             "Stake Back Win or Lose: stake fraction returned on top of every outcome, wins included."
    )
    sweat_free_fraction = st.sidebar.slider(
        "Refund Fraction",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Fraction of stake returned. 1.0 = full stake back, 0.5 = half stake back."
    )
    stake_back_on_win = (sweat_free_mode == "Stake Back Win or Lose")
    refund_partial_wins = st.sidebar.checkbox(
        "Stake Back on Partial Win/Loss",
        value=True,
        key="refund_partial_wins",
        help="If checked (default), winning tiers that pay less than your stake (e.g. a 0.3x "
             "payout, a net 0.7x loss) are topped up by the refund fraction toward breakeven. "
             "Works under both modes: with Refund on Loss, it extends the refund to defined "
             "partial win/loss tiers instead of just complete losses. With Stake Back Win or "
             "Lose, full win tiers separately still get the full fraction stacked on top "
             "regardless of this setting. If unchecked, partial win/loss tiers get no top-up "
             "and can still lose part of the stake."
    )
else:
    sweat_free_fraction = 0.0
    stake_back_on_win = False
    refund_partial_wins = True
boost_mult = st.sidebar.number_input(
    "Global Payout Boost (e.g. 1.1 for 10%)",
    key="boost_mult",
    step=0.05
)
max_boost_dollars = st.sidebar.number_input(
    "Max Boost $ (0 = unlimited)",
    key="max_boost_dollars",
    step=5.0,
    help="Cap the boost amount. The payout increase from the boost cannot exceed this dollar amount."
)
boost_on_gross = st.sidebar.checkbox(
    "Boost on gross payout",
    key="boost_on_gross",
    help="Boosts always apply to the guaranteed floor component only — never the "
         "parimutuel overage. Checked: boost multiplies the full guaranteed payout "
         "(50% boost on a 6x floor → 9x floor + overage). Unchecked: boost "
         "multiplies only the floor's net profit (Betr-style)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parimutuel Overage (DK Pick6)")
chalk_beta = st.sidebar.slider(
    "Chalk sensitivity (β)", 0.0, 3.0, 1.0, 0.1,
    help="Scales the all-correct parimutuel overage by (q₀/q)^β, where q is the "
         "product of your true leg probabilities and q₀ the pool-typical slip "
         "(50% per standard leg, ~50%/mult for multiplier legs). Chalkier slips "
         "than typical share the pool with more winners and get less overage; "
         "contrarian slips get more. Clamped to [0.25x, 4x]. 0 disables."
)
est_lower_overage = st.sidebar.checkbox(
    "Estimate lower-tier overage", value=True,
    help="Intermediate tiers (4/5, 5/6, ...) only have guaranteed floors — DK "
         "publishes no averages for them. When checked, the top-tier overage is "
         "apportioned to lower tiers by per-winner parimutuel intensity "
         "(tier pool share ÷ expected fraction of pool entries in that tier). "
         "Unchecked = lower tiers pay floors only (conservative)."
)

st.sidebar.markdown("---")
_leg_format = _cv_get("leg_mult_format", "standard")
_ud_base = _cv_get("ud_base_price", UNDERDOG_BASE_PRICE)
_ud_shift = _cv_get("ud_price_shift", UNDERDOG_PRICE_SHIFT)
if not (isinstance(_ud_base, (int, float)) and isinstance(_ud_shift, (int, float))
        and _ud_base - _ud_shift > 0):
    _ud_base, _ud_shift = UNDERDOG_BASE_PRICE, UNDERDOG_PRICE_SHIFT
_ud_format = _leg_format == "underdog"

use_std_leg_mults = st.sidebar.checkbox(
    f"All legs at standard price ({_ud_base:.2f}x)?" if _ud_format else "All leg multipliers 1.0x?",
    key="use_std_leg_mults",
    help="Underdog prices every leg, so an unmodified leg still shows a multiplier "
         f"({_ud_base:.2f}x). Check this when no leg is boosted or discounted."
         if _ud_format else None
)

_show_78_sidebar = st.session_state.get("show_78", False)
_n_leg_inputs = 8 if _show_78_sidebar else 6
leg_mults = [1.0] * 8
leg_inputs = [_ud_base if _ud_format else 1.0] * 8

if not use_std_leg_mults:
    _leg_format = _cv_selectbox(
        st.sidebar, "Leg multiplier format", "leg_mult_format",
        ["standard", "underdog"], "standard",
        format_func=lambda v: "Standard (1.0x = unmodified)" if v == "standard"
                              else f"Underdog leg price ({_ud_base:.2f}x = unmodified)",
        help="Underdog Fantasy shows each leg's price instead of a 1.0x-scale modifier. "
             "Picking that format lets you type the prices straight off the slip."
    )
    _ud_format = _leg_format == "underdog"

    if _ud_format:
        with st.sidebar.expander("Leg price scale"):
            st.caption(
                "Leg price = offset + (unmodified price − offset) × modifier. "
                "Defaults reproduce Underdog's payouts: 2.02x → 1.088x, "
                "2.04x → 1.100x, 2.55x → 1.400x — so 1.87x + 2.04x pays "
                "3.5 × 1.1 = 3.85x and 1.87x + 2.55x pays 3.5 × 1.4 = 4.90x. "
                "Prices are only shown to 2 decimals, so a converted modifier "
                "can be off by up to ~0.3%."
            )
            # Rendered before the inputs so a reset takes effect on this run:
            # _cv_reset re-keys the widgets, so they are built fresh below.
            if st.button("Reset to defaults", key="ud_scale_reset"):
                _cv_reset("ud_base_price", UNDERDOG_BASE_PRICE)
                _cv_reset("ud_price_shift", UNDERDOG_PRICE_SHIFT)
            _ud_base = _cv_number_input(
                st, "Unmodified leg price", "ud_base_price", UNDERDOG_BASE_PRICE,
                minimum=0.01, step=0.01, format="%.2f"
            )
            _ud_shift = _cv_number_input(
                st, "Price offset", "ud_price_shift", UNDERDOG_PRICE_SHIFT,
                step=0.01, format="%.2f"
            )
            if _ud_base - _ud_shift <= 0:
                st.warning(
                    f"Unmodified price must exceed the offset — using "
                    f"{UNDERDOG_BASE_PRICE:.2f} / {UNDERDOG_PRICE_SHIFT:.2f} until it does."
                )
                _ud_base, _ud_shift = UNDERDOG_BASE_PRICE, UNDERDOG_PRICE_SHIFT

    st.sidebar.subheader("Individual Leg Prices" if _ud_format else "Individual Leg Multipliers")
    lm_cols = st.sidebar.columns(3)
    for i in range(_n_leg_inputs):
        if _ud_format:
            leg_inputs[i] = _cv_number_input(
                lm_cols[i % 3], f"Leg {i+1} price", f"ud_leg_price_{i}", float(_ud_base),
                minimum=0.0, step=0.01, format="%.2f"
            )
            leg_mults[i] = underdog_price_to_leg_mult(leg_inputs[i], _ud_base, _ud_shift)
        else:
            leg_mults[i] = _cv_number_input(
                lm_cols[i % 3], f"Leg {i+1} x", f"leg_mult_{i}", 1.0,
                minimum=0.0, step=0.01, format="%.2f"
            )
            leg_inputs[i] = leg_mults[i]

    if _ud_format:
        st.sidebar.caption(
            "Converted to 1.0x scale: "
            + ", ".join(f"{m:.3f}x" for m in leg_mults[:_n_leg_inputs])
        )

# --- MAIN PAGE ---

st.header("1. Payout Structure (Base Multipliers)")
# Row 1: 2, 3, 4 picks
c1, c2, c3 = st.columns(3)
p2 = c1.number_input("2-Pick Win", value=st.session_state.get("p2", 3.0), key="p2")
p3 = c2.number_input("3-Pick Win", value=st.session_state.get("p3", 6.0), key="p3")

with c3:
    st.markdown("**4-Pick**")
    col_a, col_b = st.columns(2)
    p4 = col_a.number_input("4/4", value=st.session_state.get("p4", 10.0), key="p4")
    p4_i = col_b.number_input("3/4", value=st.session_state.get("p4_i", 0.0), key="p4_i")

# Row 2: 5 and 6 picks
c4, c5 = st.columns(2)
with c4:
    st.markdown("**5-Pick**")
    col_a, col_b, col_c = st.columns(3)
    p5 = col_a.number_input("5/5", value=st.session_state.get("p5", 20.0), key="p5")
    p5_i = col_b.number_input("4/5", value=st.session_state.get("p5_i", 0.0), key="p5_i")
    p5_i2 = col_c.number_input("3/5", value=st.session_state.get("p5_i2", 0.0), key="p5_i2")

with c5:
    st.markdown("**6-Pick**")
    col_a, col_b, col_c = st.columns(3)
    p6 = col_a.number_input("6/6", value=st.session_state.get("p6", 40.0), key="p6")
    p6_i = col_b.number_input("5/6", value=st.session_state.get("p6_i", 0.0), key="p6_i")
    p6_i2 = col_c.number_input("4/6", value=st.session_state.get("p6_i2", 0.0), key="p6_i2")

with st.expander("Guaranteed Floors — all-correct tier (splits averages into floor + parimutuel overage)"):
    st.caption(
        "The top-tier values above are 30-day AVERAGE payouts; enter the guaranteed "
        "minimums here (DK preset floors as of 8/26/26 load automatically). "
        "Overage = average − floor. For fixed-payout sites leave floors equal to "
        "the payouts (zero overage). Intermediate-tier inputs (4/5, 5/6, ...) are "
        "already guaranteed floors."
    )
    fc = st.columns(7)
    f2 = fc[0].number_input("2-Pick Floor", value=st.session_state.get("f2", 3.0), key="f2")
    f3 = fc[1].number_input("3-Pick Floor", value=st.session_state.get("f3", 6.0), key="f3")
    f4 = fc[2].number_input("4/4 Floor", value=st.session_state.get("f4", 10.0), key="f4")
    f5 = fc[3].number_input("5/5 Floor", value=st.session_state.get("f5", 20.0), key="f5")
    f6 = fc[4].number_input("6/6 Floor", value=st.session_state.get("f6", 40.0), key="f6")
    f7 = fc[5].number_input("7/7 Floor", value=st.session_state.get("f7", 0.0), key="f7")
    f8 = fc[6].number_input("8/8 Floor", value=st.session_state.get("f8", 0.0), key="f8")

# Toggle for 7-Pick and 8-Pick options
show_78 = st.checkbox(
    "Show 7-Pick & 8-Pick options",
    key="show_78",
    help="Reveal payout inputs and leg odds for 7 and 8-pick slips."
)

if show_78:
    c6, c7 = st.columns(2)
    with c6:
        st.markdown("**7-Pick**")
        col_a, col_b, col_c = st.columns(3)
        p7 = col_a.number_input("7/7", value=st.session_state.get("p7", 0.0), key="p7")
        p7_i = col_b.number_input("6/7", value=st.session_state.get("p7_i", 0.0), key="p7_i")
        p7_i2 = col_c.number_input("5/7", value=st.session_state.get("p7_i2", 0.0), key="p7_i2")
    with c7:
        st.markdown("**8-Pick**")
        col_a, col_b, col_c = st.columns(3)
        p8 = col_a.number_input("8/8", value=st.session_state.get("p8", 0.0), key="p8")
        p8_i = col_b.number_input("7/8", value=st.session_state.get("p8_i", 0.0), key="p8_i")
        p8_i2 = col_c.number_input("6/8", value=st.session_state.get("p8_i2", 0.0), key="p8_i2")
else:
    p7 = st.session_state.get("p7", 0.0)
    p7_i = st.session_state.get("p7_i", 0.0)
    p7_i2 = st.session_state.get("p7_i2", 0.0)
    p8 = st.session_state.get("p8", 0.0)
    p8_i = st.session_state.get("p8_i", 0.0)
    p8_i2 = st.session_state.get("p8_i2", 0.0)

st.header("2. Play Odds (Win Probability)")
_n_odds = 8 if show_78 else 6
odds_cols = st.columns(_n_odds)
probs = []
for i, col in enumerate(odds_cols):
    val = col.text_input(f"Leg {i+1} Odds", value="-110", key=f"l{i}")
    prob = american_to_prob(val)
    probs.append(prob)
    if not use_std_leg_mults and _ud_format:
        col.caption(f"{prob*100:.1f}% | {leg_inputs[i]:.2f}x → x{leg_mults[i]:.3f}")
    elif not use_std_leg_mults:
        col.caption(f"{prob*100:.1f}% | x{leg_mults[i]}")
    else:
        col.caption(f"{prob*100:.1f}%")

st.markdown("---")
sc1, sc2, _ = st.columns([1, 1, 2])
scale_actual = sc1.number_input(
    "Actual Payout on Slip",
    value=None,
    min_value=0.0,
    placeholder="e.g. 23",
    help="The top payout shown on your actual slip. Leave blank to use preset values as-is."
)
scale_base = sc2.number_input(
    "Preset Base Payout",
    value=None,
    min_value=0.0,
    placeholder="e.g. 25",
    help="The top payout from the preset (the value in the payout box above). All multipliers are scaled by Actual ÷ Base."
)

if scale_actual and scale_base and scale_base > 0:
    payout_scale = scale_actual / scale_base
    st.caption(f"Scaling all payouts by {scale_actual}/{scale_base} = {payout_scale:.4f}x")
else:
    payout_scale = 1.0

if st.button("Calculate EV & Stakes", type="primary"):
    results = []

    # Define the payout structures for each slip size based on inputs
    # Format: (N, {num_wins: avg payout mult}, {num_wins: guaranteed floor mult})
    # Top tier: preset value = 30-day average payout; floor input = guaranteed
    # minimum. Intermediate tiers are floors in BOTH dicts (no average data is
    # published for them); their overage is estimated in build_tier_components.
    # The slip-scale s applies to floors and averages alike, so the overage
    # scales proportionally with the displayed slip multiplier.
    s = payout_scale
    slip_configs = [
        (2, {2: p2 * s}, {2: f2 * s}),
        (3, {3: p3 * s}, {3: f3 * s}),
        (4, {4: p4 * s, 3: p4_i * s}, {4: f4 * s, 3: p4_i * s}),
        (5, {5: p5 * s, 4: p5_i * s, 3: p5_i2 * s}, {5: f5 * s, 4: p5_i * s, 3: p5_i2 * s}),
        (6, {6: p6 * s, 5: p6_i * s, 4: p6_i2 * s}, {6: f6 * s, 5: p6_i * s, 4: p6_i2 * s}),
    ]
    if show_78:
        slip_configs.append((7, {7: p7 * s, 6: p7_i * s, 5: p7_i2 * s},
                                {7: f7 * s, 6: p7_i * s, 5: p7_i2 * s}))
        slip_configs.append((8, {8: p8 * s, 7: p8_i * s, 6: p8_i2 * s},
                                {8: f8 * s, 7: p8_i * s, 6: p8_i2 * s}))

    for n, avg_structure, floor_structure in slip_configs:
        current_probs = probs[:n]
        current_leg_mults = leg_mults[:n]

        tier_comps = build_tier_components(
            avg_structure, floor_structure, n,
            estimate_lower_tier_overage=est_lower_overage)
        chalk = chalk_overage_factor(current_probs, current_leg_mults, beta=chalk_beta)
        has_overage = any(ov > 0 for (_fl, ov) in tier_comps.values())

        def _outcomes(cap_amount, cap_stake):
            return calculate_complex_outcomes(
                current_probs,
                current_leg_mults,
                tier_comps,
                boost_mult,
                max_boost_amount=cap_amount,
                stake=cap_stake,
                boost_on_gross=boost_on_gross,
                sweat_free_fraction=sweat_free_fraction,
                stake_back_on_win=stake_back_on_win,
                refund_partial_wins=refund_partial_wins,
                chalk_factor=chalk,
            )

        _cap = (max_stake_small if n <= 3 else max_stake_large) if use_tiered_stakes else max_stake_input

        def _stake_from(outc):
            f = solve_general_kelly(outc)
            stk = bankroll * f * kelly_fraction
            return min(stk, _cap) if _cap > 0 else stk

        # First pass: no boost cap (cap depends on stake, stake on outcomes)
        outcomes = _outcomes(0.0, 1.0)
        used_stake = _stake_from(outcomes)

        # With a boost cap, iterate outcomes<->stake to a fixed point: the cap
        # per dollar depends on the stake, and the Kelly stake depends on the
        # capped outcomes. Converges in a couple of iterations.
        if max_boost_dollars > 0 and used_stake > 0:
            for _ in range(8):
                outcomes = _outcomes(max_boost_dollars, used_stake)
                new_stake = _stake_from(outcomes)
                if abs(new_stake - used_stake) < 0.01:
                    used_stake = new_stake
                    break
                used_stake = new_stake
            if used_stake > 0:
                outcomes = _outcomes(max_boost_dollars, used_stake)
            else:
                outcomes = _outcomes(0.0, 1.0)

        # Calculate Stats from (potentially capped) outcomes
        ev_decimal = sum(p * n_out for p, n_out in outcomes)

        # Win Prob (Probability of winning ANY money, i.e. net_outcome > -1)
        win_prob_any = sum(p for p, n_out in outcomes if n_out > -1.0)

        used_fraction = used_stake / bankroll if bankroll > 0 else 0
        eg_bps = calculate_expected_growth(outcomes, used_fraction)

        # Compute payout details per win tier for display
        leg_mult_product = 1.0
        for m in current_leg_mults:
            leg_mult_product *= m
        payout_details = compute_payout_details(
            tier_comps, n, boost_mult, boost_on_gross,
            max_boost_dollars, used_stake, leg_mult_product,
            chalk_factor=chalk,
            sweat_free_fraction=sweat_free_fraction,
            stake_back_on_win=stake_back_on_win,
            refund_partial_wins=refund_partial_wins
        )

        results.append({
            "Size": f"{n}-Pick",
            "EV": ev_decimal,
            "Any Win %": win_prob_any,
            "Stake": used_stake,
            "EG": eg_bps,
            "Details": payout_details,
            "Chalk": chalk,
            "HasOverage": has_overage,
        })

    # --- DISPLAY RESULTS ---
    any_overage = any(r['HasOverage'] for r in results)

    if use_tiered_stakes and (max_stake_small > 0 or max_stake_large > 0):
        _large_label = "4-8 picks" if show_78 else "4-6 picks"
        st.info(f"Stakes capped by slip size — 2-3 picks: ${max_stake_small:.2f} | {_large_label}: ${max_stake_large:.2f}")
    elif max_stake_input > 0:
        st.info(f"Stakes capped at maximum: ${max_stake_input:.2f}")

    if any_overage:
        _chalk_note = ", ".join(f"{r['Size']}: {r['Chalk']:.2f}x" for r in results if r['HasOverage'])
        st.info(f"Parimutuel overage active — payouts = guaranteed floor + overage. "
                f"Boosts apply to the floor only. Chalk factor on all-correct overage — {_chalk_note}")

    if sweat_free_enabled:
        if stake_back_on_win:
            if refund_partial_wins:
                st.success(f"Stake Back Win or Lose: {sweat_free_fraction:.0%} of stake returned on losses and full wins; partial wins topped up to breakeven.")
            else:
                st.success(f"Stake Back Win or Lose: {sweat_free_fraction:.0%} of stake returned on losses and full wins; partial wins get no top-up and can still lose stake.")
        else:
            if refund_partial_wins:
                st.success(f"Refund on Loss: Complete losses return {sweat_free_fraction:.0%} of stake; partial win/loss tiers topped up toward breakeven.")
            else:
                st.success(f"Refund on Loss: Complete losses return {sweat_free_fraction:.0%} of stake.")

    # Metrics Row
    res_cols = st.columns(len(results))
    for i, res in enumerate(results):
        res_cols[i].metric(
            label=res['Size'],
            value=f"{res['EV']*100:.1f}% EV",
            delta=f"{res['EG']:.1f} bps",
            help=f"Stake: ${res['Stake']:.2f}"
        )

    # Detailed Summary Table
    table_data = []
    for res in results:
        top_detail = res['Details'][0] if res['Details'] else None
        row = {
            "Slip Size": res['Size'],
            "EV %": f"{res['EV']*100:.2f}%",
            "Exp. Growth (bps)": f"{res['EG']:.2f}",
            "Rec. Stake": f"${res['Stake']:.2f}",
            "Hit Rate (Any Prize)": f"{res['Any Win %']*100:.1f}%",
        }
        if any_overage:
            row["Chalk ×"] = f"{res['Chalk']:.2f}" if res['HasOverage'] else "—"
        if top_detail and res['Stake'] > 0:
            row["Top Prize"] = f"${top_detail['prize_dollars']:.2f}"
            row["Top Profit"] = f"${top_detail['profit_dollars']:.2f}"
        table_data.append(row)
    st.table(table_data)

    # Payout Breakdown
    has_boost = boost_mult != 1.0
    any_capped = any(d['capped'] for res in results for d in res['Details'])
    has_any_details = any(len(res['Details']) > 0 for res in results)

    if has_any_details:
        st.subheader("Payout Breakdown (Winning Tiers)")
        if not use_std_leg_mults:
            _std_leg_label = f"standard ({_ud_base:.2f}x) leg prices" if _ud_format else "standard (1.0x) leg multipliers"
            st.caption(f"ℹ️ Payouts shown assume {_std_leg_label}. "
                       "Actual payouts vary based on which specific legs win.")
        if any_overage:
            st.caption("Floor = guaranteed minimum. Overage = estimated parimutuel extra "
                       "(chalk-adjusted on the all-correct tier; intensity-apportioned on "
                       "lower tiers). Boosts apply to the floor only — never the overage.")
        if has_boost and not boost_on_gross:
            st.caption("Boost mode: Net — boost applies to the profit portion of the guaranteed floor only.")
        elif has_boost:
            st.caption("Boost mode: Gross — boost applies to the full guaranteed floor.")

        breakdown_data = []
        for res in results:
            for detail in res['Details']:
                row = {
                    "Slip": res['Size'],
                    "Tier": detail['tier'],
                    "Floor": f"{detail['base_mult']:.2f}x",
                }
                if any_overage:
                    row["Overage (est.)"] = f"{detail['overage_mult']:.2f}x"
                    row["Avg Total"] = f"{detail['avg_mult']:.2f}x"
                if has_boost:
                    row["Boosted Payout"] = f"{detail['boosted_mult']:.2f}x"
                    if any_capped:
                        row["Eff. Payout"] = f"{detail['effective_mult']:.2f}x"
                        row["Capped?"] = "YES" if detail['capped'] else "No"
                if res['Stake'] > 0:
                    row["Prize ($)"] = f"${detail['prize_dollars']:.2f}"
                    row["Profit ($)"] = f"${detail['profit_dollars']:.2f}"
                    if has_boost:
                        row["Boost Value ($)"] = f"${detail['boost_value_dollars']:.2f}"
                breakdown_data.append(row)

        st.table(breakdown_data)
