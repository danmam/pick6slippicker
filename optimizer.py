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

def calculate_complex_outcomes(probs, leg_multipliers, payout_structure, global_boost, max_boost_amount=0.0, stake=1.0, boost_on_gross=True, sweat_free=False):
    """
    Generates all 2^N scenarios to accurately calculate EV with specific leg multipliers.

    Args:
        probs: List of win probabilities for each leg.
        leg_multipliers: List of payout multipliers for each leg (if it wins).
        payout_structure: Dict mapping number of wins (k) to Base Payout Multiplier.
                          e.g., {6: 25.0, 5: 2.0, 4: 0.4}
        global_boost: Overall boost multiplier applied to the final payout.
        max_boost_amount: Maximum dollar amount the boost can add to payout (0 = unlimited).
        stake: The stake amount used to calculate the dollar cap on boost.
        boost_on_gross: If True, boost multiplies the full payout. If False, boost
                        multiplies only the net profit (payout - 1).
        sweat_free: If True, outcomes not defined in payout_structure result in a 
                    1.0x payout (Refund) instead of 0.0x (Loss).

    Returns:
        List of (probability, net_outcome) tuples.
    """
    num_legs = len(probs)
    outcomes = []

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

        # Check if this outcome is defined in the payout structure
        if wins in payout_structure:
            base_payout = payout_structure[wins]
            
            # Normal calculation for defined payouts
            if base_payout > 0:
                # Calculate unboosted payout (what it would be with global_boost = 1.0)
                unboosted_payout = base_payout * scenario_leg_mult_product
                # Calculate fully boosted payout
                if boost_on_gross:
                    boosted_payout = unboosted_payout * global_boost
                else:
                    # Boost applies only to net profit (payout minus returned stake)
                    boosted_payout = 1.0 + (unboosted_payout - 1.0) * global_boost

                # Apply max boost cap if specified
                if max_boost_amount > 0 and stake > 0:
                    # The boost amount in multiplier terms (per $1 stake)
                    boost_amount_per_dollar = boosted_payout - unboosted_payout
                    # The max boost in multiplier terms (per $1 stake)
                    max_boost_per_dollar = max_boost_amount / stake

                    if boost_amount_per_dollar > max_boost_per_dollar:
                        # Cap the boost
                        gross_payout = unboosted_payout + max_boost_per_dollar
                    else:
                        gross_payout = boosted_payout
                else:
                    gross_payout = boosted_payout

                net_outcome = gross_payout - 1.0
            else:
                # Explicit 0.0 payout in structure (rare but possible)
                net_outcome = -1.0
                
        else:
            # Outcome NOT defined in structure (typically a Loss)
            if sweat_free:
                # Sweat Free: Refund the stake. 
                # Assumes refund is exactly 1.0x (no boosts applied to refund).
                gross_payout = 1.0
                net_outcome = 0.0
            else:
                # Standard: Loss
                gross_payout = 0.0
                net_outcome = -1.0

        outcomes.append((scenario_prob, net_outcome))

    return outcomes

def compute_payout_details(payout_structure, n_legs, global_boost, boost_on_gross, max_boost_amount, stake):
    """
    Compute payout details per win tier for display purposes.
    Assumes leg multipliers = 1.0 (the standard case).
    """
    details = []
    for wins in sorted(payout_structure.keys(), reverse=True):
        base = payout_structure[wins]
        if base <= 0:
            continue

        unboosted = base
        if boost_on_gross:
            boosted = unboosted * global_boost
        else:
            boosted = 1.0 + (unboosted - 1.0) * global_boost

        boost_delta = boosted - unboosted
        capped = False
        effective = boosted
        if max_boost_amount > 0 and stake > 0:
            max_delta_per_dollar = max_boost_amount / stake
            if boost_delta > max_delta_per_dollar:
                effective = unboosted + max_delta_per_dollar
                capped = True

        tier_label = f"{wins}/{n_legs}"
        details.append({
            'tier': tier_label,
            'base_mult': base,
            'boosted_mult': boosted,
            'effective_mult': effective,
            'capped': capped,
            'prize_dollars': effective * stake if stake > 0 else 0,
            'profit_dollars': (effective - 1) * stake if stake > 0 else 0,
            'boost_value_dollars': (effective - unboosted) * stake if stake > 0 else 0,
        })
    return details

# --- PRESETS DATA ---
PRESETS = {
    "Custom": None,
    "DK Pick6 NBA": {
        "p2": 3.1, 
        "p3": 6.2, 
        "p4": 11.1, "p4_i": 0.0, 
        "p5": 15.1, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 32.4, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 NBA Promo": {
        "p2": 3.0,
        "p3": 6.2,
        "p4": 10.29, "p4_i": 0.0,
        "p5": 13.26, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 31.51, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 CBB Promo": {
        "p2": 2.7, 
        "p3": 5.78, 
        "p4": 9.47, "p4_i": 0.0, 
        "p5": 18.33, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 37, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 UFC": {
        "p2": 3.5, 
        "p3": 6.5, 
        "p4": 10.4, "p4_i": 0.0, 
        "p5": 14.3, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 40.7, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 NHL": {
        "p2": 3.4,
        "p3": 7.5,
        "p4": 12, "p4_i": 0.0,
        "p5": 16.4, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 32.4, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 CBB": {
        "p2": 3.1,
        "p3": 5.9,
        "p4": 12.2, "p4_i": 0.0,
        "p5": 20.6, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 37, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 PGA": {
        "p2": 3.3,
        "p3": 5.7,
        "p4": 9.7, "p4_i": 0.0,
        "p5": 12.3, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 21.0, "p6_i": 1.5, "p6_i2": 0.0
    },
    "Ownersbox": {
        "p2": 3, 
        "p3": 6.0, 
        "p4": 6, "p4_i": 1.5, 
        "p5": 10, "p5_i": 2.5, "p5_i2": 0.0,
        "p6": 25, "p6_i": 3.0, "p6_i2": 0.0
    },
    "Prizepicks": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 6.0, "p4_i": 1.5,
        "p5": 10.0, "p5_i": 2.0, "p5_i2": 0.4,
        "p6": 25.0, "p6_i": 2.0, "p6_i2": 0.4
    },
    "RTSports (Mulligan)": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 10.0, "p4_i": 0.0,
        "p5": 12.0, "p5_i": 2.0, "p5_i2": 0.0,
        "p6": 25.0, "p6_i": 2.5, "p6_i2": 0.0
    },
    "RTSports (Power)": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 10.0, "p4_i": 0.0,
        "p5": 12.0, "p5_i": 2.0, "p5_i2": 0.0,
        "p6": 40.0, "p6_i": 0.0, "p6_i2": 0.0
    },
    "Betr Nukes": {
        "p2": 6.0,
        "p3": 10.0,
        "p4": 20.0, "p4_i": 0.0
    },
    "Betr Picks": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 6.0, "p4_i": 1.5,
        "p5": 10.0, "p5_i": 2.0, "p5_i2": 0.5,
        "p6": 20.0, "p6_i": 1.5, "p6_i2": 1.0
    },
    "Drafters": {
        "p2": 3.0,
        "p3": 6.0,
        "p4": 4.0, "p4_i": 2.0,
        "p5": 20.0, "p5_i": 0.0, "p5_i2": 0.0,
        "p6": 10.0, "p6_i": 2.5, "p6_i2": 1.5
    }
}

# --- STREAMLIT LAYOUT ---
st.set_page_config(page_title="Pick6/DFS Optimizer", layout="wide")
st.title("Pick6 & DFS Props EV Optimizer")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
bankroll = st.sidebar.number_input("Bankroll ($)", value=8000.0)
kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.0, 1.0, 0.25)
max_stake_input = st.sidebar.number_input("Max Stake ($)", value=0.0, help="Cap the recommended stake. If Kelly suggests a smaller stake, it will use Kelly. If Kelly suggests more, it caps at this value.")

st.sidebar.markdown("---")
st.sidebar.subheader("Promo Settings")
sweat_free = st.sidebar.checkbox(
    "Sweat Free Mode (Refund on Loss)",
    value=False,
    help="If checked, any outcome not defined in the payout structure (a loss) returns 1.0x (full stake refund) instead of 0.0x."
)
boost_mult = st.sidebar.number_input("Global Payout Boost (e.g. 1.1 for 10%)", value=1.0, step=0.05)
max_boost_dollars = st.sidebar.number_input("Max Boost $ (0 = unlimited)", value=0.0, step=5.0, help="Cap the boost amount. The payout increase from the boost cannot exceed this dollar amount.")
boost_on_gross = st.sidebar.checkbox(
    "Boost on gross payout",
    value=True,
    help="Checked: boost multiplies the full payout (e.g. 50% boost on 6x → 9x, +800). "
         "Unchecked: boost multiplies only net profit (e.g. 50% boost on 6x → 1 + 1.5×5 = 8.5x, +750)."
)

st.sidebar.markdown("---")
use_std_leg_mults = st.sidebar.checkbox("All leg multipliers 1.0x?", value=True)

leg_mults = [1.0] * 6
if not use_std_leg_mults:
    st.sidebar.subheader("Individual Leg Multipliers")
    lm_cols = st.sidebar.columns(3)
    for i in range(6):
        leg_mults[i] = lm_cols[i%3].number_input(f"Leg {i+1} x", value=1.0, step=0.01, format="%.2f")

st.sidebar.markdown("---")
selected_preset = st.sidebar.selectbox("Load Payout Preset", list(PRESETS.keys()))

if selected_preset != "Custom":
    data = PRESETS[selected_preset]
    for key, val in data.items():
        st.session_state[key] = val

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

st.header("2. Play Odds (Win Probability)")
odds_cols = st.columns(6)
probs = []
for i, col in enumerate(odds_cols):
    val = col.text_input(f"Leg {i+1} Odds", value="-110", key=f"l{i}")
    prob = american_to_prob(val)
    probs.append(prob)
    if not use_std_leg_mults:
        col.caption(f"{prob*100:.1f}% | x{leg_mults[i]}")
    else:
        col.caption(f"{prob*100:.1f}%")

if st.button("Calculate EV & Stakes", type="primary"):
    results = []
    
    # Define the payout structures for each slip size based on inputs
    # Format: {num_wins: multiplier}
    slip_configs = [
        # (N, payout_dict)
        (2, {2: p2}),
        (3, {3: p3}),
        (4, {4: p4, 3: p4_i}),
        (5, {5: p5, 4: p5_i, 3: p5_i2}),
        (6, {6: p6, 5: p6_i, 4: p6_i2}),
    ]

    for n, payout_structure in slip_configs:
        current_probs = probs[:n]
        current_leg_mults = leg_mults[:n]

        # First pass: Calculate outcomes without cap to determine stake
        outcomes_uncapped = calculate_complex_outcomes(
            current_probs,
            current_leg_mults,
            payout_structure,
            boost_mult,
            max_boost_amount=0.0,
            stake=1.0,
            boost_on_gross=boost_on_gross,
            sweat_free=sweat_free
        )

        # Determine stake from uncapped outcomes
        f_opt_uncapped = solve_general_kelly(outcomes_uncapped)
        kelly_stake = bankroll * f_opt_uncapped * kelly_fraction
        # Apply max stake cap if specified
        if max_stake_input > 0:
            used_stake = min(kelly_stake, max_stake_input)
        else:
            used_stake = kelly_stake

        # Second pass: Recalculate outcomes with cap applied using the determined stake
        if max_boost_dollars > 0 and used_stake > 0:
            outcomes = calculate_complex_outcomes(
                current_probs,
                current_leg_mults,
                payout_structure,
                boost_mult,
                max_boost_amount=max_boost_dollars,
                stake=used_stake,
                boost_on_gross=boost_on_gross,
                sweat_free=sweat_free
            )
        else:
            outcomes = outcomes_uncapped

        # Calculate Stats from (potentially capped) outcomes
        ev_decimal = sum(p * n_out for p, n_out in outcomes)

        # Win Prob (Probability of winning ANY money, i.e. net_outcome > -1)
        win_prob_any = sum(p for p, n_out in outcomes if n_out > -1.0)

        # Kelly & Growth from capped outcomes
        f_opt = solve_general_kelly(outcomes)
        kelly_stake_capped = bankroll * f_opt * kelly_fraction

        # Apply max stake cap if specified
        if max_stake_input > 0:
            used_stake = min(kelly_stake_capped, max_stake_input)
        else:
            used_stake = kelly_stake_capped

        used_fraction = used_stake / bankroll if bankroll > 0 else 0
        eg_bps = calculate_expected_growth(outcomes, used_fraction)

        # Compute payout details per win tier for display
        payout_details = compute_payout_details(
            payout_structure, n, boost_mult, boost_on_gross,
            max_boost_dollars, used_stake
        )

        results.append({
            "Size": f"{n}-Pick",
            "EV": ev_decimal,
            "Any Win %": win_prob_any,
            "Stake": used_stake,
            "EG": eg_bps,
            "Details": payout_details
        })

    # --- DISPLAY RESULTS ---
    if max_stake_input > 0:
        st.info(f"Stakes capped at maximum: ${max_stake_input:.2f}")

    if sweat_free:
        st.success("Sweat Free Mode Active: Losses (unspecified payouts) are treated as Refunds (1.0x).")

    # Metrics Row
    res_cols = st.columns(5)
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
            st.caption("ℹ️ Payouts shown assume standard (1.0x) leg multipliers. "
                       "Actual payouts vary based on which specific legs win.")
        if has_boost and not boost_on_gross:
            st.caption("Boost mode: Net — boost applies to profit portion only (payout − stake).")
        elif has_boost:
            st.caption("Boost mode: Gross — boost applies to the full payout.")

        breakdown_data = []
        for res in results:
            for detail in res['Details']:
                row = {
                    "Slip": res['Size'],
                    "Tier": detail['tier'],
                    "Base Payout": f"{detail['base_mult']:.2f}x",
                }
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
