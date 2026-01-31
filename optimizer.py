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

def calculate_complex_outcomes(probs, leg_multipliers, payout_structure, global_boost):
    """
    Generates all 2^N scenarios to accurately calculate EV with specific leg multipliers.
    
    Args:
        probs: List of win probabilities for each leg.
        leg_multipliers: List of payout multipliers for each leg (if it wins).
        payout_structure: Dict mapping number of wins (k) to Base Payout Multiplier.
                          e.g., {6: 25.0, 5: 2.0, 4: 0.4}
        global_boost: Overall boost multiplier applied to the final payout.
        
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
        
        # Determine Base Payout based on number of wins
        base_payout = payout_structure.get(wins, 0.0)
        
        if base_payout > 0:
            # Gross Payout = Base * (Product of Winning Leg Mults) * Global Boost
            gross_payout = base_payout * scenario_leg_mult_product * global_boost
            net_outcome = gross_payout - 1.0
        else:
            net_outcome = -1.0 # Loss of stake
            
        outcomes.append((scenario_prob, net_outcome))
        
    return outcomes

# --- PRESETS DATA ---
PRESETS = {
    "Custom": None,
    "DK Pick6 NBA": {
        "p2": 3.3, 
        "p3": 6.5, 
        "p4": 12.8, "p4_i": 0.0, 
        "p5": 17.4, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 35.5, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 NBA Promo": {
        "p2": 3.0,
        "p3": 6.5,
        "p4": 10.83, "p4_i": 0.0,
        "p5": 15.33, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 35.5, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 NFL Promo": {
        "p2": 3.0, 
        "p3": 6.5, 
        "p4": 10.89, "p4_i": 0.0, 
        "p5": 14.835, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 36.9, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 UFC": {
        "p2": 3.2, 
        "p3": 5.8, 
        "p4": 10.4, "p4_i": 0.0, 
        "p5": 14.3, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 34.8, "p6_i": 1.5, "p6_i2": 0.0
    },
    "DK Pick6 NFL": {
        "p2": 3.4, 
        "p3": 6.5, 
        "p4": 12.4, "p4_i": 0.0, 
        "p5": 17.7, "p5_i": 1.0, "p5_i2": 0.0,
        "p6": 37.6, "p6_i": 1.5, "p6_i2": 0.0
    },
    "Ownersbox": {
        "p2": 3.0, 
        "p3": 6.0, 
        "p4": 6.0, "p4_i": 1.5, 
        "p5": 10.0, "p5_i": 2.5, "p5_i2": 0.0,
        "p6": 25.0, "p6_i": 3.0, "p6_i2": 0.0
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
manual_stake_input = st.sidebar.number_input("Manual Stake Override ($)", value=0.0, help="Calculates growth based on this specific bet size.")
boost_mult = st.sidebar.number_input("Global Payout Boost (e.g. 1.1 for 10%)", value=1.0, step=0.05)

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
        
        # 1. Generate all outcomes (Prob, Net Payout)
        outcomes = calculate_complex_outcomes(
            current_probs, 
            current_leg_mults, 
            payout_structure, 
            boost_mult
        )
        
        # 2. Calculate Stats
        ev_decimal = sum(p * n_out for p, n_out in outcomes)
        
        # Win Prob (Probability of winning ANY money, i.e. net_outcome > -1)
        win_prob_any = sum(p for p, n_out in outcomes if n_out > -1.0)
        
        # 3. Kelly & Growth
        f_opt = solve_general_kelly(outcomes)
        
        if manual_stake_input > 0:
            used_stake = manual_stake_input
            used_fraction = used_stake / bankroll if bankroll > 0 else 0
        else:
            used_fraction = f_opt * kelly_fraction
            used_stake = bankroll * used_fraction
            
        eg_bps = calculate_expected_growth(outcomes, used_fraction)
        
        results.append({
            "Size": f"{n}-Pick",
            "EV": ev_decimal,
            "Any Win %": win_prob_any,
            "Stake": used_stake,
            "EG": eg_bps
        })

    # --- DISPLAY RESULTS ---
    if manual_stake_input > 0:
        st.info(f"Showing Expected Growth (EG) for fixed stake: ${manual_stake_input:.2f}")

    # Metrics Row
    res_cols = st.columns(5)
    for i, res in enumerate(results):
        res_cols[i].metric(
            label=res['Size'], 
            value=f"{res['EV']*100:.1f}% EV", 
            delta=f"{res['EG']:.1f} bps",
            help=f"Stake: ${res['Stake']:.2f}"
        )

    # Detailed Table
    table_data = []
    for res in results:
        table_data.append({
            "Slip Size": res['Size'],
            "EV %": f"{res['EV']*100:.2f}%",
            "Exp. Growth (bps)": f"{res['EG']:.2f}",
            "Rec. Stake": f"${res['Stake']:.2f}",
            "Hit Rate (Any Prize)": f"{res['Any Win %']*100:.1f}%"
        })
    st.table(table_data)
