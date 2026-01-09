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

def calculate_exact_wins_prob(probs, total_legs, required_wins):
    indices = range(total_legs)
    winning_combinations = itertools.combinations(indices, required_wins)
    total_prob = 0.0
    for win_indices in winning_combinations:
        scenario_prob = 1.0
        win_set = set(win_indices)
        for i in range(total_legs):
            if i in win_set:
                scenario_prob *= probs[i]
            else:
                scenario_prob *= (1 - probs[i])
        total_prob += scenario_prob
    return total_prob

def solve_general_kelly(outcomes):
    ev = sum(p * b for p, b in outcomes)
    if ev <= 0:
        return 0.0
    def kelly_derivative(f):
        return sum((p * b) / (1 + f * b) for p, b in outcomes)
    low, high = 0.0, 0.9999
    for _ in range(50):
        mid = (low + high) / 2
        if kelly_derivative(mid) > 0:
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
        # ln(1 + f * b) where b is net odds
        growth_sum += prob * math.log(1 + stake_fraction * net_odds)
    return growth_sum * 10000

# --- PRESETS DATA ---
PRESETS = {
    "Custom": None,
    "DK Pick6 NBA": {"p2": 3.3, "p3": 6.5, "p4": 12.8, "p4_i": 0.0, "p5": 17.4, "p5_i": 1.0, "p6": 35.5, "p6_i": 1.5},
    "DK Pick6 NFL": {"p2": 3.4, "p3": 6.5, "p4": 12.4, "p4_i": 0.0, "p5": 17.7, "p5_i": 1.0, "p6": 37.6, "p6_i": 1.5},
    "Ownersbox": {"p2": 3.0, "p3": 6.0, "p4": 6.0, "p4_i": 1.5, "p5": 10.0, "p5_i": 2.5, "p6": 25.0, "p6_i": 3.0},
    "DK Pick6 NBA Promo": {"p2": 3.0, "p3": 6.5, "p4": 10.83, "p4_i": 0.0, "p5": 15.33, "p5_i": 1.0, "p6": 35.5, "p6_i": 1.5},
    "DK Pick6 NFL Promo": {"p2": 3.0, "p3": 6.5, "p4": 10.89, "p4_i": 0.0, "p5": 14.835, "p5_i": 1.0, "p6": 36.9, "p6_i": 1.5},
}

# --- STREAMLIT LAYOUT ---
st.set_page_config(page_title="DK Pick6 Optimizer", layout="wide")
st.title("DraftKings Pick6 EV & Expected Growth Optimizer")

st.sidebar.header("Bankroll Management")
bankroll = st.sidebar.number_input("Current Bankroll ($)", value=8000.0)
kelly_fraction = st.sidebar.slider("Kelly Fraction (e.g. 0.25 for Quarter)", 0.0, 1.0, 0.25)
# New Input: Manual Stake Override
manual_stake_input = st.sidebar.number_input("Manual Stake Override ($)", value=0.0, help="If > 0, EG is calculated based on this stake instead of Kelly.")
boost_mult = st.sidebar.number_input("Global Payout Multiplier", value=1.0, step=0.1)

selected_preset = st.sidebar.selectbox("Load Payout Preset", list(PRESETS.keys()))

if selected_preset != "Custom":
    data = PRESETS[selected_preset]
    for key, val in data.items():
        st.session_state[key] = val

st.header("1. Payout Structure")
col1, col2, col3, col4, col5 = st.columns(5)
p2 = col1.number_input("2-Pick", value=st.session_state.get("p2", 3.0), key="p2")
p3 = col2.number_input("3-Pick", value=st.session_state.get("p3", 6.0), key="p3")
p4 = col3.number_input("4-Pick", value=st.session_state.get("p4", 10.0), key="p4")
p5 = col4.number_input("5-Pick", value=st.session_state.get("p5", 20.0), key="p5")
p6 = col5.number_input("6-Pick", value=st.session_state.get("p6", 40.0), key="p6")

icol1, icol2, icol3 = st.columns(3)
p4_i = icol1.number_input("4-Pick Insurance", value=st.session_state.get("p4_i", 0.0), key="p4_i")
p5_i = icol2.number_input("5-Pick Insurance", value=st.session_state.get("p5_i", 1.0), key="p5_i")
p6_i = icol3.number_input("6-Pick Insurance", value=st.session_state.get("p6_i", 1.5), key="p6_i")

st.header("2. Play Odds")
odds_cols = st.columns(6)
probs = []
for i, col in enumerate(odds_cols):
    val = col.text_input(f"Leg {i+1}", value="-110", key=f"l{i}")
    prob = american_to_prob(val)
    probs.append(prob)
    col.caption(f"{prob*100:.1f}% Win")

if st.button("Calculate EV & Stakes"):
    results = []
    configs = [(2, p2, 0), (3, p3, 0), (4, p4, p4_i), (5, p5, p5_i), (6, p6, p6_i)]

    for n, main_p, ins_p in configs:
        p_perfect = 1.0
        for p in probs[:n]:
            p_perfect *= p
        
        gross_perfect = main_p * boost_mult
        
        if ins_p > 0:
            p_ins = calculate_exact_wins_prob(probs[:n], n, n-1)
            p_loss = 1.0 - p_perfect - p_ins
            gross_ins = ins_p * boost_mult
            ev = (p_perfect * gross_perfect) + (p_ins * gross_ins) - 1
            outcomes = [(p_perfect, gross_perfect - 1), (p_ins, gross_ins - 1), (p_loss, -1.0)]
        else:
            p_loss = 1.0 - p_perfect
            ev = (p_perfect * gross_perfect) - 1
            outcomes = [(p_perfect, gross_perfect - 1), (p_loss, -1.0)]

        # 1. Calculate Optimal Kelly Fraction (theoretical)
        f_full = solve_general_kelly(outcomes)
        
        # 2. Determine Actual Stake and Fraction used for EG calculation
        if manual_stake_input > 0:
            # Case: User manually input a stake size
            actual_stake = manual_stake_input
            # Fraction needed for EG formula: Stake / Bankroll
            used_fraction = actual_stake / bankroll if bankroll > 0 else 0
        else:
            # Case: Use Quarter Kelly (or whatever slider is set to)
            used_fraction = f_full * kelly_fraction
            actual_stake = bankroll * used_fraction

        # 3. Calculate Expected Growth based on the fraction determined above
        eg_bps = calculate_expected_growth(outcomes, used_fraction)

        results.append({
            "Size": f"{n}-Pick",
            "EV": ev,
            "Win %": p_perfect,
            "Stake": actual_stake,
            "EG": eg_bps
        })

    st.header("Results")
    if manual_stake_input > 0:
        st.info(f"Showing Expected Growth (EG) for Manual Stake: ${manual_stake_input:.2f}")

    res_cols = st.columns(5)
    for i, res in enumerate(results):
        res_cols[i].metric(res['Size'], f"{res['EV']*100:.1f}% EV", f"${res['Stake']:.2f}")

    table_data = []
    for res in results:
        table_data.append({
            "Slip": res['Size'],
            "EV %": f"{res['EV']*100:.2f}%",
            "Expected Growth": f"{res['EG']:.2f} bps",
            "Stake": f"${res['Stake']:.2f}",
            "Win Prob": f"{res['Win %']*100:.1f}%"
        })
    st.table(table_data)
