import streamlit as st
import itertools
import math

# --- UTILITY FUNCTIONS ---
def american_to_prob(value):
    """Converts American odds or percentage to decimal probability."""
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
    """Calculates probability of getting EXACTLY required_wins correct."""
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
    """Solves for optimal Kelly fraction 'f' given mutually exclusive outcomes."""
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

# --- PRESETS DATA ---
PRESETS = {
    "Custom": None,
    "DK Pick6 NBA": {"p2": 3.3, "p3": 6.5, "p4": 12.8, "p4_i": 0.0, "p5": 17.4, "p5_i": 1.0, "p6": 35.5, "p6_i": 1.5},
    "DK Pick6 NFL": {"p2": 3.4, "p3": 6.5, "p4": 12.4, "p4_i": 0.0, "p5": 17.7, "p5_i": 1.0, "p6": 37.6, "p6_i": 1.5},
    "Ownersbox": {"p2": 3.0, "p3": 6.0, "p4": 6.0, "p4_i": 1.5, "p5": 10.0, "p5_i": 2.5, "p6": 25.0, "p6_i": 3.0},
    "DK Pick6 NBA Promo": {"p2": 3.0, "p3": 6.5, "p4": 10.83, "p4_i": 0.0, "p5": 15.33, "p5_i": 1.0, "p6": 35.5, "p6_i": 1.5},
    "DK Pick6 NFL Promo": {"p2": 3.0, "p3": 6.5, "p4": 10.89, "p4_i": 0.0, "p5": 14.835, "p5_i": 1.0, "p6": 36.9, "p6_i": 1.5},
}

# --- APP CONFIG ---
st.set_page_config(page_title="Prop Optimizer", layout="wide")
st.title("Sportsbook EV & Kelly Optimizer")

# --- SIDEBAR & PRESET LOGIC ---
st.sidebar.header("Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", value=8000.0)
kelly_fraction = 0.25 
boost_mult = st.sidebar.number_input("Payout Multiplier", value=1.0, step=0.1)

selected_preset = st.sidebar.selectbox("Load Payout Preset", list(PRESETS.keys()))

# Update session state if a preset is chosen
if selected_preset != "Custom":
    data = PRESETS[selected_preset]
    for key, val in data.items():
        st.session_state[key] = val

# --- PAYOUT INPUTS ---
st.header("1. Payout Structure (Gross Multiples)")

# We use session_state.get to allow the dropdown to "push" values into these boxes
col1, col2, col3, col4, col5 = st.columns(5)
payout_2 = col1.number_input("2-Pick", value=st.session_state.get("p2", 3.0), key="p2")
payout_3 = col2.number_input("3-Pick", value=st.session_state.get("p3", 6.0), key="p3")
payout_4 = col3.number_input("4-Pick", value=st.session_state.get("p4", 10.0), key="p4")
payout_5 = col4.number_input("5-Pick", value=st.session_state.get("p5", 20.0), key="p5")
payout_6 = col5.number_input("6-Pick", value=st.session_state.get("p6", 40.0), key="p6")

st.markdown("##### Insurance Payouts (1 Leg Wrong)")
icol1, icol2, icol3 = st.columns([1, 1, 3])
ins_4 = icol1.number_input("4-Pick Ins.", value=st.session_state.get("p4_i", 0.0), key="p4_i")
ins_5 = icol2.number_input("5-Pick Ins.", value=st.session_state.get("p5_i", 1.0), key="p5_i")
ins_6 = icol3.number_input("6-Pick Ins.", value=st.session_state.get("p6_i", 1.5), key="p6_i")

# --- ODDS INPUTS ---
st.header("2. Play Odds")
odds_cols = st.columns(6)
probs = []
for i, col in enumerate(odds_cols):
    val = col.text_input(f"Leg {i+1}", value="-110", key=f"leg_{i}")
    prob = american_to_prob(val)
    probs.append(prob)
    col.caption(f"{prob*100:.1f}%")

# --- CALCULATION ---
if st.button("Calculate EV & Stakes"):
    results = []
    
    # Generic logic for slips with insurance
    # configurations: (num_legs, main_payout, insurance_payout)
    configs = [
        (2, payout_2, 0),
        (3, payout_3, 0),
        (4, payout_4, ins_4),
        (5, payout_5, ins_5),
        (6, payout_6, ins_6)
    ]

    for n, main_p, ins_p in configs:
        p_perfect = math.prod(probs[:n])
        
        if ins_p > 0:
            p_ins = calculate_exact_wins_prob(probs[:n], n, n-1)
            p_loss = 1.0 - p_perfect - p_ins
            
            gross_perfect = main_p * boost_mult
            gross_ins = ins_p * boost_mult
            
            ev = (p_perfect * gross_perfect) + (p_ins * gross_ins) - 1
            outcomes = [
                (p_perfect, gross_perfect - 1),
                (p_ins, gross_ins - 1),
                (p_loss, -1.0)
            ]
        else:
            p_loss = 1.0 - p_perfect
            gross_perfect = main_p * boost_mult
            ev = (p_perfect * gross_perfect) - 1
            outcomes = [(p_perfect, gross_perfect - 1), (p_loss, -1.0)]

        f = solve_general_kelly(outcomes)
        results.append({
            "Size": f"{n}-Pick",
            "EV": ev,
            "Win %": p_perfect,
            "Stake": bankroll * (f * kelly_fraction)
        })

    # --- DISPLAY ---
    st.header("Results (Quarter Kelly)")
    
    # 1. Metric Cards for quick glance
    res_cols = st.columns(5)
    for i, res in enumerate(results):
        with res_cols[i]:
            st.metric(
                label=res['Size'], 
                value=f"{res['EV']*100:.2f}% EV", 
                delta=f"${res['Stake']:.2f} Stake"
            )

    # 2. Comparative Table
    st.subheader("Detailed Comparison")
    
    # Format the data for the table
    table_data = []
    for res in results:
        table_data.append({
            "Slip Size": res['Size'],
            "EV %": f"{res['EV']*100:.2f}%",
            "Win % (Perfect)": f"{res['Win %']*100:.2f}%",
            "Recommended Stake": f"${res['Stake']:.2f}",
            "Edge/Variance Ratio": round(res['EV'] / (res['Stake']/bankroll), 4) if res['Stake'] > 0 else 0
        })
    
    st.table(table_data)

    # 3. Final Recommendation Logic
    best_play = max(results, key=lambda x: x['EV'])
    if best_play['EV'] > 0:
        st.success(f"**Top Recommendation:** The {best_play['Size']} slip offers the highest EV at {best_play['EV']*100:.2f}%.")
    else:
        st.warning("No positive EV plays found with current odds/payouts.")
