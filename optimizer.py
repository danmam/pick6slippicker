import streamlit as st
import itertools
import math

def american_to_prob(value):
    """
    Converts American odds (e.g., -110, +150) or percentage input (e.g., 55) 
    into a decimal probability (0.0 - 1.0).
    """
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
    """
    Calculates the probability of getting EXACTLY 'required_wins' correct.
    """
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
    """
    Solves for the optimal Kelly fraction 'f' given a list of mutually exclusive outcomes.
    outcomes structure: [(probability, net_odds), (probability, net_odds), ...]
    
    Returns the fraction of bankroll to wager (0.0 to 1.0).
    """
    # 1. Check if Expected Value is positive. If not, Kelly is 0.
    ev = sum(p * b for p, b in outcomes)
    if ev <= 0:
        return 0.0

    # 2. Function representing the derivative of the growth rate
    # We want to find f where sum( (p * b) / (1 + f * b) ) = 0
    def kelly_derivative(f):
        return sum((p * b) / (1 + f * b) for p, b in outcomes)

    # 3. Binary Search for the root f in range [0, 0.9999]
    # We cap at roughly 0.99 to avoid dividing by zero if net_odds is -1
    low = 0.0
    high = 0.9999
    iterations = 50 
    
    # Check boundaries
    if kelly_derivative(low) <= 0: return 0.0
    
    # Simple binary search to find f where derivative is close to 0
    for _ in range(iterations):
        mid = (low + high) / 2
        val = kelly_derivative(mid)
        if val > 0:
            low = mid
        else:
            high = mid
            
    return low

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="DK Pick6 Optimizer + Full Kelly", layout="wide")

st.title("DraftKings Pick6 EV & Full Kelly Optimizer")
st.markdown("Calculates EV and precise Kelly stakes for complex insurance scenarios.")

# --- SIDEBAR: BANKROLL & GLOBAL SETTINGS ---
st.sidebar.header("Bankroll Management")
bankroll = st.sidebar.number_input("Current Bankroll ($)", value=8000.0, step=100.0)
kelly_fraction = 0.25 # Quarter Kelly

st.sidebar.header("Global Boost")
boost_mult = st.sidebar.number_input("Payout Multiplier (e.g. 1.3 for 30% boost)", value=1.0, step=0.1, format="%.2f")

# --- MAIN INPUTS: PAYOUT STRUCTURE ---
st.header("1. Payout Structure (Gross Multiples)")
st.info("Enter the payout for getting ALL legs correct.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    payout_2 = st.number_input("2-Pick Payout (x)", value=3.0)
with col2:
    payout_3 = st.number_input("3-Pick Payout (x)", value=6.0)
with col3:
    payout_4 = st.number_input("4-Pick Payout (x)", value=10.0)
with col4:
    payout_5 = st.number_input("5-Pick Payout (x)", value=20.0)
with col5:
    payout_6 = st.number_input("6-Pick Payout (x)", value=40.0)

# --- MAIN INPUTS: ODDS/PROBS ---
st.header("2. Play Odds (True Odds)")
st.markdown("Enter **American Odds** (e.g. -120) or **Probability %** (e.g. 54.5).")

odds_cols = st.columns(6)
probs = []
for i, col in enumerate(odds_cols):
    with col:
        val = st.text_input(f"Leg {i+1}", value="-110")
        prob = american_to_prob(val)
        probs.append(prob)
        st.caption(f"{prob*100:.1f}% Win Prob")

# --- CALCULATION ENGINE ---
st.divider()

if st.button("Calculate EV & Stakes"):
    results = []
    
    # --- 2 LEG SLIP (Simple Binary Kelly) ---
    p_win_2 = probs[0] * probs[1]
    gross_payout_2 = payout_2 * boost_mult
    ev_2 = (p_win_2 * gross_payout_2) - 1
    
    # Outcome: (Prob, Net_Odds)
    outcomes_2 = [
        (p_win_2, gross_payout_2 - 1),  # Win
        (1 - p_win_2, -1.0)             # Loss
    ]
    full_f_2 = solve_general_kelly(outcomes_2)
    stake_2 = bankroll * (full_f_2 * kelly_fraction)
    results.append({"Size": "2-Pick", "EV": ev_2, "Win %": p_win_2, "Stake": stake_2})

    # --- 3 LEG SLIP ---
    p_win_3 = probs[0] * probs[1] * probs[2]
    gross_payout_3 = payout_3 * boost_mult
    ev_3 = (p_win_3 * gross_payout_3) - 1
    
    outcomes_3 = [
        (p_win_3, gross_payout_3 - 1),
        (1 - p_win_3, -1.0)
    ]
    full_f_3 = solve_general_kelly(outcomes_3)
    stake_3 = bankroll * (full_f_3 * kelly_fraction)
    results.append({"Size": "3-Pick", "EV": ev_3, "Win %": p_win_3, "Stake": stake_3})

    # --- 4 LEG SLIP ---
    p_win_4 = probs[0] * probs[1] * probs[2] * probs[3]
    gross_payout_4 = payout_4 * boost_mult
    ev_4 = (p_win_4 * gross_payout_4) - 1
    
    outcomes_4 = [
        (p_win_4, gross_payout_4 - 1),
        (1 - p_win_4, -1.0)
    ]
    full_f_4 = solve_general_kelly(outcomes_4)
    stake_4 = bankroll * (full_f_4 * kelly_fraction)
    results.append({"Size": "4-Pick", "EV": ev_4, "Win %": p_win_4, "Stake": stake_4})

    # --- 5 LEG SLIP (Complex Multi-Outcome Kelly) ---
    p_win_5_perfect = probs[0] * probs[1] * probs[2] * probs[3] * probs[4]
    p_win_5_insurance = calculate_exact_wins_prob(probs[:5], 5, 4)
    p_win_5_loss = 1.0 - p_win_5_perfect - p_win_5_insurance
    
    gross_payout_5_perfect = payout_5 * boost_mult
    gross_payout_5_insurance = 1.0 * boost_mult # Insurance pays 1x Gross
    
    expected_return_5 = (p_win_5_perfect * gross_payout_5_perfect) + (p_win_5_insurance * gross_payout_5_insurance)
    ev_5 = expected_return_5 - 1
    
    # Outcomes list for General Solver: [(prob, net_odds), ...]
    outcomes_5 = [
        (p_win_5_perfect, gross_payout_5_perfect - 1),    # Perfect Win
        (p_win_5_insurance, gross_payout_5_insurance - 1),# Insurance Win (Net might be negative if boost < 1, but usually positive or 0)
        (p_win_5_loss, -1.0)                              # Loss
    ]
    
    full_f_5 = solve_general_kelly(outcomes_5)
    stake_5 = bankroll * (full_f_5 * kelly_fraction)
    results.append({"Size": "5-Pick", "EV": ev_5, "Win %": p_win_5_perfect, "Stake": stake_5})

    # --- 6 LEG SLIP (Complex Multi-Outcome Kelly) ---
    p_win_6_perfect = probs[0] * probs[1] * probs[2] * probs[3] * probs[4] * probs[5]
    p_win_6_insurance = calculate_exact_wins_prob(probs[:6], 6, 5)
    p_win_6_loss = 1.0 - p_win_6_perfect - p_win_6_insurance
    
    gross_payout_6_perfect = payout_6 * boost_mult
    gross_payout_6_insurance = 1.5 * boost_mult 
    
    expected_return_6 = (p_win_6_perfect * gross_payout_6_perfect) + (p_win_6_insurance * gross_payout_6_insurance)
    ev_6 = expected_return_6 - 1
    
    outcomes_6 = [
        (p_win_6_perfect, gross_payout_6_perfect - 1),
        (p_win_6_insurance, gross_payout_6_insurance - 1),
        (p_win_6_loss, -1.0)
    ]
    
    full_f_6 = solve_general_kelly(outcomes_6)
    stake_6 = bankroll * (full_f_6 * kelly_fraction)
    results.append({"Size": "6-Pick", "EV": ev_6, "Win %": p_win_6_perfect, "Stake": stake_6})

    # --- DISPLAY RESULTS ---
    st.header("Results (Quarter Kelly)")
    
    best_ev = -float('inf')
    best_size = ""
    
    res_cols = st.columns(5)
    for i, res in enumerate(results):
        ev_percent = res['EV'] * 100
        stake = res['Stake']
        
        if res['EV'] > best_ev:
            best_ev = res['EV']
            best_size = res['Size']

        with res_cols[i]:
            st.metric(
                label=res['Size'], 
                value=f"{ev_percent:.2f}% EV",
                delta=f"Stake: ${stake:.2f}"
            )
            st.caption(f"Perfect Hit: {res['Win %']*100:.1f}%")

    # Recommendation
    st.subheader("Recommendation")
    if best_ev > 0:
        st.success(f"**Best Play:** {best_size} Slip with {best_ev*100:.2f}% EV.")
        st.info("Stakes are calculated using a precise Generalized Kelly solver (Quarter Kelly applied).")
    else:
        st.error(f"**Best Play:** {best_size}, but all EVs are negative. Recommended stake is $0.00.")
