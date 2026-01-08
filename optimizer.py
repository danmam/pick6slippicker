import streamlit as st
import itertools
import math

# ... (Keep previous american_to_prob and calculate_exact_wins_prob functions) ...

def calculate_expected_growth(outcomes, stake_fraction):
    """
    Calculates expected growth rate G = sum(p_i * ln(1 + f * b_i)).
    Returns value in basis points (1/100th of 1%).
    """
    if stake_fraction <= 0:
        return 0.0
    
    growth_sum = 0.0
    for prob, net_odds in outcomes:
        # Growth factor is (1 + fraction * net_odds)
        # e.g., a loss is (1 + f * -1) = (1 - f)
        growth_sum += prob * math.log(1 + stake_fraction * net_odds)
    
    # Convert to basis points: Growth * 10,000
    return growth_sum * 10000

# --- INSIDE THE CALCULATION BUTTON ---
# (Inside your loop for n, main_p, ins_p in configs:)

        f_full_kelly = solve_general_kelly(outcomes)
        actual_f = f_full_kelly * kelly_fraction # Applying the 0.25 Quarter Kelly
        
        eg_bps = calculate_expected_growth(outcomes, actual_f)
        
        results.append({
            "Size": f"{n}-Pick",
            "EV": ev,
            "Win %": p_perfect,
            "Stake": bankroll * actual_f,
            "EG_bps": eg_bps
        })

# --- UPDATED DISPLAY TABLE ---
st.subheader("Detailed Comparison")
table_data = []
for res in results:
    table_data.append({
        "Slip Size": res['Size'],
        "EV %": f"{res['EV']*100:.2f}%",
        "Win % (Perfect)": f"{res['Win %']*100:.2f}%",
        "Recommended Stake": f"${res['Stake']:.2f}",
        "Exp. Growth (bps)": f"{res['EG_bps']:.2f} bps"
    })
st.table(table_data)
