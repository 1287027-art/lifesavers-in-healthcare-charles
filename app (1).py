import json
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Career Venture Lab | NUR-3 Health Education Studio",
    page_icon="🩺",
    layout="wide",
)

# --- Exact Money Helpers (Integer Cents) ---
def to_cents(amount: float | Decimal | int | str) -> int:
    d = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(d * 100)

def from_cents(cents: int) -> float:
    return float(Decimal(cents) / Decimal(100))

def fmt_usd(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    abs_c = abs(cents)
    return f"{sign}${abs_c // 100:,}.{abs_c % 100:02d}"

# --- Default State Factory (NUR-3 Handbook Baselines) ---
def create_initial_state(team_id: str = "TEAM-01", level: str = "Advanced") -> dict:
    return {
        "team_id": team_id,
        "course": "AP Calculus",
        "pathway_id": "NUR-3",
        "business_name": "Nurse-Led Health Education Simulation",
        "selected_skill": "Constrained Profit Optimization & Rate Accumulation",
        "selected_level": level,
        "state_version": 1,
        "simulation_date": "2027-01-31",
        "base_period": "monthly",
        "working_days": 20,
        "currency": "USD",
        "random_seed": 2027,
        "parameters": {
            "base_price": 25.00,
            "D0": 80,
            "staff_hours": 30.0,
            "hours_per_unit": 0.5,
            "venue_seats": 60,
            "no_show_rate": 0.0,
            "satisfaction_mean": 4.0,
            "owner_contribution_cents": 600000,
            "equipment_assets_cents": 150000,
            "pre_opening_expense_cents": 50000,
            "opening_cash_cents": 400000,
            "current_cash_cents": 400000,
            "retained_earnings_cents": -50000,
            "debt_cents": 0,
            "var_costs_cents": {
                "educator_delivery": 200,
                "materials": 200,
                "cleaning_consumables": 100,
            },
            "fixed_costs_cents": {
                "venue": 25000,
                "scheduling_software": 10000,
                "accessibility_support": 20000,
                "admin_insurance_assumption": 15000,
            },
        },
        "ledger": [],
        "processed_run_ids": [],
        "assumptions": [
            "Nonclinical general health education workshops only; no medical diagnosis or patient records.",
            "Revenue and per-seat variable costs apply only to attended seats: attended = floor(booked * (1 - no_show_rate)).",
            "Simplified accrual model with immediate payment; taxes and depreciation disabled at baseline.",
        ],
    }

# --- Deterministic NUR-3 Calculation Engine ---
def compute_nur_month(
    params: dict,
    requested: int,
    ad_cost_cents: int = 0,
    refund_cents: int = 0,
    loan_cash_cents: int = 0,
    principal_cents: int = 0,
    interest_cents: int = 0,
    capex_cents: int = 0,
    new_owner_cash_cents: int = 0,
    capacity_override: int | None = None,
) -> dict:
    if requested < 0:
        raise ValueError("Requested seats cannot be negative.")
    if ad_cost_cents < 0 or refund_cents < 0:
        raise ValueError("Costs and refunds cannot be negative.")

    price_cents = to_cents(params["base_price"])
    var_per_unit_cents = sum(params["var_costs_cents"].values())
    fixed_total_cents = sum(params["fixed_costs_cents"].values())

    staff_cap = math.floor(params["staff_hours"] / params["hours_per_unit"])
    venue_cap = int(params["venue_seats"])
    derived_capacity = min(staff_cap, venue_cap)

    effective_capacity = int(capacity_override) if capacity_override is not None else derived_capacity
    booked = min(int(requested), effective_capacity)
    unserved_demand = max(0, int(requested) - booked)
    attended = math.floor(booked * (1.0 - float(params["no_show_rate"])))
    no_shows = booked - attended
    sold = attended

    gross_revenue_cents = price_cents * sold
    net_revenue_cents = gross_revenue_cents - refund_cents
    variable_expense_cents = var_per_unit_cents * sold
    operating_profit_cents = (
        net_revenue_cents - variable_expense_cents - fixed_total_cents - ad_cost_cents
    )
    net_profit_cents = operating_profit_cents - interest_cents

    opening_cash_cents = int(params["current_cash_cents"])
    ending_cash_cents = (
        opening_cash_cents
        + net_revenue_cents
        - variable_expense_cents
        - fixed_total_cents
        - ad_cost_cents
        - interest_cents
        - principal_cents
        - capex_cents
        + new_owner_cash_cents
        + loan_cash_cents
    )

    return {
        "staff_capacity": staff_cap,
        "venue_capacity": venue_cap,
        "derived_capacity": derived_capacity,
        "effective_capacity": effective_capacity,
        "requested": int(requested),
        "booked": booked,
        "unserved_demand": unserved_demand,
        "attended": attended,
        "no_shows": no_shows,
        "sold": sold,
        "price_cents": price_cents,
        "var_per_unit_cents": var_per_unit_cents,
        "gross_revenue_cents": gross_revenue_cents,
        "net_revenue_cents": net_revenue_cents,
        "variable_expense_cents": variable_expense_cents,
        "fixed_expense_cents": fixed_total_cents,
        "ad_cost_cents": ad_cost_cents,
        "total_operating_expense_cents": variable_expense_cents + fixed_total_cents + ad_cost_cents,
        "operating_profit_cents": operating_profit_cents,
        "net_profit_cents": net_profit_cents,
        "opening_cash_cents": opening_cash_cents,
        "ending_cash_cents": ending_cash_cents,
    }

def record_simulation_step(
    state: dict,
    run_id: str,
    sim_date: str,
    requested: int,
    ad_cost_cents: int = 0,
    refund_cents: int = 0,
    loan_cash_cents: int = 0,
    principal_cents: int = 0,
    interest_cents: int = 0,
    capex_cents: int = 0,
    new_owner_cash_cents: int = 0,
    capacity_override: int | None = None,
    mode: str = "normal",
) -> tuple[bool, str, dict | None]:
    if run_id in state["processed_run_ids"]:
        return False, f"Duplicate run_id '{run_id}' rejected. No transaction recorded.", None

    res = compute_nur_month(
        params=state["parameters"],
        requested=requested,
        ad_cost_cents=ad_cost_cents,
        refund_cents=refund_cents,
        loan_cash_cents=loan_cash_cents,
        principal_cents=principal_cents,
        interest_cents=interest_cents,
        capex_cents=capex_cents,
        new_owner_cash_cents=new_owner_cash_cents,
        capacity_override=capacity_override,
    )

    next_ver = state["state_version"] + 1
    tx_base = f"TX-{next_ver:03d}"

    entries = [
        {
            "transaction_id": f"{tx_base}-REV",
            "run_id": run_id,
            "date": sim_date,
            "mode": mode,
            "event_type": "workshop_sales",
            "category": "Revenue",
            "amount_cents": res["net_revenue_cents"],
            "quantity": res["sold"],
            "unit_price_cents": res["price_cents"],
            "state_version": next_ver,
        },
        {
            "transaction_id": f"{tx_base}-VAR",
            "run_id": run_id,
            "date": sim_date,
            "mode": mode,
            "event_type": "variable_expense",
            "category": "Variable Operating Expense",
            "amount_cents": -res["variable_expense_cents"],
            "quantity": res["sold"],
            "unit_price_cents": res["var_per_unit_cents"],
            "state_version": next_ver,
        },
        {
            "transaction_id": f"{tx_base}-FIX",
            "run_id": run_id,
            "date": sim_date,
            "mode": mode,
            "event_type": "fixed_expense",
            "category": "Fixed Operating Expense",
            "amount_cents": -res["fixed_expense_cents"],
            "quantity": 1,
            "unit_price_cents": res["fixed_expense_cents"],
            "state_version": next_ver,
        },
    ]

    if ad_cost_cents > 0:
        entries.append({
            "transaction_id": f"{tx_base}-MKT",
            "run_id": run_id,
            "date": sim_date,
            "mode": mode,
            "event_type": "marketing_expense",
            "category": "Marketing Expense",
            "amount_cents": -ad_cost_cents,
            "quantity": 1,
            "unit_price_cents": ad_cost_cents,
            "state_version": next_ver,
        })

    if loan_cash_cents > 0:
        entries.append({
            "transaction_id": f"{tx_base}-LOAN",
            "run_id": run_id,
            "date": sim_date,
            "mode": mode,
            "event_type": "loan_proceeds",
            "category": "Financing (Not Revenue)",
            "amount_cents": loan_cash_cents,
            "quantity": 1,
            "unit_price_cents": loan_cash_cents,
            "state_version": next_ver,
        })

    if capex_cents > 0:
        entries.append({
            "transaction_id": f"{tx_base}-CAPEX",
            "run_id": run_id,
            "date": sim_date,
            "mode": mode,
            "event_type": "capital_equipment",
            "category": "Capital Expenditure",
            "amount_cents": -capex_cents,
            "quantity": 1,
            "unit_price_cents": capex_cents,
            "state_version": next_ver,
        })

    state["ledger"].extend(entries)
    state["processed_run_ids"].append(run_id)
    state["state_version"] = next_ver
    state["simulation_date"] = sim_date
    state["parameters"]["current_cash_cents"] = res["ending_cash_cents"]
    state["parameters"]["retained_earnings_cents"] += res["net_profit_cents"]
    state["parameters"]["debt_cents"] += loan_cash_cents - principal_cents
    state["parameters"]["equipment_assets_cents"] += capex_cents

    return True, f"Recorded run '{run_id}' -> state_version v{next_ver}.", res

# --- Session State Initialization ---
if "sim_state" not in st.session_state:
    st.session_state.sim_state = create_initial_state()

state = st.session_state.sim_state
params = state["parameters"]

# --- Sidebar: Identity, Challenge Level, Save/Load JSON ---
with st.sidebar:
    st.header("🩺 NUR-3 Control Center")
    st.caption("AP Calculus | Nurse-Led Health Education Simulation")

    new_team_id = st.text_input("Team ID", value=state["team_id"])
    if new_team_id.strip():
        state["team_id"] = new_team_id.strip()

    levels = ["Foundation", "Applied", "Advanced"]
    chosen_level = st.selectbox(
        "Teacher-Approved Challenge Level",
        levels,
        index=levels.index(state["selected_level"]),
        help="Changing levels updates equations, constraints, visualizations, and interpretation tasks.",
    )
    if chosen_level != state["selected_level"]:
        state["selected_level"] = chosen_level

    st.metric("Current State Version", f"v{state['state_version']}")
    st.metric("Spendable Cash", fmt_usd(params["current_cash_cents"]))

    st.divider()
    st.subheader("💾 Durable Save / Restore JSON")
    snapshot_payload = {
        **state,
        "snapshot_id": f"SNAP-{state['team_id']}-v{state['state_version']}",
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    st.download_button(
        label="⬇️ Export Snapshot JSON",
        data=json.dumps(snapshot_payload, indent=2),
        file_name=f"{state['team_id']}_NUR3_v{state['state_version']}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("⬆️ Restore or Import Snapshot JSON", type=["json"])
    if uploaded_file is not None:
        if st.button("Validate & Load JSON", use_container_width=True):
            try:
                imported = json.loads(uploaded_file.read().decode("utf-8"))
                required_keys = {"team_id", "pathway_id", "state_version", "parameters", "ledger"}
                if not required_keys.issubset(imported.keys()):
                    st.error("Rejected: Missing required snapshot contract fields.")
                elif imported["team_id"] != state["team_id"]:
                    st.error(
                        f"Rejected: Team mismatch (expected '{state['team_id']}', got '{imported['team_id']}')."
                    )
                elif imported["pathway_id"] != "NUR-3":
                    st.error(f"Rejected: Pathway mismatch (expected 'NUR-3', got '{imported['pathway_id']}').")
                elif int(imported["state_version"]) < int(state["state_version"]):
                    st.error(
                        f"Rejected Stale Version: Imported v{imported['state_version']} is older than active v{state['state_version']}."
                    )
                else:
                    st.session_state.sim_state = imported
                    st.success(f"Restored {imported['team_id']} at v{imported['state_version']}!")
                    st.rerun()
            except Exception as exc:
                st.error(f"Invalid JSON file: {exc}")

    if st.button("🔄 Reset to NUR-3 Opening Baseline", use_container_width=True):
        st.session_state.sim_state = create_initial_state(
            team_id=state["team_id"], level=state["selected_level"]
        )
        st.rerun()

# --- Main Header ---
st.title("Nurse-Led Health Education Simulation (`NUR-3`)")
st.info(
    "**Educational Scope:** Fictional nonclinical health education workshops only. "
    "No clinical diagnosis, treatment, or real patient data. Stage 1 deterministic core is active."
)

tabs = st.tabs([
    "1. Overview & Inputs",
    "2. AP Calculus Math Panel",
    "3. Run Simulation & Ledger",
    "4. Stage 1 Acceptance Tests",
    "5. Roadmap (Stages 2–4)",
])

# ==============================================================================
# TAB 1: OVERVIEW & INPUTS
# ==============================================================================
with tabs[0]:
    st.subheader("Startup Financing & Opening Balance Sheet")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Owner Contribution", fmt_usd(params["owner_contribution_cents"]))
    c2.metric("Equipment Assets", fmt_usd(params["equipment_assets_cents"]))
    c3.metric("Pre-Opening Expense", fmt_usd(params["pre_opening_expense_cents"]))
    c4.metric("Post-Startup Cash", fmt_usd(params["opening_cash_cents"]))

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Workshop Offerings & Capacity Resources")
        params["base_price"] = st.number_input(
            "Workshop Seat Price p₀ (USD)", min_value=1.0, max_value=500.0, value=float(params["base_price"]), step=1.0
        )
        params["D0"] = st.number_input(
            "Baseline Monthly Demand D₀ (seats/month)", min_value=0, max_value=1000, value=int(params["D0"]), step=1
        )
        params["staff_hours"] = st.number_input(
            "Nurse Educator Hours / Month", min_value=0.0, max_value=500.0, value=float(params["staff_hours"]), step=1.0
        )
        params["hours_per_unit"] = st.number_input(
            "Educator Hours per Seat", min_value=0.1, max_value=10.0, value=float(params["hours_per_unit"]), step=0.1
        )
        params["venue_seats"] = st.number_input(
            "Venue Seat Limit / Month", min_value=0, max_value=1000, value=int(params["venue_seats"]), step=1
        )
        params["no_show_rate"] = st.slider(
            "Expected No-Show Rate", min_value=0.0, max_value=0.90, value=float(params["no_show_rate"]), step=0.05
        )

    with col_right:
        st.subheader("Editable Variable & Fixed Cost Lines")
        st.markdown("**Variable Costs (per attended seat)**")
        for key in list(params["var_costs_cents"].keys()):
            val_usd = st.number_input(
                f"{key.replace('_', ' ').title()} ($/seat)",
                min_value=0.0,
                max_value=200.0,
                value=from_cents(params["var_costs_cents"][key]),
                step=0.5,
                key=f"var_{key}",
            )
            params["var_costs_cents"][key] = to_cents(val_usd)

        var_sum_cents = sum(params["var_costs_cents"].values())
        st.success(f"**Total Variable Cost (v):** {fmt_usd(var_sum_cents)} / attended seat")

        st.markdown("**Fixed Monthly Costs ($/month)**")
        for key in list(params["fixed_costs_cents"].keys()):
            val_usd = st.number_input(
                f"{key.replace('_', ' ').title()} ($/month)",
                min_value=0.0,
                max_value=5000.0,
                value=from_cents(params["fixed_costs_cents"][key]),
                step=10.0,
                key=f"fix_{key}",
            )
            params["fixed_costs_cents"][key] = to_cents(val_usd)

        fix_sum_cents = sum(params["fixed_costs_cents"].values())
        st.success(f"**Total Fixed Cost (F):** {fmt_usd(fix_sum_cents)} / month")

    derived_cap = min(math.floor(params["staff_hours"] / params["hours_per_unit"]), int(params["venue_seats"]))
    st.info(
        f"**Derived Monthly Capacity:** `min(floor({params['staff_hours']} / {params['hours_per_unit']}), {params['venue_seats']})` "
        f"= **{derived_cap} seats/month**"
    )

# ==============================================================================
# TAB 2: AP CALCULUS MATH SKILLS & CHALLENGE LEVEL PANEL
# ==============================================================================
with tabs[1]:
    st.subheader(f"AP Calculus Panel — Level: {state['selected_level']}")
    p0 = float(params["base_price"])
    D0 = float(params["D0"])
    v = from_cents(sum(params["var_costs_cents"].values()))
    F = from_cents(sum(params["fixed_costs_cents"].values()))
    cap = min(math.floor(params["staff_hours"] / params["hours_per_unit"]), int(params["venue_seats"]))

    a = 2.0 * D0
    b = D0 / p0
    p_max = 2.0 * p0
    p_corner = max(0.0, (a - cap) / b) if b > 0 else 0.0
    p_unconstrained = (a + b * v) / (2.0 * b) if b > 0 else p0

    st.markdown(
        f"**Linear Demand Model:** $q(p) = {a:.1f} - {b:.2f}p$ on domain $0 \le p \le {p_max:.2f}$ (seats/month)  \n"
        f"**Current Capacity Boundary:** $s(p) = \min({cap}, \max(0, q(p)))$ | **Capacity Corner Price:** $p = \${p_corner:.2f}$"
    )

    p_grid = np.linspace(0, p_max, 200)
    q_grid = np.maximum(0.0, a - b * p_grid)
    s_grid = np.minimum(cap, q_grid)
    profit_unconstrained = (p_grid - v) * q_grid - F
    profit_constrained = (p_grid - v) * s_grid - F
    deriv_unconstrained = a + b * v - 2.0 * b * p_grid
    deriv_constrained = np.where(p_grid < p_corner, float(cap), deriv_unconstrained)

    if state["selected_level"] == "Foundation":
        st.markdown(
            f"**Foundation Task:** Derive unconstrained profit $P(p) = (p - {v:.2f})({a:.1f} - {b:.2f}p) - {F:.2f}$ "
            f"and marginal profit $P'(p) = {a + b*v:.2f} - {2*b:.2f}p$ (units: dollars per dollar of ticket price).  \n"
            f"- Unconstrained critical point ($P'(p)=0$): **$p^* = \${p_unconstrained:.2f}$** "
            f"(gives unconstrained demand $q = {a - b*p_unconstrained:.1f}$ seats)."
        )
    elif state["selected_level"] == "Applied":
        st.markdown(
            f"**Applied Task:** Enforce the **{cap}-seat** capacity constraint to form a piecewise profit model.  \n"
            f"- For $0 \le p < {p_corner:.2f}$: $s(p) = {cap}$ (capacity-capped), so $P'(p) = {cap}$ $/$.  \n"
            f"- For ${p_corner:.2f} < p \le {p_max:.2f}$: $s(p) = {a:.1f} - {b:.2f}p$, so $P'(p) = {a + b*v:.2f} - {2*b:.2f}p$.  \n"
            f"- At the corner $p = \${p_corner:.2f}$, $P'(p)$ has a jump discontinuity (undefined). "
            f"Since unconstrained $p^* = \${p_unconstrained:.2f} < \${p_corner:.2f}$, the constrained continuous optimum occurs at **$p = \${p_corner:.2f}$**."
        )
    else:
        st.markdown(
            "**Advanced Task:** Combine constrained endpoint optimization with **time-dependent rate accumulation** and sensitivity analysis.  \n"
            "Suppose a 10-day health-education outreach campaign generates marginal revenue rate $r(t) = 100 + 20t$ ($/day) "
            "and marginal cost rate $c(t) = 60 + 5t$ ($/day) over $0 \le t \le 10$ days (isolated rate module; does not overwrite monthly ledger)."
        )
        t_max = st.slider("Integration Horizon T (days)", min_value=1, max_value=14, value=10)
        rev_acc = 100 * t_max + 10 * (t_max ** 2)
        cost_acc = 60 * t_max + 2.5 * (t_max ** 2)
        net_acc = rev_acc - cost_acc
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Accumulated Revenue ∫₀^{t_max} r(t)dt", f"${rev_acc:,.2f}")
        m2.metric(f"Accumulated Cost ∫₀^{t_max} c(t)dt", f"${cost_acc:,.2f}")
        m3.metric(f"Net Accumulated Profit", f"${net_acc:,.2f}")

    # Linked Profit & Derivative Chart
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=(
            f"Continuous Monthly Profit P(p) — {state['selected_level']} View (Continuous Model ≠ Rounded Transactions)",
            "Derivative P'(p) ($ per $ of price) — Corner Discontinuity Marked",
        ),
    )
    fig.add_trace(
        go.Scatter(x=p_grid, y=profit_unconstrained, name="Unconstrained Profit P(p)", line=dict(dash="dash", color="#64748b")),
        row=1, col=1,
    )
    if state["selected_level"] in ("Applied", "Advanced"):
        fig.add_trace(
            go.Scatter(x=p_grid, y=profit_constrained, name=f"Capacity-Constrained Profit (cap={cap})", line=dict(color="#0f766e", width=3)),
            row=1, col=1,
        )
        fig.add_vline(x=p_corner, line_dash="dot", line_color="#b45309", annotation_text=f"Capacity Corner (${p_corner:.2f})", row=1, col=1)
    fig.add_hline(y=0, line_color="#1e293b", line_width=1, row=1, col=1)

    y_deriv = deriv_unconstrained if state["selected_level"] == "Foundation" else deriv_constrained
    fig.add_trace(
        go.Scatter(x=p_grid, y=y_deriv, name="Marginal Profit P'(p)", line=dict(color="#1d4ed8", width=2.5)),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="#1e293b", line_width=1, row=2, col=1)
    fig.update_layout(height=540, template="plotly_white", legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="Ticket Price p ($/seat)", row=2, col=1)
    fig.update_yaxes(title_text="Profit ($/mo)", row=1, col=1)
    fig.update_yaxes(title_text="P'(p) ($/$)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Linked Price-Volume Table
    sample_prices = sorted(list({15.0, 20.0, p0, round(p_corner, 2), round(p_unconstrained, 2), 35.0, 40.0}))
    rows = []
    for sp in sample_prices:
        if 0 <= sp <= p_max:
            q_cont = max(0.0, a - b * sp)
            s_cont = min(float(cap), q_cont)
            s_int = min(cap, math.floor(q_cont))
            p_cont = (sp - v) * s_cont - F
            p_int = (sp - v) * s_int - F
            rows.append({
                "Price ($)": f"${sp:.2f}",
                "Continuous Demand q(p)": round(q_cont, 2),
                "Constrained Seats s(p)": round(s_cont, 2),
                "Continuous Profit ($)": f"${p_cont:,.2f}",
                "Rounded Integer Seats": s_int,
                "Integer Profit ($)": f"${p_int:,.2f}",
            })
    st.markdown("**Linked Price–Volume–Profit Table (Continuous vs. Integer Rounding)**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.warning(
        "**Transfer Check Prompt:** Before moving a slider, predict why the unconstrained derivative root "
        f"$p^* = \${p_unconstrained:.2f}$ is infeasible when capacity is {cap} seats, and check the endpoint $p = \${p_corner:.2f}$."
    )

# ==============================================================================
# TAB 3: RUN SIMULATION & LEDGER
# ==============================================================================
with tabs[2]:
    st.subheader("Advance Simulation by One Period (Normal Mode)")
    st.caption("Input changes in Tab 1 act as scenarios until you explicitly approve and record a period here.")

    rc1, rc2, rc3 = st.columns(3)
    run_id_input = rc1.text_input("Unique Run ID", value=f"RUN-{state['state_version']:03d}")
    sim_date_input = rc2.text_input("Simulation Date (YYYY-MM-DD)", value="2027-01-31")
    req_seats_input = rc3.number_input("Requested Workshop Seats", min_value=0, max_value=1000, value=int(params["D0"]))

    fc1, fc2, fc3, fc4 = st.columns(4)
    ad_usd = fc1.number_input("Approved Campaign Expense ($)", min_value=0.0, value=0.0, step=10.0)
    loan_usd = fc2.number_input("Simulated Loan Proceeds ($)", min_value=0.0, value=0.0, step=100.0)
    capex_usd = fc3.number_input("Equipment Purchase / CapEx ($)", min_value=0.0, value=0.0, step=50.0)
    refund_usd = fc4.number_input("Approved Refunds ($)", min_value=0.0, value=0.0, step=10.0)

    preview = compute_nur_month(
        params=params,
        requested=int(req_seats_input),
        ad_cost_cents=to_cents(ad_usd),
        refund_cents=to_cents(refund_usd),
        loan_cash_cents=to_cents(loan_usd),
        capex_cents=to_cents(capex_usd),
    )

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Booked / Capacity", f"{preview['booked']} / {preview['effective_capacity']}")
    p2.metric("Attended (Sold)", f"{preview['sold']}")
    p3.metric("Net Revenue", fmt_usd(preview["net_revenue_cents"]))
    p4.metric("Operating Profit", fmt_usd(preview["operating_profit_cents"]))
    p5.metric("Projected Ending Cash", fmt_usd(preview["ending_cash_cents"]))

    if st.button("✅ Approve & Advance Simulation (Record to Ledger)", type="primary"):
        ok, msg, _ = record_simulation_step(
            state=state,
            run_id=run_id_input.strip(),
            sim_date=sim_date_input.strip(),
            requested=int(req_seats_input),
            ad_cost_cents=to_cents(ad_usd),
            refund_cents=to_cents(refund_usd),
            loan_cash_cents=to_cents(loan_usd),
            capex_cents=to_cents(capex_usd),
            mode="normal",
        )
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    st.divider()
    st.subheader("Single Dated Event Ledger")
    if not state["ledger"]:
        st.info("No transactions recorded yet. Run a period above or run the Stage 1 Fixture Test in Tab 4.")
    else:
        ledger_df = pd.DataFrame(state["ledger"]).copy()
        ledger_df["amount ($)"] = ledger_df["amount_cents"].apply(fmt_usd)
        ledger_df["unit_price ($)"] = ledger_df["unit_price_cents"].apply(fmt_usd)
        display_cols = [
            "transaction_id", "run_id", "date", "mode", "event_type",
            "category", "quantity", "unit_price ($)", "amount ($)", "state_version"
        ]
        st.dataframe(ledger_df[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export Ledger CSV",
            data=ledger_df[display_cols].to_csv(index=False),
            file_name=f"{state['team_id']}_NUR3_ledger_v{state['state_version']}.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 4: STAGE 1 ACCEPTANCE TESTS
# ==============================================================================
with tabs[3]:
    st.subheader("Stage 1 Automated Verification & Fixture Replay")
    st.warning(
        "**Fixture Mode Notice:** Replaying `sample_data/monthly_fixture.csv` resets the simulation to "
        "NUR-3 opening cash ($4,000.00) and uses explicit `capacity_override` values to verify the ledger."
    )

    if st.button("▶️ Run NUR-3 Monthly Fixture & Handbook Acceptance Suite"):
        # 1. Replay 3-month fixture on a clean NUR-3 state
        test_state = create_initial_state(team_id=state["team_id"], level=state["selected_level"])
        fixture_rows = [
            ("2027-01-31", 80, 60, 0, 60, 150000, 100000, 50000, 450000),
            ("2027-02-28", 88, 60, 10000, 60, 150000, 110000, 40000, 490000),
            ("2027-03-31", 72, 45, 0, 45, 112500, 92500, 20000, 510000),
        ]
        results_table = []
        all_passed = True

        for idx, (dt, req, cap_ov, ad_c, exp_sold, exp_rev, exp_exp, exp_prof, exp_cash) in enumerate(fixture_rows, 1):
            ok, _, res = record_simulation_step(
                state=test_state,
                run_id=f"FIXTURE-M{idx}",
                sim_date=dt,
                requested=req,
                ad_cost_cents=ad_c,
                capacity_override=cap_ov,
                mode="fixture",
            )
            row_pass = (
                ok
                and res["sold"] == exp_sold
                and res["net_revenue_cents"] == exp_rev
                and res["total_operating_expense_cents"] == exp_exp
                and res["operating_profit_cents"] == exp_prof
                and res["ending_cash_cents"] == exp_cash
            )
            all_passed = all_passed and row_pass
            results_table.append({
                "Month": dt[:7],
                "Sold (Actual / Exp)": f"{res['sold']} / {exp_sold}",
                "Revenue": f"{fmt_usd(res['net_revenue_cents'])} / {fmt_usd(exp_rev)}",
                "Total Expense": f"{fmt_usd(res['total_operating_expense_cents'])} / {fmt_usd(exp_exp)}",
                "Profit": f"{fmt_usd(res['operating_profit_cents'])} / {fmt_usd(exp_prof)}",
                "Closing Cash": f"{fmt_usd(res['ending_cash_cents'])} / {fmt_usd(exp_cash)}",
                "Status": "PASS" if row_pass else "FAIL",
            })

        st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

        # 2. Additional Handbook Checks
        checks = []

        # Zero sales check
        z_state = create_initial_state()
        z_res = compute_nur_month(z_state["parameters"], requested=0)
        z_pass = z_res["net_revenue_cents"] == 0 and z_res["variable_expense_cents"] == 0 and z_res["operating_profit_cents"] == -70000
        checks.append({"Test": "Zero Sales (requested=0)", "Expected": "Rev $0, Var $0, Profit -$700.00", "Status": "PASS" if z_pass else "FAIL"})

        # Business-specific no-show check (no_show_rate = 0.20)
        ns_state = create_initial_state()
        ns_state["parameters"]["no_show_rate"] = 0.20
        ns_res = compute_nur_month(ns_state["parameters"], requested=80)
        ns_pass = (
            ns_res["booked"] == 60
            and ns_res["attended"] == 48
            and ns_res["net_revenue_cents"] == 120000
            and ns_res["variable_expense_cents"] == 24000
            and ns_res["operating_profit_cents"] == 26000
        )
        checks.append({"Test": "NUR No-Show Test (no_show_rate=0.20)", "Expected": "Booked 60, Attended 48, Rev $1,200, Var $240, Profit $260", "Status": "PASS" if ns_pass else "FAIL"})

        # Duplicate run_id protection
        dup_ok, _, _ = record_simulation_step(test_state, run_id="FIXTURE-M1", sim_date="2027-01-31", requested=80)
        checks.append({"Test": "Duplicate run_id Rejection", "Expected": "Rejected without new ledger entry", "Status": "PASS" if not dup_ok else "FAIL"})

        # Loan accounting check
        l_state = create_initial_state()
        l_res = compute_nur_month(l_state["parameters"], requested=0, loan_cash_cents=100000)
        l_pass = l_res["net_revenue_cents"] == 0 and l_res["ending_cash_cents"] == 500000
        checks.append({"Test": "Loan Proceeds Accounting", "Expected": "Cash +$1,000, Revenue $0", "Status": "PASS" if l_pass else "FAIL"})

        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
        if all_passed and all(c["Status"] == "PASS" for c in checks):
            st.success("All Stage 1 NUR-3 acceptance tests PASSED!")

# ==============================================================================
# TAB 5: HONEST ROADMAP (STAGES 2-4)
# ==============================================================================
with tabs[4]:
    st.subheader("Unimplemented Modules (Scheduled for Subsequent Stages)")
    st.markdown(
        """
        Per the Career Venture Lab build protocol, features outside **Stage 1** are explicitly marked unimplemented until Stage 1 verification is signed off:
        - **Stage 2 (Planned):** Full multi-report PDF center (Income Statement, Cash Flow, Budget Variance, Balance Sheet), Costed Marketing Studio (A/B variants & campaign IDs), Cross-Course Connections panel, and Departmental AI Recommendation import/review workflow.
        - **Stage 3 (Planned):** Frozen-snapshot Practice Assessment generator and separate teacher-controlled Graded Assessment key workflow.
        - **Stage 4 (Optional, Lowest Priority):** Fictional demonstration storefront (no real money or customer data).
        """
    )
