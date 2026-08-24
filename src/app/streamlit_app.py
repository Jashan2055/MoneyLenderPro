import streamlit as st
import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CLEAN STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Page */
    .block-container {
        max-width: 1000px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Section */
    .section {
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">💳 Credit Risk Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate loan default risk using a tuned machine learning model.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN INPUTS
# ============================================================

st.markdown(
    '<div class="section">Loan Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=500.0,
        max_value=40000.0,
        value=12000.0,
        step=500.0
    )

with col2:
    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=1000.0,
        value=65000.0,
        step=1000.0
    )

with col3:
    term = st.selectbox(
        "Loan Term",
        [36, 60],
        format_func=lambda x: f"{x} months"
    )


# ============================================================
# FINANCIAL PROFILE
# ============================================================

st.markdown(
    '<div class="section">Financial Profile</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=1.0,
        max_value=40.0,
        value=12.74,
        step=0.01
    )

with col2:
    dti = st.number_input(
        "Debt-to-Income Ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=17.6,
        step=0.1
    )

with col3:
    emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0,
        max_value=60.0,
        value=6.0,
        step=1.0
    )


# ============================================================
# ADVANCED DETAILS
# ============================================================

with st.expander("Advanced applicant details"):

    col1, col2, col3 = st.columns(3)

    with col1:
        home_ownership = st.selectbox(
            "Home Ownership",
            [
                "RENT",
                "MORTGAGE",
                "OWN",
                "OTHER",
                "NONE",
                "ANY"
            ]
        )

    with col2:
        verification_status = st.selectbox(
            "Income Verification",
            [
                "Not Verified",
                "Source Verified",
                "Verified"
            ]
        )

    with col3:
        addr_state = st.selectbox(
            "State",
            [
                "CA", "NY", "TX", "FL", "IL",
                "PA", "OH", "GA", "NC", "MI",
                "NJ", "VA", "WA", "AZ", "MA",
                "CO", "MD", "TN", "MO", "IN",
                "WI", "MN", "SC", "AL", "LA",
                "KY", "OR", "OK", "CT", "IA",
                "MS", "AR", "KS", "UT", "NV",
                "NM", "WV", "NE", "ID", "HI",
                "ME", "NH", "RI", "MT", "DE",
                "SD", "AK", "VT", "WY", "ND"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        purpose = st.selectbox(
            "Loan Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "car",
                "medical",
                "moving",
                "vacation",
                "house",
                "wedding",
                "educational",
                "renewable_energy",
                "other"
            ]
        )

    with col2:
        credit_history_years = st.number_input(
            "Credit History (years)",
            min_value=0.0,
            value=14.75,
            step=0.5
        )

    st.markdown("**Credit information**")

    col1, col2, col3 = st.columns(3)

    with col1:
        revol_util = st.number_input(
            "Revolving Utilization (%)",
            min_value=0.0,
            max_value=200.0,
            value=52.3,
            step=1.0
        )

    with col2:
        open_acc = st.number_input(
            "Open Accounts",
            min_value=0.0,
            value=11.0,
            step=1.0
        )

    with col3:
        total_acc = st.number_input(
            "Total Accounts",
            min_value=0.0,
            value=23.0,
            step=1.0
        )


# ============================================================
# INSTALLMENT CALCULATION
# ============================================================

def calculate_installment(principal, annual_rate, months):

    monthly_rate = annual_rate / 100 / 12

    if monthly_rate == 0:
        return principal / months

    return (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** months
        / ((1 + monthly_rate) ** months - 1)
    )


installment = calculate_installment(
    loan_amount,
    interest_rate,
    term
)


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    '<div class="section">Application Summary</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Loan",
        f"${loan_amount:,.0f}"
    )

with col2:
    st.metric(
        "Income",
        f"${annual_income:,.0f}"
    )

with col3:
    st.metric(
        "Monthly Payment",
        f"${installment:,.2f}"
    )

with col4:
    st.metric(
        "DTI",
        f"{dti:.1f}%"
    )


# ============================================================
# ASSESS BUTTON
# ============================================================

st.markdown("")

if st.button(
    "Assess Credit Risk",
    type="primary"
):

    payload = {

        "annual_income": annual_income,

        "loan_amount": loan_amount,

        "funded_amount": loan_amount,

        "investor_funds": loan_amount,

        "term": term,

        "interest_rate": interest_rate,

        "installment": installment,

        "dti": dti,

        "emp_length": emp_length,

        "home_ownership": home_ownership,

        "verification_status": verification_status,

        "addr_state": addr_state,

        "purpose": purpose,

        "pymnt_plan": "n",

        "revol_bal": 11130.0,

        "revol_util": revol_util,

        "total_rev_hi_lim": 24000.0,

        "open_acc": open_acc,

        "total_acc": total_acc,

        "delinq_2yrs": 0.0,

        "pub_rec": 0.0,

        "tot_cur_bal": 80452.0,

        "bc_util": 63.4,

        "credit_history_years": credit_history_years
    }

    with st.spinner("Assessing credit risk..."):

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

            if response.status_code == 422:

                st.error("Invalid application data.")

                st.json(response.json())

                st.stop()

            response.raise_for_status()

            result = response.json()

            probability = float(
                result["risk_probability"]
            )

            percentage = float(
                result["risk_percentage"]
            )

            tier = result["risk_tier"]


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.markdown(
                '<div class="section">Risk Assessment</div>',
                unsafe_allow_html=True
            )

            if tier == "Low Risk":

                icon = "🟢"

                message = (
                    "The applicant shows a relatively low "
                    "probability of default."
                )

            elif tier == "Medium Risk":

                icon = "🟡"

                message = (
                    "The applicant presents moderate credit "
                    "risk. Additional assessment may be appropriate."
                )

            else:

                icon = "🔴"

                message = (
                    "The applicant presents elevated default "
                    "risk and requires careful assessment."
                )


            # -------------------------------------------------
            # RESULT METRICS
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Risk Probability",
                    f"{percentage:.2f}%"
                )

            with col2:

                st.metric(
                    "Risk Tier",
                    f"{icon} {tier}"
                )


            # -------------------------------------------------
            # PROBABILITY
            # -------------------------------------------------

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.caption(
                f"Predicted probability of bad loan: "
                f"{percentage:.2f}%"
            )


            # -------------------------------------------------
            # INTERPRETATION
            # -------------------------------------------------

            if tier == "Low Risk":

                st.success(message)

            elif tier == "Medium Risk":

                st.warning(message)

            else:

                st.error(message)


            # -------------------------------------------------
            # MODEL INFO
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.caption("MODEL")

                st.write(
                    "Tuned XGBoost"
                )

            with col2:

                st.caption("ASSESSMENT")

                if tier == "Low Risk":

                    st.write(
                        "Lower-risk application"
                    )

                elif tier == "Medium Risk":

                    st.write(
                        "Additional assessment recommended"
                    )

                else:

                    st.write(
                        "High-risk application"
                    )


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Make sure the API is running on port 8000."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The prediction request timed out."
            )

        except requests.exceptions.HTTPError as e:

            st.error(
                f"API request failed: {e}"
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Powered by Tuned XGBoost • "
    "For decision support, not automated lending decisions."
)