import streamlit as st  # type: ignore[reportMissingImports]
import pandas as pd
from engine import PricingEngine

st.title("IAA AI-Assisted Actuarial Pricing Engine")
st.write("Calculate technical insurance premiums incorporating LightGBM Tweedie models, expense loadings, and excess relativities.")

@st.cache_resource
def load_engine():
    return PricingEngine()

engine = load_engine()

# User Input Controls
lob = st.selectbox("Line of Business", engine.config.get("lines_of_business", ["motor"]))
premium_input = st.number_input("Base Premium / Value", value=500.0)
tenure = st.number_input("Tenure / Exposure", value=1.0)
voluntary_excess = st.selectbox("Voluntary Excess", [100, 250, 500], index=1)

if st.button("Calculate Technical Premium"):
    df = pd.DataFrame([{
        "Premium": premium_input,
        "Tenure": tenure,
        "exposure": tenure,
        "voluntary_excess": voluntary_excess
    }])

    try:
        result = engine.predict_policy(df, lob=lob)

        st.subheader("Pricing & Technical Breakdown")

        # Display metrics cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Pure Risk Premium", f"${result['pure_risk_premium']:,.2f}")
        m2.metric("Excess Adjusted", f"${result['excess_adjusted_premium']:,.2f}")
        m3.metric("Technical Premium", f"${result['technical_premium']:,.2f}", delta="Final Quote")

        st.divider()

        # Organized tabular view instead of raw JSON
        st.markdown("### Component Breakdown")
        breakdown_data = {
            "Component": [
                "Pure Risk Premium",
                "Excess Relativity Factor",
                "Excess Adjusted Premium",
                "Expense Loading (20%)",
                "Profit Margin (5%)",
                "Final Technical Premium"
            ],
            "Value": [
                f"${result['pure_risk_premium']:,.2f}",
                f"{result['loading_breakdown']['excess_relativity_applied']}x",
                f"${result['excess_adjusted_premium']:,.2f}",
                f"${result['loading_breakdown']['expense_loading']:,.2f}",
                f"${result['loading_breakdown']['profit_margin']:,.2f}",
                f"${result['technical_premium']:,.2f}"
            ]
        }

        st.table(pd.DataFrame(breakdown_data))

    except Exception as e:
        st.error(f"Prediction Error: {e}")