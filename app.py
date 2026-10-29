import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "house_price_model.joblib")
    with open(BASE_DIR / "schema.json", "r") as f:
        schema = json.load(f)
    return model, schema

try:
    model, schema = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

st.title("🏠 House Price Prediction")
st.write(
    "Enter the house details below to estimate its price "
    "using a trained Linear Regression model."
)

with st.form("house_form"):
    values = {}

    st.subheader("House details")

    for col in schema["columns"]:
        if col in schema["numeric_columns"]:
            default = float(schema["defaults"].get(col, 0.0))
            values[col] = st.number_input(
                col.replace("_", " ").title(),
                value=default,
                format="%.2f",
                help=f"Enter a value for {col}."
            )

        elif col in schema["categorical_columns"]:
            options = schema["categories"].get(col, [])

            if options:
                values[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options=options
                )
            else:
                values[col] = st.text_input(
                    col.replace("_", " ").title()
                )

    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame(
        [[values[col] for col in schema["columns"]]],
        columns=schema["columns"]
    )

    try:
        prediction = float(model.predict(input_df)[0])

        st.subheader("Predicted house price")
        st.metric(
            label="Estimated Price",
            value=f"{prediction:,.2f}"
        )

        st.caption(
            "This is a model estimate, not a guaranteed market price. "
            "Accuracy depends on the training data and house characteristics."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Internship project | Linear Regression")
