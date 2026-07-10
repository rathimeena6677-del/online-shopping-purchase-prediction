import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load Model and Scaler
# ---------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Online Shopping Purchase Prediction")

st.markdown("""
Predict whether a customer is likely to purchase a product based on
their browsing behavior on an online shopping website.
""")

st.markdown("---")

st.subheader("Customer Browsing Details")

col1, col2 = st.columns(2)

with col1:

    administrative = st.number_input(
        "Administrative Pages",
        min_value=0,
        value=0
    )

    administrative_duration = st.number_input(
        "Administrative Duration",
        min_value=0.0,
        value=0.0
    )

    informational = st.number_input(
        "Informational Pages",
        min_value=0,
        value=0
    )

    informational_duration = st.number_input(
        "Informational Duration",
        min_value=0.0,
        value=0.0
    )

    product_related = st.number_input(
        "Product Related Pages",
        min_value=0,
        value=10
    )

    product_related_duration = st.number_input(
        "Product Related Duration",
        min_value=0.0,
        value=100.0
    )

    bounce_rates = st.number_input(
        "Bounce Rate",
        min_value=0.0,
        value=0.02,
        format="%.5f"
    )

    exit_rates = st.number_input(
        "Exit Rate",
        min_value=0.0,
        value=0.05,
        format="%.5f"
    )

    page_values = st.number_input(
        "Page Values",
        min_value=0.0,
        value=0.0
    )

with col2:

    special_day = st.slider(
        "Special Day",
        0.0,
        1.0,
        0.0
    )

    month = st.selectbox(
        "Month",
        ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
    )

    operating_system = st.selectbox(
        "Operating System",
        [1,2,3,4,5,6,7,8]
    )

    browser = st.selectbox(
        "Browser",
        [1,2,3,4,5,6,7,8,9,10,11,12,13]
    )

    region = st.selectbox(
        "Region",
        [1,2,3,4,5,6,7,8,9]
    )

    traffic_type = st.selectbox(
        "Traffic Type",
        list(range(1,21))
    )

    visitor = st.selectbox(
        "Visitor Type",
        ["New_Visitor","Other","Returning_Visitor"]
    )

    weekend = st.selectbox(
        "Weekend",
        ["No","Yes"]
    )

# ---------------------------
# Encoding
# ---------------------------

month_map = {
    "Aug":0,
    "Dec":1,
    "Feb":2,
    "Jul":3,
    "June":4,
    "Mar":5,
    "May":6,
    "Nov":7,
    "Oct":8,
    "Sep":9
}

visitor_map = {
    "New_Visitor":0,
    "Other":1,
    "Returning_Visitor":2
}

weekend_map = {
    "No":0,
    "Yes":1
}

# ---------------------------
# Prediction
# ---------------------------

if st.button("🔍 Predict Purchase"):

    input_data = pd.DataFrame([[
        administrative,
        administrative_duration,
        informational,
        informational_duration,
        product_related,
        product_related_duration,
        bounce_rates,
        exit_rates,
        page_values,
        special_day,
        month_map[month],
        operating_system,
        browser,
        region,
        traffic_type,
        visitor_map[visitor],
        weekend_map[weekend]
    ]], columns=[
        'Administrative',
        'Administrative_Duration',
        'Informational',
        'Informational_Duration',
        'ProductRelated',
        'ProductRelated_Duration',
        'BounceRates',
        'ExitRates',
        'PageValues',
        'SpecialDay',
        'Month',
        'OperatingSystems',
        'Browser',
        'Region',
        'TrafficType',
        'VisitorType',
        'Weekend'
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    purchase_probability = probability[0][1] * 100

    st.markdown("---")

    if prediction[0] == 1:

        st.success("✅ Customer is likely to Purchase")

        st.metric(
            "Purchase Probability",
            f"{purchase_probability:.2f}%"
        )

        st.subheader("💡 Business Recommendation")

        st.write("✔ No discount required")
        st.write("✔ Recommend premium products")
        st.write("✔ Show Frequently Bought Together")
        st.write("✔ Offer loyalty rewards")
        st.write("✔ Encourage Express Checkout")

    else:

        st.error("❌ Customer is unlikely to Purchase")

        st.metric(
            "Purchase Probability",
            f"{purchase_probability:.2f}%"
        )

        st.subheader("💡 Business Recommendation")

        st.write("🎁 Offer discount coupon")
        st.write("🚚 Provide free delivery")
        st.write("🛍 Recommend popular products")
        st.write("📢 Display personalized advertisements")
        st.write("📧 Send reminder email")

st.markdown("---")
st.caption("Developed by Rathi Meena | Machine Learning Project")
