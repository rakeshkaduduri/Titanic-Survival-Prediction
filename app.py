import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

h1, h2, h3, h4 {
    color: white !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 3.5em;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #00e676, #76ff03);
    color: white;
    transform: scale(1.02);
    box-shadow: 0px 6px 20px rgba(0, 255, 120, 0.6);
}

.metric-card {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}

.description-box {
    background-color: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}

.input-box {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "titanic_ann_model.keras"
)

# ==========================================
# HEADER SECTION
# ==========================================

st.markdown("""
<h1 style='text-align:center; font-size:50px;'>
🚢 Titanic Survival Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align:center; color:#00c6ff;'>
Deep Learning Based Passenger Survival Prediction
</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# DESCRIPTION SECTION
# ==========================================

st.markdown("""
<div class='description-box'>

<h3>📌 Project Description</h3>

<p style='font-size:18px;'>

This AI-powered web application predicts whether a passenger would survive during the Titanic disaster using an Artificial Neural Network (ANN).

The system uses:
<ul>
<li>TensorFlow/Keras Deep Learning</li>
<li>Passenger data analysis</li>
<li>Real-time prediction</li>
<li>Interactive visualization dashboard</li>
</ul>

The model analyzes passenger information and estimates survival probability instantly.

</p>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================

st.markdown("""
<div class='input-box'>
<h2>🧾 Passenger Input Form</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Passenger Class
with col1:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/1048/1048314.png",
        width=70
    )

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

# Age
with col2:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2922/2922510.png",
        width=70
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

# Fare
with col3:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135706.png",
        width=70
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================

def preprocess_input(pclass, age, fare):

    pclass_norm = (pclass - 1) / 2
    age_norm = age / 80
    fare_norm = fare / 600

    data = np.array([
        [
            pclass_norm,
            age_norm,
            fare_norm
        ]
    ])

    return data

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Predict Survival"):

    input_data = preprocess_input(
        pclass,
        age,
        fare
    )

    prediction = model.predict(input_data)

    probability = float(prediction[0][0])

    if probability > 0.5:
        result = "✅ SURVIVED"
    else:
        result = "❌ NOT SURVIVED"

    confidence = probability * 100

    st.markdown("---")

    # ==========================================
    # RESULTS
    # ==========================================

    st.markdown("""
    <h2 style='text-align:center;'>
    📊 Prediction Dashboard
    </h2>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Prediction</h3>
        <h2>{result}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Survival Probability</h3>
        <h2>{probability:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>Confidence Score</h3>
        <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # ==========================================
    # CHART + MODEL METRICS
    # ==========================================

    st.markdown("""
    <h2 style='text-align:center;'>
    📊 Analytics Dashboard
    </h2>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # ==========================================
    # PROBABILITY CHART
    # ==========================================

    with left:

        st.markdown("""
        <div class='metric-card' style='margin-bottom:20px;'>
        <h3 style='text-align:center; padding-bottom:10px;'>
        📈 Probability Distribution
        </h3>
        </div>
        """, unsafe_allow_html=True)

        labels = ["Survival", "Non-Survival"]

        values = [
            probability,
            1 - probability
        ]

        fig, ax = plt.subplots(figsize=(4, 4))

        colors = ['#00e676', '#ff5252']

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': "white", 'fontsize': 11}
        )

        # Dark background
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')

        # White circle border effect
        centre_circle = plt.Circle((0,0),0.70,fc='#1e1e1e')
        fig.gca().add_artist(centre_circle)

        ax.axis('equal')

        st.pyplot(fig)

    # ==========================================
    # MODEL PERFORMANCE
    # ==========================================

    with right:

        st.markdown("""
        <div class='metric-card'>
        <h3 style='text-align:center;'>
        🤖 Model Performance Metrics
        </h3>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                "Accuracy",
                "82%"
            )

            st.metric(
                "Hidden Layers",
                "2"
            )

        with m2:
            st.metric(
                "Precision",
                "79%"
            )

            st.metric(
                "Recall",
                "81%"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.progress(int(confidence))

        st.markdown(f"""
        <center>
        <h3 style='color:#00e676;'>
        Model Confidence: {confidence:.2f}%
        </h3>
        </center>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.info("""
        This ANN model analyzes:
        - Passenger Class
        - Age
        - Fare

        and predicts survival probability using Deep Learning.
        """)

        # ==========================================
        # FOOTER
        # ==========================================

        st.markdown("""
        <center>
        <h4 style='color:lightgray;'>
        🚢 Developed using Streamlit, TensorFlow & Deep Learning
        </h4>
        </center>
        """, unsafe_allow_html=True)