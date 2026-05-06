import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib # type: ignore
import plotly.graph_objects as go # type: ignore

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System (ডায়াবেটিস পূর্বাভাস ব্যবস্থা)",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {padding: 1rem; border-radius: 0.5rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler_svm.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None


# Header
st.title("🚑 Diabetes Risk Assessment System for Bangladesh (বাংলাদেশের জন্য ডায়াবেটিস ঝুঁকি নির্ণয় ব্যবস্থা)")
st.markdown("### Proactive Diabetes Prevention Tool (ডায়াবেটিস প্রতিরোধের সক্রিয় মাধ্যম)")

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ **Model files not found!**")
    st.info("""
    Please run the following command first:
    ```
    python diabetes-prediction
    ```
    This will train and save the model files.
    """)
    st.stop()

# Function to convert English digits to Bangla digits
def to_bangla_digits(number):
    eng = "0123456789"
    ban = "০১২৩৪৫৬৭৮৯"
    return str(number).translate(str.maketrans(eng, ban))

# Sidebar inputs
st.sidebar.title("⚙️ Patient Information (রোগীর তথ্য)")

st.sidebar.subheader("Demographics (জনসংখ্যাগত তথ্য)")
st.sidebar.markdown("\n")
age = st.sidebar.slider('Age (বয়স)', 21, 82, 30)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত বয়স: {to_bangla_digits(age)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

pregnancies = st.sidebar.number_input('Pregnancies (গর্ভবতীর সংখ্যা)', 0, 12, 0)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত গর্ভবতীর সংখ্যা: {to_bangla_digits(pregnancies)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n\n")

st.sidebar.subheader("Medical Measurements (চিকিৎসা পরিমাপ)")
st.sidebar.markdown("\n")
glucose = st.sidebar.slider('Glucose (রক্তে শর্করা)(mg/dL)', 0, 200, 120)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত রক্তে শর্করা: {to_bangla_digits(glucose)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

bp_systolic = st.sidebar.slider('Systolic Blood Pressure (সিস্টোলিক রক্তচাপ) (mm Hg) ', 0, 130, 70)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত সিস্টোলিক রক্তচাপ: {to_bangla_digits(bp_systolic)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

bp_diastolic = st.sidebar.slider('Diastolic Blood Pressure (ডায়াস্টোলিক রক্তচাপ) (mm Hg) ', 0, 90, 40)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত ডায়াস্টোলিক রক্তচাপ: {to_bangla_digits(bp_diastolic)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

skin = st.sidebar.slider('Skin Thickness (ত্বকের পুরুত্ব) (mm)', 0, 100, 20)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত ত্বকের পুরুত্ব: {to_bangla_digits(skin)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

insulin = st.sidebar.slider('Insulin (ইনসুলিন) (mu U/ml)', 0, 30, 13)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত ইনসুলিন মাত্রা: {to_bangla_digits(insulin)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

bmi = st.sidebar.number_input('Body Mass Index (BMI) (দেহের ভর সূচক)', 10.0, 35.0, 23.0, 0.1)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত BMI(দেহের ভর সূচক): {to_bangla_digits(bmi)}</p>', unsafe_allow_html=True
)
st.sidebar._markdown("\n\n\n\n")

dpf = st.sidebar.slider('Diabetes Pedigree Function (ডায়াবেটিস পারিবারিক ঝুঁকি সূচক)', 0.0, 2.5, 0.5, 0.01)
st.sidebar._markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত ডায়াবেটিস পারিবারিক ঝুঁকি সূচক: {to_bangla_digits(dpf)}</p>', unsafe_allow_html=True
)

# Predict button
st.sidebar._markdown("---\n")
predict_btn = st.sidebar.button("🔮 Predict (পূর্বাভাস দিন)", type="primary", width = 'stretch')

# Main content
if predict_btn:
    # Prepare input
    feature_names = [
    'Pregnancies', 'Age', 'BMI', 'BP_systolic', 'BP_diastolic', 'DiabetesPedigreeFunction', 'Insulin',
    'SkinThickness','Glucose'
    ]
    input_data = pd.DataFrame([[
    pregnancies, age, bmi, bp_systolic, bp_diastolic, dpf, insulin,
    skin, glucose
    ]], columns=feature_names)
    
    # Standardize
    input_std = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_std)[0]
    
    # Get probability if available
    try:
        probability = model.predict_proba(input_std)[0]
        prob_negative = probability[0] * 100
        prob_positive = probability[1] * 100
    except:
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive
    
    # Display results
    st.markdown("---")
    st.header("🎯 Prediction Results (পূর্বাভাস ফলাফল)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction box
        if prediction == 0:
            if prob_positive < 35:
                st.success("#### ✅ LOW RISK - Not Diabetic (কম ঝুঁকি-সুস্থ)")
            else:
                st.warning("#### ⚠️ MODERATE RISK - Not Diabetic (মাঝারি ঝুঁকি-সতর্ক)")
        else:
            if prob_positive > 70:
                st.error("#### 🔴 HIGH RISK - Diabetic (উচ্চ ঝুঁকি-ডায়াবেটিস)")
            else:
                st.warning("#### ⚠️ MODERATE RISK - Diabetic (মাঝারি ঝুঁকি-ডায়াবেটিস)")
        
        # Probabilities
        st.subheader("Probability Breakdown (সম্ভাব্যতা বিশ্লেষণ)")
        pcol1, pcol2 = st.columns(2)
        formatted_prob = f"{prob_positive:.1f}"
        formatted_neg = f"{prob_negative:.1f}"
        pcol1.metric("Non-Diabetic (সুস্থ)", f"{formatted_neg}% ({to_bangla_digits(formatted_neg)}%)")
        pcol2.metric("Diabetic (ডায়াবেটিস)", f"{formatted_prob}% ({to_bangla_digits(formatted_prob)}%)")
    
    with col2:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_positive,
            title={'text': "Risk Level (ঝুঁকি স্তর)"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 35], 'color': "lightgreen"},
                    {'range': [35, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.6,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=320, margin=dict(l=50, r=60, t=50, b=20))
        st.plotly_chart(fig, width = 'stretch')
    
    # Risk factors
    st.markdown("---")
    st.subheader("⚠️ Risk Factor Analysis (ঝুঁকি কারণ বিশ্লেষণ)")
    
    risk_factors = []
    positive_factors = []
    
    if glucose > 125:
        risk_factors.append("🔴 High Glucose Level (>125 mg/dL) (উচ্চ শর্করা লেভেল (>১২৫ মি.গ্রা./ডে.লি.))")
    elif glucose < 100:
        positive_factors.append("🟢 Normal Glucose Level (<100 mg/dL) (স্বাভাবিক শর্করা লেভেল (<১০০ মি.গ্রা./ডে.লি.))")
    
    if bmi > 30:
        risk_factors.append("🔴 High BMI - Obesity (>30) (উচ্চ BMI - প্রায়শই অপারেশনের প্রয়োজন হয়)")
    elif 18.5 <= bmi <= 24.9:
        positive_factors.append("🟢 Healthy BMI (18.5-24.9) (স্বাভাবিক দেহের ভর সূচক পরিসীমা (১৮.৫-২৪.৯))")
    
    if age > 45:
        risk_factors.append("🟡 Age Factor (>45) (বয়সকালের ঝুঁকি (>৪৫))")
    
    if bp_systolic > 125 or bp_diastolic > 80:
        risk_factors.append("🔴 High Blood Pressure (উচ্চ রক্তচাপ)")
    elif 90 <= bp_systolic <= 125 and 60 <= bp_diastolic <= 79:
        positive_factors.append("🟢 Normal Blood Pressure (স্বাভাবিক রক্তচাপ)")
    
    if dpf > 0.5:
        risk_factors.append("🟡 Higher Genetic Predisposition (বংশগত বা জিনগত ঝুঁকি বেশি)")
    
    if risk_factors:
        st.warning("**Identified Risk Factors (ঝুঁকি কারণ):**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    
    if positive_factors:
        st.success("**Positive Health Indicators (ভাল স্বাস্থ্য সূচক):**")
        for factor in positive_factors:
            st.markdown(f"- {factor}")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations (পরামর্শ)")
    
    if prediction == 1:
        st.error("""
        **Important Actions (গুরুত্বপূর্ণ পদক্ষেপ):**
        - Consult a healthcare professional immediately 
                 (অবিলম্বে একজন স্বাস্থ্য বিশেষজ্ঞ বা চিকিৎসকের পরামর্শ নিন।)
        - Get comprehensive diabetes screening 
                 (সম্পূর্ণ ডায়াবেটিস পরীক্ষা করান।)
        - Monitor blood glucose regularly 
                 (নিয়মিত রক্তে শর্করা পরীক্ষা করুন।)
        - Consider lifestyle modifications 
                 (জীবনযাত্রায় পরিবর্তন আনার বিষয়টি বিবেচনা করুন।)
        """)
    else:
        st.success("""
        **Maintain Healthy Practices (স্বাস্থ্যকর অভ্যাস বজায় রাখুন):**
        - Regular health check-ups 
                   (নিয়মিত স্বাস্থ্য পরীক্ষা)
        - Balanced diet 
                   (সুষম খাদ্য)
        - Exercise regularly (30+ min daily) 
                   (নিয়মিত ব্যায়াম করুন (প্রতিদিন ৩০ মিনিট বা তার বেশি))
        - Monitor weight and BMI 
                   (ওজন এবং দেহের ভর সূচক পর্যবেক্ষণ করুন)
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ MEDICAL DISCLAIMER (চিকিৎসা সংক্রান্ত সতর্কতা)**
    
    This prediction is for educational purposes only. It should NOT replace 
    professional medical advice. Always consult qualified healthcare professionals 
    for medical concerns. The developers are not responsible for any health decisions made based on this tool.       
        এই পূর্বাভাস শুধুমাত্র শিক্ষামূলক উদ্দেশ্যে তৈরি করা হয়েছে। এটি পেশাদার চিকিৎসা পরামর্শের বিকল্প নয়। স্বাস্থ্য সংক্রান্ত যেকোনো উদ্বেগের জন্য সর্বদা যোগ্য চিকিৎসা পেশাদারদের সাথে পরামর্শ করুন। এই টুলের উপর ভিত্তি করে নেওয়া কোনো স্বাস্থ্য সিদ্ধান্তের জন্য ডেভেলপাররা দায়ী নয়।
    """)
    
else:
    # Initial page
    st.markdown("---")
    st.info("👈 Enter patient information in the top-left sidebar and click **Predict** (উপরের বাম পাশে থাকা সাইডবারে প্রয়োজনীয় তথ্য দিন এবং **পূর্বাভাস দিন** বাটনে ক্লিক করুন)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type (মডেলের ধরন)", "Random Forest")
    col2.metric("Accuracy (সঠিকতা)", "~89% (প্রায় ৮৯%)")
    col3.metric("Dataset (ডেটাসেট)", "2645 samples")