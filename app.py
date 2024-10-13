import streamlit as st
import pickle
import numpy as np

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set page configuration
st.set_page_config(page_title="Laptop Predictor", page_icon="💻", layout="wide")

# Adding a custom CSS to style the app
st.markdown(
    """
    <style>
    body {
        background-color: #F8F9FA;
        color: #343A40;
    }
    .title {
        text-align: center;
        color: #007BFF;
        font-size: 40px;
        font-weight: bold;
        margin: 20px 0;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    /* Style for input elements */
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: transparent;
        border: 2px solid #007BFF;
        border-radius: 5px;
        padding: 10px;
    }
    .stSelectbox:hover, .stSlider:hover, .stNumberInput:hover {
        border-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Laptop Price Predictor</h1>', unsafe_allow_html=True)

# Create input fields
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.0)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', 
                                                 '3840x2160', '3200x1800', 
                                                 '2880x1800', '2560x1600', 
                                                 '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):
    # Prepare query
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create the query array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
    query = query.reshape(1, -1)

    # Make prediction
    try:
        predicted_price = pipe.predict(query)[0]
        st.success(f"Predicted Price: ₹{int(np.exp(predicted_price))}", icon="💰")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}", icon="🚨")
