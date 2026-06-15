import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.layers import Layer

# -------------------------------
# Custom Layer
# -------------------------------
class ECALayer(Layer):
    def __init__(self, k_size=3, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.k_size = k_size
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.conv = tf.keras.layers.Conv1D(filters=1, kernel_size=k_size, padding='same', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        y = self.avg_pool(inputs)
        y = tf.expand_dims(y, axis=1)
        y = self.conv(y)
        y = tf.squeeze(y, axis=1)
        y = self.sigmoid(y)
        return inputs * tf.expand_dims(tf.expand_dims(y, axis=1), axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'k_size': self.k_size})
        return config


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("cassava_final_model.h5", custom_objects={'ECALayer': ECALayer})

model = load_my_model()


# -------------------------------
# Labels
# -------------------------------
disease_names = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)',
    2: 'Cassava Green Mite (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

def get_className(classNo):
    return disease_names.get(classNo, "Unknown Disease")


# -------------------------------
# Prediction
# -------------------------------
def predict_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    return np.argmax(pred), np.max(pred)


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Cassava Disease Detection", layout="centered")


# -------------------------------
# 🔥 PROFESSIONAL CSS (FLASK STYLE)
# -------------------------------
st.markdown("""
<style>

/* REMOVE ALL DEFAULT SPACING */
html, body, [class*="css"] {
    margin: 0;
    padding: 0;
}

/* REMOVE STREAMLIT TOP SPACE */
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
    max-width: 100% !important;
}

/* REMOVE HEADER GAP */
header {visibility: hidden;}
footer {visibility: hidden;}

/* 🔥 PERFECT NAVBAR */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;

    background-color: #2f3e46;
    color: white;

    text-align: center;
    padding: 16px 10px;
    font-size: 17px;
    font-weight: 500;

    z-index: 9999;
}

/* PUSH CONTENT DOWN (IMPORTANT) */
.main-container {
    max-width: 900px;
    margin: auto;
    padding-top: 90px;  /* space below fixed navbar */
}

/* TITLE */
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 600;
    margin-bottom: 25px;
}

/* UPLOADER CENTER */
[data-testid="stFileUploader"] {
    display: flex;
    justify-content: center;
}

/* IMAGE CENTER */
.image-box {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

/* RESULT */
.result-box {
    text-align: center;
    font-size: 28px;
    margin-top: 20px;
}

/* BUTTON */
.stButton>button {
    display: block;
    margin: 20px auto;
    background-color: #2e7d32;
    color: white;
    padding: 10px 30px;
    font-size: 16px;
    border-radius: 5px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# NAVBAR (FULL WIDTH)
# -------------------------------
st.markdown(
    '<div class="navbar">Vocational Training Project : Cassava Leaf Disease Classification Using Deep Learning</div>',
    unsafe_allow_html=True
)

# -------------------------------
# MAIN CONTENT
# -------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title
st.markdown(
    '<div class="title">Cassava Leaf Disease Classification Using Deep Learning</div>',
    unsafe_allow_html=True
)

# uploader + rest of your code...

st.markdown('</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

# -------------------------------
# DISPLAY + PREDICTION
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    st.markdown('<div class="image-box">', unsafe_allow_html=True)
    st.image(image, width=420)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict"):
        with st.spinner("Analyzing leaf... 🌿"):
            class_index, confidence = predict_image(image)
            result = get_className(class_index)

        st.markdown(
            f'<div class="result-box">Result: {result}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="conf-box">Confidence: {confidence*100:.2f}%</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)