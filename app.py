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
# Labels & Info Database
# -------------------------------
disease_info = {
    0: {
        'name': 'Cassava Bacterial Blight (CBB)',
        'color': '#ef4444', # Red
        'description': 'A serious bacterial disease caused by Xanthomonas axonopodis pv. manihotis. It leads to leaf wilting, angular leaf spots, and gum exudation on stems.',
        'tips': [
            'Plant resistant cassava varieties.',
            'Use disease-free stem cuttings for planting.',
            'Practice crop rotation (avoid planting cassava consecutively in the same field).',
            'Sanitize tools and remove infected plants from the field.'
        ]
    },
    1: {
        'name': 'Cassava Brown Streak Disease (CBSD)',
        'color': '#f97316', # Orange
        'description': 'A viral disease transmitted by whiteflies and infected cuttings. It causes necrosis in the roots, yellowing along leaf veins, and brown streaks on stems.',
        'tips': [
            'Rogue (uproot and destroy) infected plants immediately.',
            'Select virus-free planting materials.',
            'Plant tolerant or resistant varieties.',
            'Control whitefly vectors using recommended pest management.'
        ]
    },
    2: {
        'name': 'Cassava Green Mite (CGM)',
        'color': '#eab308', # Yellow
        'description': 'A pest infestation caused by Mononychellus tanajoa. Mites feed on the undersides of young leaves, causing yellow spots, leaf deformation, and stunted growth.',
        'tips': [
            'Release predatory mites (biological control) to manage CGM populations.',
            'Plant early at the start of the wet season to allow plants to establish before mite populations peak.',
            'Use healthy, vigorous planting cuttings.'
        ]
    },
    3: {
        'name': 'Cassava Mosaic Disease (CMD)',
        'color': '#ec4899', # Pink
        'description': 'A widespread viral disease, transmitted by whiteflies and cutting propagation. It causes chlorosis (yellowing), leaf distortion, and severe stunting.',
        'tips': [
            'Cultivate certified CMD-resistant cassava varieties.',
            'Regularly inspect fields and rogue out infected plants during the first 3 months.',
            'Do not take cuttings from plants showing symptoms.'
        ]
    },
    4: {
        'name': 'Healthy Leaf',
        'color': '#10b981', # Green
        'description': 'The leaf shows no symptoms of CBB, CBSD, CGM, or CMD. It appears healthy and well-nourished.',
        'tips': [
            'Continue practicing good soil management and irrigation.',
            'Monitor fields regularly for early signs of disease vectors like whiteflies.',
            'Ensure proper spacing and weed control to promote strong airflow.'
        ]
    }
}

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
st.set_page_config(page_title="Cassava Leaf Health Classifier", layout="centered", page_icon="🌿")

# -------------------------------
# 🔥 DYNAMIC GLASSMORPHISM CSS
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Main App Container background */
.stApp {
    background: linear-gradient(135deg, #090d16 0%, #111827 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}

/* Custom header container */
.header-container {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 2rem;
    backdrop-filter: blur(8px);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto;
}

/* Custom card container for Uploader and Images */
.custom-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

/* File uploader overrides */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 2px dashed rgba(16, 185, 129, 0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #10b981 !important;
    background: rgba(16, 185, 129, 0.02) !important;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.1) !important;
}

/* Button override */
.stButton>button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
}

.stButton>button:active {
    transform: translateY(0) !important;
}

/* Result Card */
.result-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.result-title {
    font-size: 1.5rem;
    font-weight: 600;
}

.result-description {
    color: #cbd5e1;
    margin-top: 0.5rem;
    font-size: 0.95rem;
    line-height: 1.5;
}

.confidence-section {
    margin: 1.2rem 0;
}

.confidence-label {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}

.progress-bar-bg {
    background: rgba(255, 255, 255, 0.1);
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 1s ease-in-out;
}

.tips-section {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1.2rem;
}

.tips-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #34d399;
    margin-bottom: 0.5rem;
}

.tips-list {
    margin: 0;
    padding-left: 1.2rem;
    color: #cbd5e1;
}

.tips-list li {
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #070a13 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.sidebar-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #10b981;
    margin-bottom: 1rem;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.75rem;
}

.sidebar-card-title {
    font-weight: 600;
    color: #f8fafc;
    font-size: 0.85rem;
}

.sidebar-card-desc {
    color: #94a3b8;
    font-size: 0.80rem;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🌿 Diagnosis Guide</div>', unsafe_allow_html=True)
    st.markdown("This model identifies 5 leaf categories:")
    for class_id, info in disease_info.items():
        st.markdown(f"""
        <div class="sidebar-card" style="border-left: 3px solid {info['color']};">
            <div class="sidebar-card-title">{info['name']}</div>
            <div class="sidebar-card-desc">{info['description'][:85]}...</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **Model Architecture:**  
    CNN with **ECA-Net** Attention.  
    
    **Project Type:**  
    Deep Learning Research for Plant Pathology.
    """)

# -------------------------------
# APP HEADER
# -------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🌿 Cassava Leaf Health Classifier</div>
    <div class="header-subtitle">Analyze cassava leaves to detect diseases and view actionable crop care suggestions instantly.</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# UPLOAD SECTION
# -------------------------------
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PREDICTION PROCESS
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.markdown('<div class="custom-card" style="text-align: center;">', unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Run Diagnostic"):
        with st.spinner("Analyzing leaf patterns using ECA-Net... 🌿"):
            class_index, confidence = predict_image(image)
            info = disease_info.get(class_index)
            tips_html = "".join([f"<li>{tip}</li>" for tip in info['tips']])
            
            # Custom styled card for prediction results
            st.markdown(f"""
            <div class="result-card" style="border-left: 6px solid {info['color']};">
                <div class="result-title" style="color: {info['color']}">{info['name']}</div>
                <div class="result-description">{info['description']}</div>
                <div class="confidence-section">
                    <div class="confidence-label">Diagnosis Confidence: <b>{confidence*100:.2f}%</b></div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: {confidence*100}%; background: {info['color']};"></div>
                    </div>
                </div>
                <div class="tips-section">
                    <div class="tips-title">📋 Recommended Management Actions:</div>
                    <ul class="tips-list">
                        {tips_html}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)