"""
LinguaGuard-FL Interactive Web Dashboard (Streamlit)
Real-time sentiment inference across Malayalam, Hindi, Tanglish, and Romanized Kannada,
with direct model checkpoint uploading, federated architecture telemetry, and CARA research analytics.
"""

import os
import sys
import zipfile
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from configs.config import DEVICE, LABEL_MAPPING, MAX_SEQ_LENGTH, CHECKPOINT_DIR, MODEL_NAME

st.set_page_config(
    page_title="LinguaGuard-FL | Privacy-Preserving Multilingual NLP",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .positive-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .negative-badge {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .neutral-badge {
        background-color: #fef3c7;
        color: #b45309;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_sentiment_model(model_path_str: str):
    """
    Loads tokenizer and model weights with caching.
    """
    p = Path(model_path_str)
    if p.exists() and (p / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(p))
        model = AutoModelForSequenceClassification.from_pretrained(str(p))
        is_custom = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
        is_custom = False

    model.to(DEVICE)
    model.eval()
    return tokenizer, model, is_custom


def check_local_checkpoints():
    target = CHECKPOINT_DIR / "final_linguaguard_model"
    if target.exists() and (target / "config.json").exists():
        return target
    zip_path = CHECKPOINT_DIR / "final_linguaguard_model.zip"
    if zip_path.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(target)
        return target
    return None


# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=70)
st.sidebar.title("LinguaGuard-FL")
st.sidebar.markdown("**Project by:** Dhanush N (`1RUA24CSE0123`)\n**Guide:** Prof. Rahul Panja\n**Center:** CARA, RV University")
st.sidebar.markdown("---")

st.sidebar.subheader("📦 Model Status")
active_checkpoint = check_local_checkpoints()

model_dir_to_load = str(active_checkpoint) if active_checkpoint else MODEL_NAME
tokenizer, model, is_custom_model = load_sentiment_model(model_dir_to_load)

if is_custom_model:
    st.sidebar.success("🟢 **Trained Model Active**\n\nLoaded directly from project `checkpoints/final_linguaguard_model/`")
else:
    st.sidebar.warning("🟡 Active: Base DistilBERT (No local checkpoint found)")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Federated Configuration")
st.sidebar.write("• **Framework:** Flower (flwr)")
st.sidebar.write("• **Algorithm:** FedAvg")
st.sidebar.write("• **Clients:** 4 (Malayalam, Hindi, Tanglish, Kannada)")
st.sidebar.write("• **Adaptive DP:** Clip=1.0, σ=0.001")
st.sidebar.write("• **Secure Aggregation:** Mask σ=0.0005")

# Header
st.markdown('<div class="main-title">🛡️ LinguaGuard-FL Sentiment Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Privacy-Preserving Multilingual Sentiment Classification for Indic & Dravidian Languages</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Live Sentiment Predictor", "📊 CARA Research Findings", "⚙️ Architecture & Pipeline"])

with tab1:
    st.markdown("### Test Sentiment Analysis Across Languages")
    st.markdown("Input any social media comment or movie review in native script or Latin transliteration:")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    preset_text = ""
    if col_btn1.button("✨ Sample Malayalam"):
        preset_text = "ഈ സിനിമ വളരെ മനോഹരമാണ്! അഭിനയവും പാട്ടുകളും സൂപ്പർ."
    if col_btn2.button("✨ Sample Hindi"):
        preset_text = "फिल्म बहुत ही शानदार और प्रेरणादायक है, बेहतरीन अभिनय!"
    if col_btn3.button("✨ Sample Tamil / Tanglish"):
        preset_text = "Padam vera level bro, Rajini acting semma mass super!"
    if col_btn4.button("✨ Sample Romanized Kannada"):
        preset_text = "Cinema tumba chennagide, acting super aagide!"

    user_text = st.text_area("Enter review or comment:", value=preset_text, height=100)

    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_SEQ_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
                pred_idx = int(probs.argmax())

            res_label = LABEL_MAPPING.get(pred_idx, "Unknown")
            res_conf = float(probs[pred_idx])

            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                st.markdown("#### Prediction Result")
                if pred_idx == 2:
                    badge_html = f'<span class="positive-badge">POSITIVE 😊</span>'
                elif pred_idx == 0:
                    badge_html = f'<span class="negative-badge">NEGATIVE 😞</span>'
                else:
                    badge_html = f'<span class="neutral-badge">NEUTRAL / MIXED 😐</span>'

                st.markdown(badge_html, unsafe_allow_html=True)
                st.write("")
                st.metric(label="Model Confidence", value=f"{res_conf*100:.2f}%")
                st.caption(f"Input language/script detected: Multilingual DistilBERT")

            with col_res2:
                st.markdown("#### Probability Distribution")
                prob_df = pd.DataFrame({
                    "Sentiment": [LABEL_MAPPING[0], LABEL_MAPPING[1], LABEL_MAPPING[2]],
                    "Probability": [probs[0], probs[1], probs[2]]
                })
                try:
                    st.bar_chart(prob_df.set_index("Sentiment"), use_container_width=True)
                except Exception:
                    for i in [2, 1, 0]:
                        st.write(f"**{LABEL_MAPPING[i]}**: {probs[i]*100:.1f}%")
                        st.progress(float(probs[i]))

with tab2:
    st.markdown("### CARA Research Gaps & Empirical Evaluation")
    st.markdown("""
    This project addresses the **triple constraint** (Privacy + Multilingual + Low-Resource) across 4 regional Indian language clients:
    """)

    col_gap1, col_gap2 = st.columns(2)
    with col_gap1:
        st.info("**GAP 2: Low-Resource Dravidian/Indic NLP**\n\nDirectly encompasses Malayalam, Hindi, Tanglish, and Romanized Kannada under decentralized evaluation.")
        st.info("**GAP 4: Formal Privacy Guarantees**\n\nCombines Adaptive Differential Privacy (parameter clipping + Gaussian noise) with Secure Aggregation simulation.")

    with col_gap2:
        st.info("**GAP 3: Non-IID Language Heterogeneity**\n\nSimulates federation across disparate morphological structures, writing systems, and class distributions.")
        st.info("**GAP 7: Code-Switching & Dialectal Social Media**\n\nNatively processes informal Latin transliterated text common to Indian social media.")

    st.markdown("#### Literature Comparison Matrix (Thesis Table 5.2 Extended)")
    comp_df = pd.DataFrame([
        {"System": "FedPerX (Prakash et al., 2026)", "Languages": "10+ Global Languages", "Privacy": "Adaptive DP on Adapters", "Low-Resource Focus": "Partial (No langs <1K samples)"},
        {"System": "MultiFED (Singh & Thakur, 2024)", "Languages": "13 Indic Languages", "Privacy": "None (Decentralization only)", "Low-Resource Focus": "Yes (Indic, no formal privacy)"},
        {"System": "CARA RP02 (Zhao et al., ICLR 2024)", "Languages": "Global benchmarks", "Privacy": "None (Prompt Tuning)", "Low-Resource Focus": "Yes (Prompt tuning)"},
        {"System": "TextHide (Huang et al., 2020)", "Languages": "English (GLUE)", "Privacy": "Representation encryption", "Low-Resource Focus": "No (English GLUE only)"},
        {"System": "LinguaGuard-FL (This Project)", "Languages": "Malayalam, Hindi, Tanglish, Kannada", "Privacy": "Adaptive DP + Secure Aggregation", "Low-Resource Focus": "Yes — Dedicated Indian & Code-Switched"}
    ])
    st.table(comp_df)

with tab3:
    st.markdown("### System Architecture & Pipeline")
    st.markdown("""
    ```mermaid
    graph TD
        A[Central Flower Server: FedAvg] -->|1. Broadcast Global Weights| B[Client 1: Malayalam]
        A -->|1. Broadcast Global Weights| C[Client 2: Hindi]
        A -->|1. Broadcast Global Weights| D[Client 3: Tanglish]
        A -->|1. Broadcast Global Weights| E[Client 4: Romanized Kannada]

        B -->|2. Local Training| B1[Adaptive DP Clipping & Noise]
        C -->|2. Local Training| C1[Adaptive DP Clipping & Noise]
        D -->|2. Local Training| D1[Adaptive DP Clipping & Noise]
        E -->|2. Local Training| E1[Adaptive DP Clipping & Noise]

        B1 -->|3. Masking| B2[Secure Aggregation Simulation]
        C1 -->|3. Masking| C2[Secure Aggregation Simulation]
        D1 -->|3. Masking| D2[Secure Aggregation Simulation]
        E1 -->|3. Masking| E2[Secure Aggregation Simulation]

        B2 -->|4. Transmit Protected Weights| A
        C2 -->|4. Transmit Protected Weights| A
        D2 -->|4. Transmit Protected Weights| A
        E2 -->|4. Transmit Protected Weights| A
    ```
    """)
    st.write("• **Local Epochs per Round:** 2")
    st.write("• **Learning Rate:** 2e-5 (AdamW)")
    st.write("• **Backbone:** distilbert-base-multilingual-cased (134M Parameters)")
