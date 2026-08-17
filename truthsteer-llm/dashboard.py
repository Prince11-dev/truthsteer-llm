import streamlit as st
import requests
import json

st.set_page_config(page_title="TruthSteer-LLM Dashboard", layout="wide")

st.title("TruthSteer-LLM: Internal Activation Steering Engine")
st.markdown("Real-time hallucination suppression via PyTorch forward hooks in residual streams.")

st.sidebar.header("Inference & Steering Parameters")
api_url = st.sidebar.text_input("FastAPI Server URL", "http://localhost:8000")
target_layer = st.sidebar.slider("Target Intervention Layer", min_value=0, max_value=31, value=14)
alpha = st.sidebar.slider("Steering Multiplier (α)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
threshold = st.sidebar.slider("Factuality Confidence Threshold (τ)", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
max_tokens = st.sidebar.slider("Max New Tokens", min_value=32, max_value=512, value=256)

prompt = st.text_area(
    "Input Prompt:",
    "What happens if you swallow watermelon seeds?",
    height=100
)

col1, col2 = st.columns(2)

if st.button("Generate Comparison Stream", type="primary"):
    with col1:
        st.subheader("Baseline Output (Steering OFF: α = 0.0)")
        unsteered_box = st.empty()
        
    with col2:
        st.subheader(f"Steered Output (Layer {target_layer}, α = {alpha})")
        steered_box = st.empty()

    payload_unsteered = {
        "prompt": prompt,
        "layer": target_layer,
        "alpha": 0.0,
        "threshold": threshold,
        "max_tokens": max_tokens
    }
    
    payload_steered = {
        "prompt": prompt,
        "layer": target_layer,
        "alpha": alpha,
        "threshold": threshold,
        "max_tokens": max_tokens
    }

    unsteered_text = ""
    try:
        response_unsteered = requests.post(f"{api_url}/api/v1/generate", json=payload_unsteered, stream=True)
        for line in response_unsteered.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    data = json.loads(decoded[6:])
                    unsteered_text += data["token"]
                    unsteered_box.markdown(unsteered_text)
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")

    steered_text = ""
    try:
        response_steered = requests.post(f"{api_url}/api/v1/generate", json=payload_steered, stream=True)
        for line in response_steered.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    data = json.loads(decoded[6:])
                    steered_text += data["token"]
                    steered_box.markdown(steered_text)
    except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")
