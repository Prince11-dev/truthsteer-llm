<div align="center">

# 🛡️ TruthSteer-LLM
### *Real-Time Internal Activation Steering & Hallucination Suppression Engine for Large Language Models*

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 📋 Executive Summary

Large Language Models frequently suffer from confident hallucinations during generation. Conventional mitigation strategies—such as massive fine-tuning pipelines, Reinforcement Learning from Human Feedback (RLHF), or complex prompt-engineering hacks—impose prohibitive computational overhead, increase latency, and degrade model agility.

**TruthSteer-LLM** is a production-grade inference optimization and alignment framework that intervenes directly at the **internal representation level**. By isolating directional truth vectors within transformer residual streams and dynamically modulating hidden states via PyTorch forward hooks, this engine suppresses hallucinations in real time **without requiring a single parameter update or model retraining cycle**.

---

## 🏗️ System Architecture & Core Subsystems

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         TruthSteer-LLM Engine                          │
 ├─────────────────────────┬──────────────────────────────┬───────────────┤
 │  1. Vector Extraction   │    2. Confidence Probing     │ 3. Inference  │
 │  (Contrastive States)   │    (Logistic Regression)     │ (Fwd Hooks)   │
 └───────────┬─────────────┴──────────────┬───────────────┴───────┬───────┘
             │                            │                       │
             ▼                            ▼                       ▼
   `data/vectors/*.pt`         `data/probes/*.pkl`       FastAPI / Streamlit

```

### 1. Contrastive Vector Extraction (`vector_extractor.py`)

* Captures hidden state activations across target transformer layers (e.g., Layer 14 on LLaMA-3-8B).
* Contrasts residual stream representations between known factual prompts and hallucinated prompts.
* Computes and normalizes a directional **Truth Vector** encapsulating factual alignment.

### 2. Dynamic Confidence Probing (`probe_trainer.py`)

* Trains a lightweight linear logistic regression probe on internal hidden states.
* Calculates real-time factual confidence probabilities ($\tau$) during inference token generation.

### 3. Hook-Based Activation Steering (`steering_engine.py`)

* Registers PyTorch pre-forward hooks (`pre_hooks`) to inspect and manipulate residual stream states on the fly.
* Applies targeted steering multipliers ($\alpha$) selectively when internal confidence drops below the validation threshold.

---

## 📂 Project Topology

```text
truthsteer-llm/
├── data/
│   ├── vectors/             # Serialized contrastive truth direction tensors (.pt)
│   └── probes/              # Trained logistic regression probe checkpoints (.pkl)
├── src/
│   ├── __init__.py
│   ├── config.py            # Global system hyperparameters & paths
│   ├── vector_extractor.py  # Residual stream contrastive activation pipeline
│   ├── probe_trainer.py     # Linear probe model training script
│   ├── steering_engine.py   # PyTorch forward hook activation steering module
│   └── server.py            # Asynchronous FastAPI streaming backend (SSE)
├── requirements.txt         # Pinned Python package dependencies
├── .env.example             # Configuration environment variable template
└── dashboard.py             # Interactive Streamlit side-by-side diagnostic UI

```

---

## ⚙️ Technical Specifications & Tech Stack

* **Deep Learning & Core ML:** PyTorch, Hugging Face `transformers`, Scikit-Learn, NumPy
* **Backend & API:** FastAPI, Uvicorn, Pydantic v2, Server-Sent Events (SSE) for token streaming
* **Interface & Telemetry:** Streamlit, Requests
* **Execution Environment:** Windows / Linux compatible, optimized for NVIDIA CUDA / Apple Silicon / CPU fallback

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository

```bash
git clone [https://github.com/Prince11-dev/truthsteer-llm.git](https://github.com/Prince11-dev/truthsteer-llm.git)
cd truthsteer-llm

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Configure Environment Variables

Copy the template configuration file and adjust settings as necessary:

```bash
copy .env.example .env

```

### 4. Execute Vector Extraction & Probe Training Pipeline

Generate contrastive truth vectors and train the linear confidence classifier prior to launching the server:

```bash
$env:PYTHONPATH = "."
python -m src.vector_extractor
python -m src.probe_trainer

```

---

## 🖥️ Running the Services

### Terminal 1: Launch the FastAPI Asynchronous Backend

```bash
$env:PYTHONPATH = "."
python src/server.py

```

*(The inference engine runs locally at `http://localhost:8000`).*

### Terminal 2: Launch the Streamlit Diagnostic Dashboard

Open a separate terminal window to start the interactive evaluation UI:

```bash
cd C:\Users\girip\truthsteer-llm
$env:PYTHONPATH = "."
streamlit run dashboard.py

```

---

## 💡 Enterprise Value & Key Advantages

* **Zero Retraining Overhead:** Bypasses costly multi-GPU cluster training sessions by executing purely at inference time.
* **Preserved Linguistic Fluency:** Surgical vector intervention targets factual grounding without degrading stylistic diversity or creative expression.
* **Production-Ready Modularity:** Strictly typed Python codebases built with robust error handling, Pydantic validation, and clean asynchronous microservice design.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for more details.

```

```
