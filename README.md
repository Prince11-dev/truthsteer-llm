# 🛡️ TruthSteer-LLM

> **Real-time internal activation steering and hallucination suppression engine for Large Language Models.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔍 Overview

Fine-tuning LLMs or relying solely on prompt engineering to prevent hallucinations is often expensive, slow, and computationally heavy. **TruthSteer-LLM** provides a surgical, inference-time alternative by manipulating a model's internal residual stream activations directly during token generation. 

By isolating a "truth direction" vector and applying dynamic confidence probing via PyTorch forward hooks, the system suppresses hallucinations in real-time **without requiring any model retraining, weight updates, or expensive fine-tuning cycles**.

---

## 📐 Architecture & How It Works

1. **Contrastive Vector Extraction (`vector_extractor.py`)**: Contrasts residual stream hidden states between factual and hallucinated prompts across target transformer layers (e.g., Layer 14 on LLaMA-3-8B) to compute a normalized steering vector.
2. **Dynamic Confidence Probing (`probe_trainer.py`)**: Trains a lightweight logistic regression linear probe on internal model representations to calculate factual confidence dynamically ($\tau$).
3. **Inference-Time Steering (`steering_engine.py`)**: Registers PyTorch forward hooks (`pre_hooks`) to inject steering multipliers ($\alpha$) *only* when the model's internal confidence drops below the set factual threshold.
4. **Full-Stack Interface (`server.py` & `dashboard.py`)**: Exposes an asynchronous streaming API built with **FastAPI** (supporting Server-Sent Events), paired with a side-by-side comparison dashboard built with **Streamlit**.

---

## 🛠️ Tech Stack

* **Core AI / ML:** PyTorch, Hugging Face Transformers, Scikit-Learn, NumPy
* **Backend API:** FastAPI, Uvicorn, Pydantic, Server-Sent Events (SSE)
* **Frontend / UI:** Streamlit, Requests
* **Infrastructure:** Python Virtual Environments, Git

---

## 📂 Project Structure

```text
truthsteer-llm/
├── data/
│   ├── vectors/             # Saved contrastive truth vectors (.pt)
│   └── probes/              # Trained logistic regression probe checkpoints (.pkl)
├── src/
│   ├── __init__.py
│   ├── config.py            # System configuration and hyperparameters
│   ├── vector_extractor.py  # Residual stream contrastive activation extraction
│   ├── probe_trainer.py     # Linear probe training pipeline
│   ├── steering_engine.py   # PyTorch forward hook activation steering engine
│   └── server.py            # FastAPI asynchronous streaming server
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable templates
└── dashboard.py             # Streamlit comparison UI
