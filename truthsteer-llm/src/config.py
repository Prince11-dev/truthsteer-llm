import os
from dataclasses import dataclass
import torch

@dataclass
class SystemConfig:
    model_name: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # Steering Hyperparameters
    target_layer: int = int(os.getenv("TARGET_LAYER", "14"))
    steering_alpha: float = float(os.getenv("STEERING_ALPHA", "2.5"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))
    
    # Data Paths
    vector_save_path: str = os.getenv("VECTOR_SAVE_PATH", "data/vectors/truth_vector_layer14.pt")
    probe_save_path: str = os.getenv("PROBE_SAVE_PATH", "data/probes/linear_probe_layer14.pkl")
    
    # Server & API Config
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

config = SystemConfig()
