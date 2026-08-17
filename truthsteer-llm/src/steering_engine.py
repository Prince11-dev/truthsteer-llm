import os
import pickle
import torch
from typing import Generator, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from src.config import config

class SteeredLLMEngine:
    def __init__(self, model_name: str = config.model_name):
        print(f"Loading base model {model_name} in {config.dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=config.dtype,
            device_map=config.device,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        self.steering_vector: Optional[torch.Tensor] = None
        self.probe: Optional[Any] = None
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(config.vector_save_path):
            self.steering_vector = torch.load(config.vector_save_path, map_location=config.device).to(config.dtype)
            print(f"Loaded steering vector from {config.vector_save_path}")
        else:
            print(f"Warning: Steering vector file not found at {config.vector_save_path}. Running without vector steering.")

        if os.path.exists(config.probe_save_path):
            with open(config.probe_save_path, "rb") as f:
                self.probe = pickle.load(f)
            print(f"Loaded linear probe from {config.probe_save_path}")

    def _activation_steering_hook(self, alpha: float, threshold: float):
        def hook_fn(module, inputs):
            hidden_states = inputs[0]
            
            if self.steering_vector is not None and alpha > 0.0:
                last_token_act = hidden_states[:, -1, :].detach().cpu().numpy()
                
                should_steer = True
                if self.probe is not None:
                    prob_factual = self.probe.predict_proba(last_token_act)[0][1]
                    should_steer = prob_factual < threshold

                if should_steer:
                    vec = self.steering_vector.to(hidden_states.device)
                    hidden_states[:, -1, :] = hidden_states[:, -1, :] + (alpha * vec)
            
            return (hidden_states,) + inputs[1:]
        return hook_fn

    def generate_steered_stream(
        self,
        prompt: str,
        layer_idx: int = config.target_layer,
        alpha: float = config.steering_alpha,
        threshold: float = config.confidence_threshold,
        max_new_tokens: int = 256
    ) -> Generator[Dict[str, Any], None, None]:
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(config.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        target_layer = self.model.model.layers[layer_idx]
        hook_handle = target_layer.register_forward_pre_hook(self._activation_steering_hook(alpha, threshold))

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        try:
            for new_text in streamer:
                yield {
                    "token": new_text,
                    "layer": layer_idx,
                    "alpha_applied": alpha
                }
        finally:
            hook_handle.remove()
